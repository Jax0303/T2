#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""검색된 셀 문장을 쿼리에 붙여 답하게 하고 EM을 잰다.

가설 (사용자 지시, 2026-09-02): 셀 조회 쿼리에서 R@1이 .9면 셀 문장 검색이 잘 된
것이고, 그 문장을 쿼리에 붙여 답하게 하면 답변 EM도 R@1에 맞춰 올라야 한다.

이 스크립트는 그 가설을 **같은 질의 위에서** 검증 가능한 형태로 만든다. 조건마다
리더를 한 번씩 부르고, 같은 파일에 그 질의의 검색 결과(gold 셀 순위, top-k 안에
들어왔는지)를 같이 적는다. 그래서 사후에 EM을 검색 성공/실패로 갈라볼 수 있다:

  cond=gold  -- gold 셀 문장만 주입. 검색이 완벽했을 때의 천장.
  cond=topK  -- 검색 상위 K개 셀 문장 주입 (K=1,5,10). 실제 파이프라인.

`analysis/multicell_ceiling.py`와 같은 리더·디코딩·채점기를 쓴다 (Qwen2.5-7B-Instruct
rev pinned, 4-bit NF4, temp=0, seed=42, max_new_tokens=32, `phase4_summary.em`).
프롬프트 문자열도 같다. 다른 점 하나: 주입 문장이 P4_path_cell 청크가 아니라 **색인
단위 그대로의 S3c 셀 문장**이다 -- 검색이 고른 것과 리더가 읽는 것을 같게 두려는
것이고, 그래서 cond=gold 값은 §2 천장과 소수점이 다를 수 있다.

CLAUDE.md §3이 미리 적어둔 기전: 검색을 늘리면 distractor도 같이 들어온다
(gold만 48토큰 EM .888 vs 검색 509토큰 .604). 이 실행은 그 비용을 **파인튜닝된
인코더에서** 다시 재는 것이다.

  PYTHONPATH=. .venv/bin/python analysis/retrieved_answer_em.py \
      --population hitab_dev_lookup_all --embed-model models/bge-cell-ft-x0 \
      --alpha 0.9 --out results/answer_ret/ft_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np                                                   # noqa: E402
import corpus_dump_vs_cell as cdv                                    # noqa: E402
from cell_rank_dump import cell_texts                                # noqa: E402
from header_path_coverage import load_corpus                         # noqa: E402
from phase4_summary import em, em_lenient                             # noqa: E402
from qwen_equiv_k import MODEL, SYS                                  # noqa: E402
from rag_agent.retrieve.encoders import default_encoder              # noqa: E402
from rag_agent.retrieve.hybrid_index import HybridIndex, _minmax     # noqa: E402
from rag_agent.serialization.base import Chunk                       # noqa: E402

SEED, MAXNEW = 42, 32


def load_reranker(name):
    """기성품 크로스인코더. 리더보다 먼저 쓰고 먼저 놓아준다 (8GB 카드)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, max_length=512)


def build_jobs(a, C, chunks, ix, ks):
    """(query, cond) 별 주입 문맥. 검색 점수는 질의당 한 번만 계산한다."""
    pos_of = {c: n for n, c in enumerate(C.cell_owner)}
    ce = load_reranker(a.rerank_model) if a.rerank_ks else None
    jobs = []
    t0 = time.time()
    for n, q in enumerate(C.queries, 1):
        gold = [pos_of[g] for g in sorted(q["gold_cells"]) if g in pos_of]
        if not gold:
            continue
        bm = ix._bm25_scores(q["question"])
        dn = ix._dense_scores(q["question"])
        cs = (dn if a.alpha == 1.0 else bm if a.alpha == 0.0
              else a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(bm))
        order = np.argsort(-cs)
        at = np.empty(len(order), dtype=np.int64)
        at[order] = np.arange(len(order))
        ranks = sorted(int(at[g]) for g in gold)
        common = {"query_id": q["query_id"], "question": q["question"],
                  "gold_answer": str(q["answer"]), "m": len(gold),
                  "gold_ranks": ranks}
        conds = [("gold", gold)] if a.gold else []
        conds += [(f"top{k}", [int(p) for p in order[:k]]) for k in ks]
        # orcK = 완벽한 재정렬의 상한. 검색 top-K 안에 gold 가 **전부** 들어왔으면
        # 그것만 주입하고, 아니면 재정렬기가 할 수 있는 게 없으므로 1위를 준다.
        # 재정렬기는 검색이 못 찾은 셀을 만들어낼 수 없다는 뜻이다.
        conds += [(f"orc{k}", gold if ranks[-1] < k else [int(order[0])])
                  for k in (a.oracle_ks or [])]
        # rrK = 검색 top-K 를 크로스인코더로 다시 정렬하고 상위 n 개만 주입한다.
        # 풀이 고정이므로 hit@K 는 재정렬 전과 **정확히 같아야** 한다 (산술 통제).
        for K in (a.rerank_ks or []):
            pool = [int(p) for p in order[:K]]
            sc = ce.predict([(q["question"], chunks[p].text) for p in pool])
            rr = [p for _, p in sorted(zip(-np.asarray(sc), pool),
                                       key=lambda t: (t[0], t[1]))]
            at = {p: r for r, p in enumerate(rr)}
            # gold 는 이미 코퍼스 위치다 (pos_of 를 한 번 통과했다)
            common[f"rr{K}_gold_rank"] = min((at[g] for g in gold if g in at),
                                             default=10 ** 9)
            conds += [(f"rr{K}" if n == 1 else f"rr{K}_{n}", rr[:n])
                      for n in a.rerank_inject]
        for cond, cells in conds:
            jobs.append(common | {
                "cond": cond, "n_injected": len(cells),
                "gold_in_ctx": sum(1 for g in gold if g in set(cells)),
                "ctx": "\n".join(chunks[p].text for p in cells)})
        if n % 100 == 0:
            print(f"  [rank] {n}/{len(C.queries)}  {time.time()-t0:.0f}s", flush=True)
    return jobs


def summarize(path):
    """EM을 조건별로, 그리고 검색 성공 여부로 갈라 찍는다."""
    recs = [json.loads(l) for l in open(path)]
    for r in recs:
        r["is_correct"] = em(r["pred_parsed"], r["gold_answer"])
        # F: em_lenient 은 진단 전용 상위집합이다. 주지표는 여전히 is_correct.
        r["is_correct_lenient"] = em_lenient(r["pred_parsed"], r["gold_answer"])
    by = defaultdict(list)
    for r in recs:
        by[r["cond"]].append(r)
    print(f"\n{'cond':>8}{'n':>6}{'EM':>9}{'EMlenient':>9}{'gold전부주입':>13}"
          f"{'EM|주입됨':>11}{'EM|안됨':>10}{'ptok중앙':>10}")
    for cond in sorted(by, key=lambda c: (c != "gold", c[:3], len(c), c)):
        rs = by[cond]
        n = len(rs)
        full = [r for r in rs if r["gold_in_ctx"] == r["m"]]
        miss = [r for r in rs if r["gold_in_ctx"] < r["m"]]
        f = lambda s, k="is_correct": (sum(r[k] for r in s) / len(s)) if s else float("nan")
        pt = sorted(r["prompt_tokens"] for r in rs)[n // 2]
        print(f"{cond:>8}{n:>6}{f(rs):>9.4f}{f(rs, 'is_correct_lenient'):>9.4f}"
              f"{len(full)/n:>13.4f}{f(full):>11.4f}{f(miss):>10.4f}{pt:>10}")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--population", default="hitab_dev_lookup_all")
    ap.add_argument("--embed-model", default="models/bge-cell-ft-x0")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--cell-scheme", default="S3c")
    ap.add_argument("--data-dir", default="data/hitab")
    ap.add_argument("--cache-dir", default=".cache/corpus_dump_vs_cell")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--oracle-ks", type=int, nargs="*", default=[],
                    help="orcK 조건: top-K 안에 gold 가 전부 있으면 gold 만 주입. "
                         "완벽한 재정렬의 상한이다.")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-large",
                    help="rrK 조건이 쓰는 기성품 크로스인코더.")
    ap.add_argument("--rerank-ks", type=int, nargs="*", default=[],
                    help="rrK 조건: 검색 top-K 를 크로스인코더로 다시 정렬한다. "
                         "top-K 밖의 셀은 만들어낼 수 없으므로 hit@K 가 상한이다. "
                         "PREREG-2026-09-03-ce-rerank-r1.md")
    ap.add_argument("--rerank-inject", type=int, nargs="*", default=[1, 3],
                    help="재정렬 후 상위 몇 개를 주입하는가. 1 은 `rrK`, "
                         "n>1 은 `rrK_n` 이라는 조건이 된다.")
    ap.add_argument("--title-mode", default="raw", choices=["raw", "page"],
                    help="색인·주입 문장의 제목 슬롯. 인코더 학습 때와 같아야 한다.")
    ap.add_argument("--gold", type=int, default=1, help="gold 조건도 돌린다")
    ap.add_argument("--out", default="results/answer_ret/run.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="문맥만 만들고 리더를 부르지 않는다 (자체 점검 포함)")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    a.dataset, a.seed, a.rhb_question_types, a.rhb_em_only = "hitab", SEED, [], False

    out = Path(a.out)
    if a.summary_only:
        summarize(out)
        return 0

    C = load_corpus(a)
    print(f"[corpus] {len(C.tids)} tables / {len(C.cell_owner)} cells | "
          f"[pop] {len(C.queries)} queries", flush=True)
    chunks = [Chunk(table_id=t, chunk_id=f"c::{t}::{i}:{j}", text=x,
                    scheme=a.cell_scheme, kind="cell")
              for x, (t, i, j) in zip(cell_texts(C, a.cell_scheme, a.title_mode),
                                      C.cell_owner)]
    enc = cdv._CachedEncoder(default_encoder(model_name=a.embed_model), a.cache_dir,
                             f"hitab_{a.split}_{a.embed_model}")
    t0 = time.time()
    ix = HybridIndex(chunks, encoder=enc, alpha=0.5)
    print(f"[index] built in {time.time()-t0:.0f}s", flush=True)

    jobs = build_jobs(a, C, chunks, ix, a.ks)
    if a.dry_run:
        # 자체 점검: gold 조건은 gold 셀을 전부 담고, topK는 정확히 K줄이다
        for j in jobs:
            if j["cond"] == "gold":
                assert j["gold_in_ctx"] == j["m"], j["query_id"]
            elif j["cond"].startswith("orc"):
                k = int(j["cond"][3:])
                # gold 가 top-K 안이면 gold 전부, 아니면 1개(1위)만 들어간다
                assert (j["gold_in_ctx"] == j["m"]) == (max(j["gold_ranks"]) < k), \
                    (j["query_id"], j["cond"])
            elif j["cond"].startswith("rr"):
                n = int(j["cond"].split("_")[1]) if "_" in j["cond"] else 1
                assert j["n_injected"] == n, (j["query_id"], j["cond"])
            else:
                k = int(j["cond"][3:])
                assert j["n_injected"] == k, (j["query_id"], j["cond"])
                assert len(j["ctx"].splitlines()) == k, (j["query_id"], j["cond"])
        r1 = sum(1 for j in jobs if j["cond"] == "top1" and j["gold_in_ctx"])
        n = sum(1 for j in jobs if j["cond"] == "top1")
        print(f"[dry] {len(jobs)} jobs, self-check OK. "
              f"top1 gold hit {r1}/{n} = {r1/n:.4f} (= hit@1)")
        for K in (a.rerank_ks or []):
            rk = list({j["query_id"]: j[f"rr{K}_gold_rank"] for j in jobs}.values())
            print(f"[dry] rr{K} 재정렬 후 " + " ".join(
                f"hit@{k}={sum(1 for r in rk if r < k) / len(rk):.4f}"
                for k in (1, 2, 3, 5, K)))
        return 0

    import torch
    from rag_agent.llm.local_qwen import LocalQwenLLM
    # 재정렬기는 build_jobs 안에서 이미 다 썼다. 8GB 카드에 리더와 같이 두지 않는다.
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["cond"], r["query_id"])
                for r in map(json.loads, open(out))}
    todo = [j for j in jobs if (j["cond"], j["query_id"]) not in done]
    print(f"[jobs] {len(jobs)} total, {len(done)} done, {len(todo)} to run", flush=True)

    llm = LocalQwenLLM(model_name=MODEL, quantization="4bit")
    tok = llm.tokenizer
    fh = open(out, "a")
    t0 = time.time()
    for n, j in enumerate(todo, 1):
        user = f"CONTEXT:\n{j['ctx']}\n\nQUESTION: {j['question']}\n\nAnswer:"
        ptok = len(tok(tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)["input_ids"])
        t = time.time()
        raw = llm.complete(system=SYS, user=user, max_tokens=MAXNEW, temperature=0.0)
        parsed = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        fh.write(json.dumps({k: j[k] for k in
                             ("query_id", "cond", "question", "gold_answer", "m",
                              "gold_ranks", "n_injected", "gold_in_ctx")}
                            | {k: v for k, v in j.items()
                               if k.endswith("_gold_rank")} | {
            "pred_answer_raw": raw, "pred_parsed": parsed,
            "hit_token_cap": len(tok(raw, add_special_tokens=False)["input_ids"]) >= MAXNEW,
            "prompt_tokens": ptok, "latency_sec": round(time.time() - t, 3)},
            ensure_ascii=False) + "\n")
        fh.flush()
        if n % 100 == 0:
            print(f"  {n}/{len(todo)}  {(time.time()-t0)/60:.1f}min", flush=True)
    fh.close()
    summarize(out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
