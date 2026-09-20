#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""같은 LLM 필터를 8개 arm 전부에 똑같이 적용한 공정 비교.
PREREG-2026-09-21-fair-llm-filtering-hitab.md

필터는 **줄 단위 문맥 압축**이다 — 유닛 선택이 아니다. 검색이 준 문맥을 물리적 줄
(항목 안의 개행까지)로 쪼개 번호를 붙이고, 필요한 줄 번호만 LLM에게 받아 그 줄만
남겨 리더에게 준다. 셀 20개를 받는 arm이든 마크다운 청크 1덩어리를 받는 arm이든
똑같이 노이즈 제거 혜택을 받는다 — "후보 중 1개 고르기"로 하면 청크 arm은 후보가
1개뿐이라 필터가 작동조차 못 해서 불공정해진다.

질의마다 무필터/필터를 같은 실행에서 나란히 재고 한 행에 쓴다(McNemar 페어링).

  PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --selftest
  PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --limit 3
  PYTHONPATH=. .venv/bin/python scripts/fair_filter_eval.py --resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.eval.metrics import hitab_exact_match_text              # noqa: E402
from rag_agent.llm.factory import build_llm                            # noqa: E402
from scripts.answer_accuracy import (PROMPTS, check_context_limit,     # noqa: E402
                                     load_evidence)

ARMS = {  # 표시 이름 -> records stem (전부 --corpus gold, 예산 20)
    "ours": "t_sleaf_gold",
    "mt2net": "t_mt2net_gold",
    "chunk": "t_chunk_s3c_gold_v2",
    "trag_hetero": "t_trag_hetero_gold_v2",
    "rowcol": "t_rowcol_s3c_gold_v2",
    "randrow": "t_randrow_s3c_gold",
    "tablerag_path": "t_tablerag_path_v2_gold",
    "tablerag_leaf": "t_tablerag_leaf_v2_gold",
}
# 산술(2단계): **2026-09-21 사용자 지시로 폐기**. 모집단 파일과 사전등록을 지웠으므로
# --arith 경로는 지금 돌지 않는다. 폐기 사유는 모집단 결함 -- HiTab 의 aggregation 라벨이
# 참조 셀 개수를 보지 않아 216건 중 60건이 m=1 이고, 그 60건에는 이 실험이 재려던 상황이
# 없다. 복원은 `git show 2d3163f:rag-agent/<경로>`, 다시 할 거면 참조 셀 개수로 모집단을
# 정의하는 새 사전등록부터 한다. 아래 두 줄은 그때 되살릴 자리 표시로만 남긴다.
# (원래 주석: 같은 실행의 arithmetic 분할 파일. mt2net 만 그 분할 파일이 없어 같은 실행의
#  전체 records 를 쓴다 -- 모집단이 216건으로 고정돼 있으므로 필터링은 질의 순회에서 일어난다.)
ARMS_ARITH = {k: (v if k == "mt2net" else v + "_arithmetic") for k, v in ARMS.items()}
POP = ROOT / "results" / "ksweep_population_300.json"
POP_ARITH = ROOT / "results" / "fair_filter_arith_population_216.json"
RECORDS = ROOT / "results" / "retrieval_accuracy"
OUT_DIR = ROOT / "results" / "fair_filter_20260921"
OUT_DIR_ARITH = ROOT / "results" / "fair_filter_arith_20260921"

FILTER_SYS = (
    "You select which context lines are needed to answer a question about a table. "
    "Reply with only the line numbers, comma-separated, in increasing order. "
    "Include the line that holds the value AND any header or label lines needed to "
    "interpret that value. Do not explain, do not repeat the lines — output numbers only."
)
# 산술은 피연산자가 여럿이라 값 줄 하나만 남기면 과제가 성립하지 않는다. 사전등록에
# 변경 사실과 이유를 적고 실행 전에 고정했다(결과를 보고 고친 것이 아니다).
FILTER_SYS_ARITH = (
    "You select which context lines are needed to answer a question about a table. "
    "The question requires a CALCULATION over SEVERAL values, so you must include "
    "EVERY line that holds a value the calculation needs — not just one. "
    "Reply with only the line numbers, comma-separated, in increasing order. "
    "Include the lines that hold the values AND any header or label lines needed to "
    "interpret them. Do not explain, do not repeat the lines — output numbers only."
)


def split_lines(context) -> list[str]:
    """문맥을 물리적 줄로 쪼갠다. 청크 arm은 항목 하나가 마크다운 표 여러 행이라
    항목 안의 개행까지 쪼개야 셀 arm과 같은 입자가 된다."""
    return [ln for entry in (context or []) for ln in str(entry).split("\n") if ln.strip()]


def parse_keep(raw: str, n: int) -> list[int]:
    """LLM 출력에서 1..n 범위의 줄 번호를 뽑는다. 하나도 못 뽑으면 빈 리스트(=폴백)."""
    seen = sorted({int(x) for x in re.findall(r"\d+", raw or "") if 1 <= int(x) <= n})
    return seen


def selftest() -> int:
    assert split_lines(["a", "b"]) == ["a", "b"]
    assert split_lines(["h1\nr1\nr2"]) == ["h1", "r1", "r2"]          # 청크가 여러 줄로
    assert split_lines(["a\n\n  \nb"]) == ["a", "b"]                  # 빈 줄 제거
    assert split_lines(None) == [] and split_lines([]) == []
    assert parse_keep("1, 3", 5) == [1, 3]
    assert parse_keep("Lines 2 and 2 and 7", 5) == [2]                # 범위 밖·중복 제거
    assert parse_keep("none", 5) == [] and parse_keep("", 5) == []    # 폴백 신호
    assert parse_keep("0, 6", 5) == []                                # 경계 밖
    # 폴백 규칙: 빈 결과면 원본 전체를 쓴다 (= 필터 미적용과 동일)
    lines = ["a", "b", "c"]
    keep = parse_keep("nope", len(lines))
    assert ([lines[i - 1] for i in keep] or lines) == lines
    # 근거잔존 대리지표: 표기 변종을 흡수해야 한다
    assert answer_in(["the value is 36"], [36.0]) == 1                # 36.0 vs "36"
    assert answer_in(["the value is 36.0"], [36]) == 1
    assert answer_in(["the value is 52.1"], [52.1]) == 1
    assert answer_in(["nothing here"], [36.0]) == 0
    assert answer_in(["a 3", "b 4"], [3, 4]) == 1 and answer_in(["a 3"], [3, 4]) == 0
    assert answer_in(["jia-a league"], ["Jia-A League"]) == 1
    print("selftest OK")
    return 0


def _forms(v) -> set[str]:
    """한 정답 값의 표기 변종. 문맥은 36 이라 쓰고 정답은 36.0 인 경우가 흔해서,
    변종을 안 만들면 근거가 남아 있는데도 없다고 세게 된다."""
    s = str(v).strip().lower()
    out = {s}
    try:
        f = float(s)
        out.add(repr(f))
        if f == int(f):
            out.add(str(int(f)))
        out.add(f"{f:g}")
    except ValueError:
        pass
    return {x for x in out if x}


def answer_in(lines, answer) -> int:
    """대리 지표: 정답 값이 문맥에 남아 있는가. 유닛 기준 gold 판정이 아니라 arm을
    가로질러 쓸 수 있는 근사치다 — 무필터/필터 양쪽에 똑같이 적용해 편향을 상쇄한다."""
    blob = " ".join(lines).lower()
    vals = answer if isinstance(answer, list) else [answer]
    return int(all(any(f in blob for f in _forms(v)) for v in vals)) if vals else 0


def operand_values(rec, tabs, data_dir="data/hitab") -> list:
    """산술 질의의 피연산자 셀 값들. 산술은 정답 값이 계산 결과라 문맥에 없으므로
    `answer_in` 을 정답으로 재면 항상 0 이 된다 — 근거 잔존은 피연산자로 재야 한다."""
    import rag_agent.bench.hitab_grid as hg
    tid = rec["table_id"]
    tab = tabs.get(tid) or tabs.setdefault(tid, hg.load_table(tid, data_dir))
    t = tab.table
    return [t.data[i][j] for _tid, i, j in rec.get("gold_cells", [])
            if 0 <= i < t.n_rows and 0 <= j < t.n_cols]


def mcnemar(a: dict, b: dict) -> dict:
    """페어드 이항검정. a/b 는 query_id -> 0/1."""
    from scipy.stats import binomtest
    qs = sorted(set(a) & set(b))
    a_only = sum(a[q] and not b[q] for q in qs)
    b_only = sum(b[q] and not a[q] for q in qs)
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return {"n": len(qs), "a_only": a_only, "b_only": b_only,
            "discordant": n, "p_value": round(p, 6)}


def report(path: Path) -> int:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_arm: dict[str, list] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    def acc(rs, key):
        return round(sum(x[key] for x in rs) / len(rs), 4) if rs else None

    out = {"n_rows": len(rows), "arms": {}}
    for name, rs in by_arm.items():
        base = {x["query_id"]: x["correct_base"] for x in rs}
        filt = {x["query_id"]: x["correct_filtered"] for x in rs}
        hit = [x for x in rs if x["retrieval_correct"]]
        out["arms"][name] = {
            "n": len(rs),
            "retrieval_accuracy": acc(rs, "retrieval_correct"),
            "answer_base": acc(rs, "correct_base"),
            "answer_filtered": acc(rs, "correct_filtered"),
            "delta": round((acc(rs, "correct_filtered") or 0) - (acc(rs, "correct_base") or 0), 4),
            "mcnemar_base_vs_filtered": mcnemar(base, filt),
            # 3단 분해: ① 검색 ② 필터가 근거 남김(대리) ③ 리더
            "evidence_kept_base": acc(rs, "answer_in_base"),
            "evidence_kept_filtered": acc(rs, "answer_in_filtered"),
            "answer_given_retrieval_hit_base": acc(hit, "correct_base"),
            "answer_given_retrieval_hit_filtered": acc(hit, "correct_filtered"),
            # 산술 전용: 피연산자가 문맥에 전부 남아 있는가 (정답 값은 계산 결과라 문맥에 없다)
            "operands_kept_base": acc(rs, "operands_in_base") if rs[0].get("n_operands") else None,
            "operands_kept_filtered": (acc(rs, "operands_in_filtered")
                                       if rs[0].get("n_operands") else None),
            "operand_loss_by_filter": (
                round(sum(1 for x in rs if x.get("operands_in_base")
                          and not x.get("operands_in_filtered")) / len(rs), 4)
                if rs[0].get("n_operands") else None),
            "filter_fallback_n": sum(x["filter_fallback"] for x in rs),
            "filter_input_tokens_mean": round(
                sum(x["filter_input_tokens"] or 0 for x in rs) / len(rs), 1),
            "lines_mean": round(sum(x["n_lines"] for x in rs) / len(rs), 1),
            "lines_kept_mean": round(sum(x["n_kept"] for x in rs) / len(rs), 1),
        }
    rank = sorted(out["arms"], key=lambda k: out["arms"][k]["answer_filtered"], reverse=True)
    out["rank_after_filter"] = rank
    out["ours_is_first"] = rank[0] == "ours"
    if "ours" in by_arm:  # ours 대 나머지, 필터 적용 후 페어드 비교
        ours = {x["query_id"]: x["correct_filtered"] for x in by_arm["ours"]}
        out["ours_vs_others_filtered"] = {
            k: mcnemar(ours, {x["query_id"]: x["correct_filtered"] for x in v})
            for k, v in by_arm.items() if k != "ours"}
    path.with_name("summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="rows.jsonl 을 읽어 요약만 낸다")
    ap.add_argument("--task", default="lookup", choices=["lookup", "arithmetic"],
                    help="lookup=1단계(단일조회 n=300), arithmetic=2단계(산술 n=216)")
    ap.add_argument("--model", default="local:Qwen/Qwen2.5-7B-Instruct?quantization=4bit")
    ap.add_argument("--limit", type=int, default=0, help="arm 당 질의 수 (스모크용)")
    ap.add_argument("--arms", default="", help="쉼표로 고른 arm만 (기본: 전부)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    arith = a.task == "arithmetic"
    if not a.out:
        a.out = str((OUT_DIR_ARITH if arith else OUT_DIR) / "rows.jsonl")
    if a.report:
        return report(Path(a.out))

    all_arms, pop_file, filter_sys = ((ARMS_ARITH, POP_ARITH, FILTER_SYS_ARITH) if arith
                                      else (ARMS, POP, FILTER_SYS))
    arms = {k: v for k, v in all_arms.items()
            if not a.arms or k in {s.strip() for s in a.arms.split(",")}}
    pop = json.load(open(pop_file))["query_ids"]
    if a.limit:
        pop = pop[:a.limit]

    loaded = {}
    for name, stem in arms.items():
        recs, meta = load_evidence(RECORDS / f"{stem}_records.jsonl")
        missing = [q for q in pop if q not in recs]
        if missing:
            raise SystemExit(f"{name}: 모집단 {len(missing)}건이 records 에 없다")
        loaded[name] = (recs, meta)
    print(f"[pop] n={len(pop)} × arm {len(loaded)}개 = {len(pop) * len(loaded)}행", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        if not a.resume:
            raise SystemExit(f"{out} 있음 — 이어 쓰려면 --resume")
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                       # 쓰다 끊긴 마지막 줄
            done.add((row["arm"], row["query_id"]))
        print(f"[resume] {len(done)}행 건너뜀", flush=True)

    llm = build_llm(a.model)
    limit = getattr(llm, "context_limit", 0)
    tabs: dict = {}                      # 피연산자 조회용 표 캐시 (산술에서만 쓴다)
    t0, n_done = time.time(), 0
    total = len(pop) * len(loaded) - len(done)
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        for name, (recs, _meta) in loaded.items():
            for q in pop:
                if (name, q) in done:
                    continue
                r = recs[q]
                lines = split_lines(r.get("context"))
                ops = operand_values(r, tabs) if arith else []
                # 무필터 리더
                base_user = "Context:\n" + "\n".join(lines) + f"\n\nQuestion: {r['question']}\nAnswer:"
                n_tok_base = (llm.n_prompt_tokens(PROMPTS["neutral"], base_user)
                              if hasattr(llm, "n_prompt_tokens") else None)
                check_context_limit(n_tok_base, a.max_tokens, limit)
                pred_base = llm.complete(PROMPTS["neutral"], base_user,
                                         max_tokens=a.max_tokens, temperature=0.0)
                # 필터
                numbered = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
                f_user = f"Question: {r['question']}\n\nLines:\n{numbered}\n\nLine numbers needed:"
                n_tok_filter = (llm.n_prompt_tokens(filter_sys, f_user)
                                if hasattr(llm, "n_prompt_tokens") else None)
                check_context_limit(n_tok_filter, a.max_tokens, limit)
                raw = llm.complete(filter_sys, f_user, max_tokens=a.max_tokens, temperature=0.0)
                keep = parse_keep(raw, len(lines))
                kept = [lines[i - 1] for i in keep] or lines
                fallback = int(not keep)
                # 필터 후 리더
                f_read = "Context:\n" + "\n".join(kept) + f"\n\nQuestion: {r['question']}\nAnswer:"
                pred_filt = llm.complete(PROMPTS["neutral"], f_read,
                                         max_tokens=a.max_tokens, temperature=0.0)
                row = {
                    "arm": name, "query_id": q, "question": r["question"],
                    "answer": r["answer"], "retrieval_correct": r["correct"],
                    "n_lines": len(lines), "n_kept": len(kept),
                    "filter_input_tokens": n_tok_filter, "reader_input_tokens": n_tok_base,
                    "filter_raw": raw, "filter_kept_idx": keep, "filter_fallback": fallback,
                    "answer_in_base": answer_in(lines, r["answer"]),
                    "answer_in_filtered": answer_in(kept, r["answer"]),
                    "operands_in_base": (answer_in(lines, ops) if arith else None),
                    "operands_in_filtered": (answer_in(kept, ops) if arith else None),
                    "n_operands": (len(ops) if arith else None),
                    "pred_base": pred_base, "pred_filtered": pred_filt,
                    "correct_base": int(hitab_exact_match_text(pred_base, r["answer"])),
                    "correct_filtered": int(hitab_exact_match_text(pred_filt, r["answer"])),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == total:
                    el = time.time() - t0
                    print(f"  {name} {n_done}/{total}  {el:.0f}s  "
                          f"(남은 예상 {el / n_done * (total - n_done) / 60:.0f}분)", flush=True)
    print(f"[done] {n_done}행, {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
