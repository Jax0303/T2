#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""검색 정확도 — MultiHiertt: 다중 셀 조회와 산술 질의.

`scripts/retrieval_accuracy.py` 와 **같은 지표**를 MultiHiertt 에 놓는다
(`CLAUDE.md` §0.1): k 사다리 없이 운영점 하나(리더에게 주는 셀 20개)에서
질의를 맞았다/틀렸다로 채점하고 그 비율만 보고한다. recall@k / all-covered@k 는
쓰지 않는다.

    검색 정확도 = (배달된 문맥이 정답 근거 셀 전부를 담은 질의 수) / (질의 수)

**MultiHiertt 를 쓰는 이유는 분모가 아니라 gold 의 모양이다.** HiTab 주지표는
gold 셀이 정확히 하나라 지표가 "그 한 칸이 예산 안에 있나"로 떨어진다. MultiHiertt
는 질의당 supporting fact 가 여러 개이고 문서 안에 표가 여럿이라, 같은 지표가
"필요한 셀을 **빠짐없이** 회수했나"를 묻게 된다. 그래서 여기서 재는 것은:

  조회 m>=2  (다중 셀 조회)  — 계산 없이 여러 칸을 동시에 회수해야 하는 질의
  산술 m>=2  (산술 질의)     — 피연산자 전부를 회수해야 계산이 성립하는 질의

`m=1` 두 층도 같은 표에 싣는다. m 이 오를 때 지표가 어떻게 무너지는지가 이
데이터셋이 추가하는 정보이고, m=1 만 싣거나 m>=2 만 싣는 것은 둘 다 체리피킹이다.

## `any` 는 주지표가 아니다 — 과대평가의 크기를 재기 위해 함께 싣는다

"gold 중 하나가 예산 안에 있으면 성공"으로 채점하면 질의당 평균 여러 개인
supporting fact 중 한 개만 찾아도 통과한다. 그 규칙으로도 같이 채점해서
**두 수치의 차이를 보고한다.** 주지표는 언제나 `all` 이다.

## 두 개의 검색 범위 — 둘 다 싣는다

`corpus`  이 모집단 문서 전부의 셀을 한 색인에 넣고 코퍼스 전역에서 찾는다.
          본 방법의 주장(별도 표 검색 단계 없음)이 서는 범위.
`doc`     후보를 질의 자신의 문서로 제한한다. MultiHiertt 의 과제 정의이자
          MT2Net 의 범위(문서 내 재정렬). 같은 색인·같은 예산, 마스크만 다르다.

## 정답 근거는 데이터셋이 정한다

`table_evidence` 의 `"{표}-{행}-{열}"` 좌표가 gold 셀 집합이다. 좌표 규약은
데이터셋 자신의 셀 렌더링(`table_description`)과 대조해 실측 확인했다 — 300문서
848셀 전부 값 일치(쉼표 정규화 후), gold 의 0.71% 만 우리 파서가 헤더로 본 영역에
떨어진다. 그 0.71% 는 `gold_in_header` 로 사유를 붙여 제외하고 개수를 보고하며,
제외분을 **오답으로 세는 하한**도 함께 낸다 (제외가 유리하게 작동하지 않도록).

`text_evidence` 가 있는 질의(4,209+713건)는 **셀 색인이 원리적으로 못 담는다** —
문단이 색인 단위에 없다. 모집단에서 빼고 개수를 보고한다. 이것은 이 계측기의
한계이지 MultiHiertt 의 결과가 아니다.

  PYTHONPATH=. python3 scripts/retrieval_accuracy_mh.py

`--metric-mode strict_fixed_budget|strict_no_k` 는 이 파일이 아니라 `mh_arms.py` 가 처리한다. 이 계측기의
선택은 리더에게 가지 않고, Strict Recall 은 리더에게 실제로 넘어간 셀을 채점해야 한다.
인자는 그대로 넘긴다 (`mh_arms.py --help`).

  PYTHONPATH=. python3 scripts/retrieval_accuracy_mh.py --metric-mode strict_no_k --split validation
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                    # noqa: E402

from retrieval_accuracy import budget_select                          # noqa: E402
from rag_agent.reconstruct import (guess_n_header_cols,               # noqa: E402
                                   guess_n_header_rows, parse_html_table,
                                   reconstruct_col_paths, reconstruct_row_paths)
from rag_agent.retrieve.encoders import _tokenize, default_encoder    # noqa: E402
from rag_agent.retrieve.hybrid_index import _minmax                   # noqa: E402
from rag_agent.retrieve.sparse_bm25 import SparseBM25                 # noqa: E402
from rag_agent.serialization.caption import caption_sentence          # noqa: E402
from rag_agent.serialization.templates import STRUCTURAL_COMPACT      # noqa: E402

# `table_description` 은 데이터셋 자신의 셀 문장이다. 끝의 "... is <값> ." 에서
# 값을 뽑아 우리 파서가 같은 좌표에서 읽은 값과 대조한다 — gold 충실도를
# 가정하지 않고 실측하기 위한 것이지 색인에 쓰는 텍스트가 아니다.
_DESC_VAL = re.compile(r"\bis\s+(.+?)\s*\.\s*$")
# MultiHiertt 표는 캡션·제목을 싣지 않는다. s3c 는 제목이 없으면 S2 경로 문자열로
# 떨어지도록 이미 정의돼 있다 (templates.py STRUCTURAL_COMPACT). 제목을 만들어
# 넣지 않는다 — 없는 것을 지어내면 그 arm 은 이 데이터셋의 수치가 아니다.
NO_TITLE = ""


def _norm(s: str) -> str:
    return re.sub(r"[,\s]", "", s or "")


def load_population(split: str):
    """(queries, docs) — 표 근거만 있는 질의 전부와 그 문서.

    층 이름은 데이터셋 필드로만 정한다: `program` 이 비면 조회, 있으면 산술이고
    `m` 은 `table_evidence` 의 셀 수다. 우리가 나눈 분류가 아니다.
    """
    from datasets import load_dataset
    rows = load_dataset("bevaya/MultiHiertt", split=split)

    queries, docs, skipped = [], {}, Counter()
    for row in rows:
        ev = row.get("table_evidence") or []
        if row.get("text_evidence"):
            skipped["needs_text_evidence"] += 1
            continue
        if not ev:
            skipped["no_table_evidence"] += 1
            continue
        coords = []
        for e in ev:
            parts = e.split("-")
            if len(parts) != 3:
                coords = None
                break
            coords.append(tuple(int(x) for x in parts))
        if coords is None:
            skipped["bad_evidence_coord"] += 1
            continue
        arith = bool((row.get("program") or "").strip())
        queries.append({
            "uid": row["uid"], "question": row["question"],
            "answer": row.get("answer"), "coords": coords,
            "kind": "arith" if arith else "lookup",
            "n_gold_tables": len({c[0] for c in coords}),
            "desc": row["table_description"],
        })
        docs[row["uid"]] = row["tables"]
    return queries, docs, skipped


def build_corpus(docs):
    """(texts, covers, unit_uid, tables) — 문서 전부의 모든 표를 셀 단위로 색인.

    헤더 경로는 평평한 격자에서 자가복원한다 (`rag_agent.reconstruct`) —
    데이터셋이 준 정답 구조를 쓰지 않는다. HiTab arm 과 같은 규칙이다.
    """
    texts, covers, unit_uid, tables = [], [], [], {}
    for uid in sorted(docs):
        for t_idx, html in enumerate(docs[uid]):
            grid = parse_html_table(html)
            if len(grid) < 3 or len(grid[0]) < 2:
                continue
            # 행 경계를 먼저(열 기본값 1), 그 경계로 열 경계를 잡는다 —
            # gold nhc 가 있는 tree_reconstruct_hitab_raw.py 에서 검증된 순서.
            nhr = max(1, min(guess_n_header_rows(grid, n_header_cols=1), len(grid) - 1))
            nhc = max(1, min(guess_n_header_cols(grid, n_header_rows=nhr),
                             len(grid[0]) - 1))
            rows = reconstruct_row_paths(grid, nhr, nhc)
            cols = reconstruct_col_paths(grid, nhr, nhc)
            tables[(uid, t_idx)] = (nhr, nhc)
            for r in range(nhr, len(grid)):
                rp = rows[r - nhr] if (r - nhr) < len(rows) else []
                for c in range(nhc, len(grid[0])):
                    v = (grid[r][c] or "").strip()
                    if not v:
                        continue
                    cp = cols[c - nhc] if (c - nhc) < len(cols) else []
                    texts.append(caption_sentence(NO_TITLE, rp, cp, value=v,
                                                  template=STRUCTURAL_COMPACT))
                    covers.append(frozenset([(uid, t_idx, r, c)]))
                    unit_uid.append(uid)
    return texts, covers, unit_uid, tables


def resolve_gold(queries, cell_index, tables):
    """질의마다 gold 셀 집합과 제외 사유. 제외는 이름을 붙여 세고 숨기지 않는다."""
    for q in queries:
        gold, why = set(), None
        for t_idx, r, c in q["coords"]:
            key = (q["uid"], t_idx, r, c)
            if key in cell_index:
                gold.add(key)
                continue
            hdr = tables.get((q["uid"], t_idx))
            if hdr is None:
                why = why or "gold_table_unparsed"
            elif r < hdr[0] or c < hdr[1]:
                why = why or "gold_in_header"      # 데이터셀 색인이 담을 수 없는 칸
            else:
                why = why or "gold_cell_missing"   # 빈 칸으로 파싱됨
        q["gold"] = gold
        q["excluded"] = why
    return queries


def audit_gold_values(queries, docs):
    """gold 좌표가 데이터셋 자신의 렌더링과 같은 값을 가리키는지 실측한다."""
    ok = mismatch = nokey = 0
    grids_of: dict = {}
    for q in queries:
        desc = json.loads(q["desc"])
        grids = grids_of.get(q["uid"])
        if grids is None:
            grids = grids_of[q["uid"]] = [parse_html_table(h) for h in docs[q["uid"]]]
        for t_idx, r, c in q["coords"]:
            m = _DESC_VAL.search(desc.get(f"{t_idx}-{r}-{c}", ""))
            if m is None:
                nokey += 1
                continue
            g = grids[t_idx] if t_idx < len(grids) else []
            got = g[r][c].strip() if r < len(g) and c < len(g[r]) else None
            ok += int(got is not None and _norm(m.group(1)) == _norm(got))
            mismatch += int(got is None or _norm(m.group(1)) != _norm(got))
    return {"gold_cells_checked": ok + mismatch + nokey, "value_match": ok,
            "value_mismatch": mismatch, "no_description_key": nokey}


def layer(q) -> str:
    return f"{q['kind']}_m{'1' if len(q['gold']) == 1 else '2+'}"


def rate(hits, n):
    return round(sum(hits) / n, 4) if n else None


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--metric-mode", default="accuracy",
                     choices=["accuracy", "strict_fixed_budget", "strict_no_k"])
    if pre.parse_known_args()[0].metric_mode != "accuracy":
        import mh_arms
        return mh_arms.main()
    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train",
                    help="MultiHiertt test 는 gold 를 공개하지 않는다. train 이 "
                         "조회 m>=2 를 n=368 로 주는 유일한 분할이다 "
                         "(validation 은 48건).")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="dense 가중. HiTab arm 에 고정된 값을 그대로 쓴다 — "
                         "이 데이터셋에서 다시 고르지 않는다 (그렇게 고른 "
                         "results/mh/VERDICT.md 의 alpha=0.8 은 낙관적이었다).")
    ap.add_argument("--budget", type=int, default=20, help="문맥 크기, 셀 개수")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="문서 수 상한 (0 = 전부). 배관 점검용이며 보고용 수치는 0.")
    ap.add_argument("--shard", type=int, default=50000,
                    help="임베딩을 이 크기로 나눠 캐시에 저장한다 — CPU 인코딩은 "
                         "몇 시간이라 중단되면 남은 조각만 다시 돈다.")
    ap.add_argument("--cache-dir", default=".cache/retrieval_accuracy_mh")
    ap.add_argument("--out-dir", default="results/retrieval_accuracy_mh")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    t0 = time.time()
    queries, docs, skipped = load_population(a.split)
    if a.max_docs:
        keep = set(sorted(docs)[:a.max_docs])
        queries = [q for q in queries if q["uid"] in keep]
        docs = {u: docs[u] for u in keep}
    print(f"[pop] {len(queries)} queries / {len(docs)} docs "
          f"(제외: {dict(skipped)})  {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    texts, covers, unit_uid, tables = build_corpus(docs)
    cell_index = {next(iter(cv)): i for i, cv in enumerate(covers)}
    uid_arr = np.array(unit_uid)
    queries = resolve_gold(queries, cell_index, tables)
    print(f"[corpus] {len(tables)} tables / {len(texts)} cells "
          f"({time.time() - t0:.0f}s)", flush=True)

    audit = audit_gold_values(queries, docs)
    print(f"[audit] {audit}", flush=True)
    for q in queries:                          # 문서당 ~18KB, 대조가 끝나면 안 쓴다
        q.pop("desc", None)

    t0 = time.time()
    bm = SparseBM25(_tokenize(t) for t in texts)
    print(f"[bm25] {len(bm.vocab)} terms in {time.time() - t0:.0f}s", flush=True)

    enc = default_encoder(model_name=a.embed_model)
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    from hashlib import md5
    n_cells = len(texts)
    key = md5("\x00".join(texts).encode()).hexdigest()[:16]
    # 셰이드를 리스트로 모았다가 concatenate 하면 피크가 두 배(1.3GB x 2)라
    # 이 상자에서 OOM 으로 죽는다 — 2026-09-10 에 인코딩 9셰이드를 다 끝내고
    # 마지막 줄에서 죽었다. 미리 잡아 두고 제자리에 채운다.
    emb = np.empty((n_cells, enc.encode(texts[:1]).shape[1]), dtype=np.float32)
    for s in range(0, n_cells, a.shard):
        f = cache / f"{enc.name.replace('/', '_')}_{n_cells}_{key}_{s}.npy"
        if f.exists():
            v = np.load(f)
        else:
            t0 = time.time()
            v = enc.encode(texts[s:s + a.shard])
            np.save(f, v)
            print(f"[dense] {s}..{s + len(v)} in {time.time() - t0:.0f}s", flush=True)
        emb[s:s + len(v)] = v
        del v
    # 색인이 선 뒤로 이 둘은 쓰이지 않는다. 채점 루프가 429k 셀 x 2,908 질의를
    # 도는 동안 붙들고 있을 이유가 없다 (budget_select 는 dump=0 이면 texts 를
    # 읽지 않는다).
    del cell_index
    texts = [""] * n_cells

    # 진단용 gold_rank 상한. 예산 20 을 채우는 데 필요한 것보다 넉넉하게 잡아
    # 전체 argsort 를 피한다 (코퍼스 40만 x 질의 2,908).
    TOP = 512
    recs, t0 = [], time.time()
    for n, q in enumerate(queries, 1):
        if q["excluded"] or not q["gold"]:
            recs.append({"uid": q["uid"], "kind": q["kind"], "m": len(q["coords"]),
                         "excluded": q["excluded"] or "gold_empty"})
            continue
        sp = bm.get_scores(_tokenize(q["question"]))
        dn = emb @ enc.encode_query([q["question"]])[0].astype(np.float32)
        sc = a.alpha * _minmax(dn) + (1 - a.alpha) * _minmax(sp)
        r = {"uid": q["uid"], "kind": q["kind"], "m": len(q["gold"]),
             "m_annotated": len(q["coords"]), "n_gold_tables": q["n_gold_tables"],
             "layer": layer(q)}
        for scope in ("corpus", "doc"):
            if scope == "doc":
                sel = np.flatnonzero(uid_arr == q["uid"])
                s2 = sc[sel]
                k = min(TOP, len(sel) - 1)
                top = sel[np.argpartition(-s2, k)[:k + 1]]
                order = top[np.argsort(-sc[top])]
            else:
                top = np.argpartition(-sc, TOP)[:TOP + 1]
                order = top[np.argsort(-sc[top])]
            got, n_ctx, _ = budget_select(order, covers, texts, a.budget, 0)
            # 마지막 gold 가 나오는 순위. 판정에는 쓰지 않는 진단값이다
            # (`CLAUDE.md` §0.1: 주지표는 질의 단위 정확도 하나).
            seen, last = set(), None
            for i, p in enumerate(order, 1):
                seen |= covers[p] & q["gold"]
                if len(seen) == len(q["gold"]):
                    last = i
                    break
            r[scope] = {
                "correct": int(q["gold"] <= got),          # 주지표: all
                "any": int(bool(q["gold"] & got)),          # 과대평가 진단
                "n_gold_in_context": len(q["gold"] & got),
                "cells_in_context": n_ctx,
                "gold_rank_last": last,
                "gold_doc_in_context": int(any(c[0] == q["uid"] for c in got)),
            }
        recs.append(r)
        if n % 100 == 0:
            print(f"  {n}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)

    scored = [r for r in recs if "corpus" in r]
    excl = Counter(r["excluded"] for r in recs if "excluded" in r)
    by = defaultdict(list)
    for r in scored:
        by[r["layer"]].append(r)
        by["ALL"].append(r)
    for r in scored:                       # gold 가 표를 가로지르는 층
        by["gold_multi_table" if r["n_gold_tables"] > 1 else "gold_single_table"].append(r)

    def block(rows):
        out = {"n": len(rows)}
        for scope in ("corpus", "doc"):
            allm = [x[scope]["correct"] for x in rows]
            anym = [x[scope]["any"] for x in rows]
            out[scope] = {
                "accuracy_all": rate(allm, len(rows)),
                "accuracy_any_DIAGNOSTIC": rate(anym, len(rows)),
                # any 가 all 을 얼마나 부풀리는지. 이 데이터셋을 쓰는 이유이므로
                # 수치로 둔다. per-cell recall 은 §0.1 금지라 쓰지 않는다.
                "any_minus_all": (round(rate(anym, len(rows)) - rate(allm, len(rows)), 4)
                                  if rows else None),
                "gold_doc_in_context": rate([x[scope]["gold_doc_in_context"]
                                             for x in rows], len(rows)),
            }
        out["mean_m"] = round(sum(x["m"] for x in rows) / len(rows), 3) if rows else None
        out["m_over_budget"] = sum(1 for x in rows if x["m"] > a.budget)
        return out

    # 제외를 오답으로 세는 하한. 제외가 유리하게 작동하지 않는다는 것을 수치로 둔다.
    lower = {}
    for name in ("lookup_m1", "lookup_m2+", "arith_m1", "arith_m2+", "ALL"):
        rows = by.get(name, [])
        n_ex = (sum(1 for r in recs if "excluded" in r) if name == "ALL"
                else sum(1 for r in recs if "excluded" in r
                         and r["kind"] == name.split("_")[0]
                         and (r["m"] == 1 if name.endswith("m1") else r["m"] >= 2)))
        lower[name] = {
            "n_incl_excluded": len(rows) + n_ex,
            "corpus_all": (round(sum(x["corpus"]["correct"] for x in rows)
                                 / (len(rows) + n_ex), 4) if rows or n_ex else None),
            "doc_all": (round(sum(x["doc"]["correct"] for x in rows)
                              / (len(rows) + n_ex), 4) if rows or n_ex else None),
        }

    summary = {
        "dataset": "MultiHiertt (Zhao et al., ACL 2022) via bevaya/MultiHiertt",
        "split": a.split,
        "metric": "질의 단위 맞았다/틀렸다, 운영점 하나 (CLAUDE.md §0.1). "
                  "주지표 = all (gold 셀 전부 문맥에). any 는 진단값.",
        "unit": "cell", "template": "s3c (제목 없음 -> S2 경로 문자열)",
        "hierarchy_source": "self-reconstructed (rag_agent.reconstruct)",
        "encoder": enc.name, "query_prefix": enc.query_prefix,
        "alpha": a.alpha, "budget_cells": a.budget,
        "n_docs": len(docs), "n_tables": len(tables), "n_cells": n_cells,
        "population_skipped_upstream": dict(skipped),
        "n_queries": len(queries), "n_scored": len(scored),
        "n_excluded": sum(excl.values()), "excluded_by_reason": dict(excl),
        "gold_value_audit": audit,
        "by_layer": {k: block(v) for k, v in sorted(by.items())},
        "lower_bound_excluded_as_wrong": lower,
    }
    tag = a.tag or f"{a.split}_cell_s3c_a{a.alpha}_k{a.budget}"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(out / f"{tag}_records.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote -> {out / tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
