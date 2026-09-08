#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색). 리더 없음.

`CLAUDE.md` §3 1단계가 요구하는 보고 형태. 검색은 코퍼스 전체 셀을 한 번에 정렬하고
(표 검색 단계가 따로 없다, `results/tabsig/VERDICT.md` §0), 여기서 두 축을 갈라 읽는다:

  표@k          gold 표가 셀 투표 순위 k 안        (`table_rank_cellvote`)
  셀|표@k       gold 표 **안에서만** 셀을 정렬했을 때 gold 셀 전부가 k 안
                (`ranks_in_table` = 오라클 표 게이팅; "표를 맞혔을 때 셀을 맞히는가")
  전체@k        코퍼스 전체 정렬에서 gold 셀 전부가 k 안 (= all-covered@k, 주지표)
  표@1×셀|표@k  표 1등을 먼저 고르고 그 안에서 셀을 고르는 엄격한 2단계의 값.
                전체@k 보다 낮다 — 다른 표의 셀이 1등이어도 gold 가 k 안이면 전체@k 는 맞음.

입력은 `analysis/cell_rank_dump.py` 의 `*_ranks.jsonl`. 파일명에서 모집단을 읽는다.

  PYTHONPATH=. .venv/bin/python analysis/stage1_board.py results/p0_confirm/*_ranks.jsonl \
      results/stage1/*_ranks.jsonl --out results/stage1/BOARD.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KS = (1, 3, 5, 10, 20, 50)
CS = (1, 3, 5, 10)
BUCKETS = ("1", "2", "3-5", "6+")
POP = re.compile(r"^(hitab_(?:dev|test|train(?:_sel|_fit)?)_[a-z_]+?)_S3c(?:_page|_sig|_drop)?(_clean)?_")


def load(path):
    return [json.loads(l) for l in open(path)]


def split_total(split):
    """스플릿에 있는 질의 총수 — 모집단이 아니라 데이터셋 전부. 없으면 None."""
    p = Path(f"data/hitab/data/{split}_samples.jsonl")
    return sum(1 for _ in open(p)) if p.exists() else None


def row(name, R):
    n = len(R)
    t1 = [r for r in R if r["table_rank_cellvote"] == 1]
    tab = {k: sum(r["table_rank_cellvote"] <= k for r in R) / n for k in (1, 3, 10)}
    cell = {k: sum(max(r["ranks_in_table"]) < k for r in R) / n for k in KS}
    full = {k: sum(max(r["ranks"]) < k for r in R) / n for k in KS}
    casc = {k: sum(max(r["ranks_in_table"]) < k for r in t1) / n for k in KS}
    m = sorted(r["m"] for r in R)
    miss = [r for r in R if max(r["ranks"]) >= 10]
    tw = sum(r["table_rank_cellvote"] != 1 for r in miss)
    # 셀 단위 recall (부분 점수) — all-covered 는 하나만 놓쳐도 0 이라 "얼마나 가까웠나"가 안 보인다.
    gcells = [x for r in R for x in r["ranks"]]
    crec = {k: sum(x < k for x in gcells) / len(gcells) for k in KS}
    # 요구 셀당 같은 여유: 예산 = c·m 셀. c=10 은 m=1 인 조회에서 @10 과 정의상 같다.
    budget = {c: sum(max(r["ranks"]) < c * r["m"] for r in R) / n for c in CS}
    # 논거가 걸린 불변식: m=1 뿐인 모집단에서 c=10 은 @10 과 같은 값이어야 한다.
    assert m[-1] > 1 or budget[10] == full[10], (name, budget[10], full[10])
    # m 층화 — 질의 종류 간 공정 비교는 **이것으로** 한다. 고정 K 는 여유(K/m)가 종류마다 달라
    # 그 자체로는 공정 비교가 안 된다. 층화는 gold 를 채점 시점의 분류에만 쓰므로 오라클이 아니다.
    bucket = lambda x: "1" if x == 1 else ("2" if x == 2 else ("3-5" if x <= 5 else "6+"))
    by_m = {}
    for b in BUCKETS:
        sub = [r for r in R if bucket(r["m"]) == b]
        by_m[b] = {"n": len(sub), "slack": (20 * len(sub) / sum(r["m"] for r in sub)) if sub else None,
                   "full20": (sum(max(r["ranks"]) < 20 for r in sub) / len(sub)) if sub else None,
                   "full10": (sum(max(r["ranks"]) < 10 for r in sub) / len(sub)) if sub else None}
    # m>k 라 어떤 검색기로도 0 인 건수
    impossible = {k: sum(r["m"] > k for r in R) for k in KS}
    return {"pop": name, "n": n, "m_median": m[n // 2], "m_max": m[-1],
            "table": tab, "cell_given_table": cell, "full": full, "cascade": casc,
            "cell_recall": crec, "budget_cm": budget, "by_m": by_m, "impossible": impossible,
            "miss10": len(miss), "miss10_table_wrong": tw,
            "miss10_table_right": len(miss) - tw}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/stage1/BOARD.md")
    a = ap.parse_args()
    rows, seen = [], {}
    for f in a.files:
        m = POP.match(Path(f).name)
        name = (m.group(1) + (m.group(2) or "")) if m else Path(f).stem
        recs = load(f)
        rows.append(row(name, recs) | {"file": f})
        s_ = "test" if "_test_" in name else "dev" if "_dev_" in name else name
        seen.setdefault(s_, {}).update({d["query_id"]: d for d in recs})

    L = ["# 1단계 보드 — 쿼리 종류 × (표 검색, 표 조건부 셀 검색)", "",
         "계측기 `analysis/stage1_board.py`. 리더 없음. 각 행의 출처 파일은 맨 아래.", "",
         "**주지표는 `전체@20`** — 고정 예산이고 오라클이 아니다. K 는 리더에 넘기는 셀 개수이며 모든 질의에 "
         "같은 값을 준다. K=20 인 이유는 사전 논거다: gold 셀 수 m 의 최대(dev 12 / test 11)보다 크게 잡으면 정의상 못 "
         "맞히는 질의가 생긴다 (아래 불가능 건수 표). 20 은 등록된 사다리 (1,3,5,10,20,50) 에서 12 이상의 "
         "가장 낮은 칸이다.", "",
         "⚠️ **@20 은 질의 종류 간 공정 비교 도구가 아니다.** 고정 K 가 없앤 것은 \"정의상 불가능\" 뿐이고, "
         "여유 K/m 은 여전히 종류마다 다르다 (조회 m=1 은 20배, 산술 m=6+ 는 약 2배 — 아래 층화 표). "
         "**질의 종류 간 비교는 m 층화로 읽는다.** @20 은 이 시스템이 실제로 낼 수 있는 값을 보고하는 열이다.", "",
         "⚠️ **`전체@10` 은 `PREREG-2026-09-06-cell90.md` 의 사전등록 값이고 그 판정은 그대로 남는다** "
         "(산술 목표 .9 는 어느 arm 에서도 @10 미달 — 값은 그 판정 파일에서 읽을 것). @20 이 더 높다고 \"목표 달성\"이라고 쓰지 말 것 — "
         "못 넘긴 것을 보고 K 를 올린 것이 된다. **@20 에 대한 판정은 아직 돌리지 않은 실험의 사전등록에서 "
         "목표값을 먼저 정한 뒤에 한다.**", "",
         "| 모집단 | n | m 중앙/최대 | **전체@20** | 전체@10 | 셀recall@20 | 표@1 | 표@3 | 셀\\|표@1 | "
         "셀\\|표@3 | 셀\\|표@5 | 셀\\|표@10 | 셀\\|표@20 | 표@1×셀\\|표@10 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c, t, f, s, cr = (r["cell_given_table"], r["table"], r["full"], r["cascade"],
                          r["cell_recall"])
        L.append(f"| `{r['pop']}` | {r['n']} | {r['m_median']}/{r['m_max']} | **{f[20]:.4f}** | "
                 f"{f[10]:.4f} | {cr[20]:.4f} | {t[1]:.4f} | {t[3]:.4f} | {c[1]:.4f} | "
                 f"{c[3]:.4f} | {c[5]:.4f} | {c[10]:.4f} | {c[20]:.4f} | {s[10]:.4f} |")
    L += ["", "`셀recall@20` 은 gold 셀 단위 회수율(부분 점수)이다 — `전체@20` 은 하나만 놓쳐도 0 이라 "
          "\"얼마나 가까웠나\"가 안 보인다. `@1`·`@3` 열은 **조회형(m=1)에만 의미가 있다** — 산술·다중은 "
          "m>k 라 정의상 0 인 질의가 대부분이다 (아래 표)."]
    L += ["", "## 전체@10 실패의 위치", "",
          "| 모집단 | 실패 | 표를 틀림 | 표는 맞고 셀을 틀림 |", "|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| `{r['pop']}` | {r['miss10']} | {r['miss10_table_wrong']} | "
                 f"{r['miss10_table_right']} |")
    L += ["", "## m 층화 — **질의 종류 간 비교는 여기서 읽는다**", "",
          "같은 지표를 요구 셀 수 m 으로 쪼갠 것. 새 지표가 아니고 **오라클도 아니다** — gold 를 채점 시점의 "
          "분류에만 쓰고 시스템이 무엇을 몇 개 꺼낼지는 안 건드린다 (예산을 m 으로 정하는 `@(c·m)` 과 여기가 "
          "다르다). 칸은 `전체@20 (n, 여유 20/m)`. 같은 m 끼리 비교할 것 — **여유가 같아야 같은 시험이다.**", "",
          "| 모집단 | m=1 | m=2 | m=3–5 | m≥6 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        cells = []
        for b in BUCKETS:
            d = r["by_m"][b]
            cells.append("—" if d["full20"] is None
                         else f"{d['full20']:.4f} ({d['n']}, {d['slack']:.1f}×)")
        L.append(f"| `{r['pop']}` | " + " | ".join(cells) + " |")
    L += ["", "## 질의 종류 간 비교 — 요구 셀당 같은 여유 `all-covered@(c·m)`", "",
          "**사후 지표** (사전등록 주지표는 전체@10 그대로). k 를 셀 수로 고정하면 정답 셀이 1개인 조회와 "
          "최대 12개인 산술이 같은 시험을 보지 않는다. 예산을 요구량 m 에 비례시켜 정답 셀 1개당 c 칸을 준다. "
          "c=10 은 m=1 인 조회에서 @10 과 **정의상 동일**하므로(아래 표에서 확인) 사전등록 조회 숫자를 바꾸지 "
          "않는 유일한 c 다 — 결과를 보고 고른 상수가 아니다. c≥1 에서는 m>k 로 불가능한 질의가 없다.", "",
          "⚠️ **오라클 예산이다 — 논문 표에 싣지 말 것.** k=c·m 을 정하려면 gold 셀 개수 m 을 알아야 하는데 "
          "배포된 시스템은 m 을 모른다 (셀|표@k 와 같은 종류의 오라클). 게다가 **없어도 된다** — 위 m 층화가 "
          "같은 결론(여유가 같으면 산술은 조회만큼 한다)을 오라클 없이 준다. 이 표는 그 결론을 한 숫자로 "
          "요약한 기록으로만 남긴다. 답변 성능의 대리 지표로도 전체@10 보다 나쁘다 "
          "(`VERDICT_METRIC.md` §3 한계 절).", "",
          "| 모집단 | n | m 최대 | c=1 | c=3 | c=5 | c=10 | (참고) 전체@10 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        b = r["budget_cm"]
        L.append(f"| `{r['pop']}` | {r['n']} | {r['m_max']} | " +
                 " | ".join(f"{b[c]:.4f}" for c in CS) + f" | {r['full'][10]:.4f} |")
    L += ["", "## 각 k 에서 `m>k` 라 정의상 0 인 건수", "",
          "분모는 옮기지 않는다 (모집단은 감사에서 고정). 세어서 밝히기만 한다. "
          "이 수가 큰 칸의 @k 는 검색기가 아니라 정의를 재고 있다.", "",
          "| 모집단 | " + " | ".join(f"@{k}" for k in KS) + " |",
          "|---|" + "---:|" * len(KS)]
    for r in rows:
        i = r["impossible"]
        L.append(f"| `{r['pop']}` | " +
                 " | ".join(f"**{i[k]}**" if i[k] > r["n"] // 2 else str(i[k]) for k in KS) + " |")
    L += ["", "## 스플릿 전체를 분모로 — \"전체 질의 중 몇 개를 검색했나\"", "",
          "위의 모든 표는 분모가 **모집단**(조회 / 다중조회 / 산술)이다. 스플릿에 있으면서 그 셋 어디에도 "
          "안 들어가는 질의는 어느 칸에서도 채점되지 않는다. 두 가지 이유로 빠진다:", "",
          "- **gold 표의 헤더 트리가 안 선다** → 그 표의 셀은 코퍼스에 색인조차 안 된다. "
          "검색은 정의상 실패다 — 아래 `하한` 은 이것을 실패로 센다.",
          "- **정답 셀이 격자 좌표로 안 풀린다** (`argmax`/`pair-argmax` 계열이 대부분) → 맞았는지 틀렸는지를 "
          "**채점할 수가 없다**. 실패로도 성공으로도 셀 수 없어서 하한/상한을 갈라 적는다.", "",
          "**논문에 \"검색 성공률\"을 한 숫자로 쓸 거면 `하한` 을 쓸 것.** 위 표 값은 채점 가능한 "
          "질의만 본 값이고, 그 선별은 우리가 했다.", "",
          "| 스플릿 | 스플릿 전체 | 채점됨 | 안 채점 | 전체@20 성공 | **하한@20** | 채점분@20 | 하한@10 | 채점분@10 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sp in sorted(seen):
        R = list(seen[sp].values())
        N, n = split_total(sp), len(R)
        ok = {k: sum(max(r["ranks"]) < k for r in R) for k in (10, 20)}
        tot = f"{N}" if N else "—"
        lo = (lambda k: f"**{ok[k] / N:.4f}**" if N else "—")
        L.append(f"| `{sp}` | {tot} | {n} | {N - n if N else '—'} | {ok[20]} | {lo(20)} | "
                 f"{ok[20] / n:.4f} | {(f'{ok[10] / N:.4f}' if N else '—')} | {ok[10] / n:.4f} |")
    L += ["", "`하한` = 성공 / 스플릿 전체 질의. `채점분` = 성공 / 채점된 질의 (위 표들과 같은 분모). "
          "참값은 둘 사이에 있고, 그 폭이 곧 채점 불가 질의의 몫이다. 스플릿 전체 수는 "
          "`data/hitab/data/<split>_samples.jsonl` 줄 수 — 데이터가 없으면 `—`.", ""]
    L += ["", "## 출처", ""] + [f"- `{r['pop']}` ← `{r['file']}`" for r in rows]
    L += ["", "셀|표@k 는 **오라클** 표 게이팅이다 — 표를 맞힌다고 가정하고 그 표 안에서만 "
          "정렬한 값. 표@1×셀|표@k 는 실제 2단계 캐스케이드가 내는 값이고, 전체@k 는 "
          "현 파이프라인(코퍼스 전체 한 번 정렬)의 값이다. m>k 인 질의는 정의상 @k 에서 0이다."]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
