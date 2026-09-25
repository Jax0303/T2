"""사람 판정 뒤 자동 검증. 사전등록: PREREG-2026-09-26-judging-criteria.md §5.
(가)=예 인 문항마다 (나)를 쉼표로 나눈 조각이 모두 들어 있는 검색 단위 수를 센다.
  mh    : 단위 = MultiHiertt train 문서 묶음 1,105개(본문 문단+표 내용이 같은 문서를 하나로 묶음).
          글 = 본문 문단 + 표(HTML 태그 제거).
  hitab : 단위 = HiTab test 표 538개. 글 = 색인 제목(with_page_title) + 비어 있지 않은 셀마다 '행 경로 > 열 경로'.
조각 찾기 = 소문자·HTML 엔티티 풀기·공백 정리 뒤, 앞뒤가 영문자·숫자가 아닌 자리에서 문자열 일치.
문단·표·경로 사이는 ' | ' 로 이어 조각이 경계를 넘어 맞지 않게 한다.
검증 예 = 조각을 모두 담은 단위가 정확히 1개이고 그것이 정답 단위. 0개·2개 이상·1개지만 정답 아님은 검증 아니오.
(가)=아니오 는 검증 '해당 없음'. 조각이 질문에 없으면 '질문에 모두 있음'=아니오 로 적는다(검증은 그대로 한다).
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/judge_verify.py {mh|hitab} <판정.csv> <출력 stem>
출력: <stem>.csv (문항별 사람 판정과 검증 나란히), <stem>.json (사람 (가) × 검증 표). 이미 있으면 멈춘다.
"""
import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["번호", "query_id", "질문", "사람 (가)", "(나)", "조각 수", "질문에 모두 있음",
          "모든 조각을 담은 단위 수", "정답 단위 포함", "검증"]


def norm(s):
    return " ".join(html.unescape(s).lower().split())


def pieces(cell):
    return [p for p in (norm(x) for x in cell.split(",")) if p]


def has(text, p):
    return re.search(rf"(?<![0-9a-z]){re.escape(p)}(?![0-9a-z])", text) is not None


def verify(ps, texts, gold):
    hits = [k for k, t in texts.items() if all(has(t, p) for p in ps)]
    return len(hits), gold in hits, "예" if hits == [gold] else "아니오"


def mh_units():
    argv, sys.argv = sys.argv, sys.argv[:1]
    sys.path.insert(0, str(ROOT / "scripts"))
    import mh_arms as mh
    sys.argv = argv
    _, docs, _ = mh.load_population("train")
    key = {u: hashlib.sha256(json.dumps([d[2], d[0]]).encode()).hexdigest() for u, d in docs.items()}
    texts = {}
    for u in sorted(docs):
        texts.setdefault(key[u], norm(" | ".join([*docs[u][2], *(re.sub(r"<[^>]+>", " ", t) for t in docs[u][0])])))
    assert len(texts) == 1105
    return texts, key


def hitab_units():
    sys.path.insert(0, str(ROOT))
    from rag_agent.bench import hitab_grid as hg
    from rag_agent.serialization.caption import with_page_title
    pages = json.loads((ROOT / "results/tableconf/totto_page_titles.json").read_text())
    recs = map(json.loads, open(ROOT / "results/retrieval_accuracy/t_s3c_split_labelabl_records.jsonl"))
    gold = {r["query_id"]: r["table_id"] for r in recs}
    texts = {}
    for tid in sorted(set(gold.values())):
        tab = hg.load_table(tid, str(ROOT / "data/hitab"))
        t = tab.table
        lines = [with_page_title(tab.title, pages.get(tid))]
        lines += [" > ".join([*t.row_path(i), *t.col_path(j)])
                  for i in range(t.n_rows) for j in range(t.n_cols) if str(t.data[i][j]).strip()]
        texts[tid] = norm(" | ".join(lines))
    assert len(texts) == 538
    return texts, gold


if __name__ == "__main__":
    ds, src, stem = sys.argv[1:4]
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    col = lambda r, prefix: next(v for k, v in r.items() if k.startswith(prefix)).strip()   # noqa: E731
    texts, gold = {"mh": mh_units, "hitab": hitab_units}[ds]()
    out = []
    for r in rows:
        a, words, q = col(r, "(가)"), col(r, "(나)"), r.get("question") or r["질문 원문"]
        if a not in ("예", "아니오"):
            raise SystemExit(f"{r['query_id']}: (가)가 예/아니오가 아니다: {a!r}")
        row = {"번호": r.get("번호") or r.get("no"), "query_id": r["query_id"], "질문": q, "사람 (가)": a, "(나)": words}
        if a == "예":
            ps = pieces(words)
            if not ps:
                raise SystemExit(f"{r['query_id']}: (가)=예 인데 (나)가 비었다")
            n, gin, v = verify(ps, texts, gold[r["query_id"]])
            row |= {"조각 수": len(ps), "질문에 모두 있음": "예" if all(has(norm(q), p) for p in ps) else "아니오",
                    "모든 조각을 담은 단위 수": n, "정답 단위 포함": "예" if gin else "아니오", "검증": v}
        else:
            row["검증"] = "해당 없음"
        out.append(row)
    with open(f"{stem}.csv", "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    summary = {"dataset": ds, "source": src, "units": len(texts), "rows": len(out),
               "human_by_verify": dict(Counter(f"{x['사람 (가)']}|{x['검증']}" for x in out)),
               "question_missing_pieces": sum(x.get("질문에 모두 있음") == "아니오" for x in out)}
    with open(f"{stem}.json", "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
