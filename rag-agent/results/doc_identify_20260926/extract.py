"""2026-09-26 문서 특정 가능성 판정용 표본 + 회사 이름 잔존 집계 (리더 없음, 검색 없음).
모집단 = MultiHiertt train 채점 대상 2,878문항(mh_arms: 표 근거만, gold 해석 성공; 머리글 규칙 v2, 라벨 없음 = c).
1) SEED 로 100문항 무작위 추출 -> sample100.csv. 첫 문단 = paragraphs 중 비어 있지 않고 `## Table N ##` 표시가 아닌
   첫 문단의 앞 200자. 정답 셀 경로 = 행 머리글 경로 > 열 머리글 경로(c 셀 문장의 경로 부분), 여러 셀은 ' || '.
   판정 열 (가)(나)는 비워 둔다.
2) 회사 이름 규칙(COMPANY): 대문자로 시작하는 단어 1~6개(사이 of/and/& 허용) + 회사 접미사
   (& Co./Inc/Incorporated/Corporation/Corp/Company/Companies/Co./LLC/L.L.C./L.P./Ltd/Limited/plc/PLC/N.A./Bancorp/Holding(s),
   첫 글자 대문자 꼴과 전부 대문자 꼴. 소문자 company 는 제외).
   이름 안의 마지막 The/Our/This/That/Its/Their 뒤만 이름으로 본다("The Company" 는 이름 없음).
   남은 단어가 전부 GENERIC 이면 버린다. 검사 대상 = 본문 문단(표 표시 제외), 표(HTML 태그 제거), 첫 문단.
   문서 = 2,878문항의 문서를 내용(paragraphs+tables) 기준으로 중복 제거한 것. 표본 50개 = 그 문서에서 SEED 로 추출.
3) 같은 규칙을 질문 문장에 적용. 문서 회사 이름 = 본문 문단 + 표에서 찾은 이름. 이름 비교는 NORM
   (소문자, '.' ',' 삭제, 공백 정리) 뒤 문자열 일치. 질문 이름 중 하나라도 정답 문서 이름 집합에 있으면 같음.
   sample100.csv 에 question_has_company / question_company / same_as_gold_doc_company 열 추가.
4) 회사 이름(NORM)별 문서 수(중복 제거 문서 기준), 문서별 서로 다른 이름 수.
실행: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python results/doc_identify_20260926/extract.py
"""
import csv
import hashlib
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import mh_arms as mh                                                  # noqa: E402

SEED = 20260926
MARK = re.compile(r"^\s*## Table \d+ ##\s*$")
SUFFIX = (r"(?:&\s+Co\.?|&\s+CO\.?|Inc\.?|Incorporated|Corporation|Corp\.?|Companies|Company|Co\.|LLC|L\.L\.C\.|L\.P\."
          r"|Ltd\.?|Limited|plc|PLC|N\.A\.|Bancorp(?:oration)?|Holdings?"
          r"|INC\.?|INCORPORATED|CORPORATION|CORP\.?|COMPANIES|COMPANY|CO\.|LTD\.?|LIMITED|BANCORP|HOLDINGS?)")
WORD = r"[A-Z][A-Za-z0-9&.'\-]*"
COMPANY = re.compile(rf"\b({WORD}(?:\s+(?:(?:of|and|&)\s+)?{WORD}){{0,5}}),?\s+{SUFFIX}(?![A-Za-z])")
DET = {"The", "THE", "Our", "OUR", "This", "That", "Its", "Their"}
GENERIC = {"parent", "holding", "holdings", "total", "consolidated", "combined", "operating", "subsidiary",
           "subsidiaries", "registrant", "insurance", "life", "non-life", "financial", "statements", "cable",
           "television", "other", "each", "such", "domestic", "foreign", "trust", "investment", "reporting",
           "acquired", "new", "management", "service", "services", "company", "companies", "corporation",
           "of", "and", "&"}


def companies(text):
    out = []
    for m in COMPANY.finditer(text):
        words = m.group(1).split()
        cut = max((k for k, w in enumerate(words) if w in DET), default=-1)
        words = words[cut + 1:]
        if words and not all(w.lower() in GENERIC for w in words):
            out.append(" ".join(words) + m.group(0)[len(m.group(1)):])
    return out


def first_paragraph(paras):
    return next((" ".join(p.split()) for p in paras if p.strip() and not MARK.match(p)), "")


def norm(name):
    return " ".join(name.lower().replace(".", "").replace(",", "").split())


def found_in(u):
    paras, tabs = docs[u][2], docs[u][0]
    body = [p for p in paras if p.strip() and not MARK.match(p)]
    return {"first_paragraph": companies(first_paragraph(paras)),
            "paragraphs": [c for p in body for c in companies(p)],
            "tables": [c for t in tabs for c in companies(re.sub(r"<[^>]+>", " ", t))]}


def doc_names(f):
    return {norm(c) for c in f["paragraphs"] + f["tables"]}


queries, docs, _ = mh.load_population("train")
tables, hdr = mh.build_tables(docs, "v2", "none")
live = {(tid, i, j) for tid, tab in tables.items() for i, r in enumerate(tab.table.data)
        for j, v in enumerate(r) if str(v).strip()}
pop = [q for q in mh.resolve_gold(queries, tables, hdr, live) if not q["excluded"] and q["gold"]]
assert len(pop) == 2878


def path(cell):
    t = tables[cell[0]].table
    return " > ".join([*t.row_path(cell[1]), *t.col_path(cell[2])])


# ---- 1) 100문항
sample = random.Random(SEED).sample(sorted(pop, key=lambda q: q["uid"]), 100)
with open(OUT / "sample100.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["no", "query_id", "question", "first_paragraph_200", "gold_cell_paths", "gold_cell_ids",
                "(가) 질문만으로 문서 특정 가능(예/아니오)", "(나) 근거 단어",
                "question_has_company", "question_company", "same_as_gold_doc_company"])
    for n, q in enumerate(sample, 1):
        cells = sorted(q["gold"])
        qn = companies(q["question"])
        same = ("예" if {norm(c) for c in qn} & doc_names(found_in(q["uid"])) else "아니오") if qn else ""
        w.writerow([n, q["uid"], q["question"], first_paragraph(docs[q["uid"]][2])[:200],
                    " || ".join(path(c) for c in cells), " || ".join(mh.cell_id(c, hdr) for c in cells), "", "",
                    "예" if qn else "아니오", " | ".join(qn), same])

# ---- 2) 회사 이름
key_of = {}
for q in pop:
    paras, tabs = docs[q["uid"]][2], docs[q["uid"]][0]
    key_of[q["uid"]] = hashlib.sha256(json.dumps([paras, tabs]).encode()).hexdigest()
uniq = {}
for q in sorted(pop, key=lambda q: q["uid"]):
    uniq.setdefault(key_of[q["uid"]], q["uid"])
found = {k: found_in(u) for k, u in uniq.items()}
scopes = {"first_paragraph": lambda f: f["first_paragraph"], "paragraphs": lambda f: f["paragraphs"],
          "tables": lambda f: f["tables"], "paragraphs_or_tables": lambda f: f["paragraphs"] or f["tables"]}
summary = {"seed": SEED, "n_questions": len(pop), "n_unique_documents": len(uniq), "rule": __doc__, "by_scope": {}}
for s, get in scopes.items():
    d = sum(bool(get(f)) for f in found.values())
    qn = sum(bool(get(found[key_of[q["uid"]]])) for q in pop)
    summary["by_scope"][s] = {"documents": d, "documents_ratio": round(d / len(uniq), 4),
                              "questions": qn, "questions_ratio": round(qn / len(pop), 4)}
fp_len = [len(first_paragraph(docs[u][2])) for u in uniq.values()]
summary["first_paragraph_chars"] = {"median": st.median(fp_len), "mean": round(st.fmean(fp_len), 1),
                                    "under_50": sum(x < 50 for x in fp_len)}
# ---- 3) 질문의 회사 이름
names_of = {k: doc_names(f) for k, f in found.items()}
rows = []
for q in pop:
    qn = companies(q["question"])
    if qn:
        rows.append({"uid": q["uid"], "question": q["question"], "question_company": qn,
                     "same_as_gold_doc_company": bool({norm(c) for c in qn} & names_of[key_of[q["uid"]]]),
                     "gold_doc_company": sorted(names_of[key_of[q["uid"]]])})
with open(OUT / "question_companies.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
same_n = sum(r["same_as_gold_doc_company"] for r in rows)
summary["questions_with_company"] = {"questions": len(rows), "questions_ratio": round(len(rows) / len(pop), 4),
                                     "same_as_gold_doc_company": same_n,
                                     "same_ratio_of_with_company": round(same_n / len(rows), 4) if rows else None}

# ---- 4) 회사 이름별 문서 수
per_name = Counter(n for s in names_of.values() for n in s)
top = per_name.most_common(10)
summary["documents_per_company_name"] = {
    "distinct_names": len(per_name), "median": st.median(per_name.values()),
    "mean": round(st.fmean(per_name.values()), 2), "max": top[0][1], "top10": top,
    "names_in_one_document": sum(c == 1 for c in per_name.values()),
    "documents_with_any_name": sum(bool(s) for s in names_of.values()),
    "documents_with_one_name": sum(len(s) == 1 for s in names_of.values())}
(OUT / "company_names.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

s50 = random.Random(SEED).sample(sorted(uniq), 50)
with open(OUT / "sample50_first_paragraph.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["no", "doc_uid", "first_paragraph_200", "company_in_first_paragraph", "company_anywhere_first3"])
    for n, k in enumerate(s50, 1):
        fnd = found[k]
        anyw = list(dict.fromkeys(fnd["paragraphs"] + fnd["tables"]))[:3]
        w.writerow([n, uniq[k], first_paragraph(docs[uniq[k]][2])[:200], " | ".join(dict.fromkeys(fnd["first_paragraph"])),
                    " | ".join(anyw)])

print(json.dumps({k: v for k, v in summary.items() if k != "rule"}, indent=1, ensure_ascii=False))
