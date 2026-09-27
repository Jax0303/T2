"""논문 4장(04_setup.md 21·23줄) MultiHiertt 사본·필드 대조 수치 재계산 (2026-09-28).

실행:  cd rag-agent && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/recheck_20260928/mh_release_compare.py
출력:  results/recheck_20260928/mh_release_compare.json
임베딩·검색·생성 없음. 캐시된 두 데이터 파일과 기존 검색 레코드만 읽는다.

계산 전에 고정한 조건 (원고 문장 기준)
- 재포장본: bevaya/MultiHiertt train (캐시 snapshot ae21d2d, 검색 실행과 같음).
- 공식 릴리스: yilunzhao/MultiHiertt@f18473d multihiertt_data/train.json (scripts/mh_arms.py OFFICIAL_REV).
- 23줄 "train 7,830건 전수를 대조": 두 파일에 함께 있는 uid 수. 비교 필드 — 표(tables), 문단(paragraphs),
  근거(table_evidence, text_evidence), 프로그램(program). 값이 같은지(==)로 센다. 질문·table_description 은 참고.
- 23줄 "정답 값만 유효숫자 6자리로 잘려": 정답이 다른 문항에서, 재포장본 정답(수)이 공식 정답을 유효숫자 k자리로
  반올림한 값(Python '{:.kg}')과 같은 k 를 1~15 에서 찾는다. 모든 수치 정답 쌍에서 성립하는 k 의 집합을 보고한다.
  정답 비교: 공식 정답이 수면 float 로, 문자열이면 문자열 그대로 비교.
- 21줄 "프로그램 주석 유무 구분이 공식 question_type 과 100% 일치": 대상은 원고가 같은 문장에서 네 그룹
  (212·367·71·2,235)으로 나누는 채점 2,885 질의 = 최종 규칙 재실행 레코드(results/rerun_20260926/mh/
  mh_train_s3c_records.jsonl)에서 excluded 가 없는 질의. 대응: kind arith ↔ arithmetic, lookup ↔ span_selection
  (공식 train 의 question_type 값은 이 둘뿐). 같은 문단이 앞 문장에서 7,830 → 2,908 → 2,885 로 좁히므로 "질의"는 2,885 로
  고정한다. 참고로 2,908(제외 전)과 7,830(train 전체, 공식 program 유무로 구분)도 센다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from datasets import load_dataset                          # noqa: E402
from huggingface_hub import hf_hub_download                # noqa: E402
from mh_arms import OFFICIAL_REPO, OFFICIAL_REV            # noqa: E402

OUT = ROOT / "results/recheck_20260928/mh_release_compare.json"
REC = "results/rerun_20260926/mh/mh_train_s3c_records.jsonl"

off = {d["uid"]: d for d in json.loads(Path(hf_hub_download(
    OFFICIAL_REPO, "multihiertt_data/train.json", repo_type="dataset", revision=OFFICIAL_REV)).read_text(encoding="utf-8"))}
bev = {r["uid"]: r for r in load_dataset("bevaya/MultiHiertt", split="train")}
common = sorted(off.keys() & bev.keys())

fields = {"tables": lambda o: o["tables"], "paragraphs": lambda o: o["paragraphs"],
          "table_evidence": lambda o: o["qa"]["table_evidence"], "text_evidence": lambda o: o["qa"]["text_evidence"],
          "program": lambda o: o["qa"]["program"], "question(참고)": lambda o: o["qa"]["question"],
          "table_description(참고)": lambda o: o["table_description"]}
bkey = {"question(참고)": "question", "table_description(참고)": "table_description"}
parse = lambda v: json.loads(v) if isinstance(v, str) and v[:1] == "{" else v     # 재포장본 table_description 은 JSON 문자열
same = {f: sum(1 for u in common if get(off[u]) == parse(bev[u][bkey.get(f, f)])) for f, get in fields.items()}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


diff, numeric = [], []
for u in common:
    o, b = off[u]["qa"]["answer"], bev[u]["answer"]
    eq = (num(b) == float(o)) if isinstance(o, (int, float)) else (str(o) == b)
    if not eq:
        diff.append(u)
        if isinstance(o, (int, float)) and num(b) is not None:
            numeric.append((float(o), num(b)))


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a))


k_all = [k for k in range(1, 16) if all(close(float(f"{o:.{k}g}"), b) for o, b in numeric)]
out = {
    "conditions": __doc__.split("계산 전에 고정한 조건 (원고 문장 기준)")[1].strip(),
    "n_official_train": len(off), "n_bevaya_train": len(bev), "n_compared_common_uid": len(common),
    "fields_equal": same, "fields_all_equal": {f: same[f] == len(common) for f in same},
    "answer_differs": len(diff), "answer_differs_numeric_pairs": len(numeric),
    "sig_digits_k_matching_all_differing_numeric": k_all,
    "example_346930.6": [[u, off[u]["qa"]["answer"], bev[u]["answer"]] for u in diff if off[u]["qa"]["answer"] == 346930.6],
}

rows = [json.loads(l) for l in open(ROOT / REC, encoding="utf-8") if l.strip()]
scored = [r for r in rows if "excluded" not in r]
want = {"arith": "arithmetic", "lookup": "span_selection"}
agree = sum(1 for r in scored if off[r["query_id"]]["qa"]["question_type"] == want[r["kind"]])
out["question_type_agreement"] = {"records": REC, "n_scored": len(scored), "agree": agree, "rate": agree / len(scored),
                                  "question_type_values_official_train": sorted({d["qa"]["question_type"] for d in off.values()})}
kind = lambda d: "arith" if (d["qa"]["program"] or "").strip() else "lookup"
out["question_type_agreement_참고"] = {
    "records_2908_제외_전": sum(1 for r in rows if off[r["query_id"]]["qa"]["question_type"] == want[r["kind"]]) / len(rows),
    "train_7830": sum(1 for d in off.values() if d["qa"]["question_type"] == want[kind(d)]) / len(off)}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "conditions"}, ensure_ascii=False, indent=2))
