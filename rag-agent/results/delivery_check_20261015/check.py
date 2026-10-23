"""검색 결과 → LLM 입력 전달 점검 (2026-10-15, 사용자 요구 2번: "오류 없다고 했는데 확인").
(1) 답변 기록의 입력 해시(context_sha256 = digest(context)) == 전달 레코드 본문 — 6개 882 실행 + 순서 시험 2개.
(2) 본 방법 레코드: 앞 20줄 == 검색 기록(retrieval_r100) 상위 20 문장, 줄 수 == 셀 수, 토큰 ≤ LangChain 토큰(예산 규칙), 정답 포함 판정 재현.
(3) LangChain 레코드: 본문 == 청크 기록 결합 본문, 셀 수 == 상위 4 청크 합집합, 정답 포함 판정 재현.
(4) 검색을 처음부터 다시 돌린 20문항(uids20.json, 새 임베딩 캐시)과 기록 대조 — 본 방법은
    scripts/mh_arms.py --split train --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --k-ladder 1,5,10,20
      --dump-ranked 100 --unit cell --template s3c --only-uids uids20.json (--cache-dir 새 디렉터리) → fresh_ours_records_20.jsonl
    LangChain 은 scripts/baselines/langchain_chunk.py --stage retrieve --limit 20 → fresh_langchain_records_20.jsonl
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/delivery_check_20261015/check.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rag_agent.eval.artifacts import digest                                       # noqa: E402

HERE = Path(__file__).resolve().parent
L = lambda p: {r["query_id"]: r for r in map(json.loads, open(ROOT / p, encoding="utf-8"))}
recs = {"ours": L("results/cellform_20261012/C_ours_budget_records.jsonl"), "langchain": L("results/cellform_20261012/A_chunk_md_records.jsonl"),
        "fulldoc": L("results/gpt4o_20261013/fulldoc_records.jsonl"), "closed": L("results/gpt4o_20261013/closed_records.jsonl")}
runs = {"Qwen 본 방법": ("ours", "results/cellform_20261012/C_ours_budget"), "Qwen LangChain": ("langchain", "results/cellform_20261012/A_chunk_md"),
        "GPT-4o 본 방법": ("ours", "results/gpt4o_20261013/ans_ours"), "GPT-4o LangChain": ("langchain", "results/gpt4o_20261013/ans_langchain"),
        "GPT-4o 표 전부": ("fulldoc", "results/gpt4o_20261013/ans_fulldoc"), "GPT-4o 질문만": ("closed", "results/gpt4o_20261013/ans_closed"),
        "Qwen 본 방법 Context-first": ("ours", "results/order_pilot_20261015/C_ours_cf"), "Qwen LangChain Context-first": ("langchain", "results/order_pilot_20261015/A_chunk_cf")}
out = ["(1) 답변 기록 입력 해시 == 전달 레코드 본문"]
for name, (rk, d) in runs.items():
    A = L(d + "/langchain_chunk_answer.jsonl"); R = recs[rk]
    out.append(f"  {name}: 해시 일치 {sum(A[q]['context_sha256'] == digest(R[q]['context']) for q in A)}/{len(A)}, "
               f"정답 포함 표시 일치 {sum(A[q]['retrieval_correct'] == R[q]['correct'] for q in A)}/{len(A)}")
R100 = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl", encoding="utf-8")) if "doc" in r}
LC = L("results/mh_only_20260928/baselines/langchain_chunk/langchain_chunk_records.jsonl")
O, A_ = recs["ours"], recs["langchain"]
c = dict(pre=0, nl=0, bud=0, gold=0)
for q, r in O.items():
    lines = r["context"].split("\n"); n = len(lines); k = min(20, n)
    c["pre"] += lines[:k] == R100[q]["doc"]["context"][:k]
    c["nl"] += n == r["cells_in_context"]
    c["bud"] += r["n_tok_ctx"] <= A_[q]["n_tok_ctx"]
    c["gold"] += (set(r["gold_ids"]) <= {u[0] for u in R100[q]["doc"]["ranked_units"][:n]}) == bool(r["correct"])
out.append(f"(2) 본 방법 레코드 882: 앞 20줄==검색 기록 {c['pre']}, 줄 수==셀 수 {c['nl']}, 토큰≤LangChain {c['bud']}, 정답 포함 재현 {c['gold']}")
u4 = lambda q: set().union(*(set(u["cells"]) for u in LC[q]["ranked_units"][:4]))
out.append(f"(3) LangChain 레코드 882: 본문==청크 기록 {sum(A_[q]['context'] == LC[q]['context'] for q in A_)}, "
           f"셀 수==상위4 합집합 {sum(A_[q]['cells_in_context'] == len(u4(q)) for q in A_)}, 정답 포함 재현 {sum((set(A_[q]['gold_ids']) <= u4(q)) == bool(A_[q]['correct']) for q in A_)}")
fo = {r["query_id"]: r for r in map(json.loads, open(HERE / "fresh_ours_records_20.jsonl", encoding="utf-8")) if "doc" in r}
fl = {r["query_id"]: r for r in map(json.loads, open(HERE / "fresh_langchain_records_20.jsonl", encoding="utf-8"))}
out.append(f"(4) 새 검색 20문항: 본 방법 상위 100 셀 순서 같음 {sum([u[0] for u in fo[q]['doc']['ranked_units'][:100]] == [u[0] for u in R100[q]['doc']['ranked_units'][:100]] for q in fo)}/{len(fo)}, "
           f"882 입력 앞 20줄과 같음 {sum(O[q]['context'].split(chr(10))[:min(20, len(O[q]['context'].split(chr(10))))] == fo[q]['doc']['context'][:min(20, len(O[q]['context'].split(chr(10))))] for q in fo)}/{len(fo)} | "
           f"LangChain 상위 4 청크 같음 {sum([u['cells'] for u in fl[q]['ranked_units'][:4]] == [u['cells'] for u in LC[q]['ranked_units'][:4]] for q in fl)}/{len(fl)}, 882 입력 본문과 같음 {sum(fl[q]['context'] == A_[q]['context'] for q in fl)}/{len(fl)}")
print("\n".join(out)); (HERE / "check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
