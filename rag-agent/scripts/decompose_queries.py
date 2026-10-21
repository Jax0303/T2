"""질문 쪼개기 — PREREG-2026-10-21-decompose-dev.md. 질문마다 "필요한 표 값 하나당 검색 문구 하나"를 같은 LLM(비생각, temperature 0)에게 받아 JSON {uid: [문구...]} 로 쓴다.
정답·표 내용은 주지 않는다(질문만). 문구 0개면 원 질문만 쓰게 빈 목록.
  PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 ~/vllm-venv/bin/python scripts/decompose_queries.py --model Qwen/Qwen3.5-4B --records a.jsonl --out sub.json
"""
import argparse, json, re, time
from pathlib import Path
INSTR = ("The question below asks about numbers in the tables of a financial report. Break it into the separate table values it needs. "
         "Write one short search phrase per value, naming the row item and the column (such as the year or category) the way they would appear in the table. "
         "Write between 1 and 4 phrases, one per line, and nothing else.")
def parse(raw):
    out = []
    for l in (raw or "").split("\n"):
        l = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip().strip('"')
        if l and len(out) < 4: out.append(l)
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--records", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.6); a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    recs = [json.loads(l) for l in open(a.records, encoding="utf-8") if l.strip()]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=2048, seed=42, max_num_seqs=64, language_model_only=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR + "\n\nQuestion: " + r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in recs]
    t0 = time.time(); outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160, seed=42), use_tqdm=False)
    sub = {r["query_id"]: parse(o.outputs[0].text) for r, o in zip(recs, outs)}
    Path(a.out).write_text(json.dumps(sub, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.out).with_suffix(".raw.jsonl").write_text("\n".join(json.dumps({"query_id": r["query_id"], "question": r["question"], "raw": o.outputs[0].text}, ensure_ascii=False) for r, o in zip(recs, outs)), encoding="utf-8")
    n = [len(v) for v in sub.values()]
    print(f"[decompose] {len(sub)}건 {time.time()-t0:.0f}s, 문구 수 평균 {sum(n)/len(n):.2f}, 0개 {sum(x==0 for x in n)}, 분포 {dict(sorted(__import__('collections').Counter(n).items()))}")
if __name__ == "__main__": main()
