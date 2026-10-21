"""출력 상한에 걸려 잘린 답변에 'Final answer:' 를 덧붙여 32 토큰 안에 답을 쓰게 하는 마무리 단계(budget forcing; Muennighoff 외 2025 s1 의 방식) — PREREG-2026-10-22-budget-forcing.md.
입력 = prompt_sweep 출력 jsonl(들) + 그 레코드 파일. finished=False 인 행만 골라 (원 프롬프트 + 생성문 + "\nFinal answer:") 를 그대로 이어 greedy 32 토큰 생성.
출력 = 같은 디렉터리에 <variant>.forced.jsonl (전 행 복사, 잘린 행은 raw 끝에 "\nFinal answer: <이어 쓴 글>" 추가, forced=1). 세 입력·모든 실행에 같은 규칙.
  PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 ~/vllm-venv/bin/python scripts/force_final.py --model Qwen/Qwen3.5-4B --jobs jobs.json
  jobs.json = [{"out": "<run dir>/<arm>/<variant>.jsonl", "records": "<records jsonl>", "variant": "<variant>"}, ...]
"""
import argparse, json, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "scripts")]
import prompt_variants as pv
FINAL = re.compile(r"final answer\**\s*[:：]", re.I)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--jobs", required=True); ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.89); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    jobs = json.load(open(a.jobs)); todo = []
    for j in jobs:
        rows = [json.loads(l) for l in open(j["out"], encoding="utf-8") if l.strip()]
        recs = {r["query_id"]: r for r in map(json.loads, open(j["records"], encoding="utf-8"))}
        for r in rows:
            if not r["finished"] and not FINAL.search(r["raw"] or ""):
                todo.append((j, r, recs[r["query_id"]]))
        j["_rows"] = rows
    print(f"[force] 잘린 행 {len(todo)}개 / 파일 {len(jobs)}개", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    LIMIT = 8192 - a.max_new_tokens - 8; tail = tok("\nFinal answer:", add_special_tokens=False)["input_ids"]
    prompts = []; n_cut = 0
    for j, r, rec in todo:                                                 # 토큰 id 로 이어 붙인다(디코드 왕복 없음). 넘치면 생성문 앞부분을 잘라 끝부분을 보존.
        chat_ids = tok(tok.apply_chat_template([{"role": "user", "content": pv.build(j["variant"], rec["question"], rec["context"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
        raw_ids = tok(r["raw"], add_special_tokens=False)["input_ids"]
        room = LIMIT - len(chat_ids) - len(tail)
        if len(raw_ids) > room: raw_ids = raw_ids[-room:]; n_cut += 1
        prompts.append({"prompt_token_ids": chat_ids + raw_ids + tail})
    print(f"[force] 프롬프트 {len(prompts)}개, 생성문 앞을 자른 것 {n_cut}, 최대 길이 {max((len(p['prompt_token_ids']) for p in prompts), default=0)}", flush=True)
    if a.dry_run: return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=8192, seed=42, enable_prefix_caching=True, max_num_seqs=16, language_model_only=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, seed=42)
    t0 = time.time(); outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
    for (j, r, rec), o in zip(todo, outs):
        cont = o.outputs[0].text.split("\n")[0].strip()
        r["raw"] = r["raw"] + "\nFinal answer: " + cont; r["forced"] = 1; r["forced_text"] = cont; r["finished"] = True
    for j in jobs:
        p = Path(j["out"]); q = p.with_suffix(".forced.jsonl")
        with open(q, "w", encoding="utf-8") as f:
            for r in j["_rows"]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[force] {q}: 전 {len(j['_rows'])}행, 덧붙임 {sum(r.get('forced', 0) for r in j['_rows'])}", flush=True)
    print(f"[force] 끝 {time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__": main()
