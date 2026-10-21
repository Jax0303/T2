"""셀 고르기 단계(줄 단위 선택) — PREREG-2026-10-21-select-dev.md.
입력 레코드의 context 를 줄로 쪼개 번호를 붙여 같은 LLM(비생각, temperature 0)에 주고, 답에 필요한 줄 번호만 받아 그 줄의 원문만 남긴 레코드를 쓴다.
세 입력(LangChain 청크·잘라낸 표·셀 문장 목록)에 같은 지시문·같은 규칙. 구조 줄 규칙(마크다운 표에만 해당, A·D 같게): 어떤 표의 값 줄을 하나라도 남기면
그 표의 제목 줄("Table N")·머리글 줄(표 블록 첫 줄)·구분 줄("| --- |")도 남긴다. 번호를 하나도 못 받으면 빈 문맥(자동 대체 없음, empty=1).
  PYTHONPATH=.:scripts HF_HUB_OFFLINE=1 ~/vllm-venv/bin/python scripts/select_lines.py --model Qwen/Qwen3.5-4B --records a.jsonl,b.jsonl --out-dir DIR --arm A,D
  --selftest 로 재조립 규칙만 점검.
"""
import argparse, hashlib, json, re, sys, time
from pathlib import Path

INSTR = ("Below is a question about tables from a financial report, followed by numbered context lines. "
         "Select the lines needed to answer the question. If the question needs a calculation or a comparison over several values, "
         "include EVERY line that holds a value the calculation or comparison needs, and any header or label lines needed to interpret those values. "
         "Reply with only the line numbers, comma-separated, in increasing order. Do not explain and do not repeat the lines.")

def split_lines(ctx):
    return [l for l in ctx.split("\n") if l.strip()]

def is_row(l): return l.lstrip().startswith("|")
def is_sep(l): return is_row(l) and set(l.replace("|", "").replace("-", "").strip()) == set()

def structure_closure(lines, keep):
    """마크다운 표 블록마다: 값 줄이 하나라도 남으면 제목 줄·머리글 줄·구분 줄을 더한다."""
    keep = set(keep); n = len(lines); i = 0
    while i < n:
        if not is_row(lines[i]): i += 1; continue
        j = i
        while j < n and is_row(lines[j]): j += 1
        block = list(range(i, j)); header = [block[0]] + [k for k in block[1:] if is_sep(lines[k])]
        title = [i - 1] if i > 0 and re.match(r"^Table \d+\s*$", lines[i - 1]) else []
        if any(k in keep for k in block if k not in header): keep.update(header + title)
        i = j
    return sorted(keep)

def parse_ids(raw, n):
    nums = [int(x) for x in re.findall(r"\d+", raw or "")]
    return sorted({x for x in nums if 1 <= x <= n}), sum(1 for x in nums if not 1 <= x <= n)

def rebuild(ctx, raw):
    lines = split_lines(ctx); ids, bad = parse_ids(raw, len(lines))
    keep0 = [i - 1 for i in ids]; keep = structure_closure(lines, keep0) if keep0 else []
    return "\n".join(lines[k] for k in keep), {"n_lines": len(lines), "n_sel": len(keep0), "n_kept": len(keep), "n_invalid_ids": bad, "empty": int(not keep)}

def user_text(question, ctx):
    lines = split_lines(ctx)
    return INSTR + "\n\nQuestion: " + question + "\n\nContext lines:\n" + "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines)) + "\n\nLine numbers:"

def selftest():
    ctx = "Table 1\n| | 2009 | 2008 |\n| --- | --- | --- |\n| Revenue | 1 | 2 |\n| Cost | 3 | 4 |\nTable 2\n| | A |\n| --- | --- |\n| X | 5 |"
    t, m = rebuild(ctx, "5"); assert t == "Table 1\n| | 2009 | 2008 |\n| --- | --- | --- |\n| Cost | 3 | 4 |", t; assert m["n_kept"] == 4
    t, m = rebuild(ctx, "2"); assert t == "| | 2009 | 2008 |", t          # 머리글만 고르면 그 줄만
    t, m = rebuild(ctx, "none"); assert t == "" and m["empty"] == 1
    a = "| H | 2009 |\n| r1 | 1 |\n| r2 | 2 |\nsome text\n| H2 | x |\n| r3 | 3 |"
    t, m = rebuild(a, "3, 6, 99"); assert t == "| H | 2009 |\n| r2 | 2 |\n| H2 | x |\n| r3 | 3 |", t; assert m["n_invalid_ids"] == 1
    c = "a > b: 1\nc > d: 2\ne > f: 3"; t, m = rebuild(c, "2,3"); assert t == "c > d: 2\ne > f: 3" and m["n_kept"] == 2
    print("selftest ok")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--records", default=""); ap.add_argument("--arm", default="")
    ap.add_argument("--out-dir", default=""); ap.add_argument("--max-new-tokens", type=int, default=512); ap.add_argument("--gpu-mem", type=float, default=0.89)
    ap.add_argument("--max-model-len", type=int, default=8192); ap.add_argument("--max-num-seqs", type=int, default=32); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"language_model_only": True} if "Qwen3.5" in a.model else {}
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len, seed=42, enable_prefix_caching=True,
              max_num_batched_tokens=4096, max_num_seqs=a.max_num_seqs, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, seed=42)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for path, arm in zip(a.records.split(","), a.arm.split(",")):
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        prompts = [tok.apply_chat_template([{"role": "user", "content": user_text(r["question"], r["context"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in recs]
        t0 = time.time(); outs = llm.generate(prompts, sp, use_tqdm=False)
        n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
        with open(out / f"{arm}F_records.jsonl", "x", encoding="utf-8") as f:
            for r, o in zip(recs, outs):
                g = o.outputs[0]; text, m = rebuild(r["context"], g.text)
                row = {**r, "context": text, "n_tok_ctx": n_tok(text), "n_tok_ctx_before": r.get("n_tok_ctx"), "context_before_sha256": hashlib.sha256(r["context"].encode()).hexdigest(),
                       "sel_raw": g.text.strip()[:2000], "sel_finished": g.finish_reason == "stop", "sel_n_tok_in": len(o.prompt_token_ids), "sel_n_tok_out": len(g.token_ids), **m}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept = [json.loads(l) for l in open(out / f"{arm}F_records.jsonl", encoding="utf-8")]
        print(f"[select] {arm}: {len(kept)}건, {time.time()-t0:.0f}s, 줄 {sum(x['n_lines'] for x in kept)/len(kept):.1f} → 남김 {sum(x['n_kept'] for x in kept)/len(kept):.1f}, "
              f"토큰 {sum(x['n_tok_ctx_before'] or 0 for x in kept)/len(kept):.0f} → {sum(x['n_tok_ctx'] for x in kept)/len(kept):.0f}, 빈 문맥 {sum(x['empty'] for x in kept)}, 잘림 {sum(not x['sel_finished'] for x in kept)}", flush=True)
    (out / "select_meta.json").write_text(json.dumps({"model": a.model, "instr": INSTR, "max_new_tokens": a.max_new_tokens, "temperature": 0, "thinking": False, "records": a.records, "arm": a.arm, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=1))

if __name__ == "__main__": main()
