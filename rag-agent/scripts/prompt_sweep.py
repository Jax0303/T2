#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""지시문 변형 × 전달 입력 답변 생성 — PREREG-2026-10-15-prompt-sweep.md.

로컬 HF 모델(bf16 무양자화), 사용자 메시지 하나(system 없음), 비생각 모드, greedy, 고정 묶음(왼쪽 패딩) 생성.
변형 지시문은 `scripts/prompt_variants.py`. 출력은 <out-dir>/<arm>/<variant>.jsonl (문항 한 줄, --resume 으로 이어 쓴다).

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/prompt_sweep.py --model Qwen/Qwen3-4B-Instruct-2507 \
      --records results/prompt_sweep_20261015/dev_C_records.jsonl --arm C --variants R0X0F0B0D0C0E0,R0X1F1B1D1C2E1 \
      --out-dir results/prompt_sweep_20261015/stage0_qwen3_4b
  --variants factorial:XFBDCE  = 그 축들의 모든 조합(나머지 축 0)
  --check-batch N              = 앞 N 문항을 한 건씩과 묶음으로 둘 다 생성해 같은지 센다(배치 확인)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import prompt_variants as pv                                        # noqa: E402

SEED = 42


def load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model, info = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda"},
                                                      output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "mtp" not in k]
    if missing:                                                     # 멀티모달 체크포인트에서 글 모델만 못 찾는 경우
        print(f"[load] AutoModelForCausalLM 이 가중치 {len(missing)}개를 못 찾음(예 {missing[:2]}) → ForConditionalGeneration", flush=True)
        from transformers import AutoModelForImageTextToText
        del model
        torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def chat(tok, user: str, thinking: bool) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def generate(tok, model, prompts: list, max_new: int, thinking: bool):
    """왼쪽 패딩 묶음 greedy 생성 → [(text, n_in, n_out, finished, think_tokens)]."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    res = []
    n_in = enc["attention_mask"].sum(1).tolist()
    end_ids = {tok.eos_token_id, tok.pad_token_id} | ({tok.convert_tokens_to_ids("<|im_end|>")} if "<|im_end|>" in tok.get_vocab() else set())
    think_end = tok.convert_tokens_to_ids("</think>") if "</think>" in tok.get_vocab() else None
    for i in range(len(prompts)):
        gen = out[i, enc["input_ids"].shape[1]:].tolist()
        finished = False
        for j, t in enumerate(gen):
            if t in end_ids:
                gen, finished = gen[:j], True
                break
        cut = 0
        if thinking and think_end is not None and think_end in gen:
            cut = len(gen) - gen[::-1].index(think_end)
        res.append((tok.decode(gen[cut:], skip_special_tokens=True).strip(), n_in[i], len(gen), finished, cut))
    return res


def batches(items, batch_tokens: int, max_batch: int, max_new: int):
    """items = [(len, idx)] 길이 내림차순 → 묶음(패딩 포함 토큰 합 ≤ batch_tokens)."""
    cur = []
    for n, i in items:
        if cur and ((len(cur) + 1) * (cur[0][0] + max_new) > batch_tokens or len(cur) >= max_batch):
            yield cur; cur = []
        cur.append((n, i))
    if cur:
        yield cur


def run_variant(a, tok, model, recs, variant: str, out_path: Path, done: dict):
    import torch
    pending = [r for r in recs if r["query_id"] not in done]
    if not pending:
        return 0
    users = [pv.build(variant, r["question"], r["context"]) for r in pending]
    prompts = [chat(tok, u, a.thinking) for u in users]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(((n, i) for i, n in enumerate(lens)), reverse=True)
    psha = hashlib.sha256((pv.instruction(variant) + "\n" + pv.examples_block(variant)).encode()).hexdigest()
    t0, n_done = time.time(), 0
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for b in batches(order, a.batch_tokens, a.max_batch, a.max_new_tokens):
            idx = [i for _, i in b]
            todo, results = [idx], {}
            while todo:                                             # 메모리 부족이면 반으로 쪼개 다시
                cur = todo.pop()
                try:
                    t = time.time()
                    res = generate(tok, model, [prompts[i] for i in cur], a.max_new_tokens, a.thinking)
                    sec = (time.time() - t) / len(cur)
                    for i, r in zip(cur, res):
                        results[i] = (r, sec, len(cur))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(cur) == 1:
                        raise
                    h = len(cur) // 2
                    print(f"  [oom] 묶음 {len(cur)} → {h}+{len(cur) - h}", flush=True)
                    todo += [cur[h:], cur[:h]]
            for i in idx:
                (text, n_in, n_out, fin, think), sec, bs = results[i]
                r = pending[i]
                row = {"query_id": r["query_id"], "arm": a.arm, "variant": variant, "model": a.model, "thinking": a.thinking,
                       "raw": text, "n_tok": n_in, "n_tok_out": n_out, "finished": fin, "think_tokens": think,
                       "layer": r["layer"], "kind": r["kind"], "m": r["m"], "retrieval_correct": r["correct"],
                       "context_sha256": hashlib.sha256(r["context"].encode()).hexdigest(), "prompt_sha256": psha,
                       "max_new_tokens": a.max_new_tokens, "batch": bs, "seconds": round(sec, 3)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
            f.flush(); os.fsync(f.fileno())
            print(f"  {variant}/{a.arm} {len(done) + n_done}/{len(recs)}  {time.time() - t0:.0f}s  (묶음 {len(idx)}, 최장 {b[0][0]})", flush=True)
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--arm", required=True, help="A = LangChain 청크 / C = 본 방법 셀 문장 (이름표)")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-tokens", type=int, default=24000, help="묶음 안 (가장 긴 입력 + 생성 상한) × 건수 상한")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="문항 부분집합(json 목록)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--thinking", action="store_true", help="생각 모드(참고 조건). </think> 뒤만 raw 에 남긴다")
    ap.add_argument("--check-batch", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 첫 문항 프롬프트만 출력")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.records, encoding="utf-8") if l.strip()]
    if a.ids:
        keep = set(json.load(open(a.ids)))
        recs = [r for r in recs if r["query_id"] in keep]
    if a.limit:
        recs = recs[:a.limit]
    if a.variants.startswith("factorial:"):
        variants = pv.all_variants(a.variants.split(":", 1)[1])
    elif a.variants == "all":
        variants = pv.all_variants()
    else:
        variants = a.variants.split(",")
    for v in variants:
        pv.parse(v)
    if a.dry_run:
        print(pv.build(variants[0], recs[0]["question"], recs[0]["context"]))
        print(f"\n[dry-run] {len(variants)} 변형 × {len(recs)} 문항")
        return

    import torch
    torch.manual_seed(SEED)
    tok, model = load_model(a.model)
    print(f"[model] {a.model} dtype={model.dtype} class={type(model).__name__} "
          f"vram={torch.cuda.memory_allocated() / 2**30:.2f}GB", flush=True)
    out_dir = Path(a.out_dir) / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "dtype": str(model.dtype), "model_class": type(model).__name__, "arm": a.arm, "records": a.records,
            "n_records": len(recs), "max_new_tokens": a.max_new_tokens, "batch_tokens": a.batch_tokens, "max_batch": a.max_batch,
            "thinking": a.thinking, "decoding": "greedy (do_sample=False), temperature 0", "seed": SEED, "system_message": None,
            "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
            "transformers": __import__("transformers").__version__, "torch": torch.__version__,
            "variants": variants, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    if a.check_batch:                                              # 배치 확인: 한 건씩 대 묶음
        v = variants[0]
        sub = recs[:a.check_batch]
        prompts = [chat(tok, pv.build(v, r["question"], r["context"]), a.thinking) for r in sub]
        single = [generate(tok, model, [p], a.max_new_tokens, a.thinking)[0][0] for p in prompts]
        batched = [x[0] for x in generate(tok, model, prompts, a.max_new_tokens, a.thinking)]
        same = sum(s == b for s, b in zip(single, batched))
        print(f"[check-batch] {v}: 한 건씩 = 묶음 {same}/{len(sub)}", flush=True)
        (out_dir / "check_batch.json").write_text(json.dumps({"variant": v, "n": len(sub), "same": same,
                                                               "single": single, "batched": batched}, ensure_ascii=False, indent=1))
    for v in variants:
        p = out_dir / f"{v}.jsonl"
        done = {}
        if p.exists():
            if not a.resume:
                raise SystemExit(f"{p} 있음 — 덮어쓰지 않는다(--resume)")
            done = {json.loads(l)["query_id"] for l in open(p, encoding="utf-8") if l.strip()}
        n = run_variant(a, tok, model, recs, v, p, done)
        print(f"[done] {v}/{a.arm}: 새로 {n}건, 합계 {len(done) + n}", flush=True)


if __name__ == "__main__":
    main()
