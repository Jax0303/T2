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

import argparse, re
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


def load_model(name: str, quantization: str = "none"):
    """bf16 무양자화가 기본. quantization=4bit 는 코드 점검(연기 시험)용 — 결과에는 쓰지 않는다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    extra = {}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model, info = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda"},
                                                      output_loading_info=True, **extra)
    missing = [k for k in info.get("missing_keys", []) if "mtp" not in k]
    if missing:                                                     # 멀티모달 체크포인트(model.language_model.*)에서 글 모델만 적재
        print(f"[load] AutoModelForCausalLM 이 가중치 {len(missing)}개를 못 찾음(예 {missing[:2]}) → 키 이름 바꿔 다시 적재", flush=True)
        del model
        torch.cuda.empty_cache()
        model, info = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda"}, output_loading_info=True,
                                                          key_mapping={r"^model\.language_model\.": "model."}, **extra)
        missing = [k for k in info.get("missing_keys", []) if "mtp" not in k]
        if missing:
            print(f"[load] 여전히 {len(missing)}개 없음(예 {missing[:2]}) → ForConditionalGeneration 전체 적재", flush=True)
            from transformers import AutoModelForImageTextToText
            del model
            torch.cuda.empty_cache()
            model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map={"": "cuda"}, **extra)
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "visual"):     # 글만 쓰므로 시각 가중치는 버린다
                inner.visual = None
                torch.cuda.empty_cache()
                print("[load] 시각 모듈 삭제", flush=True)
    print(f"[load] 못 찾은 가중치 {len(info.get('missing_keys', []))}, 안 쓴 가중치 {len(info.get('unexpected_keys', []))}", flush=True)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # WSL 은 GPU 메모리를 넘기면 오류 대신 시스템 메모리로 넘쳐 수십 배 느려진다(PREREG-2026-09-24) — 남은 메모리 안에서만 쓰게 상한을 건다
    free, total = torch.cuda.mem_get_info()
    frac = max(0.5, min(0.98, (torch.cuda.memory_allocated() + free - 400 * 2**20) / total))
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"[load] 메모리 상한 {frac:.3f} × {total / 2**30:.1f}GB", flush=True)
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


def generate_safe(tok, model, prompts: list, max_new: int, thinking: bool):
    """generate 를 메모리 부족이면 반으로 쪼개 다시 — 결과는 입력 순서, 각 항목에 실제 묶음 크기를 붙인다.
    재시도는 except 블록 밖에서 한다(블록 안에서는 예외가 실패한 할당을 붙들고 있어 연쇄 OOM 이 났다, 2026-10-15)."""
    import torch
    results, stack = [None] * len(prompts), [(0, len(prompts))]
    while stack:
        lo, hi = stack.pop()
        res = None
        try:
            res = generate(tok, model, prompts[lo:hi], max_new, thinking)
        except torch.cuda.OutOfMemoryError:
            res = None
        if res is None:
            torch.cuda.empty_cache()
            if hi - lo == 1:
                raise RuntimeError(f"한 건({lo})도 메모리에 안 들어감")
            mid = (lo + hi) // 2
            print(f"  [oom] 묶음 {hi - lo} → {mid - lo}+{hi - mid}", flush=True)
            stack += [(mid, hi), (lo, mid)]
        else:
            for i, r in enumerate(res):
                results[lo + i] = (r, hi - lo)
    return results


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
            t = time.time()
            res = generate_safe(tok, model, [prompts[i] for i in idx], a.max_new_tokens, a.thinking)
            sec = (time.time() - t) / len(idx)
            results = {i: (r, sec, bs) for i, (r, bs) in zip(idx, res)}
            for i in idx:
                (text, n_in, n_out, fin, think), sec, bs = results[i]
                r = pending[i]
                row = {"query_id": r["query_id"], "arm": a.arm, "variant": variant, "model": a.model, "quantization": a.quantization, "thinking": a.thinking,
                       "raw": text, "n_tok": n_in, "n_tok_out": n_out, "finished": fin, "think_tokens": think,
                       "layer": r["layer"], "kind": r["kind"], "m": r["m"], "retrieval_correct": r["correct"],
                       "context_sha256": hashlib.sha256(r["context"].encode()).hexdigest(), "prompt_sha256": psha,
                       "max_new_tokens": a.max_new_tokens, "batch": bs, "seconds": round(sec, 3)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
            f.flush(); os.fsync(f.fileno())
            print(f"  {variant}/{a.arm} {len(done) + n_done}/{len(recs)}  {time.time() - t0:.0f}s  (묶음 {len(idx)}, 최장 {b[0][0]})", flush=True)
    return n_done


_FINAL = re.compile(r"final answer\**\s*[:：]\s*(.*)", re.I)
_NUM = re.compile(r"-?\(?\$?\s*\d[\d,]*\.?\d*\)?%?")
def final_key(text):
    """마지막 'Final answer' 줄의 값을 투표 열쇠로: 첫 숫자(쉼표·$ 제거, 괄호 음수, 유효숫자 4자리) 없으면 소문자 글자."""
    m = list(_FINAL.finditer(text or ""))
    if not m: return None
    v = m[-1].group(1).strip()
    n = _NUM.search(v)
    if n:
        t = n.group(0); neg = t.startswith("-") or ("(" in t and ")" in t); t2 = re.sub(r"[^\d.]", "", t)
        try:
            x = float(t2) * (-1 if neg else 1); return f"{x:.4g}" + ("%" if t.endswith("%") else "")
        except ValueError: pass
    return v.lower().strip(" .") or None
def sc_vote(outputs):
    from collections import Counter
    keys = [final_key(c.text) for c in outputs]; c = Counter(k for k in keys if k is not None)
    if not c: return outputs[0], {"sc_n": len(outputs), "sc_keys": keys, "sc_win": None, "sc_votes": 0, "sc_agree": 0}
    win, votes = c.most_common(1)[0]; idx = keys.index(win)
    return outputs[idx], {"sc_n": len(outputs), "sc_keys": keys, "sc_win": win, "sc_votes": votes, "sc_agree": int(votes == len(outputs)), "sc_pick": idx,
                          "sc_n_tok_out_all": sum(len(x.token_ids) for x in outputs), "sc_finished_all": sum(x.finish_reason == "stop" for x in outputs)}

def main_vllm(a, recs_by_arm, arms, variants):
    """vLLM 경로: 변형×arm 마다 전체 문항을 한 번에 넘긴다(연속 묶음·greedy). 출력 행 형식은 hf 경로와 같다."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {}
    if "Qwen3.5" in a.model or "Qwen3_5" in a.model:
        kw["language_model_only"] = True                            # 글 모델만(시각 가중치 제외), 모델 카드 vLLM 안내
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len, seed=SEED,
              enable_prefix_caching=True, max_num_batched_tokens=a.max_num_batched_tokens, max_num_seqs=a.max_num_seqs, **kw)
    if a.n_samples > 1:
        sp = SamplingParams(n=a.n_samples, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_new_tokens, seed=SEED)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, seed=SEED)
    print(f"[model] {a.model} engine=vllm gpu_mem={a.gpu_mem} max_model_len={a.max_model_len} n_samples={a.n_samples} temperature={a.temperature if a.n_samples > 1 else 0.0}", flush=True)
    out_dir = Path(a.out_dir) / arms[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "engine": "vllm", "vllm": __import__("vllm").__version__, "dtype": "bfloat16", "arm": a.arm, "records": a.records,
            "n_records": {k: len(v) for k, v in recs_by_arm.items()}, "max_new_tokens": a.max_new_tokens, "gpu_memory_utilization": a.gpu_mem,
            "max_model_len": a.max_model_len, "max_num_batched_tokens": a.max_num_batched_tokens, "max_num_seqs": a.max_num_seqs, "thinking": a.thinking, "decoding": (f"sampling n={a.n_samples} temperature={a.temperature} top_p={a.top_p} top_k={a.top_k} + final-line vote" if a.n_samples > 1 else "greedy (temperature 0)"), "seed": SEED, "system_message": None,
            "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(), "variants": variants,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    think_end = tok.convert_tokens_to_ids("</think>") if "</think>" in tok.get_vocab() else None
    for v in variants:
        for arm in arms:
            d = Path(a.out_dir) / arm
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{v}.jsonl"
            done = set()
            if p.exists():
                if not a.resume:
                    raise SystemExit(f"{p} 있음 — 덮어쓰지 않는다(--resume)")
                done = {json.loads(l)["query_id"] for l in open(p, encoding="utf-8") if l.strip()}
            pending = [r for r in recs_by_arm[arm] if r["query_id"] not in done]
            if not pending:
                print(f"[done] {v}/{arm}: 새로 0건, 합계 {len(done)}", flush=True); continue
            prompts = [chat(tok, pv.build(v, r["question"], r["context"]), a.thinking) for r in pending]
            psha = hashlib.sha256((pv.instruction(v) + "\n" + pv.examples_block(v)).encode()).hexdigest()
            t0 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            sec = (time.time() - t0) / len(pending)
            with p.open("a", encoding="utf-8", newline="\n") as f:
                for r, o in zip(pending, outs):
                    g = o.outputs[0]; sc_info = None
                    if a.n_samples > 1:                                  # 마지막 줄 값 다수결, 이긴 표본을 raw 로 둔다(채점기 불변)
                        g, sc_info = sc_vote(o.outputs)
                    ids = list(g.token_ids)
                    cut = 0
                    if a.thinking and think_end is not None and think_end in ids:
                        cut = len(ids) - ids[::-1].index(think_end)
                    text = tok.decode(ids[cut:], skip_special_tokens=True).strip() if cut else g.text.strip()
                    row = {"query_id": r["query_id"], "arm": arm, "variant": v, "model": a.model, "engine": "vllm", "thinking": a.thinking,
                           "raw": text, "n_tok": len(o.prompt_token_ids), "n_tok_out": len(ids), "finished": g.finish_reason == "stop",
                           "think_tokens": cut, "layer": r["layer"], "kind": r["kind"], "m": r["m"], "retrieval_correct": r["correct"],
                           "context_sha256": hashlib.sha256(r["context"].encode()).hexdigest(), "prompt_sha256": psha,
                           "max_new_tokens": a.max_new_tokens, "batch": "vllm", "seconds": round(sec, 3)}
                    if sc_info: row.update(sc_info)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush(); os.fsync(f.fileno())
            print(f"[done] {v}/{arm}: 새로 {len(pending)}건, 합계 {len(done) + len(pending)}, {time.time() - t0:.0f}s ({sec:.2f}s/건)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", required=True, help="레코드 jsonl(쉼표로 여러 개, --arm 과 짝)")
    ap.add_argument("--arm", required=True, help="A = LangChain 청크 / C = 본 방법 셀 문장 (이름표, 쉼표로 여러 개)")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-tokens", type=int, default=14000, help="묶음 안 (가장 긴 입력 + 생성 상한) × 건수 상한 "
                    "(Qwen3-4B bf16: KV 캐시 토큰당 약 147KB, 가중치 7.5GB 뒤 남는 약 2.3GB → 약 15k 토큰)")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="문항 부분집합(json 목록)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--thinking", action="store_true", help="생각 모드(참고 조건). </think> 뒤만 raw 에 남긴다")
    ap.add_argument("--check-batch", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 첫 문항 프롬프트만 출력")
    ap.add_argument("--quantization", default="none", help="none(기본, 결과용) | 4bit(코드 점검용)")
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"], help="hf = transformers 정적 묶음 / vllm = vLLM 연속 묶음(bf16, greedy)")
    ap.add_argument("--gpu-mem", type=float, default=0.80, help="vllm: gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=5120, help="vllm: 최대 길이(입력+출력)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096, help="vllm: 한 스텝 prefill 토큰 상한(작을수록 활성 메모리 적게)")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="vllm: 동시 시퀀스 상한")
    ap.add_argument("--shuffle-variants", type=int, default=0, help="변형 실행 순서를 이 시드로 섞는다(0 = 이름 순)")
    ap.add_argument("--n-samples", type=int, default=1, help="vllm: 문항당 표본 수. 2 이상이면 temperature 표본 추출 + 마지막 줄 값 다수결(self-consistency, PREREG-2026-10-22-self-consistency-dev.md)")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--top-p", type=float, default=1.0); ap.add_argument("--top-k", type=int, default=-1)
    a = ap.parse_args()

    rec_paths, arms = a.records.split(","), a.arm.split(",")
    assert len(rec_paths) == len(arms), "--records 와 --arm 의 개수가 다르다"
    recs_by_arm = {}
    for arm, rp in zip(arms, rec_paths):
        recs = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
        if a.ids:
            keep = set(json.load(open(a.ids)))
            recs = [r for r in recs if r["query_id"] in keep]
        if a.limit:
            recs = recs[:a.limit]
        recs_by_arm[arm] = recs
    recs = recs_by_arm[arms[0]]
    if a.variants.startswith("factorial:"):
        variants = pv.all_variants(a.variants.split(":", 1)[1])
    elif a.variants == "all":
        variants = pv.all_variants()
    else:
        variants = a.variants.split(",")
    for v in variants:
        pv.parse(v)
    if a.shuffle_variants:                                         # 축이 고르게 섞이도록 고정 시드로 순서를 섞는다(중간 집계용)
        import random
        random.Random(a.shuffle_variants).shuffle(variants)
    if a.dry_run:
        print(pv.build(variants[0], recs[0]["question"], recs[0]["context"]))
        print(f"\n[dry-run] {len(variants)} 변형 × {len(recs)} 문항")
        return

    if a.engine == "vllm":
        return main_vllm(a, recs_by_arm, arms, variants)
    import torch
    torch.manual_seed(SEED)
    tok, model = load_model(a.model, a.quantization)
    print(f"[model] {a.model} dtype={model.dtype} class={type(model).__name__} "
          f"vram={torch.cuda.memory_allocated() / 2**30:.2f}GB", flush=True)
    out_dir = Path(a.out_dir) / arms[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "dtype": str(model.dtype), "quantization": a.quantization, "model_class": type(model).__name__, "arm": a.arm, "records": a.records,
            "n_records": {k: len(v) for k, v in recs_by_arm.items()}, "max_new_tokens": a.max_new_tokens, "batch_tokens": a.batch_tokens, "max_batch": a.max_batch,
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
        batched = [x[0][0] for x in generate_safe(tok, model, prompts, a.max_new_tokens, a.thinking)]
        same = sum(s == b for s, b in zip(single, batched))
        print(f"[check-batch] {v}: 한 건씩 = 묶음 {same}/{len(sub)}", flush=True)
        (out_dir / "check_batch.json").write_text(json.dumps({"variant": v, "n": len(sub), "same": same,
                                                               "single": single, "batched": batched}, ensure_ascii=False, indent=1))
    for v in variants:
        for arm in arms:                                           # 변형마다 모든 arm 을 돌려 중간에도 짝이 맞게
            a.arm = arm
            d = Path(a.out_dir) / arm
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{v}.jsonl"
            done = {}
            if p.exists():
                if not a.resume:
                    raise SystemExit(f"{p} 있음 — 덮어쓰지 않는다(--resume)")
                done = {json.loads(l)["query_id"] for l in open(p, encoding="utf-8") if l.strip()}
            n = run_variant(a, tok, model, recs_by_arm[arm], v, p, done)
            print(f"[done] {v}/{arm}: 새로 {n}건, 합계 {len(done) + n}", flush=True)


if __name__ == "__main__":
    main()
