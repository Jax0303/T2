#!/usr/bin/env python3
"""Phase 2 답변 측 — end-to-end EM/NM/F1 + retrieval–answer gap.

베이스라인(검색된 top-1 표를 LLM에 제공):
  oracle           : 정답 표 강제 주입 (상한)
  nocontext        : 표 미제공 (하한)
  bm25             : 튜닝 BM25 top-1 표
  dense_header_path: best dense top-1 표
LLM: 로컬 Qwen2.5-7B-Instruct 4bit (GPU, rate-limit 없음), greedy, batched.
지표: EM, NM(±2% 허용 수치/부분문자열), token-F1. 클래스별 + (R@1,EM) ρ.

seed=42. 동일 (표,질문) 프롬프트는 1회만 생성(dedup).
사용: python scripts/phase2_answer_eval.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ for phase2_retrieval
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/user/T2/hart-table-retrieval")

from rag_agent.eval.metrics import exact_match, numeric_match, difficulty_class  # noqa
from rag_agent.data.loader import load_samples  # noqa
# Reuse retrieval mechanism + embedding-cache contract from the retrieval script so the
# answer-side top-1 selection can never drift from the reported R@k numbers.
from phase2_retrieval import tok, encode_corpus, table_score_matrix, EMB_MODEL  # noqa

_WORD = re.compile(r"[a-z0-9.]+")


def token_f1(pred, gold_list):
    p = _WORD.findall(str(pred or "").lower())
    best = 0.0
    for g in (gold_list if isinstance(gold_list, list) else [gold_list]):
        gt = _WORD.findall(str(g).lower())
        if not gt and not p:
            best = max(best, 1.0); continue
        if not gt or not p:
            continue
        common = 0
        gc = gt.copy()
        for w in p:
            if w in gc:
                common += 1; gc.remove(w)
        if common == 0:
            continue
        prec, rec = common / len(p), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def load_md_texts():
    d = {}
    for l in open(ROOT / "corpus" / "serialized" / "plain_markdown.records.jsonl"):
        r = json.loads(l)
        d[r["table_id"]] = r["text"]
    return d


def compute_top1(table_ids, questions, md):
    """Return {'bm25':[tid...], 'dense_header_path':[tid...]} top-1 per query."""
    table_index = {t: i for i, t in enumerate(table_ids)}
    # bm25 (best grid k1=0.9 b=0.4) — same plain_markdown text + tokenizer as phase2_retrieval
    from rank_bm25 import BM25Okapi
    corpus_tok = [tok(md[t]) for t in table_ids]
    bm = BM25Okapi(corpus_tok, k1=0.9, b=0.4)
    bm25_top1 = [table_ids[int(np.argmax(bm.get_scores(tok(q))))] for q in questions]
    # dense header_path: reuse the retrieval script's encoder/cache + batched scatter-max
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    doc_emb, owners = encode_corpus("header_path", table_ids, model)
    q_emb = model.encode(questions, batch_size=128, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    scores = table_score_matrix(q_emb, doc_emb, owners, table_index)   # [n_tables, n_q]
    dense_top1 = [table_ids[int(i)] for i in scores.argmax(axis=0)]
    return {"bm25": bm25_top1, "dense_header_path": dense_top1}


SYSTEM = ("You answer a question using the given table. "
          "Reply with ONLY the final answer value(s), no explanation, no units unless in the table.")
SYSTEM_NC = ("Answer the question with ONLY the final value(s), no explanation. "
             "If you cannot know, give your best single guess.")


def build_prompt(md_text, question):
    t = md_text
    if t and len(t) > 4000:          # ~1200 tokens; keeps KV cache small on 8GB GPU
        t = t[:4000]
    return f"Table:\n{t}\n\nQuestion: {question}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["groq", "local"], default="groq")
    ap.add_argument("--groq-model", default="llama-3.1-8b-instant")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2_answers.json"))
    args = ap.parse_args()
    np.random.seed(SEED)

    table_ids = [json.loads(l)["table_id"] for l in open(ROOT / "corpus" / "tables.jsonl")]
    samples = load_samples("data/hitab", "dev")
    if args.limit:
        samples = samples[:args.limit]
    questions = [s.get("question") or "" for s in samples]
    golds = [s.get("table_id") for s in samples]
    answers = [s.get("answer") for s in samples]
    classes = [difficulty_class(s) for s in samples]
    n = len(samples)
    print(f"answer-eval: {n} dev queries", flush=True)

    md = load_md_texts()
    top1 = compute_top1(table_ids, questions, md)
    print("computed top-1 rankings", flush=True)

    # baseline → per-query context table_id (None = nocontext)
    ctx = {
        "oracle": golds,
        "nocontext": [None] * n,
        "bm25": top1["bm25"],
        "dense_header_path": top1["dense_header_path"],
    }
    r1 = {  # R@1 per query (table == gold); nocontext's None ctx never equals a gold id
        b: [int(ctx[b][i] == golds[i]) for i in range(n)]
        for b in ctx
    }

    # ---- dedup prompts ----
    # key: (table_id or "NONE", question_idx) -> but same question text+table → same answer
    uniq = {}
    task_key = {}
    for b in ctx:
        for i in range(n):
            tid = ctx[b][i]
            key = (tid or "NONE", questions[i])
            task_key[(b, i)] = key
            if key not in uniq:
                if tid is None:
                    uniq[key] = (SYSTEM_NC, questions[i])
                else:
                    uniq[key] = (SYSTEM, build_prompt(md.get(tid, ""), questions[i]))
    # sort by user-prompt length so batches have uniform length (less pad waste / OOM)
    keys = sorted(uniq, key=lambda k: len(uniq[k][1]))
    print(f"unique prompts: {len(keys)} (of {4*n} tasks)", flush=True)

    # ---- checkpoint / resume: cache answers to disk so an 8h run survives crashes ----
    cache_path = Path(str(args.out) + ".cache.jsonl")
    ans = {}
    if cache_path.exists():
        for l in open(cache_path):
            l = l.strip()
            if not l:
                continue
            try:                       # tolerate a partial last line from a hard shutdown
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            ans[(r["t"], r["q"])] = r["a"]
    todo = [k for k in keys if k not in ans]
    print(f"cached {len(ans)}, todo {len(todo)}", flush=True)
    _cf = open(cache_path, "a", encoding="utf-8")

    def save(k, o):
        ans[k] = o
        _cf.write(json.dumps({"t": k[0], "q": k[1], "a": o}, ensure_ascii=False) + "\n")
        _cf.flush()

    t0 = time.time()
    if args.backend == "groq":
        # Concurrent Groq calls with 429 backoff (fast for thousands of long prompts).
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from rag_agent.llm.groq_llm import GroqLLM
        llm = GroqLLM(model_name=args.groq_model, retry_on_429=6)
        lock = threading.Lock()
        prog = {"done": 0}

        def work(k):
            sys_p, usr_p = uniq[k]
            try:
                o = llm.complete(sys_p, usr_p, max_tokens=48)
            except Exception:  # noqa: BLE001
                o = ""
            with lock:
                save(k, o)
                prog["done"] += 1
                d = prog["done"]
            if d % 500 == 0 or d == len(todo):
                el = time.time() - t0
                print(f"  gen {d}/{len(todo)} {el:.0f}s {d/max(el,1):.1f}/s", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        llm_tag = f"groq:{args.groq_model}"
    else:
        # local Qwen 4bit, small-batch greedy (8GB-safe).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        name = "Qwen/Qwen2.5-7B-Instruct"
        tokz = AutoTokenizer.from_pretrained(name)
        tokz.padding_side = "left"
        if tokz.pad_token is None:
            tokz.pad_token = tokz.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                                     dtype=torch.bfloat16, device_map="auto")
        model.eval()

        def gen_batch(pairs):
            prompts = [tokz.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True) for s, u in pairs]
            enc = tokz(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=4096).to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                         pad_token_id=tokz.pad_token_id)
            gen = out_ids[:, enc["input_ids"].shape[1]:]
            return [tokz.decode(g, skip_special_tokens=True).strip() for g in gen]

        BUDGET, MAXB = 2600, 96   # small KV to avoid CPU offload on 8GB
        est = {k: len(uniq[k][1]) // 3 + 48 for k in todo}
        batches, cur, cur_max = [], [], 0
        for k in todo:
            nm_max = max(cur_max, est[k])
            if cur and (nm_max * (len(cur) + 1) > BUDGET or len(cur) >= MAXB):
                batches.append(cur); cur, cur_max = [k], est[k]
            else:
                cur.append(k); cur_max = nm_max
        if cur:
            batches.append(cur)
        done = 0
        for bi, batch_keys in enumerate(batches):
            for k, o in zip(batch_keys, gen_batch([uniq[k] for k in batch_keys])):
                save(k, o)
            done += len(batch_keys)
            if bi % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  gen {done}/{len(todo)} (b{bi+1}/{len(batches)}) {el:.0f}s "
                      f"{done/max(el,1):.1f}/s", flush=True)
        llm_tag = "local:Qwen2.5-7B-Instruct-4bit"

    # ---- score ----
    out = {"config": {"n_eval": n, "llm": llm_tag,
                      "context": "top-1 table, plain_markdown", "seed": SEED},
           "overall": {}, "by_class": {}, "per_query": {}}
    for b in ctx:
        em = nm = f1 = 0.0
        nm_list = []
        cls_acc = defaultdict(lambda: [0, 0])  # nm_correct, n
        for i in range(n):
            pred = ans[task_key[(b, i)]]
            e = exact_match(pred, answers[i])
            m = numeric_match(pred, answers[i])
            f = token_f1(pred, answers[i])
            em += e; nm += m; f1 += f
            nm_list.append(int(m))  # use NM as primary correctness
            cls_acc[classes[i]][0] += int(m); cls_acc[classes[i]][1] += 1
        out["overall"][b] = {"EM": em / n, "NM": nm / n, "F1": f1 / n,
                             "R@1": sum(r1[b]) / n}
        out["per_query"][b] = {"nm": nm_list, "r1": r1[b]}
        out["by_class"][b] = {c: {"nm": v[0] / v[1], "n": v[1]} for c, v in cls_acc.items()}
        print(f"  {b:18} EM={out['overall'][b]['EM']:.3f} NM={out['overall'][b]['NM']:.3f} "
              f"F1={out['overall'][b]['F1']:.3f} R@1={out['overall'][b]['R@1']:.3f}", flush=True)

    # ---- retrieval–answer gap ----
    # across-baseline (R@1, NM) Spearman
    bl = list(ctx)
    xs = [out["overall"][b]["R@1"] for b in bl]
    ys = [out["overall"][b]["NM"] for b in bl]
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(xs, ys)
    # per-query conditional: P(correct | gold in top1) vs P(correct | not), for retrieval baselines
    cond = {}
    for b in ["bm25", "dense_header_path"]:
        hit = [out["per_query"][b]["nm"][i] for i in range(n) if r1[b][i] == 1]
        miss = [out["per_query"][b]["nm"][i] for i in range(n) if r1[b][i] == 0]
        cond[b] = {"P_correct_given_hit": (sum(hit) / len(hit)) if hit else None,
                   "n_hit": len(hit),
                   "P_correct_given_miss": (sum(miss) / len(miss)) if miss else None,
                   "n_miss": len(miss)}
    out["gap"] = {"across_baseline_spearman_rho": float(rho), "p": float(p_rho),
                  "points": {b: {"R@1": xs[i], "NM": ys[i]} for i, b in enumerate(bl)},
                  "per_query_conditional": cond}

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
