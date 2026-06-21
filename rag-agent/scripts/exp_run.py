#!/usr/bin/env python3
"""EXPERIMENT_PROMPT GATE-3~6. 숫자/케이스만 산출. 해석·프레이밍 없음.

메인 모델: gpt-oss-120b (host=Groq; Cerebras 키 payment_required로 동일모델 대체).
temp=0, seed=42 고정. 4칸(flat-A/flat-B/hier-A/hier-B) × 30 = 120 호출.

산출: results/codegen_raw.jsonl, results/codegen_stats.json, results/wrong_cases_for_review.md
"""
from __future__ import annotations
import json, os, re, subprocess, tempfile, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np

RES = Path("results")
MODEL = os.environ.get("EXP_MODEL", "openai/gpt-oss-120b")
HOST = os.environ.get("EXP_HOST", "groq")
SEED = 42
PYBIN = "/home/user/T2/hart-table-retrieval/.venv/bin/python3"

PROMPT_TMPL = """아래 표와 질문이 주어진다. 질문에 답하는 파이썬 코드를 작성하라.
- 표는 변수 `table`에 (행들의 리스트 또는 적절한 자료구조로) 주어진다고 가정한다.
- 코드는 마지막에 정답을 변수 `answer`에 저장해야 한다.
- 코드만 출력하고 설명은 하지 마라.

[표]
{table}

[질문]
{question}"""


def llm_complete(prompt, model=MODEL, host=HOST):
    if host == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        key = os.environ["GROQ_API_KEY"]
    else:
        url = "https://api.cerebras.ai/v1/chat/completions"
        key = Path(".cerebras_key").read_text().strip()
    body = json.dumps({"model": model, "temperature": 0, "seed": SEED,
                       "max_tokens": 1500, "reasoning_effort": "low",
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(12):
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": "curl/8.4.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 500, 502):
                ra = e.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else min(2 ** attempt + 1, 30)
                time.sleep(min(wait + 0.5, 35)); continue
            raise          # 413 등 비재시도 에러는 호출부에서 처리
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(2 ** attempt + 1, 30)); continue
    raise RuntimeError("LLM call failed after retries")


def extract_code(raw):
    m = re.search(r"```python\s*\n(.*?)\n```", raw, re.DOTALL)
    if not m:
        m = re.search(r"```\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        return m.group(1).strip(), True
    if "answer" in raw and "=" in raw:        # 코드펜스 없이 코드만 온 경우
        return raw.strip(), True
    return raw.strip(), False


# ---------- 샌드박스 ----------
def run_code(code, table_obj):
    wrapper = ("import json,math,re,statistics\n"
               "from collections import Counter,defaultdict\n"
               f"table = {repr(table_obj)}\n"
               + code + "\n"
               "import json as _j\n"
               "try:\n"
               "    print('__ANSWER__'+_j.dumps(answer, default=str))\n"
               "except NameError:\n"
               "    print('__NOANSWER__')\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(wrapper); tmp = f.name
    try:
        p = subprocess.run([PYBIN, tmp], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        os.unlink(tmp); return None, "TIMEOUT"
    os.unlink(tmp)
    out = p.stdout
    if "__ANSWER__" in out:
        line = [l for l in out.splitlines() if l.startswith("__ANSWER__")][-1]
        try:
            return json.loads(line[len("__ANSWER__"):]), None
        except Exception:
            return line[len("__ANSWER__"):], None
    if "__NOANSWER__" in out:
        return None, "NO_ANSWER_VAR"
    err = (p.stderr or "").strip().splitlines()
    return None, (err[-1] if err else "UNKNOWN_ERROR")


# ---------- 채점 ----------
def norm_item(x):
    s = str(x).strip().lower()
    s2 = s.replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        return ("num", round(float(s2), 4))
    except ValueError:
        return ("str", s)


def to_set(ans):
    if isinstance(ans, (list, tuple, set)):
        items = list(ans)
    else:
        items = [ans]
    return {norm_item(x) for x in items if not (isinstance(x, str) and x.strip() == "")}


def is_correct(pred, gold):
    if pred is None:
        return False
    return to_set(pred) == to_set(gold)


# ---------- 메인 ----------
def main():
    inputs = [json.loads(l) for l in open(RES / "exp_inputs.jsonl", encoding="utf-8")]
    cache_path = RES / "exp_codegen_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            r = json.loads(l); cache[(r["id"], r["condition"])] = r
    cf = open(cache_path, "a", encoding="utf-8")

    # ---- GATE-3: 코드 생성 ----
    rows, extract_ok, extract_fail = [], 0, 0
    for inp in inputs:
        for cond in ("A", "B"):
            key = (inp["id"], cond)
            if key in cache:
                code, ok, raw = cache[key]["code"], cache[key]["extract_ok"], cache[key].get("raw", "")
            else:
                table_text = inp[f"{cond}_text"]
                prompt = PROMPT_TMPL.format(table=table_text, question=inp["question"])
                try:
                    raw = llm_complete(prompt)
                    code, ok = extract_code(raw)
                except Exception as ex:                 # 한 건 실패가 전체 중단 안 되게
                    raw, code, ok = f"<<ERROR:{ex}>>", "", False
                    print(f"  [warn] gen 실패 id={inp['id']} {cond}: {ex}", flush=True)
                rec = {"id": inp["id"], "condition": cond, "extract_ok": ok,
                       "code": code, "raw": raw[:200]}
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n"); cf.flush()
                print(f"  gen {len(cache)+1}/120 id={inp['id']} {cond} ok={ok}", flush=True)
                cache[key] = rec
                time.sleep(0.3)
            extract_ok += int(ok); extract_fail += int(not ok)
            rows.append({"input": inp, "condition": cond, "code": code, "extract_ok": ok})
    cf.close()
    total = extract_ok + extract_fail
    print("=" * 60); print("GATE-3  코드 추출")
    print(f"  호출 {total} | 추출성공 {extract_ok} | 실패 {extract_fail} | 실패율 {extract_fail/total:.1%}")
    if extract_fail / total > 0.20:
        print("  STOP: 추출 실패율 20% 초과 — 프롬프트/파싱 점검 필요"); return

    # ---- GATE-4: 실행·채점 ----
    raw_out = []
    cells = {"flat-A": [], "flat-B": [], "hier-A": [], "hier-B": []}
    for r in rows:
        inp, cond = r["input"], r["condition"]
        table_obj = inp[f"{cond}_table"]
        pred, err = run_code(r["code"], table_obj)
        ok = is_correct(pred, inp["gold_answer"])
        cellkey = f"{inp['complexity']}-{cond}"
        cells[cellkey].append(int(ok))
        rec = {"id": inp["id"], "complexity": inp["complexity"], "condition": cond,
               "question": inp["question"], "generated_code": r["code"],
               "exec_answer": pred, "gold_answer": inp["gold_answer"],
               "correct": ok, "error": err}
        raw_out.append(rec)
    with open(RES / "codegen_raw.jsonl", "w", encoding="utf-8") as f:
        for rec in raw_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n" + "=" * 60); print("GATE-4  칸별 정확도 (정답수/30)")
    for k in ("flat-A", "flat-B", "hier-A", "hier-B"):
        v = cells[k]; print(f"  {k}: {sum(v)}/{len(v)} = {np.mean(v):.3f}")

    # ---- GATE-5: bootstrap 통계 ----
    def boot_ci(vec, B=10000):
        a = np.array(vec); rng = np.random.default_rng(SEED); n = len(a)
        bs = np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])
        return float(a.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def paired_ci(va, vb, B=10000):  # vb - va
        d = np.array(vb) - np.array(va); rng = np.random.default_rng(SEED); n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}

    stats = {"model": MODEL, "host": HOST, "seed": SEED, "cells": {}, "contrasts": {}}
    for k in cells:
        m, ci = boot_ci(cells[k]); stats["cells"][k] = {"acc": m, "ci95": ci, "n": len(cells[k])}
    stats["contrasts"]["hier_B_minus_A"] = paired_ci(cells["hier-A"], cells["hier-B"])
    stats["contrasts"]["flat_B_minus_A"] = paired_ci(cells["flat-A"], cells["flat-B"])
    # 차이의 차이: (hierB-hierA) - (flatB-flatA), 두 표본 독립 → 차이벡터 직접비교 불가, bootstrap으로
    rng = np.random.default_rng(SEED)
    dh = np.array(cells["hier-B"]) - np.array(cells["hier-A"])
    dfl = np.array(cells["flat-B"]) - np.array(cells["flat-A"])
    bs = np.array([dh[rng.integers(0, len(dh), len(dh))].mean() - dfl[rng.integers(0, len(dfl), len(dfl))].mean()
                   for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    stats["contrasts"]["diff_in_diff"] = {"delta": float(dh.mean() - dfl.mean()),
                                          "ci95": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}
    json.dump(stats, open(RES / "codegen_stats.json", "w"), indent=2)
    print("\n" + "=" * 60); print("GATE-5  대조 (점추정 + 95% CI)")
    for name in ("hier_B_minus_A", "flat_B_minus_A", "diff_in_diff"):
        c = stats["contrasts"][name]
        print(f"  {name}: {c['delta']:+.3f}  CI{[round(x,3) for x in c['ci95']]}  sig={c['sig']}")

    # ---- GATE-6: 오답 덤프 ----
    md = ["# 틀린 케이스 (사람 검토용) — 자동 분류하지 않음\n",
          f"model={MODEL} host={HOST} seed={SEED}\n",
          "분류 코드: (0)채점노이즈 / (1)표읽기오류 / (2)로직오류\n"]
    wrong_counts = {}
    for cellkey in ("flat-A", "flat-B", "hier-A", "hier-B"):
        comp, cond = cellkey.split("-")
        ws = [r for r in raw_out if r["complexity"] == comp and r["condition"] == cond and not r["correct"]]
        wrong_counts[cellkey] = len(ws)
        md.append(f"\n## {cellkey} 틀린 케이스 ({len(ws)}개)\n")
        for r in ws:
            hint = ""
            e = (r["error"] or "")
            if any(k in e for k in ("KeyError", "IndexError", "no row", "no column", "not in")):
                hint = "  (읽기오류 후보: 존재하지 않는 행/열 참조 — 확정 아님)"
            md.append(f"### case {r['id']}\n질문: {r['question']}\n정답: {r['gold_answer']}\n"
                      f"실행결과: {r['exec_answer']}\n에러: {r['error']}{hint}\n"
                      f"생성코드:\n```python\n{r['generated_code']}\n```\n"
                      f"[분류: ___ ]\n")
    (RES / "wrong_cases_for_review.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "=" * 60); print("GATE-6  칸별 틀린 케이스 수")
    for k, v in wrong_counts.items():
        print(f"  {k}: {v}")
    print(f"  -> results/wrong_cases_for_review.md")


if __name__ == "__main__":
    main()
