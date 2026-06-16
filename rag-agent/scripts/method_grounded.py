#!/usr/bin/env python3
"""제안 방법: Grounding-trace 기반 자가수정 codegen (vs 나이브 codegen).

동기(측정): 나이브 codegen 실패의 82%가 '코드 실행됨 but 값 틀림'(예외 0건).
원인 = 헤더 자유추측이 빗나가도 조용히 잘못된 값 반환 → 모델이 자기 grounding 오류를 모름.
기존 Self-Debugging/LEVER는 '예외/테스트' 신호로만 고침 → 여기선 안 잡힘.

제안:
  (1) 스키마 바인딩  : 실제 row/col 헤더경로 인벤토리를 프롬프트에 주입(자유추측 금지)
  (2) grounding 추적 : 셀 접근 API가 (요청→매칭 경로/인덱스→값, 모호/빈값 플래그) 로그
  (3) trace 자가수정 : 코드가 에러 없이 끝나도 trace에 ⚠(NO_MATCH/EMPTY)나 result 비정상이면
                       trace를 모델에 되먹여 재생성(≤k)  ← 조용한 오류를 잡는 핵심 노블티
  (4) 숫자 verifier  : 최종 result 형태 점검(정답 미사용)

세 모드 비교(동일 모델·질의셋·채점, 검색=gold 고정 → 답변단계만 격리):
  --mode naive     : 느슨한 cell(str,str) 1-shot, 스키마/트레이스/리페어 없음
  --mode grounded  : 위 (1)-(4) 전부 (스키마+trace+repair)
  --mode hpir      : grounded + HPIR 헤더경로 바인딩 힌트 주입
                     (쿼리를 표의 실제 헤더경로로 사전 해소 → 제안 바인딩을 프롬프트에 주입)

사용: python scripts/method_grounded.py --mode hpir --per-class 15 [--repairs 2] [--retrieval gold]
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag_agent.data.loader import load_samples, load_table          # noqa: E402
from rag_agent.stores.original_store import build_original_table, _to_float  # noqa: E402
from rag_agent.eval.metrics import numeric_match, exact_match, difficulty_class  # noqa: E402
from rag_agent.query import resolve_against_table                    # noqa: E402

SEED = 42
HARD = ["multi_op_formula", "arithmetic_agg", "pair_or_topk_arg", "single_arg", "comparison_or_count"]


# ───────────────────────── traced header-path API ─────────────────────────
class TracedTable:
    """OriginalTable 위에 grounding 추적을 입힌 API. 생성코드가 이 함수들만 사용."""

    def __init__(self, ot):
        self.ot = ot
        self.trace = []

    def _log(self, op, req, matched, value, flag=""):
        self.trace.append({"op": op, "req": req, "matched": matched,
                           "value": value, "flag": flag})

    def cell(self, row_header, col_header):
        res = self.ot.resolve(str(row_header), str(col_header))
        if res is None:
            self._log("cell", [str(row_header), str(col_header)], None, None, "NO_MATCH")
            return None
        r, c, v = res
        num = _to_float(v)
        self._log("cell", [str(row_header), str(col_header)],
                  {"row_path": self.ot.row_path(r), "col_path": self.ot.col_path(c)},
                  v, "" if (v not in (None, "")) else "EMPTY")
        return num if num is not None else v

    def col_values(self, col_header, row_filter=None):
        cols = self.ot.find_cols_by_header(str(col_header)) or self.ot._fuzzy_find_cols(str(col_header))
        if not cols:
            self._log("col_values", [str(col_header), row_filter], None, [], "NO_MATCH")
            return []
        c = cols[0]
        vals = []
        for r in range(self.ot.n_rows):
            if row_filter and not self.ot._match_path(str(row_filter), self.ot.row_path(r)):
                continue
            num = self.ot.cell_num(r, c)
            if num is not None:
                vals.append(num)
        self._log("col_values", [str(col_header), row_filter],
                  {"col_path": self.ot.col_path(c), "n": len(vals)}, vals[:8],
                  "" if vals else "EMPTY")
        return vals

    def row_values(self, row_header, col_filter=None):
        rows = self.ot.find_rows_by_header(str(row_header)) or self.ot._fuzzy_find_rows(str(row_header))
        if not rows:
            self._log("row_values", [str(row_header), col_filter], None, [], "NO_MATCH")
            return []
        r = rows[0]
        vals = []
        for c in range(self.ot.n_cols):
            if col_filter and not self.ot._match_path(str(col_filter), self.ot.col_path(c)):
                continue
            num = self.ot.cell_num(r, c)
            if num is not None:
                vals.append(num)
        self._log("row_values", [str(row_header), col_filter],
                  {"row_path": self.ot.row_path(r), "n": len(vals)}, vals[:8],
                  "" if vals else "EMPTY")
        return vals

    # schema (스키마 바인딩용): 실제 존재하는 헤더경로 인벤토리
    def list_rows(self, cap=40):
        seen, out = set(), []
        for r in range(self.ot.n_rows):
            p = " > ".join(self.ot.row_path(r))
            if p and p not in seen:
                seen.add(p); out.append(p)
        return out[:cap]

    def list_cols(self, cap=40):
        seen, out = set(), []
        for c in range(self.ot.n_cols):
            p = " > ".join(self.ot.col_path(c))
            if p and p not in seen:
                seen.add(p); out.append(p)
        return out[:cap]


# ───────────────────────── sandbox exec ─────────────────────────
class _Timeout(Exception):
    pass


_SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
         for k in ["len", "sum", "min", "max", "abs", "round", "sorted", "range", "float",
                   "int", "str", "list", "dict", "set", "tuple", "enumerate", "zip", "map",
                   "filter", "any", "all", "bool", "print"]}


def run_code(code, api, timeout=6):
    """code 실행 → (result, error_str). api = {name: fn}."""
    g = {"__builtins__": _SAFE}
    g.update(api)

    def _h(s, f):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(timeout)
    err = ""
    try:
        exec(code, g)
    except _Timeout:
        err = "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return g.get("result", None), err


def strip_code(txt):
    m = re.search(r"```(?:python)?\s*(.+?)```", txt, re.DOTALL)
    code = m.group(1) if m else txt
    return code.strip()


# ───────────────────────── prompts ─────────────────────────
NAIVE_SYS = (
    "You write short Python to compute the answer from a table. "
    "Helpers available: cell(row_header, col_header) -> number, "
    "col_values(col_header, row_filter=None) -> list, "
    "row_values(row_header, col_filter=None) -> list. "
    "Assign the final answer to a variable named `result`. Output ONLY a python code block.")

GROUNDED_SYS = (
    "You write short Python to compute the answer from a hierarchical table via a typed API. "
    "RULES: bind ONLY to header paths listed in ROW HEADERS / COL HEADERS (do not invent strings; "
    "you may use any segment of a listed path). "
    "API: cell(row_header, col_header) -> number|None ; "
    "col_values(col_header, row_filter=None) -> list[number] ; "
    "row_values(row_header, col_filter=None) -> list[number] ; "
    "list_rows(), list_cols() -> available header paths. "
    "Assign the final answer to `result`. Output ONLY a python code block.")


def build_user(title, question, tt: TracedTable, grounded: bool, binding_hint: str = ""):
    s = f"Table title: {title}\nQuestion: {question}\n"
    if grounded:
        rows = tt.list_rows(); cols = tt.list_cols()
        s += "\nROW HEADERS (row_header 후보):\n- " + "\n- ".join(rows)
        s += "\n\nCOL HEADERS (col_header 후보):\n- " + "\n- ".join(cols)
    if binding_hint:
        # HPIR: pre-resolved header-path bindings for THIS query (a prior, not a constraint).
        s += ("\n\nHPIR SUGGESTED BINDINGS (resolved from the question; "
              "prefer these if correct, but verify against the lists above):\n" + binding_hint)
    s += "\n\nWrite python that sets `result` to the answer value."
    return s


def trace_feedback(code, trace, result, err):
    """grounding trace를 사람이 읽는 피드백으로 — 예외가 없어도 ⚠를 노출."""
    lines = ["Your previous code:", "```python", code, "```", "", "EXECUTION TRACE (grounding):"]
    warned = False
    for t in trace:
        flag = t["flag"]
        if flag:
            warned = True
        m = t["matched"]
        msum = "NO MATCH" if m is None else json.dumps(m, ensure_ascii=False)
        lines.append(f"  {t['op']}({t['req']}) -> matched={msum} value={t['value']} {('⚠'+flag) if flag else 'ok'}")
    if err:
        lines.append(f"\nERROR: {err}")
    lines.append(f"\nRESULT = {result!r}")
    issues = []
    if err:
        issues.append("execution error")
    if result is None or result == "" or result == []:
        issues.append("result is empty/None")
    if warned:
        issues.append("some header lookups did NOT match or returned empty (⚠) — you likely bound to a wrong/absent header")
    lines.append("\nPROBLEMS: " + ("; ".join(issues) if issues else "result present but verify it answers the question and the bound cells are the intended ones."))
    lines.append("Rebind to correct header paths from the lists and fix the computation. Output ONLY a corrected python code block, set `result`.")
    return "\n".join(lines)


def needs_repair(trace, result, err):
    if err:
        return True
    if result is None or result == "" or result == []:
        return True
    if any(t["flag"] for t in trace):   # 조용한 grounding 오류 신호
        return True
    return False


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "grounded", "hpir"], required=True)
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--repairs", type=int, default=2, help="grounded 자가수정 최대 횟수")
    ap.add_argument("--retrieval", choices=["gold"], default="gold")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--llm", default="local:Qwen/Qwen2.5-7B-Instruct",
                    help="e.g. groq:openai/gpt-oss-120b (CPU env), local:Qwen/... (GPU)")
    ap.add_argument("--hitab-dir", default="data/hitab")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import random
    rng = random.Random(SEED)
    samples = load_samples(args.hitab_dir, "dev")
    buckets = defaultdict(list)
    for s in samples:
        c = difficulty_class(s)
        if c in HARD:
            buckets[c].append(s)
    chosen = []
    for c in HARD:
        b = buckets[c][:]; rng.shuffle(b)
        chosen += [(c, s) for s in b[:args.per_class]]
    print(f"[{args.mode}] eval {len(chosen)} queries (per_class={args.per_class}, repairs={args.repairs})", flush=True)

    from rag_agent.llm.factory import build_llm
    llm = build_llm(args.llm)
    grounded = args.mode in ("grounded", "hpir")
    sys_p = GROUNDED_SYS if grounded else NAIVE_SYS

    def _gen(u):
        """LLM call with exponential backoff on transient (rate-limit) errors."""
        delay = 4.0
        for _ in range(6):
            try:
                return strip_code(llm.complete(sys_p, u, max_tokens=args.max_tokens))
            except Exception:  # noqa: BLE001
                time.sleep(delay); delay = min(delay * 2, 60)
        return strip_code(llm.complete(sys_p, u, max_tokens=args.max_tokens))

    def _quality(result, err, trace):
        flags = sum(1 for t in trace if t.get("flag"))
        nonempty = result is not None and result != "" and result != []
        return (not bool(err), bool(nonempty), -flags)

    rows = []
    t0 = time.time()
    for i, (cls, s) in enumerate(chosen, 1):
        q = s.get("question") or ""
        gold = s.get("answer")
        tid = s.get("table_id")
        raw = load_table(tid, args.hitab_dir)
        if not raw:
            rows.append({"class": cls, "query": q, "correct": False, "skip": "no_table"})
            continue
        ot = build_original_table(raw)
        tt = TracedTable(ot)
        api = {"cell": tt.cell, "col_values": tt.col_values, "row_values": tt.row_values,
               "list_rows": tt.list_rows, "list_cols": tt.list_cols}

        binding_hint = ""
        if args.mode == "hpir":
            intent = resolve_against_table(q, ot)
            binding_hint = intent.binding_hint()
        user = build_user(ot.title, q, tt, grounded=grounded, binding_hint=binding_hint)
        code = _gen(user)
        tt.trace = []
        result, err = run_code(code, api)
        n_repair = 0
        best = (result, err, code, _quality(result, err, tt.trace))
        if grounded:
            while n_repair < args.repairs and needs_repair(tt.trace, result, err):
                fb = trace_feedback(code, tt.trace, result, err)
                code = _gen(user + "\n\n" + fb)
                tt.trace = []
                result, err = run_code(code, api)
                n_repair += 1
                qy = _quality(result, err, tt.trace)
                if qy > best[3]:
                    best = (result, err, code, qy)
            result, err, code = best[0], best[1], best[2]   # over-correction safeguard

        pred = "" if result is None else str(result)
        ok = numeric_match(pred, gold) or exact_match(pred, gold)
        rows.append({"class": cls, "query": q, "gold": gold, "pred": pred,
                     "correct": bool(ok), "err": err, "n_repair": n_repair,
                     "code": code, "trace_flags": [t["flag"] for t in tt.trace if t["flag"]]})
        if i % 10 == 0 or i == len(chosen):
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  {i}/{len(chosen)} NM={acc:.3f} {time.time()-t0:.0f}s", flush=True)

    n = len(rows)
    nm = sum(r["correct"] for r in rows) / n
    by_class = {}
    for c in HARD:
        cr = [r for r in rows if r["class"] == c]
        if cr:
            by_class[c] = {"n": len(cr), "NM": round(sum(x["correct"] for x in cr) / len(cr), 4)}
    out = {"config": {"mode": args.mode, "per_class": args.per_class, "repairs": args.repairs,
                      "retrieval": args.retrieval, "llm": args.llm, "seed": SEED},
           "overall": {"n": n, "NM": round(nm, 4)}, "by_class": by_class, "rows": rows}
    outp = ROOT / "results" / (args.out or f"method_{args.mode}_pc{args.per_class}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[{args.mode}] NM={nm:.4f} (n={n}) → {outp}", flush=True)
    print("by_class:", json.dumps(by_class, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
