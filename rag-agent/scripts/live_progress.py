#!/usr/bin/env python3
"""Tail a codegen_chunk_eval log and render a live ASCII bar chart per progress
update. Each render is flushed as one burst so the Monitor groups it into a
single chat notification. stdlib only."""
import re
import sys
import time
from pathlib import Path

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chunk70b.log"
CONDS = ["flat_values", "flat_leaf", "header_path"]
LABEL = {"flat_values": "S0 values only ", "flat_leaf": "S1 leaf header ",
         "header_path": "S2 header-path "}
PROG = re.compile(r"(\d+)/(\d+)\s+flat_values=([\d.]+)\s+flat_leaf=([\d.]+)\s+header_path=([\d.]+)\s+(\d+)s")
SPLIT = re.compile(r"=== (flat|hier): n=(\d+)")
DONE = re.compile(r"wrote ")


def bar(v, width=24):
    fill = int(round(v * width))
    return "█" * fill + "·" * (width - fill)


def render(split, i, n, accs, secs):
    out = [f"┌─ {split.upper()} table  [{i}/{n}]  {secs}s ─────────────"]
    for c in CONDS:
        v = accs[c]
        out.append(f"│ {LABEL[c]} {bar(v)} {v*100:4.0f}%")
    gap = accs["header_path"] - accs["flat_leaf"]
    tag = "  ← 계층보존 효과" if split == "hier" else "  (대조군)"
    out.append(f"│ S2−S1 gap = {gap*100:+4.0f}%{tag}")
    out.append("└" + "─" * 44)
    print("\n".join(out), flush=True)


def main():
    p = Path(LOG)
    while not p.exists():
        time.sleep(0.5)
    split, n = "?", 0
    seen = set()
    with open(p) as f:
        f.seek(0)
        idle = 0
        while True:
            line = f.readline()
            if not line:
                if idle > 1200:   # ~10min of silence after start -> stop
                    break
                idle += 1
                time.sleep(0.5)
                continue
            idle = 0
            m = SPLIT.search(line)
            if m:
                split, n = m.group(1), int(m.group(2))
                print(f"\n▶ {split.upper()} 시작 (n={n})", flush=True)
                continue
            m = PROG.search(line)
            if m:
                i = int(m.group(1))
                key = (split, i)
                if key in seen:
                    continue
                seen.add(key)
                accs = {"flat_values": float(m.group(3)),
                        "flat_leaf": float(m.group(4)),
                        "header_path": float(m.group(5))}
                render(split, i, n, accs, m.group(6))
                continue
            if DONE.search(line):
                print("\n✅ 실험 완료 — 결과 저장됨", flush=True)
                break


if __name__ == "__main__":
    main()
