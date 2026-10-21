"""잘린 답변 마무리 대상 목록(jobs.json) — 있는 출력 파일만 넣는다. PREREG-2026-10-22-budget-forcing.md"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; P = "R1X1F1B2D1C1E0"
FD = ROOT / "results/format_dev_20261017"; SD = ROOT / "results/sentence_dev_20261019"; MC = ROOT / "results/mh_confirm_20261019"; MP = ROOT / "results/mh_multiprompt_20261022"; CF = ROOT / "results/cellform_20261012"
sets = {
 "dev": [(SD / f"run{i}", a, rec, P) for i in (1, 2) for a, rec in (("A", ROOT / "results/prompt_sweep_20261015/dev_A_records.jsonl"), ("D", FD / "D_records.jsonl"), ("Cp", SD / "Cp_records.jsonl"))],
 "600": [(MC / f"run{i}", a, MC / f"{a}_records.jsonl", P) for i in (1, 2, 3) for a in "DCA"],
 "882": [(FD / f"final882_cap4096_run{i}", a, rec, P) for i in (1, 2, 3) for a, rec in (("D", FD / "D882_records.jsonl"), ("C", CF / "C_ours_budget_records.jsonl"), ("A", CF / "A_chunk_md_records.jsonl"))],
 "600b": [(MP / f"run{i}", a, MP / f"{a}_records.jsonl", v) for i in (1, 2, 3) for a in "DCA" for v in ("R0X0F0B0D0C0E0", "R0X0F0B2D0C1E0", "R1X1F1B2D1C1E0", "R1X3F1B2D1C1E0")],
}
jobs = []; skipped = 0
for name, L in sets.items():
    for d, a, rec, v in L:
        out = d / a / f"{v}.jsonl"
        if out.exists() and rec.exists(): jobs.append({"set": name, "out": str(out), "records": str(rec), "variant": v, "arm": a, "run": d.name})
        else: skipped += 1
json.dump(jobs, open(Path(__file__).resolve().parent / "jobs.json", "w"), indent=1)
print(f"[jobs] {len(jobs)}개 (없어서 뺀 것 {skipped})")
