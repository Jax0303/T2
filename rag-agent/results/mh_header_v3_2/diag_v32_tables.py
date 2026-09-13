# v3.1 -> v3.2: which tables changed and why (row paths only; col paths unchanged per preserve_after.json)
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/user/T2-1/rag-agent")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analysis.header_confirmed_errors import judge_all, compare  # noqa: E402
import mh_arms as M  # noqa: E402

_, docs, _ = M.load_population("train")


def paths(name, rules=None):
    j = judge_all(M.build_tables(docs, name, rules=rules)[0])
    return j


j2, j31, j32 = paths("v2"), paths("v3.1"), paths("v3.2")
_, k31 = compare(j2, j31)
e_new = {t for t, *_ in k31["E"]["rule_made"]}
n_new = {t for t, *_ in k31["N"]["rule_made"]}
rows = [(t, r) for t in j31 for r in set(j31[t]) & set(j32[t]) if j31[t][r][3] != j32[t][r][3]]
tables = {t for t, _ in rows}
print("rows", len(rows), "tables", len(tables), "E_new_tables", len(tables & e_new), "N_new_tables", len(tables & n_new),
      "other", len(tables - e_new - n_new))
s4n, s4n2 = paths("v2", {"S4n"}), paths("v2", {"S4n2"})
s5n, s5n2 = paths("v2", {"S5n"}), paths("v2", {"S5n2"})
cause = Counter()
out = []
for t, r in sorted(rows):
    grp = "E_new" if t in e_new else "N_new" if t in n_new else "other"
    c4 = s4n[t].get(r, (0, 0, 0, None))[3] != s4n2[t].get(r, (0, 0, 0, None))[3]
    c5 = s5n[t].get(r, (0, 0, 0, None))[3] != s5n2[t].get(r, (0, 0, 0, None))[3]
    back = j32[t][r][3] == j2[t][r][3]
    cause[(grp, "S4alone" if c4 else "", "S5alone" if c5 else "", "back_to_v2" if back else "new_path")] += 1
    if grp == "other" and len(out) < 12:
        out.append({"t": t, "r": r, "v2": j2[t][r][3], "v3.1": j31[t][r][3], "v3.2": j32[t][r][3]})
for k, v in sorted(cause.items()):
    print(k, v)
for o in out:
    print(json.dumps(o, ensure_ascii=False))
