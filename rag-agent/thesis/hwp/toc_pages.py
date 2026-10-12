"""렌더링한 쪽 글(text.json)에서 목차 항목의 쪽 번호를 찾는다. 쪽 번호 = 그 쪽 꼬리말의 "-N-".
  python3 thesis/hwp/toc_pages.py <pages_dir> -> thesis/hwp/toc_pages.json"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
pages = {int(k): v for k, v in json.load(open(Path(sys.argv[1]) / "text.json")).items()}
items = json.load(open(HERE / "toc_items.json"))
norm = lambda s: re.sub(r"[\s·]+", "", s)
def footer(p):
    for l in reversed(pages[p]):
        m = re.fullmatch(r"-([ivxⅰ-ⅹ\d]+)-", norm(l))
        if m: return m.group(1)
    return None
def find(title, start=1):
    t = norm(title)
    for p in sorted(pages):
        if p < start: continue
        if any(norm(l).startswith(t[:40]) for l in pages[p]): return p
    return None
lists_end = max(p for p in pages if any(norm(l) == "그림목차" for l in pages[p]))
body0 = find("제 1 장 서 론", lists_end + 1)
out, miss = {}, []
for key, title in (("국문초록", "국문초록"), ("영문초록", "ABSTRACT")):
    p = find(title, lists_end + 1); out[key] = footer(p) if p else None
for lvl, title in items["toc"]:
    p = find(title if not title.startswith("제 1 장") else "제 1 장 서 론", body0)
    (out.__setitem__(title, footer(p)) if p else miss.append(title))
for c in items["tables"] + items["figures"]:
    p = find(c, body0)
    (out.__setitem__(c, footer(p)) if p else miss.append(c))
json.dump(out, open(HERE / "toc_pages.json", "w"), ensure_ascii=False, indent=1)
print("found", len(out), "missing", miss)
