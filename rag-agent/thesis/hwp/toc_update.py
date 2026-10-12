"""목차·표 목차·그림 목차의 쪽 번호를 한글이 저장한 줄 배치(linesegarray)에서 다시 계산해 고친다.

  본문(section2) 최상위 문단의 줄마다 vertpos 가 앞 줄보다 작아지거나 문단에 pageBreak="1" 이 있으면 새 쪽으로 센다.
  본문 첫 쪽이 1쪽이다. 한글에서 열어 저장한 파일(캐시 빠진 문단 0)만 받는다.
  2026-10-01: hwp MCP 렌더러는 표 높이를 한글과 다르게 잡아 4.1.2 이후 쪽이 1쪽씩 틀렸다(사용자 대조) → 렌더러를 쓰지 않는다.

  python3 thesis/hwp/toc_update.py <in.hwpx> <out.hwpx>      (out 을 주지 않으면 비교만 출력)
"""
import re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

src, dst = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
norm = lambda s: re.sub(r"[\s·]+", "", s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
SUB = re.compile(r"<hp:subList\b|</hp:subList>")


def own(p):
    """표·그림 안의 하위 문단을 뺀 최상위 문단 자신."""
    while True:
        m = re.search(r"<hp:subList\b", p)
        if not m:
            return p
        depth = 0
        for t in SUB.finditer(p, m.start()):
            depth += -1 if t.group().startswith("</") else 1
            if depth == 0:
                p = p[:m.start()] + p[t.end():]
                break


z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
B = files["Contents/section2.xml"].decode("utf-8")
first_page = {}            # 정규화한 문단 글 → 그 문단 첫 줄의 쪽
page, prev, missing = 1, None, 0
for a, b in top_paras(B)[2]:
    p = B[a:b]
    vs = [int(v) for v in re.findall(r'<hp:lineseg textpos="\d+" vertpos="(-?\d+)"', own(p))]
    missing += not vs
    if re.match(r'<hp:p [^>]*pageBreak="1"', p) and prev is not None:
        page, prev = page + 1, None
    for k, v in enumerate(vs):
        if prev is not None and v < prev:
            page += 1
        if k == 0:
            first_page.setdefault(norm(text_of(p)), page)
        prev = v
assert missing == 0, f"줄 배치 캐시가 없는 문단 {missing}개 — 한글에서 열어 저장한 파일을 넣을 것"
print(f"본문 마지막 쪽 {page}")

S = files["Contents/section0.xml"].decode("utf-8")
out, pos, changed, kept = [], 0, [], []
for a, b in top_paras(S)[2]:
    p = S[a:b]
    m = re.search(r'(<hp:tab [^>]*leader="3"[^>]*/>)(?:</hp:t></hp:run><hp:run [^>]*><hp:t>)?([ivx\d]+)</hp:t>', p)
    assert m or 'leader="3"' not in p, re.sub(r"<[^>]+>", "", p)[:60]
    if m:
        title = re.sub(r"<[^>]+>", "", p[:m.start()]).strip()
        new = first_page.get(norm(title))
        if new is None:                     # 국문초록·영문초록(앞부분 쪽 번호)은 그대로
            kept.append(title)
        elif str(new) != m.group(2):
            changed.append((title, m.group(2), new))
            p = p[:m.start(2)] + str(new) + p[m.end(2):]
            p = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p, flags=re.S)   # 바뀐 문단은 캐시 삭제
    out.append(S[pos:a]); out.append(p); pos = b
out.append(S[pos:])
for c in changed:
    print(f"{c[1]:>4} → {c[2]:<4} {c[0]}")
print(f"바뀐 항목 {len(changed)}, 본문에 없어 그대로 둔 항목 {kept}")
if dst:
    files["Contents/section0.xml"] = "".join(out).encode("utf-8")
    with zipfile.ZipFile(dst, "w") as zo:
        zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name, data in files.items():
            zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    print("wrote", dst)
