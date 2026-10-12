"""2026-10-01 원고 배치 수정 (v16 → v17). 사용자 대조 결과(저장된 줄 배치 기준) 반영.

  1 표 제목과 표가 다른 쪽으로 갈린 곳(표 4-2·4-7·A-2)과 그림과 제목이 갈린 곳(그림 3-4):
    표 제목 문단 14개를 '다음 문단과 함께'(새 문단 모양 47 = 38 + keepWithNext)로,
    단위 줄이 뒤따르는 표 문단 2개(표 4-5·4-6)도 47로,
    그림 문단 6개를 46(= 41 + keepWithNext, 그림 3-3 네 문단이 이미 쓰는 모양)으로.
  2 절 제목 36개를 '다음 문단과 함께'(새 문단 모양 48 = 29 + keepWithNext) — 위 이동으로 쪽이 밀려도 제목만 쪽 끝에 남지 않게.
  3 부록 E 마지막 문단의 마지막 줄(17자)이 57쪽으로 넘어감 → 같은 뜻으로 약 30자 줄임.

  목차 쪽 번호는 한글에서 이 파일을 열어 저장한 뒤 toc_update.py 로 갱신한다.
  .venv/bin/python thesis/hwp/edit_20261001i.py <입력.hwpx> <출력.hwpx>
"""
import re, sys, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

LSA = re.compile(r"<hp:linesegarray>.*?</hp:linesegarray>", re.S)
src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
H = files["Contents/header.xml"].decode("utf-8")
S = files["Contents/section2.xml"].decode("utf-8")

# ---------------------------------------------------------------- 새 문단 모양(목록 맨 끝에 붙인다)
cnt = int(re.search(r'<hh:paraProperties itemCnt="(\d+)"', H).group(1))
assert cnt == 47
def with_keep(src_id, new_id):
    m = re.search(rf'<hh:paraPr id="{src_id}".*?</hh:paraPr>', H, re.S).group(0)
    assert m.count('keepWithNext="0"') == 1
    return m.replace(f'id="{src_id}"', f'id="{new_id}"', 1).replace('keepWithNext="0"', 'keepWithNext="1"')
end = H.index("</hh:paraProperties>")
H = H[:end] + with_keep(38, 47) + with_keep(29, 48) + H[end:]
H = H.replace(f'<hh:paraProperties itemCnt="{cnt}"', f'<hh:paraProperties itemCnt="{cnt + 2}"', 1)
g = lambda i: re.search(rf'<hh:paraPr id="{i}".*?</hh:paraPr>', H, re.S).group(0)
assert g(41).replace('id="41"', "").replace('keepWithNext="0"', "K") == g(46).replace('id="46"', "").replace('keepWithNext="1"', "K")

# ---------------------------------------------------------------- 문단 모양 바꾸기
sp = top_paras(S)[2]
seg = [S[a:b] for a, b in sp]
pp = lambda q: re.match(r'<hp:p [^>]*paraPrIDRef="(\d+)"', q).group(1)
setp = lambda q, new: re.sub(r'(<hp:p [^>]*paraPrIDRef=")\d+(")', lambda m: m.group(1) + new + m.group(2), q, count=1)
n = {"표 제목": 0, "단위 앞 표": 0, "그림": 0, "절 제목": 0}
for i, q in enumerate(seg):
    t = text_of(q)
    if pp(q) == "38" and re.match(r"\[표 [0-9A-E]-\d+\] ", t):
        seg[i] = setp(q, "47"); n["표 제목"] += 1
    elif pp(q) == "38" and "<hp:tbl" in q and text_of(seg[i + 1]).startswith("(") and "단위" in text_of(seg[i + 1]):
        seg[i] = setp(q, "47"); n["단위 앞 표"] += 1
    elif pp(q) == "41" and "<hp:pic" in q:
        assert text_of(seg[i + 1]).startswith("[그림 ")
        seg[i] = setp(q, "46"); n["그림"] += 1
    elif pp(q) == "29" and re.match(r"\d\.\d(\.\d)? ", t):
        seg[i] = setp(q, "48"); n["절 제목"] += 1
assert n == {"표 제목": 14, "단위 앞 표": 2, "그림": 6, "절 제목": 36}, n

# ---------------------------------------------------------------- 부록 E 마지막 문단 줄이기
last = len(seg) - 1
assert text_of(seg[last]).startswith("머리글 경로를 제거하면 재현율@20이")
for old, new in (("셀 값만 색인하면 \"$14\"나 \"-35\"와 같은 숫자 문자열만 남으므로, 질문에 나오는 항목 이름과 연도가 색인에 없어 "
                  "질문과 셀을 연결할 수 없다.",
                  "셀 값만 색인하면 \"$14\"나 \"-35\"와 같은 숫자만 남아, 질문의 항목 이름과 연도로 셀을 찾을 수 없다."),
                 ("추가하는 효과이며, 같은 셀을 머리글 없이 LLM에 입력하는 효과는 별도로 검증하지 않았다.",
                  "추가하는 효과이며 머리글 없는 셀을 LLM에 입력하는 효과는 검증하지 않았다.")):
    assert seg[last].count(escape(old)) == 1, old
    seg[last] = seg[last].replace(escape(old), escape(new))
print("마지막 문단", len(text_of(S[sp[last][0]:sp[last][1]])), "→", len(text_of(seg[last])), "자")

# ---------------------------------------------------------------- 바뀐 문단의 줄 배치 캐시 삭제
out, pos, changed = [], 0, 0
for (a, b), q in zip(sp, seg):
    if q != S[a:b]:
        q = LSA.sub("", q); changed += 1
    out += [S[pos:a], q]; pos = b
out.append(S[pos:])
S = "".join(out)
print("바뀐 문단", changed)

files["Contents/header.xml"] = H.encode("utf-8")
files["Contents/section2.xml"] = S.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst)
