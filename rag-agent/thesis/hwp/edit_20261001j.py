"""2026-10-01 원고 배치 수정 (v17 → v18). 한글에서 저장한 v17 의 줄 배치로 확인한 두 곳.

  1 12쪽 끝에 '2.3' 절 제목과 그 뒤 빈 줄만 남고 2.3.1 은 13쪽으로 갈림:
    절 제목(다음 문단과 함께) 뒤의 빈 줄 문단이 다음 제목과 묶이지 않아서다.
    '절 제목 – 빈 줄 – 하위 절 제목' 의 빈 줄 문단 7개를 새 문단 모양 49(= 37 + keepWithNext)로.
  2 4장 마지막 문단의 마지막 줄 '인하였다.'(5자)만 42쪽으로 넘어감:
    바로 앞의 빈 줄 문단(4.3.3 '따라서 …' 와 '이상의 결과로부터 …' 사이) 하나를 지워 한 줄을 확보한다(글은 그대로).

  .venv/bin/python thesis/hwp/edit_20261001j.py <입력.hwpx> <출력.hwpx>
"""
import re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

LSA = re.compile(r"<hp:linesegarray>.*?</hp:linesegarray>", re.S)
src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
H = files["Contents/header.xml"].decode("utf-8")
S = files["Contents/section2.xml"].decode("utf-8")

cnt = int(re.search(r'<hh:paraProperties itemCnt="(\d+)"', H).group(1))
assert cnt == 49
m = re.search(r'<hh:paraPr id="37".*?</hh:paraPr>', H, re.S).group(0)
assert m.count('keepWithNext="0"') == 1
end = H.index("</hh:paraProperties>")
H = H[:end] + m.replace('id="37"', 'id="49"', 1).replace('keepWithNext="0"', 'keepWithNext="1"') + H[end:]
H = H.replace(f'<hh:paraProperties itemCnt="{cnt}"', f'<hh:paraProperties itemCnt="{cnt + 1}"', 1)

sp = top_paras(S)[2]
seg = [S[a:b] for a, b in sp]
txt = [text_of(q) for q in seg]
pp = lambda q: re.match(r'<hp:p [^>]*paraPrIDRef="(\d+)"', q).group(1)

# 1 절 제목 – 빈 줄 – 하위 절 제목
hit = [i + 1 for i in range(len(seg) - 2)
       if re.match(r"\d\.\d ", txt[i]) and not txt[i + 1].strip() and re.match(r"\d\.\d\.\d ", txt[i + 2])]
assert len(hit) == 7 and all(pp(seg[i]) == "37" and "<hp:tbl" not in seg[i] and "<hp:pic" not in seg[i] for i in hit), hit
for i in hit:
    seg[i] = re.sub(r'(<hp:p [^>]*paraPrIDRef=")37(")', r"\g<1>49\2", seg[i], count=1)

# 2 4장 마지막 문단 앞의 빈 줄 삭제
k = next(i for i, t in enumerate(txt) if t.startswith("이상의 결과로부터, 셀마다 머리글 경로를"))
assert not txt[k - 1].strip() and txt[k - 2].startswith("따라서 검색 단계에서 정답 셀을 누락") and "<hp:tbl" not in seg[k - 1]
drop = k - 1

out, pos, changed = [], 0, 0
for i, ((a, b), q) in enumerate(zip(sp, seg)):
    out.append(S[pos:a]); pos = b
    if i == drop:
        continue
    if q != S[a:b]:
        q = LSA.sub("", q); changed += 1
    out.append(q)
out.append(S[pos:])
S = "".join(out)
print("빈 줄 문단 묶음", len(hit), "바뀐 문단", changed, "지운 문단 1")

files["Contents/header.xml"] = H.encode("utf-8")
files["Contents/section2.xml"] = S.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst)
