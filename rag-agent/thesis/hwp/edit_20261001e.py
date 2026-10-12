"""2026-10-01 원고 수정 (v11 → v12). 3.3.1 끝에 [그림 3-3] 머리글 경로 구성의 예를 단계마다 그림 하나(4개)로 넣는다.

  그림: make_figs.fig3_3_header_path → fig/fig3_3_step1~4.png, 표는 [그림 3-4]와 같은 값(make_figs.T34_ROWS).
  그림 문단은 [그림 3-4] 그림 문단(image5)을 복제하고, 비어 있던 [그림 3-3] 그림 자리 문단을 대신한다.
  3.3.1 마지막 문단 끝에 [그림 3-3] 참조 문장을 붙인다.
  그림 네 문단이 쪽 경계에서 나뉘지 않게, 그림 문단 모양(41)을 복제해 '다음 문단과 함께'(keepWithNext)를 켠 문단 모양을 새로 만들어 쓴다.

  .venv/bin/python thesis/hwp/edit_20261001e.py <입력.hwpx> <출력.hwpx>
"""
import re, struct, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

FIGS = [Path(__file__).resolve().parent.parent / f"fig/fig3_3_step{k}.png" for k in range(1, 5)]
TAIL = "이 규칙은 경로가 같은 행과 열에만 적용하며, 나머지 셀의 경로는 변경하지 않는다."
REF = (" [그림 3-3]은 표 하나에서 머리글 행과 머리글 열을 판정하고 셀의 행 경로와 열 경로를 구성한 후, "
       "문서 안의 다른 표와 경로가 같을 때 경로를 확장하는 과정을 단계별로 나타낸 것이다.")
NOLSA = lambda p: re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p, flags=re.S)

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
S = files["Contents/section2.xml"].decode("utf-8")
assert "[그림 3-3]은" not in S and not any(f"BinData/image{7 + k}.png" in files for k in range(4))

sp = top_paras(S)[2]
txt = [text_of(S[a:b]) for a, b in sp]
cap = txt.index("[그림 3-3] 머리글 경로 구성의 예")
slot, last = sp[cap - 1], sp[cap - 2]                     # 비어 있는 그림 자리, 3.3.1 마지막 문단
assert txt[cap - 1] == "" and "<hp:pic" not in S[slot[0]:slot[1]] and 'paraPrIDRef="41"' in S[slot[0]:slot[1]]
assert txt[cap - 2].endswith(TAIL) and S[last[0]:last[1]].count("<hp:t>") == 1

# [그림 3-4] 그림 문단을 틀로 쓴다
pic = [(a, b) for a, b in sp if 'binaryItemIDRef="image5"' in S[a:b]]
assert len(pic) == 1
tmpl = NOLSA(S[pic[0][0]:pic[0][1]])
W = 36848
paras = []
for k, f in enumerate(FIGS):
    w_px, h_px = struct.unpack(">II", f.read_bytes()[16:24])
    H = round(W * h_px / w_px)
    n = 101 + k
    p = tmpl.replace('binaryItemIDRef="image5"', f'binaryItemIDRef="image{7 + k}"')
    p = p.replace('id="1500000015"', f'id="{1500000000 + n}"').replace('instid="17"', f'instid="{n}"').replace('zOrder="20"', f'zOrder="{n}"')
    p = re.sub(r'<hp:orgSz width="\d+" height="\d+"/>', f'<hp:orgSz width="{W}" height="{H}"/>', p)
    p = re.sub(r'<hp:curSz width="\d+" height="\d+"/>', f'<hp:curSz width="{W}" height="{H}"/>', p)
    p = re.sub(r'<hc:scaMatrix [^>]*/>', '<hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>', p)
    p = re.sub(r'centerX="\d+" centerY="\d+"', f'centerX="{W // 2}" centerY="{H // 2}"', p)
    p = re.sub(r'<hc:pt1 x="\d+" y="0"/><hc:pt2 x="\d+" y="\d+"/><hc:pt3 x="0" y="\d+"/>',
               f'<hc:pt1 x="{W}" y="0"/><hc:pt2 x="{W}" y="{H}"/><hc:pt3 x="0" y="{H}"/>', p)
    p = re.sub(r'<hp:imgClip left="0" right="\d+" top="0" bottom="\d+"/>',
               f'<hp:imgClip left="0" right="{w_px * 75}" top="0" bottom="{h_px * 75}"/>', p)
    p = re.sub(r'<hp:imgDim dimwidth="\d+" dimheight="\d+"/>', f'<hp:imgDim dimwidth="{w_px * 75}" dimheight="{h_px * 75}"/>', p)
    p = re.sub(r'<hp:sz width="\d+" widthRelTo="ABSOLUTE" height="\d+"', f'<hp:sz width="{W}" widthRelTo="ABSOLUTE" height="{H}"', p)
    assert f'image{7 + k}' in p and "image5" not in p and f'height="{H}"' in p
    paras.append(p)

# 그림 문단 모양 41 + keepWithNext → 새 문단 모양(네 그림과 캡션이 한 쪽에 모이게)
Hd = files["Contents/header.xml"].decode("utf-8")
m = re.search(r'<hh:paraProperties itemCnt="(\d+)"', Hd)
new_id = int(m.group(1))
assert f'<hh:paraPr id="{new_id}"' not in Hd and f'<hh:paraPr id="{new_id - 1}"' in Hd
p41 = re.search(r'<hh:paraPr id="41".*?</hh:paraPr>', Hd, re.S).group(0)
assert p41.count('keepWithNext="0"') == 1
# 새 문단 모양은 목록 맨 끝에 붙인다(id 순서 = 목록 순서를 유지)
assert Hd.count("</hh:paraProperties>") == 1
Hd = Hd.replace("</hh:paraProperties>", p41.replace('<hh:paraPr id="41"', f'<hh:paraPr id="{new_id}"')
                .replace('keepWithNext="0"', 'keepWithNext="1"') + "</hh:paraProperties>")
Hd = Hd.replace(m.group(0), f'<hh:paraProperties itemCnt="{new_id + 1}"', 1)
files["Contents/header.xml"] = Hd.encode("utf-8")
paras = [p.replace('paraPrIDRef="41"', f'paraPrIDRef="{new_id}"', 1) for p in paras]

# 뒤에서부터 바꿔야 앞 위치가 안 밀린다
S = S[:slot[0]] + "".join(paras) + S[slot[1]:]
p = NOLSA(S[last[0]:last[1]]).replace(TAIL, TAIL + REF)
S = S[:last[0]] + p + S[last[1]:]

hpf = files["Contents/content.hpf"].decode("utf-8")
items = "".join(f'<opf:item id="image{7 + k}" href="BinData/image{7 + k}.png" media-type="image/png" isEmbeded="1"/>'
                for k in range(4))
assert hpf.count('<opf:item id="section2"') == 1
files["Contents/content.hpf"] = hpf.replace('<opf:item id="section2"', items + '<opf:item id="section2"').encode("utf-8")
for k, f in enumerate(FIGS):
    files[f"BinData/image{7 + k}.png"] = f.read_bytes()
files["Contents/section2.xml"] = S.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst, "— [그림 3-3] 그림", len(paras), "개")
