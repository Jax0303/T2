"""2026-10-01 원고 수정 (v12 → v13). '머리글 행/머리글 열'(위치) → '위쪽 머리글/왼쪽 머리글'.

  '머리글 행'(2011·2010 이 있는 줄)과 '열 머리글'(2011·2010 자체)이 순서만 바뀐 같은 단어라 헷갈린다는 사용자 지적.
  HiTab 데이터셋 용어(top header / left header, top_header_rows_num / left_header_columns_num)를 따른다.
  행 머리글·열 머리글(경로 쪽 표현)은 그대로 둔다. 3.3.1 첫 문단에 두 용어의 관계를 한 문장으로 정의한다.

  .venv/bin/python thesis/hwp/edit_20261001f.py <입력.hwpx> <출력.hwpx>
"""
import re, sys, zipfile
from pathlib import Path
from xml.sax.saxutils import escape, unescape

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

EDITS = [   # (바꿀 글, 새 글) — 각각 원고에서 한 번만 나와야 한다
    ("머리글 경로 구성기: 문서의 각 표에서 머리글 행과 머리글 열을 판정하고",
     "머리글 경로 구성기: 문서의 각 표에서 위쪽 머리글과 왼쪽 머리글을 판정하고"),
    ("계층형 표는 머리글 행, 머리글 열, 데이터 칸으로 구성된다. 머리글 행은 표의 위쪽에서 각 열의 이름을 적은 행이고, "
     "머리글 열은 표의 왼쪽에서 각 행의 이름을 적은 열이며, 나머지 칸이 값을 담은 데이터 칸이다. 머리글 행이 두 줄 이상이면 "
     "윗줄의 머리글이 아랫줄의 여러 머리글을 묶는 상위 머리글이 되며, 행 쪽의 머리글도 같은 방식으로 여러 층을 이룬다.",
     "계층형 표는 위쪽 머리글(top header), 왼쪽 머리글(left header), 데이터 칸으로 구성된다. 위쪽 머리글은 표의 위쪽 몇 줄에 "
     "각 열의 이름을 적은 영역이고, 왼쪽 머리글은 표의 왼쪽 몇 열에 각 행의 이름을 적은 영역이며, 나머지 칸이 값을 담은 데이터 "
     "칸이다. 위쪽 머리글의 칸은 열의 이름이므로 열 머리글이라 하고, 왼쪽 머리글의 칸은 행의 이름이므로 행 머리글이라 한다. "
     "위쪽 머리글이 두 줄 이상이면 윗줄의 머리글이 아랫줄의 여러 머리글을 묶는 상위 머리글이 되며, 행 머리글도 같은 방식으로 "
     "여러 층을 이룬다."),
    ("먼저 표의 어느 행이 머리글 행이고 어느 열이 머리글 열인지 알아야 한다.",
     "먼저 표의 위쪽 몇 줄이 위쪽 머리글이고 왼쪽 몇 열이 왼쪽 머리글인지 알아야 한다."),
    ("숫자인지 여부를 이용하여 머리글 행과 머리글 열을 판정한다.", "숫자인지 여부를 이용하여 위쪽 머리글과 왼쪽 머리글을 판정한다."),
    ("머리글 행은 데이터가 처음 나타나는 행을 찾아 그 위의 행을 모두 머리글 행으로 판정한다.",
     "위쪽 머리글은 데이터가 처음 나타나는 행을 찾아 그 위의 행을 모두 위쪽 머리글로 판정한다."),
    ("재무 보고서의 표는 머리글 행의 첫 칸을 비워 두고", "재무 보고서의 표는 위쪽 머리글의 첫 칸을 비워 두고"),
    ("첫 행이 머리글 행이 된다.", "첫 행이 위쪽 머리글이 된다."),
    ("행은 머리글 행에 포함한다.", "행은 위쪽 머리글에 포함한다."),
    ("머리글 행에는 연도나 항목 이름과 같은 글자가 있고", "위쪽 머리글에는 연도나 항목 이름과 같은 글자가 있고"),
    ("머리글 열의 수는 머리글 행을 판정한 후에 정한다. 머리글 열이 여러 개이면 각 머리글 열의 이름은 위쪽 머리글 행에 적히고, "
     "마지막 머리글 행에서는 그 위치가 비어 있다. 따라서 머리글 행이 두 줄 이상이면 마지막 머리글 행에서 왼쪽부터 연속으로 "
     "비어 있는 칸의 수를 머리글 열의 수로 본다. 머리글 행이 한 줄이거나 이러한 빈 칸이 없으면, 머리글 행 아래 칸의 절반 이상이 "
     "숫자인 첫 열의 바로 앞 열까지를 머리글 열로 본다. 머리글 열은 최소 1개로 하며, 평가 문서의 표 가운데 95.2%는 첫 열 하나가 "
     "머리글 열로 판정되었다.",
     "왼쪽 머리글의 열 수는 위쪽 머리글을 판정한 후에 정한다. 왼쪽 머리글이 여러 열이면 각 열의 이름은 위쪽 머리글의 윗줄에 "
     "적히고, 위쪽 머리글의 마지막 줄에서는 그 위치가 비어 있다. 따라서 위쪽 머리글이 두 줄 이상이면 그 마지막 줄에서 왼쪽부터 "
     "연속으로 비어 있는 칸의 수를 왼쪽 머리글의 열 수로 본다. 위쪽 머리글이 한 줄이거나 이러한 빈 칸이 없으면, 위쪽 머리글 아래 "
     "칸의 절반 이상이 숫자인 첫 열의 바로 앞 열까지를 왼쪽 머리글로 본다. 왼쪽 머리글은 최소 1열로 하며, 평가 문서의 표 가운데 "
     "95.2%는 첫 열 하나가 왼쪽 머리글로 판정되었다."),
    ("규칙으로 판정한 머리글 행의 수는 91.1%의 표에서, 머리글 열의 수는 91.9%의 표에서",
     "규칙으로 판정한 위쪽 머리글의 줄 수는 91.1%의 표에서, 왼쪽 머리글의 열 수는 91.9%의 표에서"),
    ("머리글 행의 수가 99.0% 일치하였다", "위쪽 머리글의 줄 수가 99.0% 일치하였다"),
    ("[그림 3-3]은 표 하나에서 머리글 행과 머리글 열을 판정하고", "[그림 3-3]은 표 하나에서 위쪽 머리글과 왼쪽 머리글을 판정하고"),
    ("먼저 문서의 각 표에서 머리글 행과 머리글 열을 판정하고", "먼저 문서의 각 표에서 위쪽 머리글과 왼쪽 머리글을 판정하고"),
]
NOLSA = lambda p: re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p, flags=re.S)

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
S = files["Contents/section2.xml"].decode("utf-8")

for old, new in EDITS:
    hit = [(a, b) for a, b in top_paras(S)[2] if old in unescape(text_of(S[a:b]))]
    assert len(hit) == 1, old
    a, b = hit[0]
    p = S[a:b]
    assert p.count("<hp:t>") == 1 and unescape(text_of(p)).count(old) == 1, old
    t = re.search(r"<hp:t>(.*?)</hp:t>", p, re.S).group(1)
    t = escape(unescape(t).replace(old, new))
    p = re.sub(r"<hp:t>.*?</hp:t>", lambda m: f"<hp:t>{t}</hp:t>", p, count=1, flags=re.S)
    S = S[:a] + NOLSA(p) + S[b:]

body = unescape("".join(text_of(S[a:b]) for a, b in top_paras(S)[2]))
assert "머리글 행" not in body and "머리글 열" not in body, "남은 머리글 행/열"
for sec in ("section0", "section1"):
    assert not re.search("머리글 (행|열)", unescape(text_of(files[f"Contents/{sec}.xml"].decode("utf-8")))), sec

files["Contents/section2.xml"] = S.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst, "— 문장", len(EDITS), "곳")
