"""2026-10-01 원고 수정 (v9 → v10). 1.3절에 셀 문장을 임베딩하여 벡터 데이터베이스에 저장한다는 내용을 명시.

  코드(scripts/mh_arms.py)는 셀 문장 벡터를 행렬로 저장하고 질문 벡터와의 내적을 전부 계산한다 —
  2.1.2절 벡터 데이터베이스 정의("조각 수가 적으면 모든 벡터와의 유사도를 직접 계산")에 해당.

  .venv/bin/python thesis/hwp/edit_20261001c.py <입력.hwpx> <출력.hwpx>
"""
import re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

OLD = "셀 문장은 임베딩하여 색인하고,"
NEW = "생성한 셀 문장은 임베딩 모델로 임베딩하여 벡터 데이터베이스에 저장하고,"

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
S = files["Contents/section2.xml"].decode("utf-8")

bs, be, sp = top_paras(S)
hit = [(a, b) for a, b in sp if OLD in text_of(S[a:b])]
assert len(hit) == 1 and text_of(S[hit[0][0]:hit[0][1]]).startswith("본 논문에서는 계층형 표가")
a, b = hit[0]
p = S[a:b]
assert p.count("<hp:t>") == 1 and p.count(OLD) == 1
# 글이 바뀐 문단은 줄 배치 캐시(linesegarray)를 빼서 한글이 다시 계산하게 한다
p = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p.replace(OLD, NEW), flags=re.S)
S = S[:a] + p + S[b:]

files["Contents/section2.xml"] = S.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst)
