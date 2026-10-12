"""2026-10-01 원고 수정 (v10 → v11). 3.2절·3.3.3절에 벡터 데이터베이스를 명시하고 [그림 3-2]를 교체.

  [그림 3-2] = BinData/image4.png, make_figs.fig3_2_system 으로 다시 그린 그림(③ 뒤에 벡터 데이터베이스 상자).
  새 그림도 1516×636 px 라 hwpx 의 그림 크기 값은 그대로 둔다.

  .venv/bin/python thesis/hwp/edit_20261001d.py <입력.hwpx> <출력.hwpx>
"""
import re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

FIG = Path(__file__).resolve().parent.parent / "fig/fig3_2_system.png"
EDITS = [   # (바꿀 글, 새 글) — 각각 원고 전체에서 한 번만 나와야 한다
    ("⦁ 색인: 셀 문장을 임베딩 모델로 벡터화하여 저장하고,",
     "⦁ 색인: 셀 문장을 임베딩 모델로 768차원 벡터로 변환하여 벡터 데이터베이스에 저장하고,"),
    ("⦁ 하이브리드 검색기: 질문과 셀 문장의 코사인 유사도와 BM25 점수를 가중합하여 순위를 매기고, 상위 20개 셀 문장을 선택한다.",
     "⦁ 하이브리드 검색기: 질문을 같은 임베딩 모델로 벡터화하여 벡터 데이터베이스의 셀 문장 벡터와 코사인 유사도를 계산하고, "
     "이를 BM25 점수와 가중합하여 순위를 매긴 후 상위 20개 셀 문장을 선택한다."),
    ("768차원 벡터로 변환하여 저장한다.", "768차원 벡터로 변환하여 벡터 데이터베이스에 저장한다."),
    ("색인과 BM25 단어 통계는 평가 문서 전체의 셀 문장으로 구축하고,",
     "벡터 데이터베이스와 BM25 색인(단어 통계 포함)은 평가 문서 전체의 셀 문장으로 구축하고,"),
]

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
S = files["Contents/section2.xml"].decode("utf-8")

for old, new in EDITS:
    assert S.count(old) == 1, old
    hit = [(a, b) for a, b in top_paras(S)[2] if old in text_of(S[a:b])]
    assert len(hit) == 1, old
    a, b = hit[0]
    p = S[a:b]
    assert p.count("<hp:t>") == 1
    # 글이 바뀐 문단은 줄 배치 캐시(linesegarray)를 빼서 한글이 다시 계산하게 한다
    p = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p.replace(old, new), flags=re.S)
    S = S[:a] + p + S[b:]

old_png = files["BinData/image4.png"]
assert old_png[16:24] == FIG.read_bytes()[16:24]   # PNG IHDR 가로·세로가 같아야 그림 크기 값을 안 고쳐도 된다
files["BinData/image4.png"] = FIG.read_bytes()
files["Contents/section2.xml"] = S.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst, "— 문장", len(EDITS), "곳, [그림 3-2] 교체")
