"""2026-10-01 원고 수정 (v14 → v15). 최종 검토.

  1 [표 4-7] 같은 수의 셀을 전달할 때의 재현율 삭제(사용자: 답변 정확도와 무관해 의미가 적음) — 캡션·표·단위 줄·설명 문단,
    표 목차 줄, 5장의 같은 셀 수 문장. 설명 문단의 재현율@1·셀 정밀도@20 최대값(58.6%·11.9%)은 [표 4-6] 설명 끝으로 옮기고
    58.6% 의 근거를 바로잡음(평균 2.37개의 역수가 아니라 문항별 1/정답 셀 수의 평균, results/equal_cells_20261001/equal_cells.py).
    [표 4-8] → [표 4-7].
  2 문어체: 2.2·4.3.1 RAPTOR 설명의 '나누다' → '분할하다'.
  3 5장 한계 넷째: 원고에 정의되지 않은 '형식 허용 채점 규칙' → '표기 차이를 허용하는 채점 규칙(4.2.2절)'.
  4 3.3.4 '10 ~ 75개' → '10~75개'(다른 범위 표기와 통일).
  5 제출서·인준서 날짜 '26년' → '2026년'(표지·속표지는 v3 에서 고침).

  .venv/bin/python thesis/hwp/edit_20261001h.py <입력.hwpx> <출력.hwpx>
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
S = {k: files[f"Contents/{k}.xml"].decode("utf-8") for k in ("section0", "section1", "section2")}
ORIG = dict(S)


def sub(sec, old, new, count=1):
    """글(이스케이프 전) 치환. 원고에서 정확히 count 번 나와야 한다."""
    o, n = escape(old), escape(new)
    assert S[sec].count(o) == count, (sec, old, S[sec].count(o))
    S[sec] = S[sec].replace(o, n)


# ================================================================ 1 [표 4-7] 삭제
sp = top_paras(S["section2"])[2]
txt = [text_of(S["section2"][x:y]) for x, y in sp]
i_cap = txt.index("[표 4-7] 같은 수의 셀을 전달할 때의 재현율(200문항)")
assert "<hp:tbl" in S["section2"][sp[i_cap + 1][0]:sp[i_cap + 1][1]]
assert txt[i_cap + 2] == "(전달 셀 수는 문항당 평균, 재현율 단위: %)"
assert txt[i_cap + 3].startswith("[표 4-5]의 k는 검색 단위의 개수이므로")
assert txt[i_cap + 4].startswith("2,885문항 전체에서도 같은 경향이 나타났다.")
S["section2"] = S["section2"][:sp[i_cap][0]] + S["section2"][sp[i_cap + 3][1]:]

sub("section2", "제안하는 방법은 검색 단위가 셀 하나이므로 셀 정밀도가 [표 4-5]의 정밀도와 같다.",
    "제안하는 방법은 검색 단위가 셀 하나이므로 셀 정밀도가 [표 4-5]의 정밀도와 같다. 또한 제안하는 방법의 재현율@1은 셀 "
    "1개로 계산하므로 정답 셀이 2개 이상인 문항에서는 100%가 될 수 없으며, 200문항에서 재현율@1이 가질 수 있는 최대값은 "
    "58.6%이다. 문항당 정답 셀은 평균 2.37개이므로 셀 20개를 전달할 때 셀 정밀도@20의 최대값은 11.9%이다.")
sub("section2", " 전달하는 셀 수를 같게 맞추면, 평균 46.3개 셀을 전달하는 LangChain 청크의 1위 청크는 재현율이 76.8%였고 "
    "제안하는 방법은 98.6%였다.", "")
sub("section2", "[표 4-8]은 모든 비교군을 평가한", "[표 4-7]은 모든 비교군을 평가한")
sub("section2", "[표 4-8] 답변 정확도(200문항)", "[표 4-7] 답변 정확도(200문항)")
sub("section2", "[표 4-5]와 [표 4-8]을 함께 보면", "[표 4-5]와 [표 4-7]을 함께 보면")
assert "[표 4-8]" not in S["section2"] and "98.6" not in S["section2"]

sp0 = top_paras(S["section0"])[2]
e47 = [(x, y) for x, y in sp0 if "[표 4-7] 같은 수의 셀을 전달할 때의 재현율(200문항)" in S["section0"][x:y]]
assert len(e47) == 1
S["section0"] = S["section0"][:e47[0][0]] + S["section0"][e47[0][1]:]
sub("section0", "[표 4-8] 답변 정확도(200문항)", "[표 4-7] 답변 정확도(200문항)")

# ================================================================ 2 문어체
sub("section2", "줄바꿈에서 문장으로 나눈 후,", "줄바꿈에서 문장으로 분할한 후,")
sub("section2", "쉼표, 세미콜론, 콜론에서 다시 나눈다.", "쉼표, 세미콜론, 콜론에서 다시 분할한다.")
sub("section2", "마침표에서 문장을 나누고 100토큰을 넘는 문장은 쉼표에서 다시 나누므로",
    "마침표에서 문장을 분할하고 100토큰을 넘는 문장은 쉼표에서 다시 분할하므로")

# ================================================================ 3·4
sub("section2", "넷째, 조회 문항의 형식 허용 채점 규칙은", "넷째, 조회 문항에서 표기 차이를 허용하는 채점 규칙(4.2.2절)은")
sub("section2", "10 ~ 75개를", "10~75개를")

# ================================================================ 5 제출서·인준서 날짜(누름틀 안의 글)
FIELD26 = "</hp:fieldBegin></hp:ctrl><hp:t>26</hp:t>"  # 목차의 제4장 쪽 번호 26 과 구별
assert S["section0"].count(FIELD26) == 2
S["section0"] = S["section0"].replace(FIELD26, "</hp:fieldBegin></hp:ctrl><hp:t>2026</hp:t>")

# ================================================================ 바뀐 최상위 문단의 줄 배치 캐시 삭제
for sec in S:
    old = {ORIG[sec][x:y] for x, y in top_paras(ORIG[sec])[2]}
    out, pos, n = [], 0, 0
    for x, y in top_paras(S[sec])[2]:
        q = S[sec][x:y]
        if q not in old:
            q = LSA.sub("", q); n += 1
        out += [S[sec][pos:x], q]; pos = y
    out.append(S[sec][pos:])
    S[sec] = "".join(out)
    print(f"{sec}: 바뀐 문단 {n}")

for sec in S:
    files[f"Contents/{sec}.xml"] = S[sec].encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst)
