"""2026-10-01 원고 수정 (v6 → v7). 발표자료 대조 결과 반영.

  제목 변경 · MRR 삭제 · 1.3 목표 문장 · 5장 한계 다섯째·향후 연구 · 선행연구 4건 · α 근거(PREREG-2026-10-01-alpha-dev.md)
  · [그림 3-3] 머리글 경로 구성의 예 신설(기존 3-3 → 3-4) · 참고문헌 [19]~[23]

  .venv/bin/python thesis/hwp/edit_20261001.py <입력.hwpx> <출력.hwpx>
"""
import re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "thesis/fig/fig3_3_header_path.png"

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
S = {k: files[f"Contents/{k}.xml"].decode("utf-8") for k in ("section0", "section1", "section2")}


def sub1(sec, old, new):
    """정확히 한 번 나오는 문자열만 바꾼다."""
    n = S[sec].count(old)
    assert n == 1, f"{sec}: {n}회 — {old[:60]!r}"
    S[sec] = S[sec].replace(old, new)


# ---------------------------------------------------------------- 제목 (표지 3곳·인준지·초록 2곳)
OLD_KO, NEW_KO = "폐쇄된 환경에서 계층형 표 구조의 셀 문장 임베딩 기반 RAG", "계층형 표 문서의 셀 문장 검색 단위 기반 RAG"
OLD_EN, NEW_EN = ("Cell-Sentence Embedding RAG for Hierarchical Tables in a Closed Environment",
                  "RAG with Cell-Sentence Retrieval Units for Documents with Hierarchical Tables")
for sec in ("section0", "section1"):
    assert OLD_KO in S[sec] and OLD_EN in S[sec]
    S[sec] = S[sec].replace(OLD_KO, NEW_KO).replace(OLD_EN, NEW_EN)
assert "Closed Environment" not in S["section0"] + S["section1"] + S["section2"]

# ---------------------------------------------------------------- MRR 삭제
sub1("section2", "검색 성능은 재현율(Recall@k), 정밀도(Precision@k), 평균 역순위(Mean Reciprocal Rank, MRR)로 측정한다.",
     "검색 성능은 재현율(Recall@k)과 정밀도(Precision@k)로 측정한다.")
# 4.2.2: r_i 수식 개체와 평균 역순위 문장 제거
m = re.search(r'은 조건이 참이면 1, 거짓이면 0이다\. 정답 셀을 하나 이상 포함한 첫 검색 단위의 순위를 </hp:t>'
              r'<hp:equation .*?<hp:script>r _\{i\}</hp:script></hp:equation>'
              r'<hp:t>라 하면, 평균 역순위는 다음과 같다\. 정답 셀을 포함한 단위가 없으면 해당 문항의 점수는 0이다\. </hp:t>',
              S["section2"], re.S)
assert m
S["section2"] = S["section2"][:m.start()] + "은 조건이 참이면 1, 거짓이면 0이다.</hp:t>" + S["section2"][m.end():]
# MRR 수식 문단 삭제
bs, be, sp = top_paras(S["section2"])
mrr_eq = [(a, b) for a, b in sp if "<hp:script>MRR`=`" in S["section2"][a:b]]
assert len(mrr_eq) == 1
a, b = mrr_eq[0]
S["section2"] = S["section2"][:a] + S["section2"][b:]
# 4.3.1 본문
sub1("section2", "각각 27.8%와 92.4%, MRR은 0.721이었고, LangChain 청크는 72.4%와 100.0%, 0.898, TableRAG(Yu)는 77.5%와 100.0%, 0.939였다.",
     "각각 27.8%와 92.4%였고, LangChain 청크는 72.4%와 100.0%, TableRAG(Yu)는 77.5%와 100.0%였다.")
# 부록 E 본문
sub1("section2", "재현율이 93.9%에서 24.7%로, MRR이 0.703에서 0.112로, 답변 정확도가", "재현율이 93.9%에서 24.7%로, 답변 정확도가")
sub1("section2", "재현율은 92.4%에서 23.3%로, MRR은 0.721에서 0.102로 하락하였다.", "재현율은 92.4%에서 23.3%로 하락하였다.")


def edit_tables(xml, fn):
    out, pos = [], 0
    for m in re.finditer(r"<hp:tbl\b.*?</hp:tbl>", xml, re.S):
        out.append(xml[pos:m.start()]); out.append(fn(m.group(0))); pos = m.end()
    out.append(xml[pos:])
    return "".join(out)


def del_col(t, col, give_to):
    """col 번째 열을 지우고 그 너비를 give_to 열에 더한다."""
    def tr_fn(tr):
        tcs = list(re.finditer(r"<hp:tc\b.*?</hp:tc>", tr, re.S))
        removed = 0; parts = []; pos = 0
        for m in tcs:
            tc = m.group(0)
            c = int(re.search(r'colAddr="(\d+)"', tc).group(1))
            if c == col:
                removed = int(re.search(r'<hp:cellSz width="(\d+)"', tc).group(1)); new = ""
            else:
                new = tc
                if c > col:
                    new = new.replace(f'colAddr="{c}"', f'colAddr="{c - 1}"', 1)
            parts.append(tr[pos:m.start()]); parts.append(new); pos = m.end()
        parts.append(tr[pos:])
        s = "".join(parts)
        assert removed
        # give_to 열 너비 확장 (colAddr 재번호 후 기준)
        def widen(m2):
            tc = m2.group(0)
            if re.search(rf'colAddr="{give_to}"', tc):
                w = int(re.search(r'<hp:cellSz width="(\d+)"', tc).group(1))
                tc = tc.replace(f'<hp:cellSz width="{w}"', f'<hp:cellSz width="{w + removed}"', 1)
            return tc
        return re.sub(r"<hp:tc\b.*?</hp:tc>", widen, s, flags=re.S)
    t = re.sub(r"<hp:tr>.*?</hp:tr>", lambda m: tr_fn(m.group(0)), t, flags=re.S)
    n = int(re.search(r'colCnt="(\d+)"', t).group(1))
    return t.replace(f'colCnt="{n}"', f'colCnt="{n - 1}"', 1)


def del_row(t, row):
    trs = list(re.finditer(r"<hp:tr>.*?</hp:tr>", t, re.S))
    tr = trs[row].group(0)
    h = int(re.search(r'<hp:cellSz width="\d+" height="(\d+)"', tr).group(1))
    t = t[:trs[row].start()] + t[trs[row].end():]
    for r in range(row + 1, len(trs)):
        t = t.replace(f'rowAddr="{r}"', f'rowAddr="{r - 1}"')
    n = int(re.search(r'rowCnt="(\d+)"', t).group(1))
    t = t.replace(f'rowCnt="{n}"', f'rowCnt="{n - 1}"', 1)
    H = int(re.search(r'<hp:sz width="\d+" widthRelTo="ABSOLUTE" height="(\d+)"', t).group(1))
    return t.replace(f'height="{H}" heightRelTo', f'height="{H - h}" heightRelTo', 1)


done = {"col": 0, "row": 0}


def table_fn(t):
    if "<hp:t>MRR</hp:t>" not in t:
        return t
    cnt = int(re.search(r'colCnt="(\d+)"', t).group(1))
    if cnt == 11:                           # 표 4-5: MRR 열(9) 삭제, 너비는 '방법' 열로
        done["col"] += 1
        return del_col(t, 9, 0)
    if cnt == 3:                            # 표 E-1: MRR 행(4) 삭제
        done["row"] += 1
        return del_row(t, 4)
    raise AssertionError(t[:200])


S["section2"] = edit_tables(S["section2"], table_fn)
assert done == {"col": 1, "row": 1}, done
assert "MRR" not in S["section2"] and "역순위" not in S["section2"]

# ---------------------------------------------------------------- 1.3 목표 문장 (발표자료 연구 목표와 통일)
sub1("section2", "이 셀 문장을 검색 단위로 삼는다. 표에 캡션이 없는 경우에는",
     "이 셀 문장을 검색 단위로 삼는다. 이를 통해 정답 셀의 재현율을 유지하면서 LLM에 전달하는 셀 수를 줄이고, 전달되는 각 셀 값에 행·열 머리글 경로를 결합하는 것이 본 논문의 목표이다. 표에 캡션이 없는 경우에는")

# ---------------------------------------------------------------- 5장: 검색 수치, 한계 다섯째, 향후 연구
sub1("section2", "TableRAG(Chen)과의 차이는 통계적으로 유의함을 확인하였다. 셀 문장에서 머리글 경로를 제거하면",
     "TableRAG(Chen)과의 차이는 통계적으로 유의함을 확인하였다. 검색 단계에서 제안하는 방법은 문항당 20개 셀을 전달하여 상위 20개 셀의 재현율 93.9%, 셀 정밀도 11.0%를 기록하였고, LangChain 청크는 상위 4개 청크로 문항당 평균 126.8개 셀을 전달하여 재현율 99.4%, 셀 정밀도 2.0%였다. 셀 문장에서 머리글 경로를 제거하면")
sub1("section2", "넷째, 조회 문항의 형식 허용 채점 규칙은 제안하는 방법의 오답을 검토한 후 결정하였다.",
     "넷째, 조회 문항의 형식 허용 채점 규칙은 제안하는 방법의 오답을 검토한 후 결정하였다. 다섯째, 제안하는 방법과 청크 분할 방법의 답변 정확도 차이에는 LLM에 전달하는 셀 수의 차이와 셀 값에 머리글 경로를 결합한 형식의 차이가 함께 포함되어 있으며, 두 요인의 기여를 분리하여 측정하지는 않았다.")
sub1("section2", "계산 도구를 결합하는 방법을 연구할 계획이다.",
     "계산 도구를 결합하는 방법을 연구할 계획이다. 또한 셀 문장으로 문서의 모든 셀을 전달하는 조건과 검색된 셀만 마크다운 부분 표로 전달하는 조건을 추가하여 전달 셀 수와 셀 문장 형식의 기여를 분리하고, 답변 정확도 평가를 더 많은 문항으로 확장할 계획이다.")

# ---------------------------------------------------------------- 선행연구 (2.1.3 Docling, 2.1.5 Shi, 2.3.1 SMART, 2.4.2 Yu 지적, 3.1 정리)
sub1("section2", "표가 여러 청크로 분할되면 첫 청크를 제외한 청크에는 열 머리글이 포함되지 않는다.",
     "표가 여러 청크로 분할되면 첫 청크를 제외한 청크에는 열 머리글이 포함되지 않는다. 이를 완화하기 위해 표가 여러 청크로 분할될 때 각 청크의 앞에 열 머리글 줄을 반복하는 문서 처리 도구도 있다[20]. 그러나 이 방식도 청크가 검색 단위이므로 청크 안의 답에 필요하지 않은 셀이 함께 전달되고, 여러 층의 머리글이 각 셀 값과 같은 줄에 결합되지는 않는다.")
sub1("section2", "잘 활용하지 못한다는 것을 보였다[7].",
     "잘 활용하지 못한다는 것을 보였다[7]. Shi 등은 산술 문제에 질문과 무관한 문장을 하나만 추가하여도 LLM의 정확도가 크게 저하된다는 것을 보였다[19].")
sub1("section2", "본 논문은 MultiHiertt를 주 평가 데이터로 사용하고, 캡션의 효과는 HiTab으로 확인한다.",
     "본 논문은 MultiHiertt를 주 평가 데이터로 사용하고, 캡션의 효과는 HiTab으로 확인한다. 계층형 표의 여러 층 머리글을 LLM이 해석하기 쉽게 변환하는 연구도 있다. Zhang 등은 여러 층의 머리글을 한 층의 서술자로 평탄화한 후 LLM이 추론하도록 하는 SMART를 제안하였다[21]. 이 방법은 표 전체를 LLM에 입력하므로, 문서 안의 여러 표에서 답에 필요한 셀을 검색하는 단계는 포함하지 않는다.")
sub1("section2", "하위 질의의 답을 취합하여 최종 답을 도출할 때까지 이 과정을 반복한다.",
     "하위 질의의 답을 취합하여 최종 답을 도출할 때까지 이 과정을 반복한다. Yu 등은 표를 평탄화하여 청크로 분할하면 표의 구조가 손상되어 정보가 손실된다고 지적하였으나[14], 이를 청크 대신 SQL 실행으로 보완한다.")
sub1("section2", "따라서 계층형 표가 포함된 문서에서는 셀 값과 그 값을 해석하는 데 필요한 머리글을 하나의 검색 단위에 함께 포함하는 방법이 필요하다.",
     "정리하면, 전달하는 셀 수를 줄이는 기존 방법(TableRAG(Chen)의 셀 검색, ITR의 행·열 검색)은 셀에서 행 머리글이나 여러 층의 머리글을 제거하고, 머리글을 유지하는 기존 방법(청크 분할, 머리글 줄을 반복하는 분할[20], 머리글 평탄화[21])은 표 전체나 청크 단위로 전달한다. 셀 단위의 검색과 셀마다 계층 머리글 경로의 결합을 함께 만족하는 검색 단위는 확인되지 않았다. 따라서 계층형 표가 포함된 문서에서는 셀 값과 그 값을 해석하는 데 필요한 머리글을 하나의 검색 단위에 함께 포함하는 방법이 필요하다.")

# ---------------------------------------------------------------- 3.3.4 α 근거
sub1("section2", "재무 보고서의 질문에는 항목 이름과 연도가 그대로 나타나는 경우가 많으므로 두 점수를 함께 사용한다.",
     "재무 보고서의 질문에는 항목 이름과 연도가 그대로 나타나는 경우가 많으므로 두 점수를 함께 사용한다. 코사인 유사도의 가중치 0.7은 Bruch 등[22]이 같은 형태의 결합에서 권고한 0.6~0.8의 범위에 속하며, Wang 등[23]도 이 가중치에서 밀집 검색 단독보다 높은 검색 성능을 보고하였다. 이 값은 MultiHiertt 검증 집합 332문항에서 가중치를 0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0으로 바꾸어 상위 20개 셀의 재현율을 비교하여 확인하였다. 재현율은 각각 90.5%, 92.9%, 93.4%, 93.4%, 93.2%, 93.0%, 92.6%, 91.0%로 0.5에서 0.9 사이에서는 차이가 0.8%p 이내였고, BM25 점수만 사용하거나 코사인 유사도만 사용하는 경우가 가장 낮았다.")

# ---------------------------------------------------------------- 참고문헌 [19]~[23]
REFS = [
    '[19] Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E. H., Schärli, N. and Zhou, D., "Large Language Models Can Be Easily Distracted by Irrelevant Context," Proceedings of the 40th International Conference on Machine Learning, PMLR Vol. 202, pp. 31210-31227, 2023.',
    '[20] Auer, C., Lysak, M., Nassar, A., Dolfi, M., Livathinos, N., Vagenas, P., Berrospi Ramis, C., Omenetti, M., Lindlbauer, F., Dinkla, K., Mishra, L., Kim, Y., Gupta, S., Teixeira de Lima, R., Weber, V., Morin, L., Meijer, I., Kuropiatnyk, V. and Staar, P. W. J., "Docling Technical Report," arXiv:2408.09869, 2024 (청크마다 표 머리글 반복: Docling 문서 "Chunking", https://docling-project.github.io/docling/concepts/chunking/, 2026년 10월 1일 확인).',
    '[21] Zhang, Y., Chen, J., Xu, K., Wei, W. and Zou, S., "SMART: Semantic Header Flattening and Pseudo-Code-Style Reasoning for LLM-based Complex Table Question Answering," Findings of the Association for Computational Linguistics: ACL 2026, pp. 29483-29499, 2026.',
    '[22] Bruch, S., Gai, S. and Ingber, A., "An Analysis of Fusion Functions for Hybrid Retrieval," ACM Transactions on Information Systems, Vol. 42, No. 1, Article 20, 2023.',
    '[23] Wang, S., Zhuang, S. and Zuccon, G., "BERT-based Dense Retrievers Require Interpolation with BM25 for Effective Passage Retrieval," Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval, pp. 317-324, 2021.',
]
bs, be, sp = top_paras(S["section2"])
ref18 = [(a, b) for a, b in sp if text_of(S["section2"][a:b]).startswith("[18] Zhao")]
assert len(ref18) == 1
a, b = ref18[0]
p18 = S["section2"][a:b]
t18 = text_of(p18)
assert p18.count(t18) == 1
new_refs = "".join(p18.replace(t18, r) for r in REFS)
S["section2"] = S["section2"][:b] + new_refs + S["section2"][b:]

# ---------------------------------------------------------------- [그림 3-3] 머리글 경로 구성의 예 (기존 3-3 → 3-4)
sub1("section2", "[그림 3-3]은 표의 일부에서 셀 문장이 생성되는 예이다.", "[그림 3-4]는 표의 일부에서 셀 문장이 생성되는 예이다.")
sub1("section2", "<hp:t>[그림 3-3] 셀 문장 생성의 예</hp:t>", "<hp:t>[그림 3-4] 셀 문장 생성의 예</hp:t>")
sub1("section2", "이 규칙은 경로가 같은 행과 열에만 적용되며, 나머지 셀의 경로는 변경하지 않는다.",
     "이 규칙은 경로가 같은 행과 열에만 적용되며, 나머지 셀의 경로는 변경하지 않는다. [그림 3-3]은 표 하나에서 머리글 행과 머리글 열을 판정하고, 셀 하나의 행 경로와 열 경로를 구성한 후, 문서 안의 다른 표와 경로가 같은 셀의 경로를 확장하는 과정의 예이다.")
# 그림 문단·캡션 문단은 기존 [그림 3-4](image5) 것을 복제
bs, be, sp = top_paras(S["section2"])
pic = [(i, a, b) for i, (a, b) in enumerate(sp) if 'binaryItemIDRef="image5"' in S["section2"][a:b]]
assert len(pic) == 1
i5, a5, b5 = pic[0]
cap_a, cap_b = sp[i5 + 1]
assert "[그림 3-4] 셀 문장 생성의 예" in S["section2"][cap_a:cap_b]
pic_xml, cap_xml = S["section2"][a5:b5], S["section2"][cap_a:cap_b]
# 새 그림 크기: 너비는 같고(36848), 높이는 픽셀 비율로
import struct
w_px, h_px = struct.unpack(">II", FIG.read_bytes()[16:24])
W = 36848; H = round(W * h_px / w_px)
old_h = int(re.search(r'<hp:orgSz width="36848" height="(\d+)"/>', pic_xml).group(1))
old_cur = re.search(r'<hp:curSz width="(\d+)" height="(\d+)"/>', pic_xml)
new_pic = pic_xml.replace('binaryItemIDRef="image5"', 'binaryItemIDRef="image7"')
new_pic = new_pic.replace(f'<hp:orgSz width="36848" height="{old_h}"/>', f'<hp:orgSz width="36848" height="{H}"/>')
new_pic = new_pic.replace(f'<hp:curSz width="{old_cur.group(1)}" height="{old_cur.group(2)}"/>', f'<hp:curSz width="36848" height="{H}"/>')
new_pic = re.sub(r'<hc:scaMatrix e1="[^"]*" e2="0" e3="0" e4="0" e5="[^"]*" e6="0"/>',
                 '<hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>', new_pic)
new_pic = re.sub(r'centerX="\d+" centerY="\d+"', f'centerX="{W // 2}" centerY="{H // 2}"', new_pic)
new_pic = re.sub(r'<hc:pt2 x="36848" y="\d+"/><hc:pt3 x="0" y="\d+"/>', f'<hc:pt2 x="36848" y="{H}"/><hc:pt3 x="0" y="{H}"/>', new_pic)
new_pic = re.sub(r'<hp:imgClip left="0" right="\d+" top="0" bottom="\d+"/>', f'<hp:imgClip left="0" right="{w_px * 75}" top="0" bottom="{h_px * 75}"/>', new_pic)
new_pic = re.sub(r'<hp:imgDim dimwidth="\d+" dimheight="\d+"/>', f'<hp:imgDim dimwidth="{w_px * 75}" dimheight="{h_px * 75}"/>', new_pic)
new_pic = re.sub(r'<hp:sz width="\d+" widthRelTo="ABSOLUTE" height="\d+"', f'<hp:sz width="36848" widthRelTo="ABSOLUTE" height="{H}"', new_pic)
new_pic = re.sub(r'vertsize="\d+" textheight="\d+" baseline="\d+"', f'vertsize="{H}" textheight="{H}" baseline="{round(H * 0.85)}"', new_pic)
new_pic = new_pic.replace('id="1500000015"', 'id="1500000099"').replace('instid="17"', 'instid="99"').replace('zOrder="20"', 'zOrder="99"')
assert new_pic != pic_xml and 'image7' in new_pic
new_cap = cap_xml.replace("[그림 3-4] 셀 문장 생성의 예", "[그림 3-3] 머리글 경로 구성의 예")
# 삽입 위치: 3.3.1 마지막 문단([그림 3-3]은 표 하나에서 …) 바로 뒤
tail = [(i, a, b) for i, (a, b) in enumerate(sp) if "[그림 3-3]은 표 하나에서" in text_of(S["section2"][a:b])]
assert len(tail) == 1
_, ta, tb = tail[0]
S["section2"] = S["section2"][:tb] + new_pic + new_cap + S["section2"][tb:]

# 그림 목차 (section0)
sub1("section0", "[그림 3-3] 셀 문장 생성의 예 <hp:tab width=\"20328\" leader=\"3\" type=\"2\"/>21</hp:t></hp:run>",
     "[그림 3-3] 머리글 경로 구성의 예 <hp:tab width=\"20328\" leader=\"3\" type=\"2\"/>20</hp:t></hp:run>")
bs0, be0, sp0 = top_paras(S["section0"])
e = [(a, b) for a, b in sp0 if "[그림 3-3] 머리글 경로 구성의 예" in S["section0"][a:b]]
assert len(e) == 1
a, b = e[0]
S["section0"] = S["section0"][:b] + S["section0"][a:b].replace("[그림 3-3] 머리글 경로 구성의 예", "[그림 3-4] 셀 문장 생성의 예").replace(
    'type="2"/>20</hp:t>', 'type="2"/>21</hp:t>') + S["section0"][b:]

# ---------------------------------------------------------------- 목차 쪽 번호: 참고문헌이 3쪽→4쪽, 부록 이후 +1
# (hwp MCP 렌더러의 v6·v7 쪽 비교 기준 — 한글에서 확인 필요)
TOC = {"부록 A. 통계 검정 방법과 결과": ("42", "43"), "부록 B. LLM 지시문": ("44", "45"),
       "부록 C. 표에 캡션이 있는 경우의 효과(HiTab)": ("45", "46"), "부록 D. 공식 근거 셀만 입력한 결과": ("47", "48"),
       "부록 E. 머리글 경로 제거 실험": ("48", "49"),
       "[표 A-1] 보조 비교의 McNemar 검정(200문항)": ("42", "43"),
       "[표 A-2] 답변 정확도의 McNemar 검정(200문항, 제안 방법 대비)": ("42", "43"),
       "[표 C-1] 셀 문장의 캡션 유무에 따른 재현율@20(HiTab 991문항)": ("45", "46"),
       "[표 C-2] 캡션 유무의 McNemar 검정(HiTab 991문항)": ("45", "46"),
       "[표 D-1] 공식 근거 셀만 입력한 답변 정확도(200문항)": ("47", "48"),
       "[표 E-1] 머리글 경로 제거 실험(200문항)": ("48", "49")}
sub1("section0", '<hp:t><hp:tab width="30808" leader="3" type="2"/>41</hp:t>',
     '<hp:t><hp:tab width="30808" leader="3" type="2"/>42</hp:t>')        # "부    록" 줄
for k, (o, n) in TOC.items():
    pat = re.compile(re.escape(k) + r"( ?<hp:tab [^>]*/>)" + o + "</hp:t>")
    assert len(pat.findall(S["section0"])) == 1, k
    S["section0"] = pat.sub(lambda m: k + m.group(1) + n + "</hp:t>", S["section0"])

# ---------------------------------------------------------------- 바뀐 문단의 줄 배치 캐시 삭제
# linesegarray 는 한글의 줄 배치 캐시다. 글을 바꾼 문단에 옛 캐시가 남으면(특히 글이 짧아져 textpos 가
# 글자 수를 넘으면) 한글이 "손상된 파일"로 연다. v6 편집 도구처럼 바뀐 문단의 캐시를 지워 한글이 다시 계산하게 한다.
ORIG = {k: v.decode("utf-8") for k, v in ((k, z.read(f"Contents/{k}.xml")) for k in S)}
for sec in S:
    old = {ORIG[sec][a:b] for a, b in top_paras(ORIG[sec])[2]}
    _, _, sp = top_paras(S[sec])
    out, pos, n = [], 0, 0
    for a, b in sp:
        p = S[sec][a:b]
        if p not in old:
            p = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", p, flags=re.S); n += 1
        out.append(S[sec][pos:a]); out.append(p); pos = b
    out.append(S[sec][pos:])
    S[sec] = "".join(out)
    print(f"{sec}: 캐시 지운 문단 {n}")

# ---------------------------------------------------------------- 저장
files["Contents/section0.xml"] = S["section0"].encode("utf-8")
files["Contents/section1.xml"] = S["section1"].encode("utf-8")
files["Contents/section2.xml"] = S["section2"].encode("utf-8")
files["BinData/image7.png"] = FIG.read_bytes()
hpf = files["Contents/content.hpf"].decode("utf-8")
assert 'id="image7"' not in hpf
hpf = hpf.replace('<opf:item id="section2"', '<opf:item id="image7" href="BinData/image7.png" media-type="image/png" isEmbeded="1"/><opf:item id="section2"', 1)
files["Contents/content.hpf"] = hpf.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print("wrote", dst)

