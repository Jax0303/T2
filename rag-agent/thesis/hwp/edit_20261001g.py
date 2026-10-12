"""2026-10-01 원고 수정 (v13 → v14). 전체 검토(2026-10-01) 1·3·4·5번 반영 + 같은 셀 수 재현율 표 신설.

  1 문제 정의·연구 목표: 1.1 에 둘째 문제(불필요한 셀) 문단, 1.3 에 "기존 연구는 둘 중 하나씩" 공백 서술,
    "재현율을 유지하면서" → "크게 낮추지 않으면서", 5장 첫 문단·국문/영문 초록에 둘째 문제.
  2 [표 4-7] 같은 수의 셀을 전달할 때의 재현율(200문항) 신설, 기존 [표 4-7] 답변 정확도 → [표 4-8].
    수치: results/equal_cells_20261001/equal_cells.py(기존 순위 기록만 사용, 새 실행 없음).
  3 부록 C 모순 문장, 2.2 RAPTOR 분할 설명(vendor/raptor/raptor/utils.py split_text 와 일치),
    문항당 정답 셀 수(2,885 2.65 / 200 2.37), 1위 청크 셀 수(46.3·43.3), 3.2 검색 범위 문장 위치,
    1.3 "원래의 파이프라인대로", 1.1 "분리된다" → "분리될 수 있다", 4.3.3 보조 비교 46.0%, 142.6 의 집합 표기.
    표 4-5 이중 반올림 두 칸: 제안 R@5 73.5 → 73.4(정확히 73.448%), LangChain R@1 76.7 → 76.8(정확히 76.75%).
    표 E-1 R@5 도 73.4.
  4 [그림 3-1] 청크 1 에 "December 31"·"Consumer" 줄, 청크 2 에 "Commercial" 줄(실제 청크 글과 같게), 설명 문장 수정.
  5 용어: LangChain 청크, 검색 단위, 정답 셀, 재현율@k, 제안하는 방법(본문), 행·열 머리글 경로(이하 행·열 경로),
    상위 행 머리글, 본 논문, "이상의 결과로부터".

  .venv/bin/python thesis/hwp/edit_20261001g.py <입력.hwpx> <출력.hwpx>
"""
import re, struct, sys, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

FIG31 = Path(__file__).resolve().parent.parent / "fig/fig3_1_chunk_split.png"
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


def para_of(sec, starts):
    hit = [(a, b) for a, b in top_paras(S[sec])[2] if text_of(S[sec][a:b]).startswith(escape(starts))]
    assert len(hit) == 1, starts
    return hit[0]


def body_para(template, text):
    """글자 모양 하나짜리 본문 문단 틀로 새 문단. 줄 배치 캐시는 넣지 않는다."""
    head = template[:template.index("<hp:t>")]
    return f"{head}<hp:t>{escape(text)}</hp:t></hp:run></hp:p>"


# ================================================================ 1 문제 정의·연구 목표
sub("section2", "그러나 계층형 표를 청크로 분할하면 셀 값과 머리글이 분리된다.",
    "그러나 계층형 표를 청크로 분할하면 셀 값과 머리글이 분리될 수 있다.")
a, b = para_of("section2", "그러나 계층형 표를 청크로 분할하면")
P2 = ("청크 단위의 검색에는 전달하는 셀이 많다는 문제도 있다. 답에 필요한 셀 몇 개를 전달하기 위해 그 셀이 포함된 청크의 "
      "다른 셀이 모두 함께 전달되기 때문이다. "
      "4장의 답변 평가 200문항에서 답에 필요한 정답 셀은 문항당 평균 2.37개였으나, LangChain 튜토리얼과 같이 상위 4개 청크를 "
      "전달하면 문항당 평균 126.8개의 셀이 전달되어 그 가운데 정답 셀의 비율은 2.0%였다(4.3.3절). 질문과 무관한 내용이 입력에 "
      "함께 포함되면 LLM의 정확도가 저하될 수 있으므로[6][19], 답에 필요한 셀을 포함하면서도 전달하는 셀 수를 줄일 필요가 있다.")
S["section2"] = S["section2"][:b] + body_para(S["section2"][a:b], P2) + S["section2"][b:]
sub("section2", "표를 청크 단위로 분할하는 대신, 각 셀이 자신의 머리글 정보를 가진 채로 검색되도록 하는 방법이 요구된다.",
    "표를 청크 단위로 분할하는 대신, 답에 필요한 셀을 적은 수로 검색하면서 각 셀이 자신의 머리글 정보를 가진 채로 "
    "전달되도록 하는 방법이 요구된다.")
sub("section2", "기존의 RAG는 표를 청크 단위로 분할하여 색인하기 때문에, 청크 경계에서 셀 값과 머리글이 분리되면 검색기가 "
    "질문이 가리키는 셀을 검색하기 어렵다. 따라서 제안하는 방법은",
    "1.1절에서 설명한 것처럼 청크 단위의 검색에는 셀 값과 머리글이 분리될 수 있다는 문제와 답에 필요하지 않은 셀이 많이 "
    "전달된다는 문제가 있다. 기존 연구는 이 두 문제 가운데 하나씩을 다루었다. 전달하는 셀 수를 줄이는 방법인 Chen 등[13]이 "
    "제안한 TableRAG의 셀 검색과 Lin 등[12]이 제안한 표 안 검색기의 행·열 검색은 검색 단위에서 행 머리글이나 여러 층의 "
    "머리글을 제거한다. 머리글을 유지하는 방법인 청크마다 표의 머리글 줄을 반복하는 분할[20]과 여러 층의 머리글을 "
    "한 층으로 평탄화하는 방법[21]은 청크나 표 전체를 단위로 전달하므로 전달하는 셀 수가 줄지 않는다. 셀 하나를 검색 단위로 "
    "삼으면서 셀마다 계층 머리글 경로를 결합하는 방법은 확인되지 않았다(3.1절). 따라서 제안하는 방법은")
sub("section2", "이를 통해 정답 셀의 재현율을 유지하면서", "이를 통해 정답 셀의 재현율을 크게 낮추지 않으면서")
sub("section2", "LangChain 공식 튜토리얼의 청크 분할 방식[2], RAPTOR[9], Chen 등[13]과 Yu 등[14]의 TableRAG(이하 TableRAG(Chen), "
    "TableRAG(Yu))를 원래의 파이프라인대로 구현하고, 모든 방법에 같은 LLM과 같은 임베딩 모델을 사용하여 검색 성능과 "
    "답변 정확도를 비교한다.",
    "LangChain 공식 튜토리얼의 청크 분할 방식[2](이하 LangChain 청크), RAPTOR[9], Chen 등[13]과 Yu 등[14]의 TableRAG"
    "(이하 TableRAG(Chen), TableRAG(Yu))를 각 방법의 공식 코드나 튜토리얼의 파이프라인에 따라 구현하되, 모든 방법에 같은 "
    "LLM과 같은 임베딩 모델을 사용하고 검색 범위를 질문이 속한 문서로 같게 하여 검색 성능과 답변 정확도를 비교한다.")
sub("section2", "11.8%에서 정답 셀이 열 머리글 줄이 없는 청크에만 포함되어 있었다. 이처럼 셀 값과 머리글이 분리되면 검색기는 "
    "질문이 가리키는 셀을 검색하기 어려우므로, 본 논문은 셀마다 머리글을 함께 포함하는 검색 단위에 초점을 둔다.",
    "11.8%에서 정답 셀이 열 머리글 줄이 없는 청크에만 포함되어 있었다. 또한 청크 단위로 전달하면 답에 필요한 셀 몇 개와 "
    "함께 청크 안의 다른 셀이 모두 전달되며, LangChain 청크는 문항당 평균 126.8개의 셀을 전달하였다. 기존의 표 대상 방법은 "
    "셀 수를 줄이면 머리글이 제거되고, 머리글을 유지하면 청크나 표 전체를 전달하였다. 따라서 본 논문은 셀 하나를 검색 단위로 "
    "삼고 셀마다 머리글 경로를 함께 포함하는 방법에 초점을 둔다.")
# 초록
sub("section1", "셀 값과 머리글이 분리되면 검색기는 질문이 가리키는 셀을 검색하기 어렵다. 따라서 본 학위논문에서는",
    "셀 값과 머리글이 분리되면 검색기는 질문이 가리키는 셀을 검색하기 어렵다. 또한 청크 단위로 전달하면 답에 필요한 셀 "
    "몇 개와 함께 청크 안의 다른 셀이 모두 LLM에 전달된다. 따라서 본 논문에서는")
sub("section1", "LangChain 청크 분할, RAPTOR, TableRAG(Chen)과의 차이는 통계적으로 유의하였다.",
    "LangChain 청크 분할, RAPTOR, TableRAG(Chen)과의 차이는 통계적으로 유의하였다. 제안하는 방법은 문항당 20개의 셀을 "
    "LLM에 전달하였으며, LangChain 청크 분할은 문항당 평균 126.8개의 셀을 전달하였다.")
sub("section1", "Once a value is separated from its headers, the retriever has difficulty finding the cells a question refers to.",
    "Once a value is separated from its headers, the retriever has difficulty finding the cells a question refers to. "
    "Chunk-level retrieval also passes all the other cells in each retrieved chunk to the LLM together with the few cells "
    "needed for the answer.")
sub("section1", "and TableRAG (Chen) are statistically significant.",
    "and TableRAG (Chen) are statistically significant. The proposed method passes 20 cells per question to the LLM, "
    "whereas LangChain chunking passes 126.8 cells on average.")

# ================================================================ 3 수치·설명
sub("section2", "RAPTOR는 먼저 문서를 100토큰 길이의 잎 청크로 분할하고 임베딩한다. 이때 문장이 100토큰 한도를 넘으면 문장을 "
    "분할하지 않고 다음 청크로 이동한다.",
    "RAPTOR는 먼저 문서를 마침표, 물음표, 느낌표, 줄바꿈에서 문장으로 나눈 후, 문장들을 100토큰 이하의 잎 청크로 묶어 "
    "임베딩한다. 문장을 추가하면 100토큰을 넘는 경우 그 문장은 다음 청크로 넘기고, 한 문장이 100토큰을 넘으면 쉼표, 세미콜론, "
    "콜론에서 다시 나눈다.")
sub("section2", "RAPTOR의 원본 청크 분할기는 마침표, 쉼표 등에서 문장을 분할하므로",
    "RAPTOR의 원본 청크 분할기는 2.2절과 같이 마침표에서 문장을 나누고 100토큰을 넘는 문장은 쉼표에서 다시 나누므로")
sub("section2", "캡션의 효과가 표가 많은 색인에서 더 큰 것은, 캡션이 셀이 어느 표에 속하는지를 나타내어 서로 다른 표의 유사한 셀을 "
    "구별하게 하기 때문이다. 한 표 안에서는 모든 셀의 캡션이 같으므로 캡션으로 셀을 구별할 수 없다.",
    "캡션의 효과는 표 538개를 하나의 색인에 포함한 검색에서 더 컸으며, 이는 캡션이 셀이 어느 표에 속하는지를 나타내어 "
    "서로 다른 표의 유사한 셀을 구별하게 하기 때문으로 해석된다. 한 표 안에서는 모든 셀의 캡션이 같아 캡션이 표를 구별하는 "
    "역할은 하지 않으나, 질문의 표 안 검색에서도 재현율@20이 유의하게 상승하였으며 이 상승의 원인은 본 논문에서 분석하지 "
    "않았다.")
sub("section2", "평가 집합의 문서에는 평균 4.05개의 표와 145.7개의 데이터 셀이 있다.",
    "평가 집합의 문서에는 평균 4.05개의 표와 145.7개의 데이터 셀이 있고, 문항당 정답 셀은 평균 2.65개이다.")
sub("section2", "[표 4-3]의 조회·셀 1개 그룹은 검색 평가 212문항, 추출 표본 211문항이다.",
    "[표 4-3]의 조회·셀 1개 그룹은 검색 평가 212문항, 추출 표본 211문항이다. 답변 평가 200문항의 문항당 정답 셀은 "
    "평균 2.37개이다.")
sub("section2", "LangChain 청크와 TableRAG(Yu)는 단위 하나에 평균 37.5개와 30.6개의 셀을 포함한다.",
    "LangChain 청크와 TableRAG(Yu)는 순위 목록의 단위 하나에 평균 37.5개와 30.6개의 셀을 포함하며, 1위 단위는 평균 "
    "46.3개와 43.3개의 셀을 포함한다.")
sub("section2", "벡터 데이터베이스에 저장하고, 같은 셀 문장으로 BM25 색인을 구축한다. 검색 범위는 질문의 문서이다.",
    "벡터 데이터베이스에 저장하고, 같은 셀 문장으로 BM25 색인을 구축한다.")
sub("section2", "이를 BM25 점수와 가중합하여 순위를 매긴 후 상위 20개 셀 문장을 선택한다.",
    "이를 BM25 점수와 가중합하여 질문이 속한 문서의 셀 문장 순위를 매긴 후 상위 20개 셀 문장을 선택한다.")
sub("section2", "(문서 셀 142.6개의 80%)", "(200문항 문서의 평균 셀 142.6개의 80%)")
sub("section2", "다만 LangChain 청크는 튜토리얼의 지시문을 그대로 사용하므로, 두 방법의 답변 정확도 차이에는 입력의 차이와 지시문의 "
    "차이가 함께 포함되어 있다. 지시문을 동일하게 설정한 보조 비교의 결과는 부록 A에 수록한다.",
    "다만 LangChain 청크는 튜토리얼의 지시문을 그대로 사용하므로, 두 방법의 답변 정확도 차이에는 입력의 차이와 지시문의 "
    "차이가 함께 포함되어 있다. LLM 입력만 제안하는 방법의 셀 문장 20개로 바꾸고 튜토리얼의 지시문을 그대로 사용한 보조 "
    "비교에서는 답변 정확도가 46.0%로 LangChain 청크의 39.5%보다 6.5%p 높았다(부록 A).")
# 표 4-5·E-1 이중 반올림
for old, new, n in (("<hp:t>73.5</hp:t>", "<hp:t>73.4</hp:t>", 2), ("<hp:t>76.7</hp:t>", "<hp:t>76.8</hp:t>", 1)):
    assert S["section2"].count(old) == n, old
    S["section2"] = S["section2"].replace(old, new)
sub("section2", "상위 1개 단위의 재현율이 76.7%와 82.8%로 제안하는 방법의 32.3%보다 높았고, 상위 5개 단위에서는 100%에 이르렀다.",
    "재현율@1이 76.8%와 82.8%로 제안하는 방법의 32.3%보다 높았고, 재현율@5는 100%에 이르렀다.")

# ================================================================ 4 그림 3-1 설명
sub("section2", "첫 번째 청크에는 연도를 나타내는 열 머리글 줄이 포함되어 있지만, 두 번째 청크에는 \"Total consumer\" 행의 값 "
    "18,768, 20,854, 20,839만 남는다.",
    "첫 번째 청크에는 \"December 31\"과 연도를 나타내는 열 머리글 줄, 절 제목 행 \"Consumer\"가 포함되어 있지만, 두 번째 "
    "청크의 \"Total consumer\" 줄에는 행 머리글과 값 18,768, 20,854, 20,839만 있고 열 머리글 줄은 없다.")

# ================================================================ 5 용어
sub("section2", "LangChain 튜토리얼 청크[2]", "LangChain 청크[2]")
sub("section2", "LangChain 튜토리얼 청크, RAPTOR", "LangChain 청크, RAPTOR")
sub("section2", "문서를 색인 단위로 분할하는 방법", "문서를 검색 단위로 분할하는 방법")
sub("section2", "문서를 색인 단위인 청크로", "문서를 검색 단위인 청크로")
sub("section2", "검색 단계의 색인 단위를 변경하여", "검색 단위를 변경하여")
sub("section2", "방법들이 색인하는 단위와 단위 하나에", "방법들의 검색 단위와 단위 하나에")
sub("section2", "[표 3-1] 기존 방법의 색인 단위와", "[표 3-1] 기존 방법의 검색 단위와")
sub("section0", "[표 3-1] 기존 방법의 색인 단위와", "[표 3-1] 기존 방법의 검색 단위와")
assert S["section2"].count("<hp:t>색인 단위</hp:t>") == 1
S["section2"] = S["section2"].replace("<hp:t>색인 단위</hp:t>", "<hp:t>검색 단위</hp:t>")
sub("section2", "위층 행 머리글", "상위 행 머리글", count=2)
sub("section2", "셀의 행 경로는 그 셀이 속한 행의 머리글을 상위 머리글부터 차례로 나열한 것이고, 열 경로는",
    "셀의 행 머리글 경로(이하 행 경로)는 그 셀이 속한 행의 머리글을 상위 머리글부터 차례로 나열한 것이고, "
    "열 머리글 경로(이하 열 경로)는")
# 정답 셀
sub("section2", "수치 계산이 필요한 문항에는 계산 과정도 표시되어 있다.",
    "수치 계산이 필요한 문항에는 계산 과정도 표시되어 있다. 본 논문에서는 근거로 표시된 표의 셀을 정답 셀이라 한다.")
sub("section2", "질문만 있고 정답과 근거 셀이 포함되어 있지 않아", "질문만 있고 정답과 근거 표시가 포함되어 있지 않아")
sub("section2", "정답과 근거 셀이 제공되는 학습 집합", "정답과 근거 표시가 제공되는 학습 집합")
sub("section2", "근거 셀이 빈 칸 또는 머리글 칸인 23문항", "정답 셀이 빈 칸 또는 머리글 칸인 23문항")
sub("section2", "근거 셀 수에 따라", "정답 셀 수에 따라")
for old, new, n in (("근거 셀 1개</hp:t>", "정답 셀 1개</hp:t>", 2), ("근거 셀 2개 이상</hp:t>", "정답 셀 2개 이상</hp:t>", 2)):
    assert S["section2"].count(old) == n, old
    S["section2"] = S["section2"].replace(old, new)
sub("section2", "정답 근거 셀 집합을", "정답 셀 집합을")
sub("section2", "전달된 셀 중 정답 근거 셀이 차지하는 비율인 셀 정밀도는", "전달된 셀 중 정답 셀이 차지하는 비율인 셀 정밀도@k(표에서는 셀P@k)는")
sub("section2", "재현율(Recall@k)과 정밀도(Precision@k)로 측정한다.",
    "재현율@k(Recall@k, 표에서는 R@k)와 정밀도@k(Precision@k, 표에서는 P@k)로 측정한다.")
sub("section2", "검색된 상위 20개 셀에 공식 근거 셀이 모두 포함되었는지이고, 둘째는 공식 근거 셀만 LLM에 입력하였을 때",
    "검색된 상위 20개 셀에 정답 셀이 모두 포함되었는지이고, 둘째는 정답 셀만 LLM에 입력하였을 때")
sub("section2", "⦁ 근거 셀이 상위 20개에 포함되지 않았고, 근거 셀만 입력하면 정답: 7개",
    "⦁ 정답 셀 일부가 상위 20개에 포함되지 않았고, 정답 셀만 입력하면 정답: 7개")
sub("section2", "⦁ 근거 셀이 상위 20개에 포함되지 않았고, 근거 셀만 입력하여도 오답: 10개",
    "⦁ 정답 셀 일부가 상위 20개에 포함되지 않았고, 정답 셀만 입력하여도 오답: 10개")
sub("section2", "⦁ 근거 셀이 모두 포함되었고 근거 셀만 입력하면 정답이지만", "⦁ 정답 셀이 모두 포함되었고 정답 셀만 입력하면 정답이지만")
sub("section2", "⦁ 근거 셀이 모두 포함되었으나 근거 셀만 입력하여도 오답", "⦁ 정답 셀이 모두 포함되었으나 정답 셀만 입력하여도 오답")
sub("section2", "근거 셀을 누락한 것만으로", "정답 셀을 누락한 것만으로")
sub("section2", "근거 셀 누락과 LLM의 오답이", "정답 셀 누락과 LLM의 오답이")
sub("section2", "부록 D. 공식 근거 셀만 입력한 결과", "부록 D. 정답 셀만 입력한 결과")
sub("section0", "부록 D. 공식 근거 셀만 입력한 결과", "부록 D. 정답 셀만 입력한 결과")
sub("section2", "MultiHiertt가 표시한 공식 근거 셀의 셀 문장만", "MultiHiertt가 근거로 표시한 정답 셀의 셀 문장만")
sub("section2", "[표 D-1] 공식 근거 셀만 입력한", "[표 D-1] 정답 셀만 입력한")
sub("section0", "[표 D-1] 공식 근거 셀만 입력한", "[표 D-1] 정답 셀만 입력한")
sub("section2", "공식 근거 셀만 입력한 정확도(%)", "정답 셀만 입력한 정확도(%)")
sub("section2", "공식 근거 셀만 입력하여도 정확도는", "정답 셀만 입력하여도 정확도는")
sub("section2", "공식 근거 셀에 답을 도출하는 데 필요한", "데이터셋이 표시한 정답 셀에 답을 도출하는 데 필요한")
sub("section2", "근거 셀만으로는 계산할 수 없다", "정답 셀만으로는 계산할 수 없다")
sub("section2", "따라서 공식 근거 셀만 입력한 조건의", "따라서 정답 셀만 입력한 조건의")
# 재현율@k
sub("section2", "상위 20개 셀의 재현율을 비교하여 확인하였다", "재현율@20을 비교하여 확인하였다")
sub("section2", "제안하는 방법은 상위 20개 단위의 재현율이 93.9%로", "제안하는 방법은 재현율@20이 93.9%로")
sub("section2", "200문항의 상위 20개 단위 재현율 81.7%는", "200문항의 재현율@20 81.7%는")
sub("section2", "제안하는 방법의 재현율은 상위 1개와 20개에서 각각 27.8%와 92.4%였고",
    "제안하는 방법의 재현율@1과 재현율@20은 각각 27.8%와 92.4%였고")
sub("section2", "LLM에 전달한 상위 4개 청크의 재현율이 99.4%로 제안하는 방법이 전달한 상위 20개 셀의 93.9%보다",
    "LLM에 전달한 상위 4개 청크의 재현율(재현율@4)이 99.4%로 제안하는 방법의 재현율@20 93.9%보다")
sub("section2", "상위 20개 셀의 재현율 93.9%, 셀 정밀도 11.0%를 기록하였고", "재현율@20 93.9%, 셀 정밀도@20 11.0%를 기록하였고")
sub("section2", "재현율 99.4%, 셀 정밀도 2.0%였다.",
    "재현율 99.4%, 셀 정밀도 2.0%였다. 전달하는 셀 수를 같게 맞추면, 평균 46.3개 셀을 전달하는 LangChain 청크의 1위 청크는 "
    "재현율이 76.8%였고 제안하는 방법은 98.6%였다.")
sub("section2", "머리글 경로를 제거하면 상위 20개 셀의 재현율이 93.9%에서 24.7%로", "머리글 경로를 제거하면 재현율@20이 93.9%에서 24.7%로")
sub("section2", "상위 5개와 20개 셀의 정밀도도 32.7%와 11.0%에서", "정밀도@5와 정밀도@20도 32.7%와 11.0%에서")
sub("section2", "상위 20개 셀의 재현율은 92.4%에서 23.3%로", "재현율@20은 92.4%에서 23.3%로")
# 본문의 '제안 방법' → '제안하는 방법'(표 안 글과 표 제목은 그대로)
sub("section2", "맞힌 문항 수는 제안 방법 93,", "맞힌 문항 수는 제안하는 방법 93,")
sub("section2", "표 전체 입력 80이며, 제안 방법 대비 유의 여부는", "표 전체 입력 80이며, 제안하는 방법 대비 유의 여부는")
sub("section2", "12문항이었으며, 제안 방법 대비 100 : 3", "12문항이었으며, 제안하는 방법 대비 100 : 3")
sub("section2", "제안 방법, 머리글 경로 제거 조건, 표 전체 입력 조건의 지시문은", "제안하는 방법, 머리글 경로 제거 조건, 표 전체 입력 조건의 지시문은")
sub("section2", "대입한 제안 방법의 문장이다", "대입한 제안하는 방법의 문장이다")
sub("section2", "위의 실험 결과 비교 분석을 통해, ", "이상의 결과로부터, ")

# ================================================================ 2 [표 4-7] 신설, 기존 4-7 → 4-8
sub("section2", "[표 4-7]은 모든 비교군을 평가한", "[표 4-8]은 모든 비교군을 평가한")
sub("section2", "[표 4-7] 답변 정확도(200문항)", "[표 4-8] 답변 정확도(200문항)")
sub("section2", "[표 4-5]와 [표 4-7]을 함께 보면", "[표 4-5]와 [표 4-8]을 함께 보면")
sub("section0", "[표 4-7] 답변 정확도(200문항)", "[표 4-8] 답변 정확도(200문항)")
assert "[표 4-7]" not in S["section2"]

sp = top_paras(S["section2"])[2]
txt = [text_of(S["section2"][x:y]) for x, y in sp]
i_cap = txt.index("[표 4-6] 셀 정밀도(200문항, k = 검색 단위 수)")
cap_xml, tbl_xml, note_xml = (S["section2"][sp[i][0]:sp[i][1]] for i in (i_cap, i_cap + 1, i_cap + 2))
assert "<hp:tbl" in tbl_xml and text_of(note_xml) == "(셀P@k 단위: %)"
i_after = next(i for i in range(i_cap, len(txt)) if txt[i].startswith("반면 LangChain 청크와 TableRAG(Yu)는"))
body_tmpl = S["section2"][sp[i_after][0]:sp[i_after][1]]

HEAD = ["비교 방법(상위 단위 수)", "전달 셀 수", "비교 방법 재현율", "제안 방법 재현율"]
ROWS = [("LangChain 청크 상위 1개", "46.3", "76.8", "98.6"), ("LangChain 청크 상위 2개", "83.9", "93.4", "99.4"),
        ("LangChain 청크 상위 4개", "126.8", "99.4", "99.4"), ("TableRAG(Yu) 상위 1개", "43.3", "82.8", "97.5"),
        ("TableRAG(Yu) 상위 2개", "79.6", "95.6", "99.4"), ("TableRAG(Yu) 상위 5개", "131.8", "100.0", "99.4"),
        ("RAPTOR 상위 1개", "11.2", "37.0", "69.0"), ("RAPTOR 상위 5개", "53.5", "68.4", "94.9"),
        ("RAPTOR 상위 20개", "112.6", "81.7", "97.9")]
WIDTHS = [12848, 8000, 8000, 8000]
assert sum(WIDTHS) == 36848
trs = re.findall(r"<hp:tr>.*?</hp:tr>", tbl_xml, re.S)
tc_head = re.findall(r"<hp:tc .*?</hp:tc>", trs[0], re.S)[0]
tc_body = re.findall(r"<hp:tc .*?</hp:tc>", trs[1], re.S)[0]
H_HEAD, H_BODY = 2632, 1507


def cell(tmpl, text, col, row, h):
    c = re.sub(r"<hp:t>.*?</hp:t>", lambda m: f"<hp:t>{escape(text)}</hp:t>", tmpl, count=1, flags=re.S)
    c = re.sub(r'<hp:cellAddr colAddr="\d+" rowAddr="\d+"/>', f'<hp:cellAddr colAddr="{col}" rowAddr="{row}"/>', c)
    c = re.sub(r'<hp:cellSz width="\d+" height="\d+"/>', f'<hp:cellSz width="{WIDTHS[col]}" height="{h}"/>', c)
    return LSA.sub("", c)


rows_xml = "<hp:tr>" + "".join(cell(tc_head, t, j, 0, H_HEAD) for j, t in enumerate(HEAD)) + "</hp:tr>"
for i, r in enumerate(ROWS, 1):
    rows_xml += "<hp:tr>" + "".join(cell(tc_body, t, j, i, H_BODY) for j, t in enumerate(r)) + "</hp:tr>"
new_tbl = tbl_xml[:tbl_xml.index("<hp:tr>")] + rows_xml + tbl_xml[tbl_xml.rindex("</hp:tr>") + len("</hp:tr>"):]
new_tbl = new_tbl.replace('id="1500000052" zOrder="61"', 'id="1500000301" zOrder="301"', 1)
new_tbl = new_tbl.replace('rowCnt="6" colCnt="5"', f'rowCnt="{len(ROWS) + 1}" colCnt="4"', 1)
new_tbl = re.sub(r'(<hp:tbl [^>]*>)<hp:sz width="36848" widthRelTo="ABSOLUTE" height="\d+"',
                 lambda m: m.group(1) + f'<hp:sz width="36848" widthRelTo="ABSOLUTE" height="{H_HEAD + H_BODY * len(ROWS)}"',
                 new_tbl, count=1)
assert 'id="1500000301"' in new_tbl and f'rowCnt="{len(ROWS) + 1}"' in new_tbl and "1500000052" not in new_tbl
new_tbl = LSA.sub("", new_tbl)
new_cap = LSA.sub("", cap_xml.replace("[표 4-6] 셀 정밀도(200문항, k = 검색 단위 수)", "[표 4-7] 같은 수의 셀을 전달할 때의 재현율(200문항)"))
new_note = LSA.sub("", note_xml.replace("(셀P@k 단위: %)", "(전달 셀 수는 문항당 평균, 재현율 단위: %)"))
EXPL = ("[표 4-5]의 k는 검색 단위의 개수이므로, 단위가 큰 방법일수록 같은 k에서 더 많은 셀을 전달한다. [표 4-7]은 전달하는 셀 수를 "
        "같게 하여 비교한 것으로, 문항마다 비교 방법의 상위 k개 단위가 전달한 셀 수와 같은 수의 셀을 제안하는 방법의 순위 목록에서 "
        "전달하였을 때의 재현율을 평균하였다. 제안하는 방법의 순위 목록은 문항마다 상위 100개 셀(문서의 셀이 100개보다 적으면 "
        "문서의 모든 셀)까지 저장되어 있으므로, 비교 방법이 이보다 많은 셀을 전달한 문항에서는 저장된 목록 전체로 계산하였다. LangChain 청크의 1위 청크는 평균 46.3개의 셀을 전달하여 재현율이 76.8%였으나, "
        "같은 수의 셀을 전달하면 제안하는 방법의 재현율은 98.6%였다. [표 4-5]의 LangChain 청크 P@1 81.0%는 이 1위 청크에 정답 셀이 "
        "하나 이상 포함된 문항의 비율이며, 같은 수의 셀을 전달한 제안하는 방법에서는 이 비율이 99.0%였다. 상위 1개와 2개 단위, 그리고 "
        "RAPTOR의 모든 k에서 제안하는 방법의 재현율이 높았고, 문서 셀 대부분을 전달하는 LangChain 청크 상위 4개와 TableRAG(Yu) "
        "상위 5개에서는 두 방법 모두 99% 이상이었다. 제안하는 방법은 셀 20개로 재현율@20 93.9%를 얻어, 평균 46.3개 셀을 전달한 "
        "LangChain 청크 1위 청크의 재현율보다 높았다. 또한 문항당 정답 셀이 평균 2.37개이므로 셀 1개를 전달하는 재현율@1은 최대 "
        "58.6%이고, 셀 20개를 전달할 때 셀 정밀도@20은 최대 11.9%이다. 제안하는 방법의 재현율@1 32.3%와 셀 정밀도@20 11.0%는 이 "
        "최대값을 기준으로 해석하여야 한다.")
new_expl = body_para(body_tmpl, EXPL)
a, b = sp[i_after]
S["section2"] = S["section2"][:b] + new_cap + new_tbl + new_note + new_expl + S["section2"][b:]

# 표 목차: [표 4-6] 줄 다음에 [표 4-7] 줄(쪽 번호는 한글에서 저장 후 toc_update 로 맞춘다)
sp0 = top_paras(S["section0"])[2]
e46 = [(x, y) for x, y in sp0 if "[표 4-6] 셀 정밀도(200문항, k = 검색 단위 수)" in S["section0"][x:y]]
assert len(e46) == 1
x, y = e46[0]
entry = S["section0"][x:y].replace("[표 4-6] 셀 정밀도(200문항, k = 검색 단위 수)", "[표 4-7] 같은 수의 셀을 전달할 때의 재현율(200문항)")
S["section0"] = S["section0"][:y] + LSA.sub("", entry) + S["section0"][y:]

# ================================================================ 그림 3-1 교체(사용자가 한글에서 자른 오른쪽 폭·배율 유지, 높이만 새 그림)
png = FIG31.read_bytes()
w_px, h_px = struct.unpack(">II", png[16:24])
m = re.search(r'<hp:pic [^>]*>(?:(?!</hp:pic>).)*?binaryItemIDRef="image3".*?</hp:pic>', S["section2"], re.S)
pic = m.group(0)
org_w, org_h = map(int, re.search(r'<hp:orgSz width="(\d+)" height="(\d+)"/>', pic).groups())
cur_w, cur_h = map(int, re.search(r'<hp:curSz width="(\d+)" height="(\d+)"/>', pic).groups())
old_hpx = int(re.search(r'dimheight="(\d+)"', pic).group(1)) // 75
assert int(re.search(r'dimwidth="(\d+)"', pic).group(1)) == w_px * 75, "그림 폭이 바뀌면 자른 폭을 다시 정해야 한다"
new_org_h = round(org_h * h_px / old_hpx)
new_cur_h = round(cur_h * h_px / old_hpx)
p = pic
p = p.replace(f'<hp:orgSz width="{org_w}" height="{org_h}"/>', f'<hp:orgSz width="{org_w}" height="{new_org_h}"/>')
p = p.replace(f'<hp:curSz width="{cur_w}" height="{cur_h}"/>', f'<hp:curSz width="{cur_w}" height="{new_cur_h}"/>')
p = re.sub(r'centerY="\d+"', f'centerY="{new_cur_h // 2}"', p)
p = re.sub(r'(<hc:scaMatrix e1="[^"]*" e2="0" e3="0" e4="0" e5=")[^"]*(" e6="0"/>)', lambda mm: mm.group(1) + f"{new_cur_h / new_org_h:.6f}" + mm.group(2), p)
p = re.sub(r'<hc:pt2 x="(\d+)" y="\d+"/><hc:pt3 x="0" y="\d+"/>', lambda mm: f'<hc:pt2 x="{mm.group(1)}" y="{new_org_h}"/><hc:pt3 x="0" y="{new_org_h}"/>', p)
p = re.sub(r'(<hp:imgClip left="0" right="\d+" top="0" bottom=")\d+("/>)', lambda mm: mm.group(1) + str(h_px * 75) + mm.group(2), p)
p = re.sub(r'dimheight="\d+"', f'dimheight="{h_px * 75}"', p)
p = re.sub(r'(<hp:sz width="\d+" widthRelTo="ABSOLUTE" height=")\d+(")', lambda mm: mm.group(1) + str(new_cur_h) + mm.group(2), p)
assert p.count(str(new_cur_h)) >= 2 and p != pic
S["section2"] = S["section2"].replace(pic, p, 1)
files["BinData/image3.png"] = png

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
