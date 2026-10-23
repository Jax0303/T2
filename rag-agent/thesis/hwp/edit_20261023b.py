"""2026-10-23 원고 수정 b (v20 → v21): 4장을 10월 주 설정(882문항, Qwen3.5-4B bf16)으로 교체 + 4.4 절 신설 + 5장·부록 A/B/D 갱신.
수치 출처: PREREG-2026-10-23-baselines-newregime.md(비교군), PREREG-2026-10-19-882-cap4096/18-D-vs-A/19-delivery-ceiling/22-reader-sweep.md(4.4).
  .venv/bin/python thesis/hwp/edit_20261023b.py thesis/thesis_working_20261023_v20.hwpx thesis/thesis_working_20261023_v21.hwpx
목차(section0)는 손대지 않는다 — 4.4 항목·쪽 번호는 한글에서 갱신.
"""
import re, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of
from hwpx_table import make_table
from scipy.stats import binomtest

LSA = re.compile(r"<hp:linesegarray>.*?</hp:linesegarray>", re.S)
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def para_like(template, text):
    head = re.match(r"(<hp:p [^>]*>)", template).group(1).replace('pageBreak="1"', 'pageBreak="0"')
    cp = re.search(r'charPrIDRef="(\d+)"', template).group(1)
    return f'{head}<hp:run charPrIDRef="{cp}"><hp:t>{esc(text)}</hp:t></hp:run></hp:p>'
def pfmt(p):
    if p >= 0.001: return f"{p:.3f}".lstrip("0") if p < 1 else "1.0"
    m, e = f"{p:.1e}".split("e"); return f"{m}×10{str(int(e)).translate(SUP)}"
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0] * m; cur = 0
    for r, i in enumerate(order): cur = max(cur, min(1.0, (m - r) * ps[i])); adj[i] = cur
    return adj

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src); files = {n: z.read(n) for n in z.namelist()}
S = files["Contents/section2.xml"].decode("utf-8")
bs, be, sp = top_paras(S); seg = [S[a:b] for a, b in sp]; txt = [text_of(q) for q in seg]
def idx(prefix, start=0):
    return next(k for k in range(start, len(txt)) if txt[k].startswith(prefix))
def table_after(caption_prefix):
    i = idx(caption_prefix); j = next(k for k in range(i + 1, i + 4) if "<hp:tbl" in seg[k]); return i, j
repl, ins_after, drop = {}, {}, set()           # 문단 번호 → 새 XML / 뒤에 끼울 XML 목록 / 지울 문단
BODY = seg[idx("본 논문에서는 계층형 표가 여러 개 포함된 문서에서")]
def body(text): return para_like(BODY, text)
def set_text(i, text):                            # 단일 run 문단의 글 교체
    repl[i] = para_like(seg[i], text)            # run 이 여러 개인 문단은 첫 run 의 글자 모양 하나로 합친다(글을 통째로 바꾸므로)

# ---------- 1.4 구성
i = idx("본 논문의 구성은 다음과 같다"); set_text(i, txt[i].replace("비교군 실험을 통해 제안하는 방법의 검색 성능과 답변 정확도를 평가한다.", "비교군 실험을 통해 제안하는 방법의 검색 성능과 답변 정확도를 평가하며, 전달 형식, 답변 정확도의 상한, 리더 LLM, 오답 원인을 분석한다."))
# ---------- 4.1.1
i = idx("모든 실험은 GPU 한 장을 장착한"); set_text(i, "모든 실험은 GPU 한 장을 장착한 컴퓨터 한 대의 로컬 환경에서 수행하였으며, 외부 API는 4.4.3절의 참고 비교(GPT-4o)에서만 사용하였다. [표 4-1]은 하드웨어 및 소프트웨어 실험 환경이고, [표 4-2]는 실험에 사용한 모델을 정리한 것이다.")
c, t = table_after("[표 4-1] 하드웨어")
repl[t] = make_table(seg[t], [["항목", "내용"], ["CPU", "Intel Core i9-13900K"], ["GPU", "NVIDIA GeForce RTX 4070 Ti (12GB)"], ["메모리", "15GB (WSL2 할당)"], ["운영체제", "Ubuntu 24.04.5 LTS (WSL2)"], ["언어", "Python 3.12.3"],
    ["라이브러리", "PyTorch 2.14.1 (CUDA 13.0), transformers 5.19.0, sentence-transformers 6.1.0, faiss 1.15.1; vLLM 0.31.0 (PyTorch 2.13.0, transformers 5.17.0)"]], col_widths=[8000, 28848])
c, t = table_after("[표 4-2] 실험에 사용한 모델")
repl[t] = make_table(seg[t], [["용도", "모델", "설정"], ["임베딩", "BAAI/bge-base-en-v1.5[4]", "768차원, 학습하지 않음"],
    ["LLM", "Qwen3.5-4B", "bf16(양자화 없음), 비생각 모드, 그리디 디코딩, vLLM 서버로 생성(제안 방법은 연속 묶음, 비교군은 한 문항씩 요청)"],
    ["재정렬", "BAAI/bge-reranker-v2-m3", "TableRAG(Yu) 비교군에서만 사용"], ["참고 비교 외부 모델", "GPT-4o (gpt-4o-2024-08-06)", "OpenAI API, temperature 0 (4.4.3절)"]], col_widths=[7000, 11000, 18848])
# ---------- 4.1.2
i = idx("평가 집합은 데이터셋의 계산 과정 표시 유무에 따라")
set_text(i, "평가 집합은 데이터셋의 계산 과정 표시 유무에 따라 조회 문항과 산술 문항으로 분류하고, 정답 셀 수에 따라 셀 1개 문항과 셀 2개 이상 문항으로 분류하여 네 그룹으로 구분한다. [표 4-3]은 평가 집합의 구성이다. 검색 성능은 2,885문항 전체로 평가한다. 답변 정확도는 각 그룹에서 최대 300문항을 고정된 난수 시드로 추출한 882문항으로 평가한다.")
c, t = table_after("[표 4-3] 평가 집합의 구성")
repl[t] = make_table(seg[t], [["문항 그룹", "정의", "검색 평가", "답변 평가"], ["조회·셀 1개", "계산 없음, 정답 셀 1개", "212", "211"], ["조회·셀 2개 이상", "계산 없음, 정답 셀 2개 이상", "367", "300"],
    ["산술·셀 1개", "계산 있음, 정답 셀 1개", "71", "71"], ["산술·셀 2개 이상", "계산 있음, 정답 셀 2개 이상", "2,235", "300"], ["전체", "", "2,885", "882"]], col_widths=[9000, 13848, 7000, 7000])
# ---------- 4.2.1
i = idx("기존 방법 비교군은 LangChain 청크[2], RAPTOR[9]")
set_text(i, "기존 방법 비교군은 LangChain 청크[2], RAPTOR[9], TableRAG(Chen)[13], TableRAG(Yu)[14]의 네 가지이며, 각 방법의 공식 코드나 공식 튜토리얼의 파이프라인을 동일하게 따른다. 모든 비교군은 제안하는 방법과 같은 조건에서 실행한다. 모든 LLM 호출에는 양자화하지 않은 Qwen3.5-4B 모델을, 모든 임베딩에는 bge-base-en-v1.5를 사용하며, 검색 범위는 질문이 속한 문서의 표로 한정한다. 각 방법의 지시문, 검색 단위, LLM 호출 횟수는 원래의 방법을 따른다. 답변 정확도는 각 방법의 전체 파이프라인으로 측정하고, TableRAG(Yu)의 검색 성능은 4.2.3절과 같이 질의 분해 전의 첫 검색 단계로 측정한다.")
c, t = table_after("[표 4-4] 비교군의 구성")
repl[t] = make_table(seg[t], [["비교군", "검색 단위와 LLM 입력", "답변 평가"], ["LangChain 청크[2]", "1,000자(겹침 200자) 청크, 상위 4개 청크를 입력", "882"], ["RAPTOR[9]", "100토큰 잎 청크와 요약 노드의 붕괴 트리 검색, 2,000토큰까지 입력", "882"],
    ["TableRAG(Chen)[13]", "LLM 질의 확장, 스키마·셀 검색, 파이썬 프로그램 실행(최대 5회)", "882"], ["TableRAG(Yu)[14]", "LLM 질의 분해, 청크 검색과 재정렬, SQL 실행(최대 5회)", "882"]], col_widths=[9000, 21848, 6000])
i = idx("RAPTOR는 원 논문의 붕괴 트리 설정인 2,000토큰을 사용한다")
set_text(i, "RAPTOR는 원 논문의 붕괴 트리 설정인 2,000토큰을 사용한다. TableRAG(Chen)과 TableRAG(Yu)는 원 논문에서 상용 LLM을 사용하였으나, 본 실험에서는 같은 조건을 위해 모든 호출에 Qwen3.5-4B를 사용한다. TableRAG(Chen)의 원본 코드는 LLM 호출 한 번의 최대 생성 길이를 128토큰으로 두는데, 로컬 모델은 파이썬 코드 앞에 풀이 문장을 길게 작성하여 코드가 중간에 절단되므로 최대 생성 길이를 TableRAG(Yu)와 같은 512토큰으로 확장하여 실행하였다. 제안하는 방법은 검색된 상위 20개 셀을, 그 셀이 속한 표의 행·열 머리글을 되살린 부분 표(이하 채운 부분 표)로 LLM에 입력하며, 지시문은 부록 B의 것을 사용한다. 전달 형식에 따른 차이는 4.4.1절에서 따로 비교한다.")
# ---------- 4.2.2 채점 문단
i = idx("답변 정확도는 LLM의 답이 정답과 일치하는 문항의 비율이다")
set_text(i, "답변 정확도는 LLM의 최종 답이 정답을 포함하는 문항의 비율이다(형식 허용 채점). 제안하는 방법은 지시문이 요구한 마지막 줄(\"Final answer:\")의 값을, 비교군은 각 파이프라인이 낸 최종 답을 채점 대상으로 한다. 조회 문항은 통화 기호·날짜 표기 등을 정규화한 뒤 정답 문자열의 포함 여부로, 산술 문항은 수치 비교로 판정하되 10의 거듭제곱 배의 척도 차이와 반올림 차이를 허용하며, 예/아니오 문항은 최종 답의 yes/no로 판정한다. 모든 방법에 같은 규칙을 적용하였고, 규칙은 공개 코드로 고정하였다. MultiHiertt 공식 채점 규칙(정확 일치)만 적용한 결과는 부록 A에 함께 수록한다. 두 방법의 정확도 차이가 우연히 생길 수 있는 정도를 넘는지는 McNemar 검정[16]으로 판정한다. McNemar 검정은 같은 문항에 대한 두 방법의 채점 결과를 대응시켜, 한 방법만 맞힌 문항 수와 다른 방법만 맞힌 문항 수가 같다고 볼 수 있는지를 검정한다. 두 방법이 모두 정답이거나 모두 오답인 문항은 검정에 영향을 주지 않는다. 기존 방법 비교군 네 개와의 비교는 하나의 묶음으로 보고 Holm 보정[17]한 p값을 함께 보고한다.")
# ---------- 4.2.3
i = idx("실험은 다음 두 가지로 구성한다")
set_text(i, "실험은 다음 세 가지로 구성한다. 첫째, 제안하는 방법과 기존 방법 비교군의 검색 성능을 비교한다(4.3.1절). RAPTOR와 TableRAG(Chen)은 색인이나 검색 단계에서 LLM을 호출하므로 200문항에서 검색 성능을 측정하고, 나머지 방법은 2,885문항 전체에서도 측정한다. 이 검색 성능 표의 RAPTOR와 TableRAG(Chen) 수치는 이전 설정의 LLM(Qwen3-8B 4비트)으로 트리 요약과 질의 확장을 수행한 결과이다. TableRAG(Yu)의 검색 성능은 LLM의 질의 분해 없이 질문 자체로 수행한 첫 번째 검색(후보 30개 검색 후 재정렬)의 순위 목록으로 측정한다. 둘째, 882문항에서 제안하는 방법과 기존 방법 비교군의 답변 정확도를 비교한다(4.3.2절). 제안하는 방법은 vLLM 연속 묶음 실행이 실행마다 조금씩 달라지므로 같은 조건으로 3회 실행하여 문항마다 2회 이상 맞힌 경우를 정답으로 판정하고, 비교군 파이프라인은 한 문항씩 요청하여 1회 실행한다. 셋째, 같은 882문항에서 전달 형식, 답변 정확도의 상한, 리더 LLM, 오답 원인을 분석한다(4.4절). 셀 문장에서 머리글 경로를 제거한 조건과의 비교는 부록 E에 수록한다.")
# ---------- 4.3.2 표 4-7 + 문단
i = idx("[표 4-7]은 모든 비교군을 평가한 200문항에서의"); set_text(i, "[표 4-7]은 882문항에서의 답변 정확도이며, 4.2.2절의 채점 기준으로 채점하였다.")
c, t = table_after("[표 4-7] 답변 정확도"); set_text(c, "[표 4-7] 답변 정확도(882문항)")
R47 = [["방법", "조회·셀 1개", "조회·셀 2개 이상", "산술·셀 1개", "산술·셀 2개 이상", "맞힌 수", "정확도(%)"],
       ["제안 방법", "152/211", "245/300", "51/71", "172/300", "620", "70.3"], ["LangChain 청크", "131/211", "199/300", "40/71", "109/300", "479", "54.3"],
       ["RAPTOR", "129/211", "193/300", "30/71", "83/300", "435", "49.3"], ["TableRAG(Chen)", "37/211", "81/300", "4/71", "16/300", "138", "15.6"], ["TableRAG(Yu)", "135/211", "221/300", "42/71", "124/300", "522", "59.2"]]
repl[t] = make_table(seg[t], R47)
i = idx("[그림 4-1] 방법별 답변 정확도"); set_text(i, "[그림 4-1] 방법별 답변 정확도(882문항)")
i = idx("[그림 4-1]에서 제안하는 방법은 200문항 가운데")
set_text(i, "[그림 4-1]에서 제안하는 방법은 882문항 가운데 620문항을 맞혀 70.3%의 정확도를 보였고, TableRAG(Yu) 59.2%, LangChain 청크 54.3%, RAPTOR 49.3%, TableRAG(Chen) 15.6%보다 높았다. 네 비교군과의 차이는 모두 Holm 보정 후에도 유의하였다(부록 A). 조회 문항과 산술 문항으로 구분하면, 제안하는 방법은 조회 511문항에서 77.7%, 산술 371문항에서 60.1%로 두 유형 모두에서 가장 높았다. TableRAG(Yu)는 각각 69.7%와 44.7%, LangChain 청크는 64.6%와 40.2%였다. 검색 없이 문서의 표 전체를 같은 지시문으로 입력한 참고 조건은 68.4%(603문항)로 제안하는 방법과의 차이가 우연 범위였다(p=.11).")
i = idx("TableRAG(Chen)은 LLM이 작성한 파이썬 코드를 실행하고")
set_text(i, "TableRAG(Chen)은 LLM이 작성한 파이썬 코드를 실행하고 그 결과를 바탕으로 코드를 다시 작성하는 과정을 최대 5회 반복하여 답을 구한다. 882문항 가운데 848문항(96.1%)에서 이 코드가 한 번 이상 실행 오류를 일으켰고, 652문항(73.9%)에서는 5회 안에 최종 답을 도출하지 못하였다. 실행 오류 1,716건 가운데 1,046건은 모델이 코드 앞에 \"python\"을 붙여 이름 오류가 난 것이고 503건은 셸 명령 형태나 여러 줄 코드로 인한 구문 오류였다. 즉 로컬 4B 모델이 원본 ReAct 형식(한 줄 파이썬 코드)을 따르지 않은 것이 주된 원인이며, 이는 셀·스키마 검색 자체의 열세가 아니라 해당 파이프라인이 요구하는 형식 준수 능력의 문제이다. TableRAG(Yu)는 882문항에서 문항당 평균 5.4회 LLM을 호출하였고, 182문항(20.6%)에서 SQL 실행 오류가 한 번 이상 발생하였으며, 146문항(16.6%)에서는 최종 답을 도출하지 못하였다.")
# ---------- 4.3.3
i = idx("[표 4-5]와 [표 4-7]을 함께 보면")
set_text(i, "[표 4-5]와 [표 4-7]을 함께 보면, 검색 단계의 재현율이 높은 방법이 답변 정확도도 높은 것은 아니었다. LangChain 청크는 LLM에 전달한 상위 4개 청크의 재현율(재현율@4)이 99.4%로 제안하는 방법의 재현율@20 93.9%보다 높았으나, 답변 정확도는 54.3%로 제안하는 방법의 70.3%보다 낮았다. 상위 4개 청크는 문서의 표 대부분을 포함하므로, LangChain 청크는 문항당 평균 126.8개의 셀을 LLM에 전달하고, 그 가운데 정답 셀의 비율은 2.0%이다. 제안하는 방법은 셀 20개를 전달하며 그 가운데 정답 셀의 비율은 11.0%이다. 다만 LangChain 청크는 튜토리얼의 지시문을 그대로 사용하므로, 두 방법의 답변 정확도 차이에는 입력의 차이와 지시문의 차이가 함께 포함되어 있다. 지시문을 제안하는 방법과 같게 통일하고 입력만 LangChain의 상위 4개 청크로 둔 보조 비교에서는 68.0%(600문항)로 제안하는 방법의 70.3%와의 차이가 우연 범위였다(p=.085, 4.4.1절·부록 A). 즉 LangChain 청크와의 16%p 차이의 대부분은 지시문에서 오고, 전달 형식에서 오는 차이는 2~3%p이다.")
i = idx("제안하는 방법의 오답 85문항의 원인을 확인하기 위해")
set_text(i, "제안하는 방법의 오답이 검색 단계와 답변 단계 중 어디에서 생기는지는 4.4.2절(정답 셀만 입력한 상한)과 4.4.4절(오답 원인 분류)에서 다룬다. 정답 셀만 입력하여도 75.3%에 그치므로, 검색·전달 단계에서 더 줄일 수 있는 손실은 5%p 안이다.")
j = idx("⦁ 정답 셀 일부가 상위 20개에 포함되지 않았고, 정답 셀만 입력하면")
for k in range(j, j + 7): drop.add(k)                      # 분류 bullet 6개 + "따라서 …" 문단
assert txt[j + 6].startswith("따라서 검색 단계에서 정답 셀을 누락한")
i = idx("이상의 결과로부터, 셀마다 머리글 경로를 포함한")
set_text(i, "이상의 결과로부터, 셀마다 머리글 경로를 포함한 셀 문장을 검색 단위로 사용하면 정답 셀을 적은 수의 다른 셀과 함께 LLM에 전달할 수 있고, 같은 LLM을 사용하는 LangChain 청크, RAPTOR, TableRAG(Chen), TableRAG(Yu)보다 유의하게 높은 답변 정확도를 얻을 수 있음을 확인하였다.")
# ---------- 4.4 신설 (제 5 장 앞)
H2 = seg[idx("4.3 실험 결과 및 분석")]; H3 = seg[idx("4.3.3 실험 결과 비교 분석")]; CAP = seg[idx("[표 4-7] 답변 정확도")]; TBL = seg[table_after("[표 4-7] 답변 정확도")[1]]
def h2(t): return para_like(H2, t)
def h3(t): return para_like(H3, t)
def cap(t): return para_like(CAP, t)
def tbl(rows, w=None): return make_table(TBL, rows, col_widths=w)
blank = LSA.sub("", seg[idx("제 5 장 결론") - 1]) if not txt[idx("제 5 장 결론") - 1].strip() else body("")
new44 = [h2("4.4 전달 형식, 답변 정확도의 상한, 리더 LLM, 오답 원인"),
 h3("4.4.1 전달 형식"),
 body("검색된 상위 20개 셀을 LLM에 전달하는 형식 세 가지를 같은 지시문(부록 B)과 같은 LLM으로 비교하였다. 셀 문장 목록은 검색 단위를 그대로 나열한 것이고, 채운 부분 표는 검색된 셀이 속한 표의 행·열 머리글을 되살려 부분 표로 표시한 것(제안 방법의 설정)이며, LangChain 청크는 비교군의 상위 4개 청크를 같은 지시문으로 입력한 것이다. vLLM 연속 묶음 실행은 같은 조건에서도 실행마다 팔당 ±4~9문항 달라지므로 3회 실행하여 문항마다 2회 이상 맞힘(다수결)으로 판정하였다. [표 4-8]은 출력 상한 3,072토큰과 4,096토큰에서의 결과이다."),
 cap("[표 4-8] 전달 형식에 따른 답변 정확도(882문항, 3회 실행 다수결)"),
 tbl([["전달 형식", "출력 상한 3,072토큰", "출력 상한 4,096토큰"], ["채운 부분 표(제안 방법)", "629 (71.3%)", "620 (70.3%)"], ["셀 문장 목록", "585 (66.3%)", "588 (66.7%)"], ["LangChain 청크", "601 (68.1%)", "600 (68.0%)"], ["채운 부분 표 대 LangChain 청크 McNemar", "71 : 43, p=.011", "71 : 51, p=.085"]], [14848, 11000, 11000]),
 body("같은 20개 셀을 부분 표로 되살려 전달하면 셀 문장 목록보다 3~4%p 높았고(두 상한 모두 p<.05), LangChain 청크보다는 3,072토큰에서 우연이 아니었으나 4,096토큰에서는 우연 범위였다(여섯 실행 모두 채운 부분 표가 높음). 평가 집합 밖의 새 문항 600개에서도 채운 부분 표 399문항(66.5%) 대 LangChain 청크 376문항(62.7%)으로 세 실행 모두 p<.05로 재현되었다. 이득은 정답 셀이 2개 이상인 문항에 몰렸고(조회 셀 2개 이상 +16, 산술 셀 2개 이상 +7~15문항), 셀 1개 문항에서는 차이가 없었다. 표마다 캡션이 있는 HiTab 300문항(부록 C의 설정)에서는 257문항(85.7%) 대 247문항(82.3%)으로 우연 범위였다(p=.17)."),
 h3("4.4.2 답변 정확도의 상한"),
 body("검색이 완벽하다고 가정했을 때의 상한을 재기 위해, 데이터셋이 근거로 표시한 정답 셀만 같은 형식과 지시문으로 입력하였다(1회 실행). [표 4-9]와 같이 정답 셀만 주어도 정확도는 75.3%(664문항)에 그쳤고, 정답 표 전체나 정답 표 안으로 제한한 채운 부분 표도 73~74%였다. 따라서 실제 검색 결과(70.3~71.3%)와 상한의 차이는 4~5%p이며, 검색·전달 단계에서 더 줄일 수 있는 손실은 그 안이다. 남은 25%의 원인은 4.4.4절에서 분류한다. 세부 결과는 부록 D에 수록한다."),
 cap("[표 4-9] 입력 조건에 따른 답변 정확도(882문항)"),
 tbl([["입력", "맞힌 수", "정확도(%)"], ["정답 셀만(상한)", "664", "75.3"], ["정답 표 전체", "645", "73.1"], ["정답 표 안으로 제한한 채운 부분 표", "651", "73.8"], ["채운 부분 표(실제 검색, 3,072토큰 다수결)", "629", "71.3"]], [20848, 8000, 8000]),
 h3("4.4.3 리더 LLM에 따른 답변 정확도"),
 body("같은 882문항, 같은 두 입력(채운 부분 표, 정답 셀만), 같은 지시문, 같은 출력 상한 4,096토큰에서 리더 LLM만 바꾸었다. 로컬 모델은 모두 bf16 무양자화이고, 외부 모델 GPT-4o는 OpenAI API를 temperature 0으로 호출하였다(seed 없음). [표 4-10]에서 GPT-4o와 제안 방법의 Qwen3.5-4B는 두 입력 모두에서 구별되지 않았다(채운 부분 표 b:c 69:63, p=.66; 정답 셀만 42:44, p=.91). 둘 다 틀린 문항이 193개로 각 모델 오답의 약 4분의 3이 겹친다. 반면 같은 회사의 전 세대 4B 모델(Qwen3-4B-Instruct-2507)은 두 입력 모두 6~10%p 낮았고(p<.001), 다른 계열의 3.8B 모델(Phi-4-mini-instruct)은 풀이를 거의 쓰지 않아 크게 낮았다. 즉 리더에 따른 차이는 실제로 크지만, 제안 방법의 70%대 정확도는 모델 크기의 한계가 아니다."),
 cap("[표 4-10] 리더 LLM에 따른 답변 정확도(882문항)"),
 tbl([["리더 LLM", "크기·구동", "채운 부분 표", "정답 셀만", "데이터 쪽 100문항(정답 셀만)"], ["Qwen3.5-4B(제안 방법)", "4B, 로컬 bf16", "620 (70.3%)", "664 (75.3%)", "0 (정의상)"], ["GPT-4o (gpt-4o-2024-08-06)", "외부 API", "626 (71.0%)", "662·665 (75.1·75.4%)", "10 (10%)"],
      ["Qwen3-4B-Instruct-2507", "4B, 로컬 bf16", "565 (64.1%)", "579 (65.6%)", "5 (5%)"], ["Phi-4-mini-instruct", "3.8B, 로컬 bf16", "286 (32.4%)", "400 (45.4%)", "14 (14%)"]], [10848, 6000, 6000, 7000, 7000]),
 body("외부 모델을 사용할 때 관찰된 것은 다음과 같다. 같은 882문항을 같은 조건으로 두 번 호출하면 출력 문자열이 같은 문항은 270개(30.6%)뿐이었고 정오가 뒤집힌 문항은 29개(3.3%)였으며, 응답에 기록된 백엔드 식별자(system_fingerprint)는 한 실행 안에서 4종, 두 실행에서 6종이 섞였다. 실행 중 프로젝트 지출 한도로 한 차례 중단되었다. 로컬 모델을 한 문항씩 greedy로 두 번 생성하면 50문항의 출력이 글자 단위로 같았다(연속 묶음 실행과는 26문항만 같으나 정확도는 같음). 882문항을 네 조건으로 외부에 보내는 데 문서 표 내용을 포함한 입력 토큰 310만 개가 전송되었다(1.1절)."),
 h3("4.4.4 정답 셀만 주어도 틀리는 문항의 원인"),
 body("정답 셀만 입력한 조건에서 Qwen3.5-4B가 틀린 218문항을 사람이 전부 읽고 원인을 분류하였다(한 사람 판정, 사후 분석). [표 4-11]에서 데이터셋 쪽 원인(정답 셀 표시가 계산에 필요한 셀을 다 담지 않음, 정답 규칙·정답 오류)은 100문항(46%), LLM의 해석·계산 오류는 94문항(43%), 채점 형식·출력 잘림은 24문항(11%)이었다."),
 cap("[표 4-11] 정답 셀만 입력하여도 틀린 218문항의 원인"),
 tbl([["원인", "문항 수", "비율(%)"], ["LLM의 해석·계산 오류", "94", "43"], ["데이터셋의 정답 셀 표시가 계산에 필요한 셀을 다 담지 않음", "56", "26"], ["데이터셋의 정답 규칙·정답 오류", "44", "20"], ["채점 형식·반올림", "14", "6"], ["출력 잘림", "10", "5"]], [24848, 6000, 6000]),
 body("데이터셋 쪽으로 분류한 100문항이 모델이 아니라 문항의 문제라는 판정은 다른 리더로 확인하였다. 같은 100문항을 정답 셀만 주었을 때 GPT-4o는 10문항, Qwen3-4B-Instruct-2507은 5문항, Phi-4-mini-instruct는 14문항만 맞혀 모두 50% 미만이었다([표 4-10]). 따라서 정답 셀만 주어도 틀리는 218문항 가운데 100문항(882문항의 11.3%p)은 리더를 바꾸어도 회복되지 않는 데이터 쪽 몫이고, 나머지 118문항(13.4%p)이 모델 쪽 몫이다. 규칙으로 만든 자동 분류는 출력 잘림과 형식 오류만 가려내고 데이터 쪽과 모델 쪽을 구분하지 못하였으므로(사람 판정 218건과 대조), 분류는 사람 판정에 따른다. 이 100문항을 뺀 정확도는 제안 방법 리더의 오답에서 뽑은 집합이라 그 리더에게 유리하게 치우치므로 제시하지 않는다."),
 h3("4.4.5 지시문의 영향"),
 body("제안 방법의 지시문(부록 B)은 검증 집합 395문항에서 147개 변형 가운데 고른 것이며, 882문항은 선택에 쓰지 않았다. 같은 입력에서 LangChain 튜토리얼의 원문 지시문은 Qwen3.5-4B에서 56.6~58.8%였고 선택한 지시문은 78.8~80.2%(출력 상한 1,536토큰)로, 지시문의 효과가 전달 형식의 효과보다 훨씬 크다. 평가 집합 밖의 새 문항 600개에서 지시문 4개로 채운 부분 표와 LangChain 청크를 다시 비교하면 12쌍 가운데 11쌍에서 채운 부분 표가 높았으나 유의한 것은 원문 지시문의 2쌍뿐이었다. 따라서 4.4.1절의 전달 형식 이득은 선택한 지시문에 한정된 결과로 읽어야 한다."),
 blank]
ins_after[idx("제 5 장 결론") - 1] = new44 if txt[idx("제 5 장 결론") - 1].strip() else new44[:-1]   # 5장 앞 빈 줄 앞에 끼움
# ---------- 5장
i = idx("제안하는 방법의 성능을 평가하기 위해 MultiHiertt 데이터셋에서")
set_text(i, "제안하는 방법의 성능을 평가하기 위해 MultiHiertt 데이터셋에서 LangChain 청크, RAPTOR, 두 가지 TableRAG를 같은 LLM과 같은 임베딩 모델로 실행하여 검색 성능과 답변 정확도를 비교하였다. 실험을 통해 제안하는 방법의 답변 정확도는 882문항에서 70.3%로 TableRAG(Yu)(59.2%), LangChain 청크(54.3%), RAPTOR(49.3%), TableRAG(Chen)(15.6%)보다 높았으며, 네 비교군과의 차이는 모두 통계적으로 유의함을 확인하였다. 검색 단계에서 제안하는 방법은 문항당 20개 셀을 전달하여 재현율@20 93.9%, 셀 정밀도@20 11.0%를 기록하였고, LangChain 청크는 상위 4개 청크로 문항당 평균 126.8개 셀을 전달하여 재현율 99.4%, 셀 정밀도 2.0%였다. 검색된 셀을 부분 표로 되살려 전달하면 셀 문장 목록보다 3~4%p 높았고, 정답 셀만 입력한 상한은 75.3%로 실제 검색 결과와의 차이가 5%p 안이었다. 정답 셀만 주어도 틀리는 문항의 46%는 데이터셋 쪽 원인이었고, 외부 상용 모델(GPT-4o)로 리더를 바꾸어도 정확도는 같았다. 셀 문장에서 머리글 경로를 제거하면 답변 정확도는 16.5%로 하락하였다(부록 E). 또한 표마다 제목이 주어지는 HiTab에서는 셀 문장에 캡션을 추가하면 표 538개를 하나의 색인에 포함한 검색의 재현율@20이 81.2%에서 91.4%로 상승하였다.")
i = idx("본 연구의 한계는 다음과 같다")
set_text(i, "본 연구의 한계는 다음과 같다. 첫째, MultiHiertt의 공개 평가 집합에는 정답이 없어 학습 집합에서 평가 집합을 구성하였으며, 머리글 판정 규칙은 이 집합의 표를 검토하여 결정하였다. 둘째, 기존 방법 비교군은 각 방법의 원래 지시문을 사용하므로, 비교군과의 답변 정확도 차이에는 검색 단위와 함께 지시문의 차이가 혼재되어 있다. 지시문을 동일하게 설정한 비교는 LangChain 청크에 대해서만 수행하였고, 그 차이는 우연 범위였다(4.4.1절). 셋째, 제안하는 방법의 지시문은 검증 집합에서 고른 것이며, 전달 형식의 이득은 이 지시문에 한정된 결과이다(4.4.5절). 넷째, 비교군 파이프라인은 1회 실행이고 TableRAG(Chen)의 낮은 정확도는 로컬 4B 모델이 원본 코드 형식을 따르지 못한 결과이므로, 그 수치는 검색 단위의 비교로 읽을 수 없다. 다섯째, 오답 원인 분류는 한 사람의 사후 판정이다.")
i = idx("향후 연구로는 표 제목이 없는 문서에서")
set_text(i, "향후 연구로는 표 제목이 없는 문서에서 표를 구별하는 정보를 본문에서 추출하여 셀 문장에 추가하는 방법과, 산술 문항에서 LLM의 계산 오류를 감소시키기 위해 계산 도구를 결합하는 방법을 연구할 계획이다. 또한 답변 정확도 평가를 공개 평가 집합의 근거 주석이 보완되는 대로 더 많은 문항으로 확장할 계획이다.")
# ---------- 부록 A
i = idx("두 방법의 정확도 차이는 같은 문항끼리 대응시킨 McNemar 검정[16]으로 판정하였으며")
set_text(i, "두 방법의 정확도 차이는 같은 문항끼리 대응시킨 McNemar 검정[16]으로 판정하였으며, 불일치 쌍에 대한 정확 이항 검정(양측)으로 p값을 계산하였다. 표에는 \"제안 방법만 맞힘 : 상대만 맞힘\" 형태로 불일치 수를 기재한다. 유의 수준은 .05이다. 기존 방법 비교군과의 비교는 882문항의 네 비교를 하나의 묶음으로 보고 Holm 보정[17]한 p값을 함께 기재한다.")
c, t = table_after("[표 A-1] 보조 비교의 McNemar 검정"); set_text(c, "[표 A-1] 보조 비교의 McNemar 검정")
p1 = binomtest(5, 92, 0.5).pvalue; p2 = binomtest(51, 122, 0.5).pvalue; p3 = binomtest(46, 124, 0.5).pvalue
repl[t] = make_table(seg[t], [["비교", "앞 조건만 맞힘 : 뒤 조건만 맞힘", "p", "Holm p"], ["제안 방법 대 머리글 경로 제거(부록 E, 200문항)", "87 : 5", pfmt(p1), "—"], ["채운 부분 표 대 LangChain 청크, 지시문 통일(882문항, 4,096토큰)", "71 : 51", pfmt(p2), "—"], ["채운 부분 표 대 셀 문장 목록(882문항, 4,096토큰)", "78 : 46", pfmt(p3), "—"]], col_widths=[16848, 9000, 6000, 5000])
c, t = table_after("[표 A-2] 답변 정확도의 McNemar 검정"); set_text(c, "[표 A-2] 답변 정확도의 McNemar 검정(882문항, 제안 방법 대비)")
pairs = [("LangChain 청크", 189, 48), ("RAPTOR", 256, 71), ("TableRAG(Chen)", 498, 16), ("TableRAG(Yu)", 136, 38)]
ps = [binomtest(min(b, c), b + c, 0.5).pvalue for _, b, c in pairs]; hs = holm(ps)
repl[t] = make_table(seg[t], [["비교 대상", "제안 방법 : 비교 대상", "p", "Holm p"]] + [[n, f"{b} : {c}", pfmt(p), pfmt(h)] for (n, b, c), p, h in zip(pairs, ps, hs)], col_widths=[12848, 10000, 7000, 7000])
i = idx("공식 채점 규칙만 적용하면 200문항에서 맞힌 문항 수는")
set_text(i, "공식 채점 규칙(정확 일치)만 적용하면 882문항에서 맞힌 문항 수는 제안 방법 482, LangChain 청크 384, RAPTOR 197, TableRAG(Chen) 40, TableRAG(Yu) 394이며, 순서는 [표 4-7]과 같다. 지시문을 통일한 LangChain 청크 입력은 475였다.")
# ---------- 부록 B
i = idx("제안하는 방법, 머리글 경로 제거 조건, 표 전체 입력 조건의 지시문은 다음과 같다")
set_text(i, "제안하는 방법과 4.4절의 모든 조건에서 사용한 지시문은 다음과 같다. 이 지시문은 검증 집합 395문항에서 역할·입력 설명·풀이 방식·규칙·답 형식 줄의 조합 147개 가운데 정확도가 가장 높은 것을 고른 것이다(4.4.5절). 머리글 경로 제거 조건(부록 E)은 이전 설정의 지시문을 사용하였다.")
i = idx("Answer the question using only the provided table context")
set_text(i, "You are a financial analyst answering questions about tables from a company's annual report. Use the following pieces of retrieved context to answer the question. The context contains cells from the report's tables; the meaning of each value is given by its row and column headers. The context contains what is needed to answer; do not say that you don't know, always give your best answer. Before answering, write out the calculation as an explicit arithmetic expression using the values from the context (for example (2,253 - 2,057) / 2,057 = 0.0953) and then evaluate it. If no calculation is needed, state the value directly. Rules: (1) Percentage change or growth rate = (new - old) / old. (2) Average = sum / count; total or sum = add all the items asked. (3) 'How many' asks for a count of the items meeting the condition. (4) Use values exactly as shown, keeping their sign (values in parentheses are negative); do not convert units. (5) Choose the cell whose row and column headers both match the question. (6) For 'which' or 'what' questions asking for a name, answer with the exact text from the headers. On the last line write 'Final answer: ' followed by only the final value, with no units or explanation.")
i = idx("사용자 입력은 \"Context:\" 줄 다음에")
set_text(i, "사용자 입력은 위 지시문 다음 줄에 \"Question: {질문}\", 그다음 줄에 \"Context: {입력}\", 마지막 줄에 \"Answer:\"를 붙인 것이며, 시스템 메시지는 쓰지 않는다. 채운 부분 표는 표마다 \"Table k\" 줄과 머리글 행·열을 포함한 markdown 표로 입력한다.")
# ---------- 부록 D
i = idx("검색이 답변 정확도의 상한을 얼마나 제한하는지 확인하기 위해")
set_text(i, "검색이 답변 정확도의 상한을 얼마나 제한하는지 확인하기 위해, 4장과 같은 882문항에서 MultiHiertt가 근거로 표시한 정답 셀만 LLM에 입력하고 답변 정확도를 측정하였다. 정답 셀이 속한 표 전체를 입력한 조건과, 검색된 상위 20개 셀을 정답 표 안으로 제한한 채운 부분 표 조건도 함께 측정하였다. LLM, 지시문, 입력 형식(채운 부분 표)은 제안하는 방법과 같고 1회 실행이며, 채점은 4.2.2절의 기준을 따른다.")
c, t = table_after("[표 D-1] 정답 셀만 입력한 답변 정확도"); set_text(c, "[표 D-1] 정답 셀만 입력한 답변 정확도(882문항)")
repl[t] = make_table(seg[t], [["문항 그룹", "문항 수", "정답 셀만", "정답 표 전체", "정답 표 안 채운 부분 표", "제안 방법(실제 검색)"], ["조회·셀 1개", "211", "168", "161", "163", "153"], ["조회·셀 2개 이상", "300", "254", "249", "248", "245"], ["산술·셀 1개", "71", "33", "50", "49", "49"], ["산술·셀 2개 이상", "300", "209", "185", "191", "182"], ["전체", "882", "664 (75.3%)", "645 (73.1%)", "651 (73.8%)", "629 (71.3%)"]], col_widths=[8848, 5000, 5500, 5500, 6500, 5500])
i = idx("정답 셀만 입력하여도 정확도는 66.5%였다")
set_text(i, "정답 셀만 입력하여도 정확도는 75.3%였다. 산술·셀 1개 문항에서는 정답 셀만 주면 오히려 낮았는데(33 대 49), 주가 수익률 표의 기준값 100이나 평균의 개수처럼 공식 계산 과정이 표의 다른 셀이나 상수를 쓰는 문항에서 정답 셀 표시가 계산에 필요한 값을 다 담지 않기 때문이다. 틀린 218문항의 원인 분류와 다른 리더에 의한 확인은 4.4.4절에 있다. 따라서 정답 셀만 입력한 조건의 정확도는 검색이 완벽할 때의 정확도 상한의 추정치로만 해석한다.")
# ---------- 조립
out, pos = [], 0
for k, (a, b) in enumerate(sp):
    out.append(S[pos:a]); pos = b
    if k in drop: continue
    q = repl.get(k, S[a:b]); out.append(q if q == S[a:b] else LSA.sub("", q))
    for x in ins_after.get(k, []): out.append(x)
out.append(S[pos:]); S2 = "".join(out)
import xml.dom.minidom; xml.dom.minidom.parseString(S2.encode("utf-8"))
files["Contents/section2.xml"] = S2.encode("utf-8")
# 그림 4-1 교체(답변 정확도 막대, 882)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, io
from matplotlib import font_manager as fm
import os
FONT = "/mnt/c/Windows/Fonts/H2GTRM.TTF"                       # make_figs.py 와 같은 글꼴(HY중고딕)
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 8, "axes.unicode_minus": False, "text.color": "#0b0b0b"})
names = ["제안 방법", "TableRAG(Yu)", "LangChain 청크", "RAPTOR", "TableRAG(Chen)"]; vals = [70.3, 59.2, 54.3, 49.3, 15.6]
fig, ax = plt.subplots(figsize=(7.5, 3.3)); bars = ax.bar(names, vals, color=["#2a78d6", "#8a8883", "#8a8883", "#8a8883", "#8a8883"], width=0.6)
for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, 85); ax.set_ylabel("답변 정확도(%)"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.04); files["BinData/image10.png"] = buf.getvalue()
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items(): zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
n2 = len(top_paras(S2)[2])
print(f"문단 {len(sp)} → {n2}; 교체 {len(repl)}, 삽입 {sum(len(v) for v in ins_after.values())}, 삭제 {len(drop)}; 표 A-2 p: {[pfmt(p) for p in ps]} Holm {[pfmt(h) for h in hs]}; wrote {dst}")
