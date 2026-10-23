"""2026-10-23 원고 수정 d (v22 → v23): 초록(한·영)·3.2 구성 요소·3.3.4 끝·5장 동작 과정의 '상위 20개 셀 문장 전달' 서술을 10월 설정으로.
  .venv/bin/python thesis/hwp/edit_20261023d.py thesis/thesis_working_20261023_v22.hwpx thesis/thesis_working_20261023_v23.hwpx
"""
import re, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of
LSA = re.compile(r"<hp:linesegarray>.*?</hp:linesegarray>", re.S)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def para_like(template, text):
    head = re.match(r"(<hp:p [^>]*>)", template).group(1); cp = re.search(r'charPrIDRef="(\d+)"', template).group(1)
    return f'{head}<hp:run charPrIDRef="{cp}"><hp:t>{esc(text)}</hp:t></hp:run></hp:p>'
src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src); files = {n: z.read(n) for n in z.namelist()}
def edit(section, changes):
    S = files[section].decode("utf-8"); bs, be, sp = top_paras(S); seg = [S[a:b] for a, b in sp]; txt = [text_of(q) for q in seg]; repl = {}
    for key, old, new in changes:
        i = next(k for k, t in enumerate(txt) if key in t); assert old in txt[i], (key[:20], old[:40])
        txt[i] = txt[i].replace(old, new); repl[i] = para_like(seg[i], txt[i])
    out, pos = [], 0
    for k, (a, b) in enumerate(sp):
        out.append(S[pos:a]); pos = b; out.append(LSA.sub("", repl[k]) if k in repl else S[a:b])
    out.append(S[pos:]); S2 = "".join(out)
    import xml.dom.minidom; xml.dom.minidom.parseString(S2.encode("utf-8")); files[section] = S2.encode("utf-8"); return len(repl)
n1 = edit("Contents/section1.xml", [
 ("하이브리드 유사도로 상위 20개 셀 문장을 검색하여 LLM에 전달한다", "하이브리드 유사도로 상위 20개 셀 문장을 검색하여 LLM에 전달한다", "하이브리드 유사도로 셀 문장의 순위를 매기고, 상위 셀을 머리글을 되살린 부분 표로 LLM에 전달한다"),
 ("MultiHiertt 데이터셋의 200문항으로 실험한 결과", "MultiHiertt 데이터셋의 200문항으로 실험한 결과, 제안하는 방법의 답변 정확도는 57.5%로 같은 LLM을 사용한 LangChain 청크 분할(39.5%), RAPTOR(38.5%), TableRAG(Chen)(13.5%), TableRAG(Yu)(53.5%)보다 높았으며, LangChain 청크 분할, RAPTOR, TableRAG(Chen)과의 차이는 통계적으로 유의하였다. 제안하는 방법은 문항당 20개의 셀을 LLM에 전달하였으며, LangChain 청크 분할은 문항당 평균 126.8개의 셀을 전달하였다.",
  "MultiHiertt 데이터셋의 882문항으로 실험한 결과, 제안하는 방법의 답변 정확도는 70.3%로 같은 LLM을 사용한 TableRAG(Yu)(59.2%), LangChain 청크 분할(54.3%), RAPTOR(49.3%), TableRAG(Chen)(15.6%)보다 높았으며, 네 비교군과의 차이는 모두 통계적으로 유의하였다. 같은 입력 길이에서 검색된 셀을 부분 표로 되살려 전달하면 셀 문장 목록보다 3~4%p 높았고, 정답 셀만 입력한 상한은 75.3%였으며, 리더를 외부 상용 모델(GPT-4o)로 바꾸어도 정확도는 같았다."),
 ("passes the top 20 cell sentences ranked by hybrid similarity to the LLM", "passes the top 20 cell sentences ranked by hybrid similarity to the LLM", "ranks cell sentences by hybrid similarity and passes the top-ranked cells to the LLM as a partial table with their headers restored"),
 ("In experiments on 200 MultiHiertt questions", "In experiments on 200 MultiHiertt questions, the proposed method reaches an answer accuracy of 57.5%, higher than LangChain chunking (39.5%), RAPTOR (38.5%), TableRAG (Chen) (13.5%), and TableRAG (Yu) (53.5%) run with the same LLM, and its differences from LangChain chunking, RAPTOR, and TableRAG (Chen) are statistically significant. The proposed method passes 20 cells per question to the LLM, whereas LangChain chunking passes 126.8 cells on average.",
  "In experiments on 882 MultiHiertt questions, the proposed method reaches an answer accuracy of 70.3%, higher than TableRAG (Yu) (59.2%), LangChain chunking (54.3%), RAPTOR (49.3%), and TableRAG (Chen) (15.6%) run with the same LLM, and its differences from all four baselines are statistically significant. At equal input length, delivering the retrieved cells as a header-restored partial table is 3-4 points more accurate than a list of cell sentences; the ceiling with only the gold cells as input is 75.3%, and replacing the reader with a commercial external model (GPT-4o) does not change the accuracy."),
])
n2 = edit("Contents/section2.xml", [
 ("⦁ 하이브리드 검색기: 질문을 같은 임베딩 모델로", "질문이 속한 문서의 셀 문장 순위를 매긴 후 상위 20개 셀 문장을 선택한다.", "질문이 속한 문서의 셀 문장 순위를 매긴다."),
 ("⦁ LLM: 상위 20개 셀 문장과 질문을 입력받아", "⦁ LLM: 상위 20개 셀 문장과 질문을 입력받아 계산 과정과 최종 답을 생성한다.", "⦁ LLM: 검색 순위 상위 셀을 머리글과 함께 되살린 채운 부분 표(3.3.5절)와 질문을 입력받아 계산 과정과 최종 답을 생성한다."),
 ("코사인 유사도는 질문과 셀 문장의 의미 유사도를 반영하고", "하이브리드 점수의 상위 20개 셀 문장을 LLM의 입력으로 전달한다. 전달하는 셀 수 20은 평가 문항과 겹치지 않는", "검색 성능은 하이브리드 점수의 상위 20개 셀을 기준으로 평가하고, 답변 단계의 입력은 3.3.5절과 같이 길이 예산 안에서 순위대로 고른다. 기준 셀 수 20은 평가 문항과 겹치지 않는"),
 ("본 논문에서 제안한 방법의 전체 동작 과정은 다음과 같다", "상위 20개 셀 문장을 LLM의 입력으로 전달하여 답을 생성한다.", "검색 순위 상위 셀을 머리글을 되살린 부분 표로 LLM에 전달하여 답을 생성한다."),
])
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items(): zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print(f"section1 {n1}, section2 {n2} 교체; wrote {dst}")
