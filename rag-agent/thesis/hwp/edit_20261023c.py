"""2026-10-23 원고 수정 c (v21 → v22): 전달 방식 서술을 10월 설정에 맞춤 — 답변 단계 입력은 '상위 20개 셀'이 아니라
'검색 순위 상위 셀을 LangChain 청크 4개와 같은 토큰 길이(T)까지 고른 채운 부분 표'(PREREG-2026-10-17-delivery-format.md 선택 규칙).
3.3.5·1.1·4.2.1·4.3.3·4.4.1·5장·부록 E 문구 교체. 수치: D 문항당 평균 108.4셀(95,593/882), T 평균 1,147토큰, C 목록 43.7셀, LangChain 127.3셀.
  .venv/bin/python thesis/hwp/edit_20261023c.py thesis/thesis_working_20261023_v21.hwpx thesis/thesis_working_20261023_v22.hwpx
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
S = files["Contents/section2.xml"].decode("utf-8"); bs, be, sp = top_paras(S); seg = [S[a:b] for a, b in sp]; txt = [text_of(q) for q in seg]
def idx(prefix): return next(k for k, t in enumerate(txt) if t.startswith(prefix))
repl = {}
def set_text(i, text): repl[i] = para_like(seg[i], text); txt[i] = text      # 같은 문단을 여러 번 고칠 수 있게 글도 갱신
def sub(prefix, old, new):
    i = idx(prefix); assert old in txt[i], (prefix[:20], old[:30]); set_text(i, txt[i].replace(old, new))
# 3.3.5
set_text(idx("상위 20개 셀 문장을 유사도 순위대로 한 줄씩 나열하여"), "검색 순위 상위 셀부터 하나씩 더해, 입력 글의 토큰 길이가 비교군 LangChain 청크 4개의 길이(T, 문항당 평균 1,147토큰)를 넘기 직전까지 셀을 고른다. 고른 셀은 표별로 묶어 그 셀들이 속한 행과 열만 남긴 markdown 부분 표로 만들고, 행과 열이 만나는 칸은 문서의 값으로 모두 채운다(이하 채운 부분 표). 표마다 \"Table k\" 줄을 앞에 두고, 표의 순서는 가장 높은 순위의 셀이 속한 표부터이다. 이 입력은 문항당 평균 108개 셀을 담는다. 같은 길이 예산에서 셀 문장을 순위대로 나열한 목록(평균 44개 셀)과의 비교는 4.4.1절에 있다. LLM 입력은 지시문(부록 B) 다음에 \"Question:\"과 질문, \"Context:\"와 채운 부분 표, \"Answer:\"를 붙인 형태이다. LLM에는 공개 모델 Qwen3.5-4B를 양자화 없이(bf16) 사용하며, 매 단계에서 확률이 가장 높은 토큰을 선택하는 그리디 디코딩(greedy decoding)으로 최대 4,096토큰을 생성한다. LLM의 답은 마지막 \"Final answer:\" 이후의 값으로 추출한다.")
set_text(idx("셀 문장 20개와 지시문, 질문을 포함한 LLM 입력은"), "채운 부분 표와 지시문, 질문을 포함한 LLM 입력은 4장의 평가 문항에서 평균 약 1,400토큰이다.")
# 1.1
sub("청크 단위의 검색에는 전달하는 셀이 많다는 문제도 있다", "4장의 답변 평가 200문항에서", "4장의 검색 평가 200문항에서")
# 4.2.1 끝 문단
sub("RAPTOR는 원 논문의 붕괴 트리 설정인 2,000토큰을 사용한다", "제안하는 방법은 검색된 상위 20개 셀을, 그 셀이 속한 표의 행·열 머리글을 되살린 부분 표(이하 채운 부분 표)로 LLM에 입력하며, 지시문은 부록 B의 것을 사용한다.", "제안하는 방법은 3.3.5절과 같이 LangChain 청크 4개와 같은 토큰 길이까지 검색 순위 상위 셀을 골라 채운 부분 표로 LLM에 입력하며, 지시문은 부록 B의 것을 사용한다. 따라서 제안하는 방법과 LangChain 청크는 LLM 입력 길이가 같다.")
# 4.3.3
sub("[표 4-5]와 [표 4-7]을 함께 보면", "제안하는 방법은 셀 20개를 전달하며 그 가운데 정답 셀의 비율은 11.0%이다.", "제안하는 방법은 검색 순위 상위 20개 셀 기준으로 재현율 93.9%, 셀 정밀도 11.0%이며, 답변 단계에는 LangChain 청크와 같은 토큰 길이까지 채운 부분 표(문항당 평균 108개 셀)를 전달하므로 두 방법의 입력 길이는 같고 셀의 조직 방식이 다르다.")
# 4.4.1 첫 문단
sub("검색된 상위 20개 셀을 LLM에 전달하는 형식 세 가지를", "검색된 상위 20개 셀을 LLM에 전달하는 형식 세 가지를", "같은 토큰 길이 예산(T) 안에서 검색 결과를 LLM에 전달하는 형식 세 가지를")
sub("같은 토큰 길이 예산(T) 안에서 검색 결과를 LLM에 전달하는 형식 세 가지를", "셀 문장 목록은 검색 단위를 그대로 나열한 것이고, 채운 부분 표는 검색된 셀이 속한 표의 행·열 머리글을 되살려 부분 표로 표시한 것(제안 방법의 설정)이며", "셀 문장 목록은 검색 순위대로 셀 문장을 T까지 나열한 것(평균 44개 셀)이고, 채운 부분 표는 같은 규칙으로 고른 셀이 속한 행·열을 되살려 값을 채운 부분 표(평균 108개 셀, 제안 방법의 설정)이며")
sub("같은 20개 셀을 부분 표로 되살려 전달하면", "같은 20개 셀을 부분 표로 되살려 전달하면", "같은 길이 예산에서 셀을 부분 표로 되살려 전달하면")
# 4.4.2
sub("검색이 완벽하다고 가정했을 때의 상한을 재기 위해, 데이터셋이 근거로 표시한 정답 셀만", "정답 표 안으로 제한한 채운 부분 표도", "검색된 셀을 정답 표 안으로 제한한 채운 부분 표도")
# 5장
sub("제안하는 방법의 성능을 평가하기 위해 MultiHiertt 데이터셋에서", "검색 단계에서 제안하는 방법은 문항당 20개 셀을 전달하여 재현율@20 93.9%, 셀 정밀도@20 11.0%를 기록하였고, LangChain 청크는 상위 4개 청크로 문항당 평균 126.8개 셀을 전달하여 재현율 99.4%, 셀 정밀도 2.0%였다. 검색된 셀을 부분 표로 되살려 전달하면 셀 문장 목록보다 3~4%p 높았고,", "검색 단계에서 제안하는 방법은 상위 20개 셀 기준 재현율@20 93.9%, 셀 정밀도@20 11.0%를 기록하였고, LangChain 청크는 상위 4개 청크로 문항당 평균 126.8개 셀을 전달하여 재현율 99.4%, 셀 정밀도 2.0%였다. 답변 단계에서 LangChain 청크와 같은 입력 길이까지 검색된 셀을 채운 부분 표로 전달하면 셀 문장 목록보다 3~4%p 높았고,")
# 부록 E
sub("머리글 경로 제거 조건은 제안하는 방법에서 색인하는 셀 문장만 셀 값으로 변경한 조건이다", "[표 E-1]은 4장과 같은 200문항에서 두 조건의 검색 성능과 답변 정확도를 비교한 것이다.", "[표 E-1]은 이전 설정(Qwen3-8B 4비트, 셀 문장 목록 20개, 200문항)에서 두 조건의 검색 성능과 답변 정확도를 비교한 것이다.")
out, pos = [], 0
for k, (a, b) in enumerate(sp):
    out.append(S[pos:a]); pos = b; q = repl.get(k, S[a:b]); out.append(q if q == S[a:b] else LSA.sub("", q))
out.append(S[pos:]); S2 = "".join(out)
import xml.dom.minidom; xml.dom.minidom.parseString(S2.encode("utf-8"))
files["Contents/section2.xml"] = S2.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items(): zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
print(f"교체 {len(repl)}; wrote {dst}")
