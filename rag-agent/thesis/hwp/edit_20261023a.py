"""2026-10-23 원고 수정 a (찐최종.hwpx → thesis_working_20261023_v20.hwpx) — 지도교수 지시 1 대응.
  1 1.1 절 끝 문장 "그리고 보안상 문서를 외부 서비스로 전송할 수 없는 기관에서는 … 필요하다." 를 지우고,
    바로 뒤에 "왜 로컬 LLM 인가" 문단(thesis/draft_20261023/01_why_local.md)을 새 문단으로 넣는다(본문 문단 모양 그대로).
  2 참고문헌 [23] 뒤에 [24]~[27] 추가(참고문헌 문단 모양 그대로).
  바뀐 문단의 linesegarray 는 지워 한글이 다시 배치하게 한다. 목차 쪽 번호는 한글에서 저장한 뒤 toc_update.py 로 따로 맞춘다.
  .venv/bin/python thesis/hwp/edit_20261023a.py thesis/찐최종.hwpx thesis/thesis_working_20261023_v20.hwpx
"""
import re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hwpx_lib import top_paras, text_of

LSA = re.compile(r"<hp:linesegarray>.*?</hp:linesegarray>", re.S)
OLD = "그리고 보안상 문서를 외부 서비스로 전송할 수 없는 기관에서는 외부 API 없이 로컬 환경의 공개 모델만으로 동작하는 구성이 필요하다."
PARA = ("검색 증강 생성은 검색된 문서 조각을 그대로 언어 모델의 입력으로 넘기므로, 외부 API 모델을 사용하면 문서 내용이 외부 서버로 전송된다. "
        "본 논문의 평가 집합(882문항, 문서 582개)에서 이 양을 재면, 문서의 표 전체를 입력하는 방식은 표 2,344개가 통째로 전송되고, "
        "검색 단위를 셀로 줄인 제안하는 방법도 질문당 문서 셀의 평균 32%가 전송되며 같은 문서에 질문이 반복되면 절반에 가까운 셀이 누적 전송된다(4.4절). "
        "기업과 기관의 내부 문서는 이러한 전송 자체가 제한된다. 삼성전자는 2023년 3월 사내 ChatGPT 사용을 허용한 지 20일 안에 반도체 설비 소스코드가 입력되는 사고를 겪었고 "
        "같은 해 5월 사내 생성형 AI 사용을 금지하였으며[24], 금융권은 망분리 규제 아래 생성형 AI를 보안 대책을 조건으로 한 규제 샌드박스 특례로만 허용하고[25], "
        "공공기관 보안 지침은 개인정보를 포함한 기관 내 민감 정보의 입력을 지양하도록 한다[26]. 전송된 입력은 공급자 정책상 남용 감시 목적으로 최대 30일 보존되고 "
        "법적 요구가 있으면 더 오래 보존될 수 있다[27]. 또한 외부 모델은 제공되는 모델 버전이 공급자에 의해 교체·종료되고, 같은 입력에 대한 출력의 재현을 보장하지 않는다(4.4절). "
        "따라서 본 논문은 외부 API 없이 로컬 환경의 공개 모델만으로 동작하는 구성을 전제로 하고, 같은 입력과 같은 지시문에서 외부 모델과의 답변 정확도 차이를 4.4절에서 함께 보고한다.")
REFS = [
    "[24] 정두용, \"[단독] 우려가 현실로…삼성전자, 챗GPT 빗장 풀자마자 '오남용' 속출,\" 이코노미스트, 2023년 3월 30일; Bloomberg, \"Samsung Bans Staff's AI Use After Spotting ChatGPT Data Leak,\" 2023년 5월 2일.",
    "[25] 금융위원회, \"「금융분야 망분리 개선 로드맵」 발표,\" 보도자료, 2024년 8월 13일.",
    "[26] 국가정보원, 「챗GPT 등 생성형 AI 기술 활용 보안 가이드라인」, 2023.",
    "[27] OpenAI, \"Your data,\" OpenAI Developer Platform, https://developers.openai.com/api/docs/guides/your-data, 2026년 10월 22일 접속.",
]

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def new_para(template, text):
    head = re.match(r"(<hp:p [^>]*>)", template).group(1)
    cp = re.search(r'charPrIDRef="(\d+)"', template).group(1)
    return f'{head}<hp:run charPrIDRef="{cp}"><hp:t>{esc(text)}</hp:t></hp:run></hp:p>'

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
S = files["Contents/section2.xml"].decode("utf-8")
bs, be, sp = top_paras(S)
seg = [S[a:b] for a, b in sp]
txt = [text_of(q) for q in seg]

i = next(k for k, t in enumerate(txt) if OLD in t)
assert seg[i].count(OLD) == 1 and seg[i].count("<hp:run") == 1, "1.1 문단 모양이 예상과 다름"
body = seg[i].replace(" " + OLD, "").replace(OLD, "")
new_i = LSA.sub("", body) + new_para(seg[i], PARA)

j = next(k for k, t in enumerate(txt) if t.startswith("[23] "))
assert seg[j].count("<hp:run") == 1 and not any(t.startswith("[24]") for t in txt), "참고문헌 모양이 예상과 다름"
new_j = seg[j] + "".join(new_para(seg[j], r) for r in REFS)

out, pos = [], 0
for k, (a, b) in enumerate(sp):
    out.append(S[pos:a]); pos = b
    out.append(new_i if k == i else new_j if k == j else S[a:b])
out.append(S[pos:])
S2 = "".join(out)
files["Contents/section2.xml"] = S2.encode("utf-8")
with zipfile.ZipFile(dst, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
    for name, data in files.items():
        zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
n2 = len(top_paras(S2)[2])
print(f"문단 {len(sp)} → {n2} (+{n2 - len(sp)}); 1.1 문단 #{i} 문장 교체 + 새 문단, 참고문헌 #{j} 뒤 {len(REFS)}개 추가; wrote {dst}")
