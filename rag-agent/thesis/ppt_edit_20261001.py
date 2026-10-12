"""발표자료를 원고 v15 에 맞춘다 (defense_20261001.pptx → defense_20261001_v2.pptx). 2026-10-01.

  - 연구 배경을 두 장으로: ① 두 문제(머리글 분리는 그림 3-1 로, 불필요한 셀은 수치와 선행연구 Cuconasu·Shi 로)
    ② 기존 연구는 한 문제씩만 해결(머리글 유지: Docling·SMART / 셀 수 감소: TableRAG(Chen)·ITR) → 둘 다 해결한 연구는 확인되지 않음.
  - 수치·용어를 원고와 일치: 표 4-5 두 칸(73.4·76.8), '재현율을 크게 낮추지 않으면서', 정답 셀, 셀P@k, 표 전체 입력(참고), GPU·모델 표기.
  - 그림을 원고 그림으로: 시스템 구조(벡터 데이터베이스 포함), 머리글 경로 구성 4단계(위쪽/왼쪽 머리글), 셀 문장 생성.
  - 유사도 슬라이드에 가중치 0.7 근거, 평가 지표 수식 오타(|G_i G|, 'Precision : :') 수정.
  - 개행: 마스터·레이아웃·슬라이드의 latinLnBrk="1"(한글 단어 잘림 허용)을 "0"으로 — 한글이 단어 중간에서 줄바꿈되지 않게 한다.
    빈 문단으로 띄우던 간격은 문단 앞 간격으로 바꾼다.

  python3 thesis/ppt_edit_20261001.py <입력.pptx> <출력.pptx>     (rag-agent 에서)
"""
import re, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
FIG = Path(__file__).resolve().parent / "fig"
SKILL = Path("/home/user/.claude/skills/synced/53f7acd4-78a1-4306-b119-44fad797e944_d71cd5b4-c7a8-4df3-a5fb-21de320d833b/pptx/scripts")
W = Path(tempfile.mkdtemp(prefix="ppt_"))
zipfile.ZipFile(SRC).extractall(W)

# 슬라이드 4 를 복제해 연구 배경 두 번째 장(slide20.xml)을 슬라이드 4 뒤에 둔다
out = subprocess.run([sys.executable, str(SKILL / "add_slide.py"), str(W), "slide4.xml", "--after", "slide4.xml"],
                     capture_output=True, text=True, check=True).stdout
assert "slide20.xml" in out, out

SL = W / "ppt/slides"
def rd(n): return (SL / f"slide{n}.xml").read_text(encoding="utf-8")
def wr(n, x): (SL / f"slide{n}.xml").write_text(x, encoding="utf-8")

EMU = 914400
def E(v): return int(round(v * EMU))
BLUE, ORANGE, INK, GRAY = "2A78D6", "C05A00", "222222", "595959"
LBLUE, LOR, LGRAY = "E3EEFB", "FBE9DC", "F2F2F2"
MONO = "Consolas"


# ---------------------------------------------------------------- DrawingML 조각
def r(t, sz=None, b=False, color=None, font=None):
    a = 'lang="ko-KR" altLang="en-US"' + (f' sz="{int(sz * 100)}"' if sz else "") + (' b="1"' if b else "") + ' dirty="0"'
    inner = (f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else "") + \
            (f'<a:latin typeface="{font}"/><a:cs typeface="{font}"/>' if font else "")
    sp = ' xml:space="preserve"' if t != t.strip() else ""
    return f"<a:r><a:rPr {a}>{inner}</a:rPr><a:t{sp}>{escape(t)}</a:t></a:r>"


def br(sz=None):
    return f'<a:br><a:rPr lang="ko-KR" altLang="en-US"{f" sz={chr(34)}{int(sz * 100)}{chr(34)}" if sz else ""}/></a:br>'


def p(runs, sz=18, bullet=None, lvl=0, before=0, algn=None, marL=None, indent=None, line=None):
    """bullet: None(없음) / '•' / '-'. runs: 글 하나 또는 r() 목록."""
    if isinstance(runs, str):
        runs = [r(runs, sz)]
    if marL is None:
        marL, indent = {None: (0, 0), "•": (0.3, -0.3), "-": (0.75, -0.3)}[bullet]
    att = f' marL="{E(marL)}" indent="{E(indent)}"' + (f' lvl="{lvl}"' if lvl else "") + (f' algn="{algn}"' if algn else "")
    kids = (f'<a:lnSpc><a:spcPct val="{int(line * 1000)}"/></a:lnSpc>' if line else "")
    kids += f'<a:spcBef><a:spcPts val="{int(before * 100)}"/></a:spcBef>'
    if bullet == "•":
        kids += '<a:buFont typeface="Arial" panose="020B0604020202020204" pitchFamily="34" charset="0"/><a:buChar char="•"/>'
    elif bullet == "-":
        kids += '<a:buFontTx/><a:buChar char="-"/>'
    else:
        kids += "<a:buNone/>"
    return f'<a:p><a:pPr{att} latinLnBrk="0">{kids}</a:pPr>{"".join(runs)}<a:endParaRPr lang="ko-KR" altLang="en-US" sz="{int(sz * 100)}" dirty="0"/></a:p>'


NEXT_ID = [200]
def nid():
    NEXT_ID[0] += 1
    return NEXT_ID[0]


def box(x, y, w, h, paras, fill=None, line=None, geom="rect", anchor="t", ins=(0.1, 0.05, 0.1, 0.05), name="텍스트"):
    i = nid()
    fl = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    ln = f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    av = '<a:gd name="adj" fmla="val 8000"/>' if geom == "roundRect" else ""
    txb = ' txBox="1"' if not fill and not line else ""
    l, t, rr, b = (E(v) for v in ins)
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{name} {i}"/><p:cNvSpPr{txb}/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(w)}" cy="{E(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}"><a:avLst>{av}</a:avLst></a:prstGeom>{fl}{ln}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="{l}" tIns="{t}" rIns="{rr}" bIns="{b}" rtlCol="0" anchor="{anchor}">'
            f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def body(x, y, w, h, paras):
    """레이아웃의 본문 개체 틀(idx 11)을 위치를 지정해 쓴다."""
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="텍스트 개체 틀"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            f'<p:nvPr><p:ph type="body" sz="quarter" idx="11"/></p:nvPr></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(w)}" cy="{E(h)}"/></a:xfrm></p:spPr>'
            f'<p:txBody><a:bodyPr><a:noAutofit/></a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def subtitle(text):
    """레이아웃의 부제목 개체 틀(idx 10, 굵은 남색)."""
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="부제목 개체 틀"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            f'<p:nvPr><p:ph type="body" sz="quarter" idx="10"/></p:nvPr></p:nvSpPr><p:spPr/>'
            f'<p:txBody><a:bodyPr><a:noAutofit/></a:bodyPr><a:lstStyle/>{p(text, sz=22)}</p:txBody></p:sp>')


def arrow(x1, y1, x2, y2, color=GRAY):
    i = nid()
    return (f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{i}" name="화살표 {i}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{E(x1)}" y="{E(y1)}"/><a:ext cx="{E(x2 - x1)}" cy="{E(y2 - y1)}"/></a:xfrm>'
            f'<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>'
            f'<a:ln w="22225"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln></p:spPr></p:cxnSp>')


def add_image(n, png):
    """그림을 media 에 복사하고 슬라이드 n 의 관계를 추가해 rId 를 돌려준다."""
    dst = W / "ppt/media" / f"thesis_{png.name}"
    shutil.copy(png, dst)
    rel = SL / "_rels" / f"slide{n}.xml.rels"
    x = rel.read_text(encoding="utf-8")
    k = 1 + max(int(v) for v in re.findall(r'Id="rId(\d+)"', x))
    x = x.replace("</Relationships>", f'<Relationship Id="rId{k}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                  f'relationships/image" Target="../media/{dst.name}"/></Relationships>')
    rel.write_text(x, encoding="utf-8")
    return f"rId{k}"


def pic(n, png, x, y, w):
    import struct
    pw, ph = struct.unpack(">II", png.read_bytes()[16:24])
    h = w * ph / pw
    rid, i = add_image(n, png), nid()
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="그림 {i}" descr="{png.stem}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/>'
            f'</p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(w)}" cy="{E(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'), h


def cell(text, head=False, fill="FFFFFF", sz=14, algn=None, b=False):
    rp = (f'<a:rPr lang="ko-KR" altLang="en-US" sz="{sz * 100}" b="{1 if head or b else 0}" dirty="0"><a:solidFill><a:srgbClr val="'
          f'{"FFFFFF" if head else INK}"/></a:solidFill><a:latin typeface="맑은 고딕"/><a:ea typeface="맑은 고딕"/>'
          f'<a:cs typeface="맑은 고딕"/></a:rPr>')
    fl = '<a:schemeClr val="accent1"/>' if head else f'<a:srgbClr val="{fill}"/>'
    return (f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:pPr latinLnBrk="0"{f" algn={chr(34)}{algn}{chr(34)}" if algn else ""}/>'
            f'<a:r>{rp}<a:t>{escape(text)}</a:t></a:r></a:p></a:txBody><a:tcPr anchor="ctr"><a:solidFill>{fl}</a:solidFill></a:tcPr></a:tc>')


def table(x, y, widths, rows, rh=0.45, sz=14):
    """slide13 실험 환경 표와 같은 모양(머리글 accent1, 줄무늬 F3F4F6)."""
    i = nid()
    grid = "".join(f'<a:gridCol w="{E(v)}"/>' for v in widths)
    trs = ""
    for k, row in enumerate(rows):
        fill = "FFFFFF" if k % 2 else "F3F4F6"
        trs += f'<a:tr h="{E(rh)}">' + "".join(cell(c, head=k == 0, fill=fill, sz=sz, b=(j == 0 and k > 0)) for j, c in enumerate(row)) + "</a:tr>"
    return (f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{i}" name="표 {i}"/><p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/>'
            f'</p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="{E(x)}" y="{E(y)}"/>'
            f'<a:ext cx="{E(sum(widths))}" cy="{E(rh * len(rows))}"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/'
            f'drawingml/2006/table"><a:tbl><a:tblPr firstRow="1"><a:tableStyleId>{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}</a:tableStyleId>'
            f'</a:tblPr><a:tblGrid>{grid}</a:tblGrid>{trs}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')


def shapes(x):
    """spTree 바로 아래 도형들 (시작, 끝, 이름). mc:AlternateContent 는 한 덩어리."""
    body_s = x.index("</p:grpSpPr>") + len("</p:grpSpPr>")
    body_e = x.rindex("</p:spTree>")
    tok = re.compile(r"<(/?)(p:sp|p:pic|p:graphicFrame|p:cxnSp|p:grpSp|mc:AlternateContent)[ >]")
    res, depth, start, i = [], 0, None, body_s
    for m in tok.finditer(x, body_s, body_e):
        if m.group(1):
            depth -= 1
            if depth == 0:
                end = x.index(">", m.end() - 1) + 1
                seg = x[start:end]
                nm = re.search(r'<p:cNvPr id="\d+" name="([^"]*)"', seg)
                res.append((start, end, nm.group(1) if nm else ""))
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return res


def drop(x, *names):
    """이름이 names 로 시작하는 도형을 지운다(각각 1개 이상 있어야 한다)."""
    for nm in names:
        hit = [(a, b) for a, b, n in shapes(x) if n.startswith(nm)]
        assert hit, nm
        for a, b in reversed(hit):
            x = x[:a] + x[b:]
    return x


def append(x, *xml):
    k = x.rindex("</p:spTree>")
    return x[:k] + "".join(xml) + x[k:]


def rep(x, old, new, count=1):
    assert x.count(old) == count, (old, x.count(old))
    return x.replace(old, new)


# ================================================================ 2 목차
x = rd(2)
x = rep(x, "<a:t>시스템 구조 및 구성요소 </a:t>", "<a:t>시스템 구조 및 구성 요소</a:t>")
wr(2, x)

# ================================================================ 4 연구 배경 (1) 두 문제
x = drop(rd(4), "텍스트 개체 틀")
fig31, h31 = pic(4, FIG / "fig3_1_chunk_split.png", 5.3, 1.85, 7.65)
x = append(x, subtitle("청크 단위 검색의 두 가지 문제"),
    body(0.39, 1.9, 4.85, 2.9, [
        p([r("① 셀 값과 머리글의 분리", 20, b=True, color=BLUE)], sz=20),
        p([r("표를 1,000자 청크로 분할하면", 16), br(16), r("아래쪽 청크에는 열 머리글 줄이 없음", 16)], sz=16, bullet="•", before=8),
        p([r("그림의 청크 2만 검색되면", 16), br(16), r("20,854가 어느 연도의 값인지 알 수 없음", 16)], sz=16, bullet="•", before=8),
        p([r("MultiHiertt 2,885문항 중 11.8%는", 16), br(16), r("정답 셀이 열 머리글 없는 청크에만 있음", 16)], sz=16, bullet="•", before=8)]),
    fig31,
    box(0.39, 4.85, 8.75, 2.35, [
        p([r("② 답에 필요 없는 셀의 대량 전달", 20, b=True, color=BLUE)], sz=20),
        p("정답 셀은 문항당 평균 2.37개이나, LangChain 청크 상위 4개는 126.8셀을 전달함", sz=16, bullet="•", before=8),
        p("재현율은 99.4%로 높아도, 전달된 셀 가운데 정답 셀은 2.0%에 그침", sz=16, bullet="•", before=6),
        p("질문과 무관한 내용이 함께 입력되면 LLM의 정확도가 저하됨(선행 연구)", sz=16, bullet="•", before=6),
        p("Cuconasu 등(SIGIR 2024): 답이 없는 유사 문서를 추가하면 정확도가 크게 하락", sz=14, bullet="-", before=3),
        p("Shi 등(ICML 2023): 무관한 문장 하나만 추가해도 산술 문제 정확도가 크게 하락", sz=14, bullet="-", before=3)],
        ins=(0.0, 0.05, 0.1, 0.05)),
    box(9.45, 5.3, 3.5, 1.75, [
        p([r("LangChain 청크 상위 4개", 14, color=GRAY)], sz=14, algn="ctr"),
        p([r("126.8셀 전달", 28, b=True, color=INK)], sz=28, algn="ctr", before=4),
        p([r("정답 셀 비율 2.0%", 20, b=True, color=ORANGE)], sz=20, algn="ctr", before=4)],
        fill=LGRAY, geom="roundRect", anchor="ctr"))
wr(4, x)

# ================================================================ 20(→5) 연구 배경 (2) 기존 연구
x = drop(rd(20), "텍스트 개체 틀")
C1, C2, C3 = (0.45, 2.75), (3.75, 4.75), (9.0, 3.9)       # (x, 폭)
Y1, Y2, H = 2.3, 3.95, 1.45
def cellbox(c, y, paras, fill, line=None):
    return box(c[0], y, c[1], H, paras, fill=fill, line=line, geom="roundRect", anchor="ctr", ins=(0.15, 0.05, 0.12, 0.05))
x = append(x, subtitle("기존 연구는 두 문제 가운데 하나만 해결함"),
    box(C1[0], 1.85, C1[1], 0.4, [p([r("문제", 14, b=True, color=GRAY)], sz=14)]),
    box(C2[0], 1.85, C2[1], 0.4, [p([r("기존 방법", 14, b=True, color=GRAY)], sz=14)]),
    box(C3[0], 1.85, C3[1], 0.4, [p([r("남는 한계", 14, b=True, color=GRAY)], sz=14)]),
    cellbox(C1, Y1, [p([r("① 셀 값과", 18, b=True), br(18), r("머리글의 분리", 18, b=True)], sz=18)], LBLUE),
    cellbox(C2, Y1, [p([r("머리글 유지", 16, b=True, color=BLUE)], sz=16),
                     p([r("청크마다 머리글 줄 반복  ", 14), r("Docling(2024)", 12, color=GRAY)], sz=14, bullet="•", before=4),
                     p([r("여러 층 머리글을 한 층으로 평탄화", 14), br(14), r("SMART(ACL Findings 2026)", 12, color=GRAY)], sz=14, bullet="•", before=2)],
             "FFFFFF", "BFBFBF"),
    cellbox(C3, Y1, [p("청크나 표 전체를 전달하므로", sz=16), p([r("전달 셀 수가 줄지 않음", 16, b=True, color=ORANGE)], sz=16, before=2)], LOR),
    cellbox(C1, Y2, [p([r("② 답에 필요 없는", 18, b=True), br(18), r("셀의 대량 전달", 18, b=True)], sz=18)], LBLUE),
    cellbox(C2, Y2, [p([r("전달 셀 수 감소", 16, b=True, color=BLUE)], sz=16),
                     p([r("셀 단위 검색  ", 14), r("TableRAG(Chen), NeurIPS 2024", 12, color=GRAY)], sz=14, bullet="•", before=4),
                     p([r("행·열 단위 검색  ", 14), r("ITR, ACL 2023", 12, color=GRAY)], sz=14, bullet="•", before=2)], "FFFFFF", "BFBFBF"),
    cellbox(C3, Y2, [p("셀에서 행 머리글이나", sz=16), p([r("여러 층 머리글이 제거됨", 16, b=True, color=ORANGE)], sz=16, before=2)], LOR),
    arrow(C1[0] + C1[1] + 0.08, Y1 + H / 2, C2[0] - 0.08, Y1 + H / 2), arrow(C2[0] + C2[1] + 0.08, Y1 + H / 2, C3[0] - 0.08, Y1 + H / 2),
    arrow(C1[0] + C1[1] + 0.08, Y2 + H / 2, C2[0] - 0.08, Y2 + H / 2), arrow(C2[0] + C2[1] + 0.08, Y2 + H / 2, C3[0] - 0.08, Y2 + H / 2),
    box(0.45, 5.7, 12.45, 0.9, [
        p([r("두 문제를 동시에 해결한 연구는 확인되지 않음", 18, b=True, color="FFFFFF"), br(18),
           r("→ 셀 하나를 검색 단위로 삼고, 셀마다 머리글 경로를 결합한 셀 문장 검색 단위를 제안", 16, color="FFFFFF")], sz=18, algn="ctr")],
        fill=BLUE, geom="roundRect", anchor="ctr"),
    box(0.45, 6.72, 12.45, 0.35, [p([r("TableRAG(Chen): Chen 등(NeurIPS 2024)의 TableRAG. 이름이 같은 Yu 등(EMNLP 2025)의 TableRAG는 "
                                      "TableRAG(Yu)로 구분하여 표기함", 11, color=GRAY)], sz=11)], ins=(0, 0, 0, 0)))
wr(20, x)

# ================================================================ 5 연구 목표
x = drop(rd(5), "텍스트 개체 틀")
x = append(x,
    box(0.45, 1.4, 12.45, 1.35, [
        p([r("목표", 16, b=True, color=BLUE)], sz=16),
        p([r("계층형 표가 포함된 문서의 RAG에서 정답 셀 재현율을 크게 낮추지 않으면서,", 20, b=True), br(20),
           r("LLM에 전달하는 셀 수를 줄이고 각 셀 값에 행·열 머리글 경로를 결합하는 검색 단위를 설계함", 20, b=True)], sz=20, before=4)],
        fill=LBLUE, geom="roundRect", anchor="ctr", ins=(0.25, 0.08, 0.2, 0.08)),
    body(0.39, 3.0, 12.5, 2.6, [
        p("셀마다 행·열 머리글 경로를 결합한 셀 문장을 검색 단위로 설계", sz=20, bullet="•"),
        p([r("기존 RAG 방법과 검색 성능·답변 정확도 비교", 20), br(20),
           r("(LangChain 청크, RAPTOR, TableRAG(Chen), TableRAG(Yu))", 16, color=GRAY)], sz=20, bullet="•", before=14),
        p("색인에 머리글 경로를 포함하는지에 따른 검색 성능과 답변 정확도의 차이 검증", sz=20, bullet="•", before=14)]),
    box(0.45, 5.35, 12.45, 0.75, [
        p([r("제약 조건  ", 16, b=True, color=GRAY), r("임베딩 모델과 LLM을 학습하지 않고, 외부 API 없이 로컬 환경의 공개 모델만 사용함", 16)], sz=16)],
        fill=LGRAY, geom="roundRect", anchor="ctr", ins=(0.25, 0.05, 0.2, 0.05)))
wr(5, x)

# ================================================================ 7 시스템 구조
x = rd(7)
x = rep(x, "<a:t>시스템 구조 </a:t>", "<a:t>시스템 구조 및 구성 요소</a:t>")
x = drop(x, "Picture 1")
f32, h32 = pic(7, FIG / "fig3_2_system.png", 0.9, 1.3, 11.55)
x = append(x, f32, box(0.9, 1.3 + h32 + 0.15, 11.55, 0.8, [
    p([r("색인 단계(문서마다 1회)  ", 15, b=True, color=BLUE),
       r("셀 문장을 벡터로 변환해 벡터 데이터베이스에 저장하고, BM25 색인도 구축", 15)], sz=15),
    p([r("질의 단계(질문마다)  ", 15, b=True, color=BLUE),
       r("질문이 속한 문서의 셀 문장 순위를 매겨 상위 20개를 LLM에 전달", 15)], sz=15, before=4)], ins=(0, 0, 0, 0)))
wr(7, x)

# ================================================================ 8 머리글 경로 구성 (원고 그림 3-3 네 단계)
x = drop(rd(8), "머리글 경로 구성 예")
cw, gx, x0 = 6.1, 0.33, 0.4
s = [pic(8, FIG / f"fig3_3_step{k}.png", x0 + (k - 1) % 2 * (cw + gx), 0, cw) for k in (1, 2, 3, 4)]
row1_y, gap = 1.3, 0.22
row2_y = row1_y + max(s[0][1], s[1][1]) + gap
def at(xml, y): return re.sub(r'(<a:off x="\d+" y=")\d+(")', lambda m: m.group(1) + str(E(y)) + m.group(2), xml, count=1)
x = append(x, at(s[0][0], row1_y), at(s[1][0], row1_y), at(s[2][0], row2_y), at(s[3][0], row2_y),
    box(0.4, row2_y + s[3][1] + 0.12, 12.5, 0.75, [
        p([r("MultiHiertt 표에는 머리글 칸 표시(<th>)가 없어, 칸의 글자가 숫자인지로 위쪽·왼쪽 머리글을 판정", 14)], sz=14, bullet="•"),
        p([r("HiTab 3,597표에 같은 규칙 적용 시 데이터셋의 머리글 구조와 위쪽 머리글 줄 수 91.1%, 왼쪽 머리글 열 수 91.9% 일치", 14)],
          sz=14, bullet="•", before=2)], ins=(0, 0, 0, 0)))
wr(8, x)

# ================================================================ 9 셀 문장 생성
x = drop(rd(9), "텍스트 개체 틀", "그림 4")
f34, h34 = pic(9, FIG / "fig3_4_cell_sentence.png", 0.9, 2.6, 11.55)
x = append(x,
    body(0.39, 1.25, 12.5, 1.3, [
        p([r("캡션이 없는 표(MultiHiertt):  ", 18, b=True), r("{행 경로} > {열 경로}: {값}", 18, font=MONO)], sz=18, bullet="•"),
        p([r("캡션이 있는 표(HiTab):  ", 18, b=True), r("In the table '{캡션}', among {행 경로}, the value of {열 경로} is {값}.", 15, font=MONO)],
          sz=18, bullet="•", before=10)]),
    f34,
    box(0.9, 2.6 + h34 + 0.15, 11.55, 1.0, [
        p("셀 1개 = 셀 문장 1개 = 벡터 1개 = 검색 단위 1개 (머리글 칸과 빈 칸은 셀 문장을 만들지 않음)", sz=16, bullet="•"),
        p("캡션 효과(HiTab): 표 538개를 한 색인에서 검색할 때 캡션을 추가하면 재현율@20 81.2% → 91.4%", sz=16, bullet="•", before=6)],
        ins=(0, 0, 0, 0)))
wr(9, x)

# ================================================================ 10 유사도 계산 예시
x = rd(10)
ch = re.search(r"<mc:Choice.*?</mc:Choice>", x, re.S).group(0)
q = re.search(r'<a:p><a:r><a:rPr lang="ko-KR" altLang="en-US" sz="1600" b="1" dirty="0"><a:solidFill><a:srgbClr val="0063BA"/>.*?</a:p>', ch, re.S).group(0)
empty = '<a:p><a:endParaRPr lang="en-US" altLang="ko-KR" dirty="0"/></a:p>'
assert ch.count(empty) == 2
new_ch = ch.replace(empty + q + empty, "".join([
    p([r("가중치 0.7: Bruch 등(TOIS 2023)이 권고한 0.6~0.8 범위에 속함", 14, color=GRAY)], sz=14, bullet="•", before=2),
    p([r("검증 집합 332문항에서 0.5~0.9는 재현율@20 차이 0.8%p 이내, 코사인만(1.0)·BM25만(0.0)이 가장 낮음", 14, color=GRAY)],
      sz=14, bullet="•", before=2),
    q.replace("<a:p>", '<a:p><a:pPr><a:spcBef><a:spcPts val="1400"/></a:spcBef></a:pPr>', 1)]), 1)
assert new_ch != ch
x = x.replace(ch, new_ch)
x = rep(x, '<a:off x="457200" y="2880360"/>', f'<a:off x="457200" y="{E(3.3)}"/>')
x = rep(x, '<a:off x="457200" y="5532120"/>', f'<a:off x="457200" y="{E(6.2)}"/>')
wr(10, x)

# ================================================================ 12 실험 환경
x = rd(12)
x = rep(x, "<a:t>NVIDIA RTX 3060 Ti (8GB)</a:t>", "<a:t>NVIDIA GeForce RTX 3060 Ti (8GB)</a:t>")
def next_cell(x, label, para):
    """표에서 글이 label 인 칸의 오른쪽 칸 내용을 para 로 바꾼다."""
    k = x.index(f"<a:t>{label}</a:t>")
    a = x.index("</a:tc>", k) + len("</a:tc>")
    b = x.index("</a:tc>", a) + len("</a:tc>")
    c = re.sub(r"<a:txBody>.*</a:txBody>", lambda m: "<a:txBody><a:bodyPr/><a:lstStyle/>" + para + "</a:txBody>", x[a:b], flags=re.S)
    return x[:a] + c + x[b:]


x = next_cell(x, "임베딩", p([r("BAAI/bge-base-en-v1.5", 15, color=INK), br(15), r("(768차원, 학습하지 않음)", 13, color=GRAY)], sz=15))
x = next_cell(x, "LLM", p([r("Qwen3-8B, NF4 4비트 양자화, 그리디 디코딩", 15, color=INK)], sz=15))
x = next_cell(x, "재정렬", p([r("BAAI/bge-reranker-v2-m3", 15, color=INK), br(15), r("(TableRAG(Yu) 비교군에서만 사용)", 13, color=GRAY)], sz=15))
wr(12, x)

# ================================================================ 13 실험 방법
x = drop(rd(13), "텍스트 개체 틀")
x = append(x,
    body(0.39, 1.25, 12.5, 1.7, [
        p([r("데이터셋: ", 18, b=True), r("MultiHiertt(재무 보고서 질의응답) — 문서당 평균 표 4.05개, 데이터 셀 145.7개", 18)], sz=18, bullet="•"),
        p("검색 성능: 2,885문항  /  답변 정확도: 200문항(조회 116개, 산술 84개)", sz=16, bullet="-", before=4),
        p([r("비교군: ", 18, b=True), r("모든 방법에 같은 LLM·임베딩 모델, 검색 범위는 질문이 속한 문서", 18)], sz=18, bullet="•", before=12)]),
    table(0.45, 2.85, [2.9, 9.55], [
        ["비교군", "검색 단위와 LLM 입력"],
        ["LangChain 청크", "1,000자(겹침 200자) 청크, 상위 4개 청크를 입력"],
        ["RAPTOR", "100토큰 잎 청크와 요약 노드의 트리 검색, 2,000토큰까지 입력"],
        ["TableRAG(Chen)", "LLM 질의 확장, 스키마·셀 검색, 파이썬 코드 실행(최대 5회)"],
        ["TableRAG(Yu)", "LLM 질의 분해, 청크 검색과 재정렬, SQL 실행(최대 5회)"],
        ["표 전체 입력(참고)", "검색 없이 문서의 표 전체를 마크다운으로 입력(답변 정확도만 평가)"]], rh=0.5, sz=15),
    box(0.45, 5.95, 12.45, 0.35, [p([r("TableRAG(Chen)·TableRAG(Yu): 이름이 같은 Chen 등(NeurIPS 2024)과 Yu 등(EMNLP 2025)의 방법을 제1저자로 구분",
                                     12, color=GRAY)], sz=12)], ins=(0, 0, 0, 0)))
wr(13, x)

# ================================================================ 14 평가 지표
x = drop(rd(14), "Rectangle 13")
ch = re.search(r"<mc:Choice.*?</mc:Choice>", x, re.S).group(0)
ps = re.findall(r"<a:p>.*?</a:p>", ch, re.S)
def text_para(t, sz):   # 수식 없는 문단 하나를 글 하나로 바꾼다(문단 모양은 유지)
    return lambda old: re.sub(r"(</a:pPr>|<a:pPr[^>]*/>).*?(<a:endParaRPr[^>]*/>|</a:p>)", lambda m: m.group(1) + r(t, sz) + m.group(2), old, count=1, flags=re.S)
fix = {"검색 성능 평가": ("검색 성능 지표 (k = 검색 단위 수)", 19),
       "Recall : ": ("재현율@k: 정답 셀 중 상위 k개 단위에 포함된 비율", 19),
       "Precision : : ": ("정밀도@k: 상위 k개 단위 중 정답 셀을 하나 이상 포함한 단위의 비율", 19),
       "CellPrecision : ": ("셀 정밀도@k: 상위 k개 단위에 포함된 셀 중 정답 셀의 비율", 19)}
new_ch = ch
for key, (t, sz) in fix.items():
    hit = [q for q in ps if "a14:m" not in q and "".join(re.findall(r"<a:t>([^<]*)</a:t>", q)).startswith(key)]
    assert len(hit) == 1, key
    new_ch = new_ch.replace(hit[0], text_para(t, sz)(hit[0]))
# 재현율 수식 분모 |G_i G| → |G_i|
m1 = re.search(r"<a14:m>.*?</a14:m>", new_ch, re.S).group(0)
dens = list(re.finditer(r"<m:den>.*?</m:den>", m1, re.S))
d = dens[-1].group(0)
d2 = re.sub(r"(</m:sSub>)<m:r>(?:(?!</m:r>).)*?<m:t>𝐺</m:t></m:r>", r"\1", d, count=1, flags=re.S)
assert d2 != d
new_ch = new_ch.replace(m1, m1.replace(d, d2))
x = x.replace(ch, new_ch)
NB = "\u00a0"  # 범례는 줄바꿈 없는 공백으로 쓰여 있다
x = rep(x, NB.join("정답 근거 셀 집합".split()), NB.join("정답 셀 집합".split()))
x = rep(x, NB.join("검색 단위가 담은 셀 집합".split()), NB.join("검색 단위에 포함된 셀 집합".split()))
x = append(x, box(0.45, 6.72, 12.45, 0.4, [p([r("답변 정확도: ", 14, b=True), r(
    "LLM의 답이 정답과 일치하는 문항의 비율(200문항). 방법 간 차이는 McNemar 검정과 Holm 보정으로 판정", 14)], sz=14)], ins=(0, 0, 0, 0)))
wr(14, x)

# ================================================================ 15·16·17 실험 결과
def head_text(x, t):
    return re.sub(r'(name="텍스트 개체 틀 3".*?<p:txBody>).*?(</p:txBody>)',
                  lambda m: m.group(1) + '<a:bodyPr><a:noAutofit/></a:bodyPr><a:lstStyle/>' + p([r(t, 18, b=True)], sz=18) + m.group(2),
                  x, count=1, flags=re.S)

x = drop(rd(15), "Rectangle 13")
x = head_text(x, "검색 성능 (200문항, R@k: 재현율@k, P@k: 정밀도@k, k = 검색 단위 수, 단위: %)")
x = rep(x, "<a:t>73.5</a:t>", "<a:t>73.4</a:t>")
x = rep(x, "<a:t>76.7</a:t>", "<a:t>76.8</a:t>")
x = append(x, box(1.7, 5.55, 9.95, 1.2, [
    p("LangChain 청크·TableRAG(Yu): 단위 하나가 30셀 이상, 문서당 4~5개 청크 → 상위 5개가 사실상 표 전체", sz=14, bullet="•"),
    p("제안 방법: 단위가 셀 1개라 정답 셀이 2개 이상이면 R@1이 100%가 될 수 없음(200문항 최대 58.6%)", sz=14, bullet="•", before=4)],
    ins=(0, 0, 0, 0)))
wr(15, x)

x = drop(rd(16), "Rectangle 13")
x = head_text(x, "셀 정밀도 (200문항, 셀P@k: 셀 정밀도@k, k = 검색 단위 수, 단위: %)")
for k in (1, 5, 10, 20):
    x = rep(x, f"<a:t>CP@{k}</a:t>", f"<a:t>셀P@{k}</a:t>")
x = append(x, box(1.7, 5.25, 9.95, 1.2, [
    p("셀 정밀도: 전달된 셀 가운데 정답 셀의 비율", sz=14, bullet="•"),
    p("LangChain 청크 상위 4개: 문항당 126.8셀 전달, 셀 정밀도 2.0%", sz=14, bullet="•", before=4),
    p("제안 방법: 20셀 전달, 셀 정밀도 11.0% (정답 셀이 문항당 평균 2.37개이므로 최대 11.9%)", sz=14, bullet="•", before=4)],
    ins=(0, 0, 0, 0)))
wr(16, x)

x = drop(rd(17), "Rectangle 13")
x = head_text(x, "답변 정확도 (200문항)")
x = rep(x, "<a:t>표 전체 입력</a:t>", "<a:t>표 전체 입력(참고)</a:t>")
x = append(x, box(0.74, 6.05, 11.85, 0.8, [
    p("제안 방법과 LangChain 청크, RAPTOR, TableRAG(Chen)의 차이는 Holm 보정 후에도 유의함(McNemar 검정)", sz=14, bullet="•"),
    p("표 전체 입력(참고)은 검색 없이 문서의 표 전체를 입력한 조건", sz=14, bullet="•", before=4)], ins=(0, 0, 0, 0)))
wr(17, x)

# ================================================================ 19 결론
x = drop(rd(19), "텍스트 개체 틀")
x = append(x, body(0.39, 1.35, 12.5, 5.6, [
    p([r("검색: ", 20, b=True), r("문항당 20셀 전달, 재현율@20 93.9%, 셀 정밀도@20 11.0%", 20)], sz=20, bullet="•"),
    p("LangChain 청크: 상위 4개 청크로 126.8셀 전달, 재현율 99.4%, 셀 정밀도 2.0%", sz=17, bullet="-", before=4),
    p([r("답변 정확도: ", 20, b=True), r("57.5%로 LangChain 청크 39.5%, RAPTOR 38.5%, TableRAG(Chen) 13.5%보다 유의하게 높음", 18)],
      sz=20, bullet="•", before=16),
    p("TableRAG(Yu) 53.5%보다 4.0%p 높음", sz=17, bullet="-", before=4),
    p([r("머리글 경로 제거: ", 20, b=True), r("재현율@20 93.9% → 24.7%, 답변 정확도 57.5% → 16.5%", 20)], sz=20, bullet="•", before=16),
    p([r("캡션(HiTab): ", 20, b=True), r("표 538개 한 색인 검색에서 캡션 추가 시 재현율@20 81.2% → 91.4%", 20)], sz=20, bullet="•", before=16),
    p([r("한계: ", 20, b=True), r("답변 정확도 차이에서 전달 셀 수 감소와 머리글 결합 형식의 기여를 분리하지 않음", 18)],
      sz=20, bullet="•", before=16),
    p([r("향후 연구: ", 20, b=True), r("형식(셀 문장/마크다운) × 전달량(상위 20셀/문서 전체) 2×2 비교, 답변 평가 문항 확대", 18)],
      sz=20, bullet="•", before=16)]))
wr(19, x)

# 지운 그림의 관계(원래 그림 파일)를 뺀다 — 이후 clean.py 가 쓰이지 않는 media 를 지운다
for n in (7, 8, 9):
    rel = SL / "_rels" / f"slide{n}.xml.rels"
    t, sx = rel.read_text(encoding="utf-8"), rd(n)
    t = re.sub(r'<Relationship Id="(rId\d+)" Type="[^"]*/image" Target="[^"]*"/>',
               lambda m: m.group(0) if f'"{m.group(1)}"' in sx else "", t)
    rel.write_text(t, encoding="utf-8")

# ================================================================ 개행: 한글 단어 잘림 허용 끄기
for f in [*(W / "ppt/slides").glob("slide*.xml"), *(W / "ppt/slideLayouts").glob("*.xml"), *(W / "ppt/slideMasters").glob("*.xml")]:
    t = f.read_text(encoding="utf-8")
    f.write_text(t.replace('latinLnBrk="1"', 'latinLnBrk="0"'), encoding="utf-8")

subprocess.run([sys.executable, str(SKILL / "clean.py"), str(W)], check=True, capture_output=True)
if DST.exists():
    DST.unlink()
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
    names = sorted(str(q.relative_to(W)) for q in W.rglob("*") if q.is_file())
    names.remove("[Content_Types].xml")
    z.write(W / "[Content_Types].xml", "[Content_Types].xml")
    for n in names:
        z.write(W / n, n)
print("wrote", DST)
