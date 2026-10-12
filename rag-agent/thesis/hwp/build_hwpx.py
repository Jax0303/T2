"""thesis/src/*.md -> thesis/hwp/thesis_revised.hwpx, 사용자 초안(draft_original.hwpx, 학교 양식)의 문단·글자 서식을 그대로 복제한다.

  python3 thesis/hwp/build_hwpx.py [--toc toc_pages.json]

- 표지·제출서·인준서(section0), 초록(section1)은 초안 문단을 두고 글자만 바꾼다.
- 본문(section2)은 첫 문단(구역 정의 포함)만 남기고 md 에서 새로 만든다. 서식 번호는 초안에서 읽은 값:
  장 제목 paraPr30/charPr9(16pt 견명조, 가운데), 절 paraPr31/charPr5(13pt 굵은 신명조), 항 paraPr31/charPr7(11pt 굵은 중고딕),
  본문 paraPr40/charPr39(11pt 신명조, 첫 줄 들여쓰기), 그림·표 제목 paraPr41/charPr38(10pt 바탕, 가운데),
  표 칸 paraPr49 + charPr45(머리, 굵게)/44(본문), 칸 테두리 borderFill 10(머리)/9(본문), 참고문헌 paraPr42(내어쓰기).
- 그림은 글자처럼 취급(treatAsChar=1)해 문단 흐름을 따른다. linesegarray(줄 배치 캐시)는 넣지 않는다 — 한글이 다시 계산한다.
- --toc: 렌더링으로 얻은 쪽 번호({"제목": 쪽})로 목차·표 목차·그림 목차를 채운다. 없으면 쪽 번호 자리를 비운다.
"""
import argparse, json, re, struct, sys, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
THESIS = HERE.parent
SRC = THESIS / "src"
sys.path.insert(0, str(HERE))
from hwpx_lib import top_paras  # noqa: E402

DRAFT = HERE / "draft_original.hwpx"
OUT = HERE / "thesis_revised.hwpx"
TEXT_W = 36848                      # 본문 폭(HWPUNIT), 초안 lineseg horzsize
PX = 75                             # HWPUNIT / pixel (96dpi)
_id = [1500000000]


def nid():
    _id[0] += 1
    return _id[0]


def clean(t):
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return escape(t)


def para(text, ppr, cpr, page_break=False):
    run = f'<hp:run charPrIDRef="{cpr}"><hp:t>{clean(text)}</hp:t></hp:run>' if text else f'<hp:run charPrIDRef="{cpr}"/>'
    return (f'<hp:p id="2147483648" paraPrIDRef="{ppr}" styleIDRef="0" pageBreak="{1 if page_break else 0}" '
            f'columnBreak="0" merged="0">{run}</hp:p>')


def blank():
    return para("", 40, 39)


EM_ASCII = 0.55


def text_em(s):
    return sum(EM_ASCII if ord(c) < 128 else 1.0 for c in s)


def table(rows, wide):
    n = len(rows[0])
    size = 900
    margin = 200 if wide else 400
    need = [max(text_em(r[j]) for r in rows) * size * (1.05 if True else 1) + 2 * margin + 100 for j in range(n)]
    # 긴 글 열은 줄바꿈을 허용: 가장 긴 단어 기준 최소 폭
    minw = [max(max((text_em(w) for w in r[j].split()), default=1) * size + 2 * margin + 100 for r in rows) for j in range(n)]
    if sum(need) <= TEXT_W:
        w = [x + (TEXT_W - sum(need)) * x / sum(need) for x in need]
    else:
        base = sum(minw)
        extra = max(0, TEXT_W - base)
        want = [max(0, a - b) for a, b in zip(need, minw)]
        w = [b + (extra * a / sum(want) if sum(want) else 0) for a, b in zip(want, minw)]
        if base > TEXT_W:
            w = [x * TEXT_W / base for x in minw]
    w = [int(x) for x in w]
    w[-1] += TEXT_W - sum(w)
    import math

    def row_h(r):                            # 칸 안 줄 수를 어림해 행 높이를 정한다(줄 간격 125%)
        lines = max(math.ceil(max(text_em(c), 1) * size * 1.05 / max(w[j] - 2 * margin, 1)) for j, c in enumerate(r))
        return max(1182, lines * 1125 + 282 + 100)
    hs = [row_h(r) for r in rows]
    h = sum(hs)
    trs = []
    for i, r in enumerate(rows):
        tcs = []
        for j, c in enumerate(r):
            head = i == 0
            num = bool(re.fullmatch(r"[\d.,%×⁻⁰¹²³⁴⁵⁶⁷⁸⁹ :<>()\-/p=—]+", c.strip())) and not head
            tcs.append(
                f'<hp:tc name="" header="{1 if head else 0}" hasMargin="1" protect="0" editable="0" dirty="0" borderFillIDRef="{10 if head else 9}">'
                f'<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" '
                f'textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">'
                f'<hp:p id="2147483648" paraPrIDRef="49" styleIDRef="18" pageBreak="0" columnBreak="0" merged="0">'
                f'<hp:run charPrIDRef="{45 if head else 44}"><hp:t>{clean(c)}</hp:t></hp:run></hp:p></hp:subList>'
                f'<hp:cellAddr colAddr="{j}" rowAddr="{i}"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
                f'<hp:cellSz width="{w[j]}" height="{hs[i]}"/><hp:cellMargin left="{margin}" right="{margin}" top="141" bottom="141"/></hp:tc>')
        trs.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
    tbl = (f'<hp:tbl id="{nid()}" zOrder="{nid() % 1000}" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
           f'dropcapstyle="None" pageBreak="CELL" repeatHeader="1" rowCnt="{len(rows)}" colCnt="{n}" cellSpacing="0" borderFillIDRef="8" noAdjust="0">'
           f'<hp:sz width="{TEXT_W}" widthRelTo="ABSOLUTE" height="{h}" heightRelTo="ABSOLUTE" protect="0"/>'
           f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" vertRelTo="PARA" horzRelTo="PARA" '
           f'vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/><hp:outMargin left="0" right="0" top="0" bottom="0"/>'
           f'<hp:inMargin left="510" right="510" top="141" bottom="141"/>' + "".join(trs) + "</hp:tbl>")
    return (f'<hp:p id="2147483648" paraPrIDRef="41" styleIDRef="18" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="38">{tbl}<hp:t/></hp:run></hp:p>')


def png_size(path):
    b = Path(path).read_bytes()
    return struct.unpack(">II", b[16:24])


def picture(bin_id, path, max_w=TEXT_W):
    pw, ph = png_size(path)
    w = max_w
    hgt = int(w * ph / pw)
    if hgt > 30000:                          # 너무 긴 그림은 높이로 줄인다
        hgt = 30000; w = int(hgt * pw / ph)
    dw, dh = pw * PX, ph * PX
    pic = (f'<hp:pic id="{nid()}" zOrder="{nid() % 1000}" numberingType="PICTURE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
           f'dropcapstyle="None" href="" groupLevel="0" instid="{nid() % 100000000}" reverse="0"><hp:offset x="0" y="0"/>'
           f'<hp:orgSz width="{w}" height="{hgt}"/><hp:curSz width="{w}" height="{hgt}"/><hp:flip horizontal="0" vertical="0"/>'
           f'<hp:rotationInfo angle="0" centerX="{w // 2}" centerY="{hgt // 2}" rotateimage="1"/><hp:renderingInfo>'
           f'<hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/><hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
           f'<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/></hp:renderingInfo>'
           f'<hc:img binaryItemIDRef="{bin_id}" bright="0" contrast="0" effect="REAL_PIC" alpha="0"/>'
           f'<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="{w}" y="0"/><hc:pt2 x="{w}" y="{hgt}"/><hc:pt3 x="0" y="{hgt}"/></hp:imgRect>'
           f'<hp:imgClip left="0" right="{dw}" top="0" bottom="{dh}"/><hp:inMargin left="0" right="0" top="0" bottom="0"/>'
           f'<hp:imgDim dimwidth="{dw}" dimheight="{dh}"/><hp:effects/>'
           f'<hp:sz width="{w}" widthRelTo="ABSOLUTE" height="{hgt}" heightRelTo="ABSOLUTE" protect="0"/>'
           f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" vertRelTo="PARA" horzRelTo="PARA" '
           f'vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/><hp:outMargin left="0" right="0" top="0" bottom="0"/>'
           f'<hp:shapeComment>그림입니다.</hp:shapeComment></hp:pic>')
    return (f'<hp:p id="2147483648" paraPrIDRef="48" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="39">{pic}<hp:t/></hp:run></hp:p>')


def split_row(line):
    body = line.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", body)]


def body_from_md(files, images):
    """md 파일들 -> (문단 XML 목록, 목차 항목[(수준, 제목)], 표 제목, 그림 제목)"""
    out, toc, tabs, figs = [], [], [], []
    first_chapter = True
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        i = 0
        para_buf = []

        def flush():
            if para_buf:
                out.append(para(" ".join(para_buf), 40, 39)); para_buf.clear()
        while i < len(lines):
            t = lines[i].strip(); i += 1
            if not t:
                flush(); continue
            m = re.match(r"^(#{1,3}) (.*)$", t)
            if m:
                flush()
                lvl, text = len(m.group(1)), m.group(2)
                if lvl == 1:
                    if text.startswith("제 "):
                        if first_chapter:
                            first_chapter = False            # 제 1 장은 초안 첫 문단(구역 정의)을 쓴다
                            out.append("__FIRST_CHAPTER__")
                        else:
                            out.append(para(text, 30, 9, page_break=True))
                        out.append(para("", 30, 9))
                        toc.append((1, text))
                    elif text == "참고문헌":
                        out.append(para(text, 41, 46, page_break=True)); out.append(blank()); toc.append((0, text))
                    elif text == "부록":
                        out.append(para("부    록", 30, 8, page_break=True)); toc.append((0, "부    록"))
                    continue
                if text.startswith("부록 "):
                    out.append(para(text, 30, 9, page_break=True)); out.append(blank()); toc.append((2, text)); continue
                if out and not out[-1].endswith('<hp:run charPrIDRef="9"/></hp:p>'):
                    out.append(blank())
                out.append(para(text, 31, 5 if lvl == 2 else 7)); toc.append((lvl, text))
                continue
            m = re.match(r"^!\[(그림 (\d+-\d+))\. (.*)\]\((.*)\)$", t)
            if m:
                flush()
                bid = f"image{1 + len(images)}"
                images.append((bid, THESIS / m.group(4)))
                out.append(picture(bid, THESIS / m.group(4)))
                cap = f"[그림 {m.group(2)}] {m.group(3)}"
                out.append(para(cap, 41, 38)); figs.append(cap)
                continue
            m = re.match(r"^표 ([A-Z\d]+-\d+)\. (.*)$", t)
            if m:
                flush()
                cap = f"[표 {m.group(1)}] {m.group(2)}"
                out.append(para(cap, 41, 38)); tabs.append(cap)
                continue
            if t.startswith("|"):
                flush()
                block = [t]
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block.append(lines[i].strip()); i += 1
                rows = [split_row(l) for k, l in enumerate(block) if not re.match(r"^\|?\s*:?-{2,}", l.strip("|").strip())]
                rows = [[c.replace("\\|", "|") for c in r] for r in rows]
                out.append(table(rows, wide=len(rows[0]) >= 8))
                continue
            if t.startswith("> "):
                flush(); out.append(para(t[2:], 48, 39)); continue
            m = re.match(r"^- (.*)$", t)
            if m:
                flush(); out.append(para("⦁ " + m.group(1), 44, 39)); continue
            if re.match(r"^\[\d+\] ", t):
                flush(); out.append(para(t, 42, 39)); continue
            para_buf.append(t)
        flush()
    return out, toc, tabs, figs


def set_text(p, new):
    """문단의 첫 <hp:t> 에 new 를 넣고 나머지 <hp:t> 는 비운다(서식 run 유지)."""
    k = [0]

    def rep(m):
        k[0] += 1
        return f"<hp:t>{escape(new) if k[0] == 1 else ''}</hp:t>"
    return re.sub(r"<hp:t>[^<]*</hp:t>|<hp:t/>", rep, p, count=0)


def drop_linesegs(x):
    return re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", x, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toc", default="")
    a = ap.parse_args()
    pages = json.loads(Path(a.toc).read_text()) if a.toc else {}
    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    ko, en = meta["ko"]["title"], meta["en"]["title"]
    z = zipfile.ZipFile(DRAFT)
    files = {i.filename: z.read(i.filename) for i in z.infolist()}

    # ---------- section2: 본문 ----------
    x2 = files["Contents/section2.xml"].decode()
    bs, be, sp = top_paras(x2)
    first = x2[sp[0][0]:sp[0][1]]
    first = re.sub(r"<hp:t>[^<]*</hp:t>", "<hp:t>제 1 장 서 론</hp:t>", first, count=1)
    md = sorted(f for f in SRC.glob("*.md") if f.name[:2].isdigit() and not f.name.startswith("00"))
    images = []
    body, toc, tabs, figs = body_from_md(md, images)
    body = [first if b == "__FIRST_CHAPTER__" else b for b in body]
    x2 = x2[:bs] + "".join(body) + x2[be:]
    files["Contents/section2.xml"] = x2.encode()

    # ---------- section1: 초록 ----------
    abs_ko = (SRC / "00a_abstract_ko.md").read_text(encoding="utf-8").split("\n", 1)[1].strip()
    abs_en = (SRC / "00b_abstract_en.md").read_text(encoding="utf-8").split("\n", 1)[1].strip()
    x1 = files["Contents/section1.xml"].decode()
    bs, be, sp = top_paras(x1)
    ps = [x1[s:e] for s, e in sp]
    rep1 = {2: ko, 4: meta["ko"]["name"], 5: meta["ko"]["dept"], 8: abs_ko,
            11: en, 13: meta["en"]["name"], 14: meta["en"]["dept"], 17: abs_en}
    for k, v in rep1.items():
        ps[k] = set_text(ps[k], v)
    x1 = x1[:bs] + "".join(ps) + x1[be:]
    files["Contents/section1.xml"] = drop_linesegs(x1).encode() if False else x1.encode()

    # ---------- section0: 표지·인준서·목차 ----------
    x0 = files["Contents/section0.xml"].decode()
    for old in ["계층형 표 문서를 위한 셀 문장 검색 기반 검색 증강 생성"]:
        x0 = x0.replace(old, escape(ko))
    for old in ["Cell-Sentence Retrieval-Augmented Generation for Documents with Hierarchical Tables",
                "ell-Sentence Retrieval-Augmented Generation for Documents with Hierarchical Tables"]:
        x0 = x0.replace(old, escape(en))
    x0 = x0.replace("석(박)사학위 논문", "석사학위 논문")
    bs, be, sp = top_paras(x0)
    ps = [x0[s:e] for s, e in sp]
    # 박사 인준서(77~90) 삭제, 목차(99~117)·표 목차(120~124)·그림 목차(127~130) 다시 만들기
    toc_chap = ps[99]; toc_sec = ps[101]; toc_back = ps[116]; toc_tab = ps[120]

    def pg(title):
        v = pages.get(title)
        return str(v) if v is not None else ""

    def entry(template, label, page, indent=""):
        p = re.sub(r"<hp:t>[^<]*(<hp:tab [^>]*/>)[^<]*</hp:t>", lambda m: f"<hp:t>{escape(indent + label)} {m.group(1)}{page}</hp:t>", template, count=1)
        return p

    def chap_entry(label, page):
        p = toc_chap
        p = re.sub(r'(<hp:run charPrIDRef="5"><hp:t>)[^<]*(</hp:t>)', lambda m: m.group(1) + escape(label) + " " + m.group(2), p, count=1)
        p = re.sub(r'(<hp:run charPrIDRef="32"><hp:t>)[^<]*(</hp:t>)', lambda m: m.group(1) + page + m.group(2), p, count=1)
        return p
    new_toc = []
    for lvl, title in toc:
        if lvl == 1:
            new_toc.append(chap_entry(title, pg(title)))
        elif lvl in (2, 3) and not title.startswith("부록 "):
            new_toc.append(entry(toc_sec, title, pg(title), "  " if lvl == 2 else "    "))
        elif title.startswith("부록 "):
            new_toc.append(entry(toc_sec, title, pg(title), "  "))
        elif lvl == 0:
            new_toc.append(re.sub(r"(<hp:t>)[^<]*(</hp:t>)", lambda m: m.group(1) + escape(title) + " " + m.group(2), toc_back, count=1)
                           .replace(">29</hp:t>", f">{pg(title)}</hp:t>"))
    tab_entries = [entry(toc_tab, c, pg(c)) for c in tabs]
    fig_entries = [entry(toc_tab, c, pg(c)) for c in figs]
    for k, key, old in ((96, "국문초록", "ⅳ"), (97, "영문초록", "ⅴ")):
        if pg(key):
            ps[k] = ps[k].replace(old + "</hp:t>", pg(key) + "</hp:t>")
    ps = ps[:77] + ps[91:99] + new_toc + ps[118:120] + tab_entries + ps[124:127] + fig_entries
    x0 = x0[:bs] + "".join(ps) + x0[be:]
    files["Contents/section0.xml"] = x0.encode()

    # ---------- 그림 등록 ----------
    hpf = files["Contents/content.hpf"].decode()
    used = set(re.findall(r'binaryItemIDRef="([^"]+)"', x0 + x1 + x2))
    for old in re.findall(r'<opf:item id="(image\d+)" href="(BinData/[^"]+)"[^>]*/>', hpf):
        if old[0] not in used:
            hpf = re.sub(rf'<opf:item id="{old[0]}" [^>]*/>', "", hpf); files.pop(old[1], None)
    items = "".join(f'<opf:item id="{b}" href="BinData/{b}.png" media-type="image/png" isEmbeded="1"/>' for b, _ in images)
    hpf = hpf.replace('<opf:item id="section2"', items + '<opf:item id="section2"')
    files["Contents/content.hpf"] = hpf.encode()
    for b, pth in images:
        files[f"BinData/{b}.png"] = Path(pth).read_bytes()

    with zipfile.ZipFile(OUT, "w") as zo:
        zo.writestr(zipfile.ZipInfo("mimetype"), files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name, data in files.items():
            zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    json.dump({"toc": toc, "tables": tabs, "figures": figs}, open(HERE / "toc_items.json", "w"), ensure_ascii=False, indent=1)
    print("wrote", OUT, "paragraphs", len(body), "images", len(images), "tables", len(tabs))


if __name__ == "__main__":
    main()
