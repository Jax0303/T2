// thesis/src/*.md -> thesis/thesis.docx  (학교 학위논문양식틀(국문, A4) 서식, 표지·제출서·인준서는 HWP 양식에서 가져온다)
// 지원하는 Markdown: # ## ### 제목, 문단, "- " 목록, "1. " 번호 목록, "> " 들여쓴 문단,
// 파이프 표(바로 앞 줄 "표 X-Y. ..."는 표 제목), ![그림 X-Y. 설명](경로), **굵게**, *기울임*, `코드`,
// 각주 [^N](본문 표시)과 "[^N]: 내용"(한 줄 정의).
//   NODE_PATH=<docx 가 설치된 node_modules> node thesis/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, PageBreak, Footer, PageNumber, TableOfContents,
  LevelFormat, ImageRun, VerticalAlign, StyleLevel, PageOrientation, TableLayoutType, FootnoteReferenceRun,
} = require("docx");

const DIR = __dirname;
const SRC = path.join(DIR, "src");
const MYEONG = "한양신명조", GYEON = "한양견명조", GOTHIC = "한양중고딕";   // 양식과 같은 글꼴 이름, 한글·영문 공통
const font = (name) => ({ ascii: name, hAnsi: name, eastAsia: name });
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 2268, HF = 567;   // A4, 여백 사방 40mm, 머리말·꼬리말 10mm
const TEXT_W = PAGE_W - 2 * MARGIN;                               // 본문 폭 130mm
const LAND_W = PAGE_H - 2 * MARGIN;                               // 가로 쪽 본문 폭 217mm
const LANDSCAPE = new Set(["5-7"]);                               // 가로 쪽 구역에 넣는 표
const LAND = { orientation: PageOrientation.LANDSCAPE }, PORT = { orientation: PageOrientation.PORTRAIT };   // 구역 나눔 표시

function runs(text, base = {}) {
  // **굵게**, *기울임*, `코드`, 각주 표시 [^N] 만 해석한다
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[\^\d+\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("[^")) out.push(new FootnoteReferenceRun(Number(t.slice(2, -1))));
    else if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: { ascii: "Consolas", hAnsi: "Consolas", eastAsia: MYEONG }, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const cellText = (s) => s.trim().replace(/\\\|/g, "|");

function splitRow(line) {
  // 이스케이프한 \| 는 칸 구분이 아니다
  const parts = [];
  let cur = "";
  const body = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  for (let i = 0; i < body.length; i++) {
    if (body[i] === "\\" && body[i + 1] === "|") { cur += "\\|"; i++; continue; }
    if (body[i] === "|") { parts.push(cur); cur = ""; continue; }
    cur += body[i];
  }
  parts.push(cur);
  return parts.map(cellText);
}

// 글자 폭(em): HY신명조(한양신명조 계열, Windows H2MJSM.TTF) 실측. ASCII 32~126 은 아래 표, 그 밖(한글·위첨자·×·−·—)은 1
const ASCII_EM = (".33,.42,.42,.83,.62,.92,.83,.25,.5,.5,.5,.83,.29,.83,.29,.38,.62,.62,.62,.62,.62,.62,.62,.62,.62,.62,.33,.33,.6,.83,.6,.5,1,"
  + ".79,.71,.71,.75,.71,.67,.75,.79,.38,.5,.79,.67,.92,.79,.75,.67,.75,.71,.67,.79,.79,.75,1,.71,.71,.67,.5,.38,.5,.5,.5,.25,"
  + ".54,.58,.54,.58,.58,.38,.58,.58,.29,.33,.58,.29,.88,.58,.58,.58,.58,.46,.54,.38,.58,.58,.83,.62,.62,.5,.58,.58,.58,.75").split(",").map(Number);
const em = (ch) => (ch >= " " && ch <= "~" ? ASCII_EM[ch.charCodeAt(0) - 32] : 1);
const sum = (a) => a.reduce((x, y) => x + y, 0);

function minColWidth(cells, size) {
  // 칸 안에서 줄을 바꿀 수 없는 가장 긴 단어의 폭(twip, 굵게는 1.05배) + 칸 좌우 여백 160
  const word = (c, bold) => Math.max(0, ...c.replace(/\*\*|`/g, "").split(/\s+/).map((w) => sum([...w].map(em))))
    * size * 10 * (bold ? 1.05 : 1);
  return Math.ceil(Math.max(...cells.map(([c, bold]) => word(c, bold)))) + 160;
}

function table(lines, textW, name) {
  const rows = lines.filter((l, i) => !(i === 1 && /^\|?\s*:?-{2,}/.test(l.trim()))).map(splitRow);
  const align = splitRow(lines[1]).map((a) => (a.endsWith(":") ? AlignmentType.RIGHT : AlignmentType.LEFT));
  const n = rows[0].length;
  const size = 17;
  // 열 너비(고정): 가장 긴 단어 폭을 먼저 주고, 남는 폭은 글자 수에 비례해 나눈다. HWP로 옮겨도 이 폭이 유지되게 고정 배치
  let need = Array.from({ length: n }, (_, j) => minColWidth(rows.map((r, i) => [r[j] || "", i === 0 || (r[j] || "").includes("**")]), size));
  if (sum(need) + 40 * n <= textW) need = need.map((w) => w + 40);      // 글꼴 차이에 대비한 여유 — 본문 폭을 넘기면 넣지 않는다
  const len = Array.from({ length: n }, (_, j) => Math.max(...rows.map((r) => Math.min((r[j] || "").length, 60)), 4));
  const extra = textW - sum(need);
  if (extra < 0) console.warn(`${name}: 열 최소 폭 합 ${(sum(need) / 56.7).toFixed(1)}mm > 본문 폭 ${(textW / 56.7).toFixed(1)}mm`);
  const tableW = Math.max(textW, sum(need));
  const widths = need.map((w, j) => Math.floor(w + (Math.max(0, extra) * len[j]) / sum(len)));
  widths[n - 1] += tableW - sum(widths);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: tableW, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders,
        verticalAlign: VerticalAlign.CENTER,
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined,
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({
          alignment: i === 0 ? AlignmentType.CENTER : align[j] || AlignmentType.LEFT,
          spacing: { line: 260, before: 0, after: 0 },
          children: runs(c, { size, bold: i === 0 ? true : undefined }),
        })],
      })),
    })),
  });
}

function caption(text, style) {
  // "표 4-1. 설명" -> "[표 4-1] 설명". 캡션 스타일이 표 목차·그림 목차 항목이 된다
  return new Paragraph({ style, children: runs(text.replace(/^(표|그림) ([A-Z\d]+-\d+)\. /, "[$1 $2] ")) });
}

const footnotes = {};   // [^N]: 정의 → Document 각주

function convert(md) {
  const out = [];
  const lines = md.split(/\r?\n/);
  let para = [], wide = false, capId = "";
  const flush = () => {
    if (!para.length) return;
    out.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 400 },
      children: runs(para.join(" ")),
    }));
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.trim();
    if (!t) { flush(); continue; }
    const fn = /^\[\^(\d+)\]: (.*)$/.exec(t);      // 각주 정의 줄은 본문에 넣지 않는다
    if (fn) { footnotes[fn[1]] = { children: [new Paragraph({ children: runs(fn[2], { size: 18 }) })] }; continue; }
    if (t === "<!-- pagebreak -->") { flush(); out.push(new Paragraph({ children: [new PageBreak()] })); continue; }
    let m;
    if ((m = /^(#{1,3}) (.*)$/.exec(t))) {
      flush();
      const lvl = m[1].length, text = m[2].replace(/^제(\d+)장/, "제 $1 장");
      const back = lvl === 1 && !text.startsWith("제 ");        // 참고문헌·부록
      out.push(new Paragraph({
        ...(back ? { style: "BackHeading" } : { heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][lvl - 1] }),
        pageBreakBefore: lvl === 1,
        children: [new TextRun(text)],
      }));
      continue;
    }
    if ((m = /^!\[(.*)\]\((.*)\)$/.exec(t))) {
      flush();
      const img = fs.readFileSync(path.join(DIR, m[2]));
      const w = img.readUInt32BE(16), h = img.readUInt32BE(20);   // PNG IHDR
      const width = 560, height = Math.round((560 * h) / w);
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
        children: [new ImageRun({ type: "png", data: img, transformation: { width, height } })] }));
      out.push(caption(m[1], "FigureCaption"));
      continue;
    }
    if (t.startsWith("|")) {
      flush();
      const block = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) block.push(lines[i++]);
      i--;
      out.push(table(block, wide ? LAND_W : TEXT_W, `표 ${capId}`));
      out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      if (wide) { out.push(PORT); wide = false; }
      continue;
    }
    if ((m = /^표 ([A-Z\d]+-\d+)\./.exec(t))) {
      flush();
      capId = m[1];
      if (LANDSCAPE.has(m[1])) { out.push(LAND); wide = true; }
      out.push(caption(t, "TableCaption"));
      continue;
    }
    if (t.startsWith("> ")) {
      flush();
      out.push(new Paragraph({ indent: { left: 700, right: 400 }, spacing: { before: 80, after: 80 },
        children: runs(t.slice(2)) }));
      continue;
    }
    if ((m = /^- (.*)$/.exec(t))) {
      flush();
      out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
        children: runs(m[1]) }));
      continue;
    }
    if ((m = /^(\d+)\. (.*)$/.exec(t))) {
      flush();
      out.push(new Paragraph({ indent: { left: 500, hanging: 300 }, alignment: AlignmentType.JUSTIFIED,
        children: runs(`${m[1]}. ${m[2]}`) }));
      continue;
    }
    para.push(t);
  }
  flush();
  return out;
}

const meta = JSON.parse(fs.readFileSync(path.join(SRC, "meta.json"), "utf8"));

function listTitle(text) {
  // 목차·표 목차·그림 목차 제목 (목차에 넣지 않는다)
  return new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: body.length > 0, spacing: { after: 360 },
    children: [new TextRun({ text, bold: true, size: 32, font: font(GOTHIC) })] });
}

function abstractHead(label, m) {
  // 국문초록·ABSTRACT 머리: 제목 16pt 굵게, 이름·학과·대학원 14pt
  const p = (text, size, bold, after = 0) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after },
    children: [new TextRun({ text, size, bold })] });
  return [new Paragraph({ style: "AbstractHeading", pageBreakBefore: true, children: [new TextRun(label)] }),
    p(m.title, 32, true, 400), p(m.name, 28), p(m.dept, 28), p(m.school, 28, false, 400)];
}

const files = fs.readdirSync(SRC).filter((f) => f.endsWith(".md")).sort();
const mds = files.map((f) => fs.readFileSync(path.join(SRC, f), "utf8"));
const body = [];
files.forEach((f, k) => {
  if (f.includes("_toc")) {
    body.push(listTitle("목   차"), new TableOfContents("목차", { hyperlink: true, headingStyleRange: "1-3",
      stylesWithLevels: [new StyleLevel("AbstractHeading", 4), new StyleLevel("BackHeading", 4)] }));
    body.push(listTitle("표  목  차"), new TableOfContents("표 목차", { hyperlink: true, stylesWithLevels: [new StyleLevel("TableCaption", 9)] }));
    if (mds.some((md) => /^!\[/m.test(md)))
      body.push(listTitle("그 림 목 차"), new TableOfContents("그림 목차", { hyperlink: true, stylesWithLevels: [new StyleLevel("FigureCaption", 9)] }));
    return;
  }
  if (f.includes("_abstract_")) {
    const en = f.includes("_en");
    body.push(...abstractHead(en ? "ABSTRACT" : "국문초록", en ? meta.en : meta.ko), ...convert(mds[k].replace(/^# .*\n/, "")));
    return;
  }
  body.push(...convert(mds[k]));
});

// 표 5-7 앞뒤에서 구역을 나눈다(가로 쪽). 쪽 번호는 이어진다
const pageProps = ({ orientation }) => ({ page: { size: { orientation },
  margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, header: HF, footer: HF } } });
const sections = [{ properties: { ...pageProps(PORT), page: { ...pageProps(PORT).page, pageNumbers: { start: 1 } } },
  footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
  children: [] }];
for (const x of body) {
  if (x === LAND || x === PORT) sections.push({ properties: pageProps(x), children: [] });
  else sections[sections.length - 1].children.push(x);
}

const captionStyle = { basedOn: "Normal", next: "Normal", run: { size: 20, bold: true },
  paragraph: { alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 } } };
const toc = (n, run, left = 0) => ({ id: `TOC${n}`, name: `toc ${n}`, basedOn: "Normal", next: "Normal", run, paragraph: { indent: { left } } });
const doc = new Document({
  features: { updateFields: true },
  footnotes,
  styles: {
    default: { document: { run: { font: font(MYEONG), size: 22 },
                           paragraph: { spacing: { line: 480, after: 60 } } } },   // 11pt, 줄간격 배수 2.0
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, font: font(GYEON) },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 360 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: font(MYEONG) },
        paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: font(GOTHIC) },
        paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 2 } },
      { id: "BackHeading", name: "BackHeading", basedOn: "Normal", next: "Normal",          // 참고문헌·부록
        run: { size: 32, bold: true, font: font(GOTHIC) },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 360 }, outlineLevel: 0 } },
      { id: "AbstractHeading", name: "AbstractHeading", basedOn: "Normal", next: "Normal",  // 국문초록·ABSTRACT
        run: { size: 28, bold: true },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 360 }, outlineLevel: 0 } },
      { id: "TableCaption", name: "TableCaption", ...captionStyle },
      { id: "FigureCaption", name: "FigureCaption", ...captionStyle },
      // 목차: 장 13pt 굵게, 중·소제목과 초록·참고문헌·부록 11pt. 표·그림 목차(9단계) 11pt 중고딕
      toc(1, { size: 26, bold: true }), toc(2, {}, 400), toc(3, {}, 800), toc(4, {}), toc(9, { font: font(GOTHIC) }),
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 280 } } } }] }] },
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(DIR, "thesis.docx"), buf);
  console.log("wrote", path.join(DIR, "thesis.docx"), buf.length, "bytes");
});
