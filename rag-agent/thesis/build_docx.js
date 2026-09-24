// thesis/src/*.md -> thesis/thesis.docx
// 지원하는 Markdown: # ## ### 제목, 문단, "- " 목록, "1. " 번호 목록, "> " 들여쓴 문단,
// 파이프 표(바로 앞 줄 "표 X-Y. ..."는 표 제목), ![그림 X-Y. 설명](경로), **굵게**, `코드`.
//   NODE_PATH=<docx 가 설치된 node_modules> node thesis/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, PageBreak, Footer, PageNumber, TableOfContents,
  LevelFormat, ImageRun, VerticalAlign,
} = require("docx");

const DIR = __dirname;
const SRC = path.join(DIR, "src");
const BODY = "Batang", HEAD = "Malgun Gothic", LATIN = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1701;              // A4, 좌우 30mm
const TEXT_W = PAGE_W - 2 * MARGIN;

function runs(text, base = {}) {
  // **굵게** 와 `코드` 만 해석한다
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: { ascii: "Consolas", hAnsi: "Consolas", eastAsia: BODY }, ...base }));
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

function table(lines) {
  const rows = lines.filter((l, i) => !(i === 1 && /^\|?\s*:?-{2,}/.test(l.trim()))).map(splitRow);
  const align = splitRow(lines[1]).map((a) => (a.endsWith(":") ? AlignmentType.RIGHT : AlignmentType.LEFT));
  const n = rows[0].length;
  // 열 너비: 글자 수에 비례, 최소 폭 보장
  const len = Array.from({ length: n }, (_, j) => Math.max(...rows.map((r) => Math.min((r[j] || "").length, 60)), 4));
  const tot = len.reduce((a, b) => a + b, 0);
  let widths = len.map((l) => Math.max(900, Math.round((TEXT_W * l) / tot)));
  const scale = TEXT_W / widths.reduce((a, b) => a + b, 0);
  widths = widths.map((w) => Math.floor(w * scale));
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
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
          children: runs(c, { size: 17, bold: i === 0 ? true : undefined }),
        })],
      })),
    })),
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true,
    spacing: { before: 200, after: 100 },
    children: runs(text, { bold: true, size: 20 }),
  });
}

function convert(md) {
  const out = [];
  const lines = md.split(/\r?\n/);
  let para = [];
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
    if (t === "<!-- pagebreak -->") { flush(); out.push(new Paragraph({ children: [new PageBreak()] })); continue; }
    let m;
    if ((m = /^(#{1,3}) (.*)$/.exec(t))) {
      flush();
      const lvl = m[1].length;
      out.push(new Paragraph({
        heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][lvl - 1],
        pageBreakBefore: lvl === 1,
        children: [new TextRun(m[2])],
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
      out.push(caption(m[1]));
      continue;
    }
    if (t.startsWith("|")) {
      flush();
      const block = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) block.push(lines[i++]);
      i--;
      out.push(table(block));
      out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    if (/^표 \d+-\d+\./.test(t) || /^표 [A-Z]-\d+\./.test(t)) { flush(); out.push(caption(t)); continue; }
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

function titlePage() {
  const meta = JSON.parse(fs.readFileSync(path.join(SRC, "meta.json"), "utf8"));
  const p = (text, size, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: opts.before || 0, after: opts.after || 0 },
    children: [new TextRun({ text, size, bold: opts.bold, font: { ascii: LATIN, hAnsi: LATIN, eastAsia: HEAD } })] });
  return [
    p(meta.degree, 28, { before: 1200, after: 1600 }),
    p(meta.title, 40, { bold: true, after: 300 }),
    p(meta.title_en, 24, { after: 2400 }),
    ...meta.lines.map((l) => p(l, 26, { after: 200 })),
  ];
}

const files = fs.readdirSync(SRC).filter((f) => f.endsWith(".md")).sort();
const body = [];
for (const f of files) {
  if (f.includes("_toc")) {
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 360 },
      children: [new TextRun({ text: "목차", bold: true, size: 32, font: { ascii: LATIN, hAnsi: LATIN, eastAsia: HEAD } })] }));
    body.push(new TableOfContents("목차", { hyperlink: true, headingStyleRange: "1-2" }));
    continue;
  }
  body.push(...convert(fs.readFileSync(path.join(SRC, f), "utf8")));
}

const doc = new Document({
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: { ascii: LATIN, hAnsi: LATIN, eastAsia: BODY }, size: 22 },
                           paragraph: { spacing: { line: 384, after: 60 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: { ascii: LATIN, hAnsi: LATIN, eastAsia: HEAD } },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 360 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: { ascii: LATIN, hAnsi: LATIN, eastAsia: HEAD } },
        paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: { ascii: LATIN, hAnsi: LATIN, eastAsia: HEAD } },
        paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 280 } } } }] }] },
  sections: [
    { properties: { page: { margin: { top: 1701, bottom: 1417, left: MARGIN, right: MARGIN } } }, children: titlePage() },
    { properties: { page: { margin: { top: 1701, bottom: 1417, left: MARGIN, right: MARGIN }, pageNumbers: { start: 1 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
      children: body },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(DIR, "thesis.docx"), buf);
  console.log("wrote", path.join(DIR, "thesis.docx"), buf.length, "bytes");
});
