"""hwpx 표 만들기 — 원고의 기존 표 문단(<hp:p>…<hp:tbl>…</hp:tbl>…</hp:p>)을 틀로 삼아 행·열 수와 글만 바꾼 새 표 문단을 만든다.
틀의 0행 셀 = 머리글 셀 모양(borderFill·문단·글자 모양), 1행 셀 = 본문 셀 모양. linesegarray 는 지워 한글이 다시 배치하게 한다.
  from hwpx_table import make_table
  xml = make_table(template_para_xml, rows=[["방법","맞힘","%"], ["제안 방법","620","70.3"], ...], col_widths=[12000, 6000, 6000])
"""
import re

_LSA = re.compile(r"<hp:linesegarray>.*?</hp:linesegarray>", re.S)
_TC = re.compile(r"<hp:tc\b.*?</hp:tc>", re.S)
_TR = re.compile(r"<hp:tr>.*?</hp:tr>", re.S)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _set_text(cell_xml, text):
    """셀 안 문단의 <hp:t> 를 하나로 만들고 글을 바꾼다(여러 run 이면 첫 run 만 남긴다)."""
    cell_xml = _LSA.sub("", cell_xml)
    runs = re.findall(r"<hp:run\b.*?</hp:run>", cell_xml, re.S)
    assert runs, "셀에 run 이 없다"
    first = runs[0]
    new_first = re.sub(r"<hp:t(?: [^>]*)?>.*?</hp:t>", f"<hp:t>{_esc(text)}</hp:t>", first, count=1, flags=re.S) if "<hp:t" in first \
        else first.replace("</hp:run>", f"<hp:t>{_esc(text)}</hp:t></hp:run>")
    for r in runs[1:]:
        cell_xml = cell_xml.replace(r, "", 1)
    return cell_xml.replace(first, new_first, 1)


def make_table(template, rows, col_widths=None, header_rows=1, row_height=None):
    n_rows, n_cols = len(rows), len(rows[0])
    assert all(len(r) == n_cols for r in rows), "행마다 열 수가 다르다"
    head = re.match(r"(<hp:p [^>]*>)", template).group(1)
    run_open = re.search(r"<hp:run [^>]*>", template).group(0)
    tbl_open = re.search(r"<hp:tbl\b[^>]*>", template).group(0)
    tbl_body = template[template.index(tbl_open) + len(tbl_open):template.index("</hp:tbl>")]
    pre = tbl_body[:tbl_body.index("<hp:tr>")]                      # <hp:sz/> <hp:pos/> <hp:outMargin/> <hp:inMargin/>
    trs = _TR.findall(tbl_body)
    cells0 = _TC.findall(trs[0]); cells1 = _TC.findall(trs[1]) if len(trs) > 1 else cells0
    t_head, t_body = cells0[0], cells1[0]
    total_w = int(re.search(r'<hp:sz width="(\d+)"', pre).group(1))
    if col_widths is None:
        col_widths = [total_w // n_cols] * n_cols
        col_widths[-1] += total_w - sum(col_widths)
    if sum(col_widths) != total_w:                                  # 상대 너비로 받아 표 너비에 맞춰 비례 조정
        s = sum(col_widths); col_widths = [int(w * total_w / s) for w in col_widths]; col_widths[-1] += total_w - sum(col_widths)
    rh = row_height or int(re.search(r'<hp:cellSz width="\d+" height="(\d+)"', t_body).group(1))
    out_rows = []
    for r, row in enumerate(rows):
        tcs = []
        for c, text in enumerate(row):
            t = t_head if r < header_rows else t_body
            t = re.sub(r'<hp:cellAddr colAddr="\d+" rowAddr="\d+"/>', f'<hp:cellAddr colAddr="{c}" rowAddr="{r}"/>', t)
            t = re.sub(r'<hp:cellSz width="\d+" height="\d+"/>', f'<hp:cellSz width="{col_widths[c]}" height="{rh}"/>', t)
            t = re.sub(r'<hp:cellSpan colSpan="\d+" rowSpan="\d+"/>', '<hp:cellSpan colSpan="1" rowSpan="1"/>', t)
            tcs.append(_set_text(t, text))
        out_rows.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
    tbl_open2 = re.sub(r'rowCnt="\d+"', f'rowCnt="{n_rows}"', tbl_open)
    tbl_open2 = re.sub(r'colCnt="\d+"', f'colCnt="{n_cols}"', tbl_open2)
    pre2 = re.sub(r'(<hp:sz width="\d+" widthRelTo="ABSOLUTE" height=")\d+(")', lambda m: f"{m.group(1)}{rh * n_rows}{m.group(2)}", pre)
    return f"{head}{run_open}{tbl_open2}{pre2}{''.join(out_rows)}</hp:tbl></hp:run></hp:p>"
