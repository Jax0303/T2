import re, zipfile
P_OPEN = re.compile(r"<hp:p[ >]"); P_CLOSE = "</hp:p>"
def top_paras(xml):
    """(start,end) spans of top-level <hp:p> elements directly under <hs:sec>."""
    body_start = xml.index(">", xml.index("<hs:sec")) + 1
    body_end = xml.rindex("</hs:sec>")
    spans = []; i = body_start
    while True:
        m = P_OPEN.search(xml, i)
        if not m or m.start() >= body_end: break
        depth = 0; j = m.start()
        tok = re.compile(r"<hp:p[ >]|</hp:p>")
        for t in tok.finditer(xml, j):
            if t.group().startswith("</"): depth -= 1
            else: depth += 1
            if depth == 0:
                spans.append((m.start(), t.end())); i = t.end(); break
    return body_start, body_end, spans
def text_of(p):
    return "".join(re.findall(r"<hp:t(?: [^>]*)?>([^<]*)</hp:t>", p))
def attrs(p):
    m = re.match(r'<hp:p [^>]*paraPrIDRef="(\d+)" styleIDRef="(\d+)"', p)
    ch = re.findall(r'charPrIDRef="(\d+)"', p[:400])
    return (m.group(1), m.group(2)) if m else None, ch[:2]
