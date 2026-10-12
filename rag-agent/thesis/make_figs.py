"""논문 그림 (그림 1-1, 1-2, 3-1~3-4, 4-1~4-3). 그림 4-x 수치는 results/ 파일에서 계산한다.
  .venv/bin/python thesis/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle, Wedge

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "fig"
BLUE, INK, MUTED, FILL = "#2a78d6", "#0b0b0b", "#52514e", "#f0efec"
LIGHT, MID = "#e3eefb", "#c6dbf5"          # 강조색 옅은 채움 / 조금 진한 채움
ORANGE, LIGHT_OR = "#c05a00", "#fbe9dc"
LINE = "#9a9893"                            # 얇은 회색 테두리
MONO = "DejaVu Sans Mono"
FONT = "/mnt/c/Windows/Fonts/H2GTRM.TTF"    # HY중고딕

font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
                     "font.size": 8, "axes.unicode_minus": False, "text.color": INK})


def canvas(w, h):
    """1 단위 = 1인치인 빈 도화지."""
    fig = plt.figure(figsize=(w, h), dpi=200)   # 저장 dpi 와 같아야 글자 폭 측정이 맞다
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, x, y, w, h, fc="white", ec=LINE, lw=0.8, r=0.06, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, pts, color=MUTED, ls="-", lw=0.9):
    """꺾인 선 화살표. pts 마지막 점에 화살촉."""
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, ls=ls, lw=lw, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=color,
                                lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def cw(fig, fs):
    """고정폭 글꼴 한 글자 진행폭(인치). 힌팅으로 반올림된 실제 폭을 렌더러로 잰다(0 200개 - 100개)."""
    r = fig.canvas.get_renderer()
    ws = []
    for n in (100, 200):
        t = fig.text(0, 0, "0" * n, family=MONO, fontsize=fs)
        ws.append(t.get_window_extent(r).width)
        t.remove()
    return (ws[1] - ws[0]) / 100 / fig.dpi


def mono(ax, x, y, s, fs, w, hi=None, **kw):
    """고정폭 한 줄. hi 부분 문자열은 강조 칸 + 굵게."""
    if hi is None:
        ax.text(x, y, s, ha="left", va="center", family=MONO, fontsize=fs, **kw)
        return None
    i = s.index(hi)
    ax.text(x, y, s[:i] + " " * len(hi) + s[i + len(hi):], ha="left", va="center", family=MONO, fontsize=fs, **kw)
    hx = x + i * w
    h = fs / 72 * 1.45
    ax.add_patch(Rectangle((hx - 0.025, y - h / 2), len(hi) * w + 0.05, h, fc=MID, ec=BLUE, lw=1.0))
    ax.text(hx, y, hi, ha="left", va="center", family=MONO, fontsize=fs, weight="bold")
    return hx + len(hi) * w / 2


def grid(ax, cells):
    """cells: (x, y, w, h, text, fc, ec, lw, fontkw)"""
    for x, y, w, h, t, fc, ec, lw, kw in cells:
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
        if t:
            ha = kw.pop("ha", "center")
            tx = x + 0.07 + kw.pop("indent", 0) if ha == "left" else x + w / 2
            ax.text(tx, y + h / 2, t, ha=ha, va="center", **kw)


# ---------------------------------------------------------------- 그림 1-1
def fig1_1_hier_table():
    fig, ax = canvas(7.5, 2.0)
    x0, top, rh = 0.1, 1.9, 0.28
    cws = [0.8, 0.8, 0.72, 0.72, 0.72, 0.72]
    xs = [x0 + sum(cws[:i]) for i in range(len(cws) + 1)]
    ys = [top - rh * (i + 1) for i in range(6)]          # 행 i 의 아래 y
    rows = [("Revenue", "Product", [1120, 740, 1250, 860]), ("Revenue", "Service", [530, 310, 590, 340]),
            ("Cost", "Product", [780, 460, 820, 510]), ("Cost", "Service", [290, 150, 320, 170])]
    on_path = {("row", "Revenue"), ("row0", "Product"), ("col", "2019"), ("sub", 3)}
    c = [(xs[0], ys[1], cws[0] + cws[1], 2 * rh, "", FILL, LINE, 0.6, {})]
    for j, yr in enumerate(["2018", "2019"]):
        fc = LIGHT if ("col", yr) in on_path else FILL
        c.append((xs[2 + 2 * j], ys[0], cws[2] * 2, rh, yr, fc, LINE, 0.6, {}))
        for k, sub in enumerate(["Domestic", "Foreign"]):
            fc = LIGHT if ("sub", 2 * j + k) in on_path else FILL
            c.append((xs[2 + 2 * j + k], ys[1], cws[2], rh, sub, fc, LINE, 0.6, {}))
    for g in range(2):
        grp = rows[2 * g][0]
        fc = LIGHT if ("row", grp) in on_path else FILL
        c.append((xs[0], ys[2 + 2 * g + 1], cws[0], 2 * rh, grp, fc, LINE, 0.6, {}))
    for i, (grp, sub, vals) in enumerate(rows):
        fc = LIGHT if (i == 0) else FILL
        c.append((xs[1], ys[2 + i], cws[1], rh, sub, fc, LINE, 0.6, {}))
        for j, v in enumerate(vals):
            hit = (i == 0 and j == 3)
            c.append((xs[2 + j], ys[2 + i], cws[2], rh, str(v), MID if hit else "white",
                      LINE, 0.6, {} if hit else {"color": MUTED}))
    grid(ax, c)
    # 대상 셀 테두리를 위에 한 번 더
    tx, ty = xs[5], ys[2]
    ax.add_patch(Rectangle((tx, ty), cws[5], rh, fc="none", ec=BLUE, lw=1.4))

    # 오른쪽 설명
    ax_x = 5.05
    arrow(ax, [(tx + cws[5] + 0.03, ty + rh / 2), (ax_x - 0.12, ty + rh / 2)], color=BLUE)
    lines = [("행 머리글 경로", "Revenue > Product"), ("열 머리글 경로", "2019 > Foreign"), ("셀 값", "860")]
    yc = ty + rh / 2
    for k, (a, b) in enumerate(lines):
        y = yc + (1 - k) * 0.24
        ax.text(ax_x, y, a, ha="left", va="center", color=MUTED)
        ax.text(ax_x + 0.95, y, b, ha="left", va="center")
    bx = ax_x - 0.08                                     # 세 줄을 묶는 괄호
    ax.plot([bx + 0.05, bx, bx, bx + 0.05], [yc + 0.36, yc + 0.36, yc - 0.36, yc - 0.36], color=BLUE, lw=0.8)
    yb = yc - 0.36 - 0.06
    arrow(ax, [(ax_x + 1.1, yb), (ax_x + 1.1, yb - 0.16)], color=MUTED)
    box(ax, ax_x - 0.1, yb - 0.52, 2.4, 0.34, fc=LIGHT, ec=BLUE, lw=1.1)
    ax.text(ax_x + 1.1, yb - 0.35, "2019년 해외 제품 매출 = 860", ha="center", va="center", fontsize=9)
    save(fig, "fig1_1_hier_table.png")


# ---------------------------------------------------------------- 그림 1-2
def glyph(ax, kind, cx, cy, c=MUTED):
    lw = 0.9
    if kind == "doc":
        ax.add_patch(Rectangle((cx - 0.08, cy - 0.1), 0.16, 0.2, fc="white", ec=c, lw=lw))
        for k in range(3):
            ax.plot([cx - 0.05, cx + 0.05], [cy + 0.05 - k * 0.05] * 2, color=c, lw=0.7)
    elif kind == "embed":
        L = [(cx - 0.09, cy + d) for d in (-0.07, 0, 0.07)]
        R = [(cx + 0.09, cy + d) for d in (-0.035, 0.035)]
        for p in L:
            for q in R:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=0.5)
        for p in L + R:
            ax.add_patch(Circle(p, 0.022, fc="white", ec=c, lw=lw, zorder=3))
    elif kind == "db":
        ax.add_patch(Rectangle((cx - 0.09, cy - 0.08), 0.18, 0.16, fc="white", ec="none"))
        ax.plot([cx - 0.09] * 2, [cy - 0.08, cy + 0.08], color=c, lw=lw)
        ax.plot([cx + 0.09] * 2, [cy - 0.08, cy + 0.08], color=c, lw=lw)
        for yy in (cy - 0.08, cy):
            ax.add_patch(matplotlib.patches.Arc((cx, yy), 0.18, 0.06, theta1=180, theta2=360, color=c, lw=lw))
        ax.add_patch(Ellipse((cx, cy + 0.08), 0.18, 0.06, fc="white", ec=c, lw=lw))
    elif kind == "user":
        ax.add_patch(Circle((cx, cy + 0.05), 0.045, fc="white", ec=c, lw=lw))
        ax.add_patch(Wedge((cx, cy - 0.1), 0.09, 0, 180, fc="white", ec=c, lw=lw))
    elif kind == "search":
        ax.add_patch(Circle((cx - 0.02, cy + 0.02), 0.06, fc="white", ec=c, lw=lw * 1.3))
        ax.plot([cx + 0.023, cx + 0.08], [cy - 0.023, cy - 0.08], color=c, lw=lw * 1.8, solid_capstyle="round")
    elif kind == "llm":
        ax.add_patch(Rectangle((cx - 0.07, cy - 0.07), 0.14, 0.14, fc="white", ec=c, lw=lw))
        ax.add_patch(Rectangle((cx - 0.03, cy - 0.03), 0.06, 0.06, fc=c, ec="none"))
        for d in (-0.035, 0.035):
            for s in (-1, 1):
                ax.plot([cx + d] * 2, [cy + s * 0.07, cy + s * 0.1], color=c, lw=0.7)
                ax.plot([cx + s * 0.07, cx + s * 0.1], [cy + d] * 2, color=c, lw=0.7)
    elif kind == "check":
        ax.add_patch(Circle((cx, cy), 0.09, fc="white", ec=c, lw=lw))
        ax.plot([cx - 0.045, cx - 0.01, cx + 0.05], [cy, cy - 0.04, cy + 0.04], color=c, lw=1.3,
                solid_capstyle="round", solid_joinstyle="round")


def fig1_2_rag():
    fig, ax = canvas(7.5, 3.55)
    bw, bh = 1.3, 0.66
    node = {"doc": (1.15, 3.1), "emb": (3.75, 3.1), "db": (6.35, 3.1),
            "user": (1.15, 1.85), "ret": (3.75, 1.85), "llm": (6.35, 1.85), "ver": (3.75, 0.5)}
    spec = {"doc": ("문서", "doc"), "emb": ("임베딩 모델", "embed"), "db": ("벡터 데이터베이스", "db"),
            "user": ("사용자", "user"), "ret": ("검색기", "search"), "llm": ("LLM", "llm"),
            "ver": ("답변 검증", "check")}
    for k, (cx, cy) in node.items():
        hi = k == "ver"
        box(ax, cx - bw / 2, cy - bh / 2, bw, bh, fc=LIGHT if hi else FILL, ec=BLUE if hi else LINE,
            lw=1.2 if hi else 0.8, r=0.08)
        glyph(ax, spec[k][1], cx, cy + 0.12, c=BLUE if hi else MUTED)
        ax.text(cx, cy - 0.19, spec[k][0], ha="center", va="center", fontsize=8.5)
    L = lambda k: node[k][0] - bw / 2
    R = lambda k: node[k][0] + bw / 2
    T = lambda k: node[k][1] + bh / 2
    B = lambda k: node[k][1] - bh / 2
    lab = dict(ha="center", va="bottom", fontsize=7.5, color=MUTED)
    arrow(ax, [(R("doc") + 0.04, 3.1), (L("emb") - 0.04, 3.1)])
    arrow(ax, [(R("emb") + 0.04, 3.1), (L("db") - 0.04, 3.1)])
    arrow(ax, [(6.35, B("db") - 0.02), (6.35, 2.49), (4.05, 2.49), (4.05, T("ret") + 0.03)])
    arrow(ax, [(R("user") + 0.04, 1.85), (L("ret") - 0.04, 1.85)])
    ax.text((R("user") + L("ret")) / 2, 1.9, "질문", **lab)
    arrow(ax, [(R("ret") + 0.04, 1.85), (L("llm") - 0.04, 1.85)])
    ax.text((R("ret") + L("llm")) / 2, 1.9, "질문 + 검색된 조각", **lab)
    arrow(ax, [(6.2, B("llm") - 0.02), (6.2, 0.62), (R("ver") + 0.03, 0.62)])
    ax.text(5.25, 0.66, "통과", **lab)
    arrow(ax, [(R("ver") + 0.03, 0.38), (6.55, 0.38), (6.55, B("llm") - 0.03)], ls=(0, (3, 2)))
    ax.text(6.62, 1.0, "재시도", ha="left", va="center", fontsize=7.5, color=MUTED)
    arrow(ax, [(L("ver") - 0.03, 0.5), (1.15, 0.5), (1.15, B("user") - 0.03)])
    ax.text(2.1, 0.54, "답변", **lab)
    save(fig, "fig1_2_rag.png")


# ---------------------------------------------------------------- 그림 3-1
def md(a, b, c, d):
    return f"| {a:<21} | {b:>11} | {c:>7} | {d:>7} |"


def fig3_1_chunk_split():
    fs, ls = 6.8, 0.19
    fig, ax = canvas(7.5, 3.15)
    w = cw(fig, fs)
    x0, pad = 0.1, 0.1
    bw = 59 * w + 2 * pad
    # 실제 LangChain 청크 글(MultiHiertt 문서 49ca72c5… 표 1)과 같은 줄 구성, 연도 열만 3개로 줄임
    c1 = [md("", "December 31", "", ""), md("(Dollars in millions)", "2011", "2010", "2009"), md("Consumer", "", "", ""),
          md("Residential mortgage", "$15,970", "$17,691", "$16,596"), md("Home equity", "2,453", "2,694", "3,804")]
    c2 = [md("Other consumer", "15", "48", "104"), md("Total consumer", "18,768", "20,854", "20,839"),
          md("Commercial", "", "", ""), md("U.S. commercial", "2,174", "3,453", "4,925")]

    def chunk(lines, ytop, label, head=0, hi=None):
        h = len(lines) * ls + 0.1
        ax.text(x0, ytop + 0.1, label, ha="left", va="bottom", fontsize=8)
        box(ax, x0, ytop - h, bw, h, fc="white", ec=LINE, lw=0.8, r=0.04)
        for k, s in enumerate(lines):
            y = ytop - 0.05 - ls * (k + 0.5)
            if k < head:
                ax.add_patch(Rectangle((x0 + 0.03, y - ls / 2), bw - 0.06, ls, fc=FILL, ec="none"))
            hx = mono(ax, x0 + pad, y, s, fs, w, hi=hi if hi and k == 1 else None)
            if hx:
                hit = (hx, y)
        return ytop - h, (hit if hi else None)

    y1, _ = chunk(c1, 2.9, "청크 1 (머리글 포함)", head=2)
    cut = y1 - 0.14
    ax.plot([x0 - 0.05, x0 + bw + 0.05], [cut, cut], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(x0 + bw + 0.1, cut, "분할 경계", ha="left", va="center", fontsize=7, color=MUTED)
    top2 = cut - 0.36
    y2, (hx, yv) = chunk(c2, top2, "청크 2 (머리글 없음)", hi="20,854")
    lx = hx - 2.5 * w          # 값의 첫 글자 위치: 아래 줄(  3,453)에서는 빈칸이라 선이 글자를 지나지 않는다
    ax.plot([lx, lx], [yv - ls / 2, y2 - 0.16], color=BLUE, lw=0.8)
    ax.text(lx, y2 - 0.19, "어느 연도의 값인지 청크 안에 없음", ha="center", va="top", fontsize=7.5, color=BLUE)

    # 오른쪽: 셀 문장
    rx, fs2 = 4.55, 7.2
    w2 = cw(fig, fs2)
    sl = ["Consumer > Total consumer >", "December 31 > 2010: 20,854"]
    sw = max(map(len, sl)) * w2 + 0.3
    sh = 2 * 0.22 + 0.14
    sy = yv - sh / 2
    ax.text(rx, sy + sh + 0.1, "셀 문장 (머리글 경로 포함)", ha="left", va="bottom", fontsize=8)
    box(ax, rx, sy, sw, sh, fc="white", ec=BLUE, lw=1.3, r=0.05)
    for k, t in enumerate(sl):
        mono(ax, rx + 0.15, sy + sh - 0.07 - 0.22 * (k + 0.5), t, fs2, w2, hi="20,854" if k else None)
    arrow(ax, [(x0 + bw + 0.08, yv), (rx - 0.08, yv)], color=MUTED)
    ax.text((x0 + bw + rx) / 2, yv + 0.05, "같은 셀", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    save(fig, "fig3_1_chunk_split.png")


# ---------------------------------------------------------------- 그림 3-2
def fig3_2_system():
    fig, ax = canvas(7.5, 3.1)
    bh = 0.78

    def lane(y0, y1, label):
        ax.add_patch(FancyBboxPatch((0.05, y0), 7.4, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#f7f6f4", ec=LINE, lw=0.6))
        ax.text(0.17, y1 - 0.1, label, ha="left", va="top", fontsize=8, color=MUTED)

    def row(items, x, y, gap):
        """items: (w, title, sub, emph). 가로로 늘어놓고 사이에 화살표. 각 상자의 (왼, 오, 중앙 x) 반환."""
        pos = []
        for k, (w, title, sub, emph) in enumerate(items):
            box(ax, x, y - bh / 2, w, bh, fc=LIGHT if emph else "white", ec=BLUE if emph else LINE,
                lw=1.3 if emph else 0.8)
            nt, ns = title.count("\n") + 1, (sub.count("\n") + 1 if sub else 0)
            th, sh = 8.2 / 72 * 1.3, 6.6 / 72 * 1.35          # 줄 높이(인치)
            yt = y + (nt * th + (0.07 + ns * sh if ns else 0)) / 2   # 제목+부제 묶음을 세로 가운데로
            ax.text(x + w / 2, yt, title, ha="center", va="top", fontsize=8.2, linespacing=1.3)
            if sub:
                ax.text(x + w / 2, yt - nt * th - 0.07, sub, ha="center", va="top", fontsize=6.6, color=MUTED,
                        linespacing=1.35)
            pos.append((x, x + w, x + w / 2))
            if k < len(items) - 1:
                arrow(ax, [(x + w + 0.03, y), (x + w + gap - 0.03, y)])
            x += w + gap
        return pos

    lane(1.62, 3.05, "색인 단계 (문서마다 1회)")
    lane(0.05, 1.48, "질의 단계 (질문마다)")
    top = row([(0.75, "계층형 표\n문서", "", False),
               (1.25, "① 머리글 경로 구성", "행 경로 · 열 경로", False),
               (1.55, "② 셀 문장 생성", "캡션이 있으면 캡션\n+ 행 경로 + 열 경로 + 값", True),
               (1.3, "③ 임베딩·색인", "bge-base-en-v1.5 벡터\n+ BM25", False)], 0.17, 2.22, 0.25)
    # ③의 결과를 저장하는 벡터 데이터베이스(그림 1-2 와 같은 아이콘)
    dx, dw = top[3][1] + 0.25, 1.35
    arrow(ax, [(top[3][1] + 0.03, 2.22), (dx - 0.03, 2.22)])
    box(ax, dx, 2.22 - bh / 2, dw, bh, fc=FILL)
    glyph(ax, "db", dx + dw / 2, 2.22 + 0.12)
    ax.text(dx + dw / 2, 2.22 - 0.19, "벡터 데이터베이스", ha="center", va="center", fontsize=8.2)
    bot = row([(0.62, "질문", "", False),
               (1.95, "④ 하이브리드 유사도 계산", "0.7 × 코사인 + 0.3 × BM25\n질문의 문서 안", False),
               (1.1, "상위 20개\n셀 문장", "", False),
               (1.8, "⑤ LLM 입력 구성·\n답변 생성", "Qwen3-8B 4비트", False),
               (0.62, "답변", "", False)], 0.17, 0.65, 0.28)
    x3, x4 = dx + dw / 2, bot[1][2]
    arrow(ax, [(x3, 2.22 - bh / 2 - 0.02), (x3, 1.55), (x4, 1.55), (x4, 0.65 + bh / 2 + 0.03)])
    save(fig, "fig3_2_system.png")


# ---------------------------------------------------------------- 그림 3-3·3-4 공통 표
# MultiHiertt train 문서 49ca72c5d69c43d9bca9d73117f85fc5 의 표 1 일부($·각주 표시 뺌). T34_ROWS2 = 같은 문서 표 2 의 같은 위치.
T34_ROWS = [("Residential mortgage", "15,970", "17,691"), ("Total consumer", "18,768", "20,854")]
T34_ROWS2 = [("Residential mortgage", "21,164", "16,768"), ("Total consumer", "24,324", "21,747")]


def t34_table(ax, x0, top, rh, cws, rows=T34_ROWS, hrow=LIGHT_OR, hcol=LIGHT, **kw):
    """그림 3-4 의 표. kw 는 칸 글자 모양. 반환: 열 경계 xs(4개), 행 아래 ys(5개)."""
    xs = [x0, x0 + cws[0], x0 + cws[0] + cws[1], x0 + sum(cws)]
    ys = [top - rh * (i + 1) for i in range(5)]
    c = [(xs[0], ys[1], cws[0], 2 * rh, "", FILL, LINE, 0.6, {}),
         (xs[1], ys[0], cws[1] + cws[2], rh, "December 31", hrow, LINE, 0.6, dict(kw)),
         (xs[1], ys[1], cws[1], rh, "2011", hrow, LINE, 0.6, dict(kw)),
         (xs[2], ys[1], cws[2], rh, "2010", hrow, LINE, 0.6, dict(kw)),
         (xs[0], ys[2], cws[0], rh, "Consumer", hcol, LINE, 0.6, {"ha": "left", **kw}),
         (xs[1], ys[2], cws[1], rh, "", "white", LINE, 0.6, {}),
         (xs[2], ys[2], cws[2], rh, "", "white", LINE, 0.6, {})]
    for i, (name, a, b) in enumerate(rows):
        y = ys[3 + i]
        c += [(xs[0], y, cws[0], rh, name, hcol, LINE, 0.6, {"ha": "left", "indent": 0.12, **kw}),
              (xs[1], y, cws[1], rh, a, "white", LINE, 0.6, dict(kw)),
              (xs[2], y, cws[2], rh, b, "white", LINE, 0.6, dict(kw))]
    grid(ax, c)
    return xs, ys


def mark(ax, x, y, w, h, ec, lw=1.3, ls="-", fc="none"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))


def segs(ax, x, y, parts, fs, w):
    """고정폭 한 줄을 (글, 색, 굵기) 조각으로."""
    for t, col, wt in parts:
        ax.text(x, y, t, ha="left", va="center", family=MONO, fontsize=fs, color=col, weight=wt)
        x += len(t) * w


def step_head(ax, x, y, num, title):
    ax.add_patch(Circle((x + 0.11, y), 0.1, fc=BLUE, ec="none"))
    ax.text(x + 0.11, y, num, ha="center", va="center", color="white", fontsize=7.5, weight="bold")
    ax.text(x + 0.3, y, title, ha="left", va="center", fontsize=8.5)


# ---------------------------------------------------------------- 그림 3-3 (머리글 경로 구성, 단계마다 그림 하나)
def fig3_3_header_path():
    """3.3.1 의 네 단계를 그림 3-4 와 같은 표로. 판정 결과는 scripts/mh_arms.build_tables(v3.3u)를 이 표에 돌린 값과 같다
    (위쪽 머리글 2줄·왼쪽 머리글 1열, 경로 Consumer > Total consumer / December 31 > 2010, 표 둘이면 Table 1·Table 2).
    원고 본문 폭(36848 HWPUNIT = 5.12인치)과 같은 폭으로 그려 한글에서 줄이지 않고 들어가게 한다 — 글자 크기 = 인쇄 크기."""
    W = 5.12
    rh, cws, tfs = 0.22, [1.4, 0.52, 0.52], 8          # 표 행 높이·열 너비·글자
    x0, tx, fs, mfs = 0.62, 3.25, 8.3, 7.6              # 표 왼쪽, 설명 왼쪽, 설명 글자, 경로 글자
    ls = 0.155                                          # 설명 줄 간격

    def page(h):
        fig, ax = canvas(W, h)
        ax.add_patch(Rectangle((0, 0), W, h, fc="white", ec="none", zorder=-1))   # 네 그림의 폭(=배율)을 같게
        return fig, ax

    def lines(ax, y, rows, size=fs):
        for t, col in rows:
            if t:
                ax.text(tx, y, t, ha="left", va="center", fontsize=size, color=col)
            y -= ls if t else ls * 0.6
        return y

    def side(ax, y0, y1, text, col):
        """표 왼쪽의 세로 괄호와 이름."""
        bx = x0 - 0.07
        ax.plot([bx + 0.04, bx, bx, bx + 0.04], [y1 - 0.02, y1 - 0.02, y0 + 0.02, y0 + 0.02], color=col, lw=0.9)
        ax.text(bx - 0.05, (y0 + y1) / 2, text, ha="right", va="center", fontsize=7.6, color=col, linespacing=1.2)

    # ---- 1 위쪽 머리글
    fig, ax = page(1.45)
    step_head(ax, 0.02, 1.33, "1", "위쪽 머리글 판정")
    xs, ys = t34_table(ax, x0, 1.13, rh, cws, hcol="white", fontsize=tfs)
    mark(ax, xs[0], ys[1], cws[0], 2 * rh, ORANGE, lw=1.1, ls=(0, (3, 2)))
    ax.text(xs[0] + cws[0] / 2 + 0.12, ys[1] + rh, "비어 있음", ha="center", va="center", fontsize=7.6, color=ORANGE)
    arrow(ax, [(xs[0] + 0.13, ys[0] + rh - 0.05), (xs[0] + 0.13, ys[2] + rh + 0.03)], color=ORANGE)
    mark(ax, xs[0], ys[2], cws[0], rh, ORANGE)
    ax.plot([xs[0] - 0.02, xs[3] + 0.02], [ys[1], ys[1]], color=ORANGE, lw=1.6)
    side(ax, ys[1], ys[0] + rh, "위쪽\n머리글", ORANGE)
    side(ax, ys[4], ys[1], "데이터", MUTED)
    lines(ax, 1.1, [("왼쪽 위 칸이 비어 있으므로", INK), ("첫 열을 위에서부터 내려가며", INK),
                    ("글자가 처음 나오는 행", INK), ("(Consumer)을 찾는다.", INK), ("", None),
                    ("이 행부터 데이터로 보고,", ORANGE), ("그 위의 2줄을 위쪽 머리글로", ORANGE),
                    ("판정한다.", ORANGE)])
    save(fig, "fig3_3_step1.png")

    # ---- 2 왼쪽 머리글
    fig, ax = page(1.45)
    step_head(ax, 0.02, 1.33, "2", "왼쪽 머리글 판정")
    xs, ys = t34_table(ax, x0, 1.13, rh, cws, fontsize=tfs)
    mark(ax, xs[0], ys[1], cws[0], rh, BLUE, lw=1.1, ls=(0, (3, 2)))
    ax.text(xs[0] + cws[0] / 2, ys[1] + rh / 2, "빈 칸 1개", ha="center", va="center", fontsize=7.6, color=BLUE)
    ax.plot([xs[1], xs[1]], [ys[4] - 0.02, ys[1] + rh + 0.02], color=BLUE, lw=1.6)
    side(ax, ys[1], ys[0] + rh, "위쪽\n머리글", ORANGE)
    ym = (ys[4] + ys[2] + rh) / 2
    ax.text(x0 - 0.12, ym, "왼쪽\n머리글", ha="right", va="center", fontsize=7.6, color=BLUE, linespacing=1.2)
    arrow(ax, [(x0 - 0.1, ym), (x0 + 0.1, ym)], color=BLUE)
    lines(ax, 1.1, [("위쪽 머리글이 2줄이므로", INK), ("그 마지막 줄(2011, 2010)에서", INK),
                    ("왼쪽부터 연속으로 비어 있는", INK), ("칸을 센다(1개).", INK), ("", None),
                    ("따라서 첫 열 1개를", BLUE), ("왼쪽 머리글로 판정한다.", BLUE)])
    save(fig, "fig3_3_step2.png")

    # ---- 3 경로 구성
    fig, ax = page(1.5)
    step_head(ax, 0.02, 1.38, "3", "행 경로와 열 경로 구성")
    xs, ys = t34_table(ax, x0, 1.18, rh, cws, fontsize=tfs)
    mark(ax, xs[2], ys[4], cws[2], rh, BLUE, lw=1.4, fc=MID)
    ax.text(xs[2] + cws[2] / 2, ys[4] + rh / 2, "20,854", ha="center", va="center", fontsize=tfs, weight="bold", zorder=4)
    mark(ax, xs[1], ys[0], cws[1] + cws[2], rh, ORANGE)
    mark(ax, xs[2], ys[1], cws[2], rh, ORANGE)
    mark(ax, xs[0], ys[2], cws[0], rh, BLUE)
    mark(ax, xs[0], ys[4], cws[0], rh, BLUE)
    w = cw(fig, mfs)
    y = lines(ax, 1.2, [("열 경로", ORANGE)])
    segs(ax, tx, y, [("December 31 > 2010", ORANGE, "normal")], mfs, w)
    y = lines(ax, y - ls, [("위쪽 머리글에서 이 셀 위의 칸", MUTED), ("(병합 칸은 두 열에 공통)", MUTED), ("", None),
                           ("행 경로", BLUE)], size=7.6)
    segs(ax, tx, y, [("Consumer > Total consumer", BLUE, "normal")], mfs, w)
    lines(ax, y - ls, [("왼쪽 머리글에서 이 셀 왼쪽의 칸", MUTED), ("(Consumer는 절 제목 행)", MUTED)], size=7.6)
    save(fig, "fig3_3_step3.png")

    # ---- 4 같은 경로 확장
    fig, ax = page(2.05)
    step_head(ax, 0.02, 1.93, "4", "같은 경로가 있으면 경로 확장")
    mrh, mcw, mt = 0.18, [1.3, 0.5, 0.5], 7.2
    top = 1.58
    for lx, rows, lab in [(0.05, T34_ROWS, "표 1"), (2.7, T34_ROWS2, "표 2 (같은 문서의 다른 표)")]:
        ax.text(lx, top + 0.06, lab, ha="left", va="bottom", fontsize=7.8)
        xs, ys = t34_table(ax, lx, top, mrh, mcw, rows=rows, fontsize=mt)
        mark(ax, xs[2], ys[4], mcw[2], mrh, BLUE, lw=1.3, fc=MID)
        ax.text(xs[2] + mcw[2] / 2, ys[4] + mrh / 2, rows[1][2], ha="center", va="center", fontsize=mt,
                weight="bold", zorder=4)
    ax.text(0.05, 0.5, "두 셀의 경로가 같으므로 두 표의 행 경로 앞에 표 순서를 추가한다.", ha="left", va="center",
            fontsize=fs, color=BLUE)
    w = cw(fig, mt)
    for k, (t, v) in enumerate([("Table 1", "20,854"), ("Table 2", "21,747")]):
        segs(ax, 0.05, 0.28 - 0.19 * k, [(t, INK, "bold"), (" > ", MUTED, "normal"),
                                        ("Consumer > Total consumer", BLUE, "normal"), (" > ", MUTED, "normal"),
                                        ("December 31 > 2010", ORANGE, "normal"), (": ", MUTED, "normal"),
                                        (v, INK, "bold")], mt, w)
    save(fig, "fig3_3_step4.png")


# ---------------------------------------------------------------- 그림 3-4
def fig3_3_cell_sentence():
    fig, ax = canvas(7.5, 1.75)
    x0, top, rh = 0.1, 1.65, 0.27
    cws = [1.45, 0.62, 0.62]
    xs, ys = t34_table(ax, x0, top, rh, cws)
    data = T34_ROWS

    fs = 6.9
    w = cw(fig, fs)
    sx, lsp = 3.55, 0.2
    right = xs[2] + cws[2]
    for i, (name, a, b) in enumerate(data):
        yc = ys[3 + i] + rh / 2 - (0.02 if i == 0 else 0.2)   # 두 쌍이 겹치지 않게 벌림
        ys_pair = [yc + lsp / 2, yc - lsp / 2]
        for (yr, val), y in zip([("2011", a), ("2010", b)], ys_pair):
            segs = [(f"Consumer > {name}", BLUE, "normal"), (" > ", MUTED, "normal"),
                    (f"December 31 > {yr}", ORANGE, "normal"), (": ", MUTED, "normal"), (val, INK, "bold")]
            x = sx
            for t, col, wt in segs:
                ax.text(x, y, t, ha="left", va="center", family=MONO, fontsize=fs, color=col, weight=wt)
                x += len(t) * w
        # 행 → 괄호
        bx = sx - 0.12
        ax.plot([bx + 0.05, bx, bx, bx + 0.05], [ys_pair[0] + 0.08, ys_pair[0] + 0.08,
                                                  ys_pair[1] - 0.08, ys_pair[1] - 0.08], color=MUTED, lw=0.8)
        arrow(ax, [(right + 0.05, ys[3 + i] + rh / 2), (bx - 0.04, yc)])
    # 범례
    ly = top - 0.12
    lx = sx
    for t, col in [("행 경로", BLUE), ("열 경로", ORANGE), ("값", INK)]:
        ax.add_patch(Rectangle((lx, ly - 0.05), 0.1, 0.1, fc=col, ec="none"))
        ax.text(lx + 0.15, ly, t, ha="left", va="center", fontsize=7.5)
        lx += 0.85
    save(fig, "fig3_4_cell_sentence.png")


# ---------------------------------------------------------------- 결과 그림 공통
def jl(p):
    return [json.loads(l) for l in open(ROOT / p)]


def chart_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE); ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3, width=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 그림 4-1
def fig4_1_recall():
    tab = json.load(open(ROOT / "results/mh_only_20260928/final/retrieval_table.json"))
    series = [("본 방법 (셀 문장)", "본 방법", BLUE, "o", "-", 2.2),
              ("LangChain 튜토리얼 청크", "LangChain 청크", "#8a8883", "s", "-", 1.2),
              ("RAPTOR", "RAPTOR", "#6f86a3", "^", "-", 1.2),
              ("TableRAG (Chen 2024)", "TableRAG(Chen)", "#b0aea9", "D", "-", 1.2),
              ("TableRAG (Yu 2025)", "TableRAG(Yu)", "#3f4d63", "v", (0, (4, 2)), 1.2)]
    ks = [1, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    chart_style(ax)
    for key, name, col, mk, ls, lw in series:
        d = tab[key]
        vals = [d["200"][f"R@{k}"] * 100 for k in ks]
        hollow = key.startswith("TableRAG (Yu")
        ax.plot(ks, vals, color=col, lw=lw, ls=ls, marker=mk, ms=6 if lw > 2 else 5,
                mfc="white" if hollow else col, mec=col, mew=1.1, zorder=4 if lw > 2 else 3,
                label=f"{name} (단위당 {d['200'].get('cells_per_unit', d['cells_per_unit']):g}셀)")
    ax.set_xticks(ks); ax.set_xlim(0, 21)
    ax.set_ylim(0, 104); ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_xlabel("k (검색 결과 상위 단위 수)", color=MUTED)
    ax.set_ylabel("재현율@k", color=MUTED)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7.5, handlelength=2.6)
    save(fig, "fig4_1_recall.png")


# ---------------------------------------------------------------- 그림 4-2
def fig4_2_header_ablation():
    ids = [r["query_id"] for r in jl("results/mh_arms/cap300_20260924/cell_uniq.jsonl")]

    def ret(p):
        d = {r["query_id"]: r["doc"]["correct"] for r in jl(p) if "doc" in r}  # 23건은 doc 없음(882 밖)
        return 100 * sum(d[i] for i in ids) / len(ids)

    def ans(p):
        a = jl(p)
        assert len(a) == len(ids) == 882
        return 100 * sum(bool(r["answer_correct"]) for r in a) / len(a)

    ours = [ret("results/mh_only_20260928/retrieval_r100/mh_train_s3c_records.jsonl"),
            ans("results/mh_arms/cap300_20260924/cell_uniq.jsonl")]
    abl = [ret("results/header_ablation_20260929/mh_train_s3c_value_records.jsonl"),
           ans("results/header_ablation_20260929/mh_value_answer.jsonl")]
    assert [round(v, 1) for v in ours] == [89.6, 46.3], ours
    assert [round(v, 1) for v in abl] == [13.3, 16.2], abl

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    chart_style(ax)
    bw = 0.34
    for off, vals, col, name in [(-bw / 2 - 0.01, ours, BLUE, "본 방법"),
                                 (bw / 2 + 0.01, abl, "#a9a7a2", "머리글 경로 제거")]:
        xs = [g + off for g in range(2)]
        ax.bar(xs, vals, width=bw, color=col, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["상위 20셀 검색 성공", "답변 정확도"], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    save(fig, "fig4_2_header_ablation.png")


# ---------------------------------------------------------------- 그림 4-3
def fig4_3_answer():
    ids = [r["query_id"] for r in jl("results/mh_only_20260928/baselines/ids200.jsonl")]
    B = "results/mh_only_20260928/baselines/"
    runs = [("제안 방법", "results/mh_arms/cap300_20260924/cell_uniq.jsonl"),
            ("LangChain 청크", B + "langchain_chunk/langchain_chunk_answer.jsonl"),
            ("RAPTOR", B + "raptor/raptor_answer_2000.jsonl"),
            ("TableRAG(Chen)", B + "tablerag_chen_mt512/tablerag_chen_answer.jsonl"),  # 최대 생성 길이 512(PREREG-2026-09-29-chen-max-tokens)
            ("TableRAG(Yu)", B + "tablerag_yu/tablerag_yu_answer.jsonl")]

    import sys
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    from mh_arms import official_answers
    from rag_agent.eval.multihiertt_em import format_match
    gold = official_answers("train")

    def acc(p):                        # 형식 허용 채점(2026-09-30)
        d = {r["query_id"]: r for r in jl(p)}
        if not all(i in d for i in ids):
            return None
        return 100 * sum(format_match(d[i]["pred"], gold[i], d[i]["kind"] == "arith") for i in ids) / len(ids)

    vals = []
    for name, p in runs:
        v = acc(p)
        if v is None:
            print(f"[알림] {name}: {p} 에 200문항이 다 없어 막대에서 뺌")
            continue
        vals.append((name, v))
    ref = acc("results/mh_arms/cap300_20260924/fulltable.jsonl")
    expect = {"제안 방법": 57.5, "LangChain 청크": 39.5, "RAPTOR": 38.5,
              "TableRAG(Chen)": 13.5, "TableRAG(Yu)": 53.5}
    for name, v in vals:
        assert name not in expect or round(v, 1) == expect[name], (name, v)
    assert round(ref, 1) == 52.5, ref
    vals.sort(key=lambda t: -t[1])

    n = len(vals)
    ypos = list(range(n))[::-1]           # 위에서부터 큰 값
    yref = -1.4
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * (n + 2) + 0.5))
    chart_style(ax)
    for (name, v), y in zip(vals, ypos):
        ax.barh(y, v, height=0.62, color=BLUE if name == "제안 방법" else "#a9a7a2", zorder=3)
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", ha="left", fontsize=8.5)
    ax.barh(yref, ref, height=0.62, color="#eceae6", edgecolor="#8a8883", hatch="////", lw=0.8, zorder=3)
    ax.text(ref + 0.8, yref, f"{ref:.1f}%", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.axhline((yref + ypos[-1]) / 2, color=LINE, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ypos + [yref])
    ax.set_yticklabels([t[0] for t in vals] + ["참고: 표 전체 입력"], fontsize=8.5)
    ax.get_yticklabels()[-1].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 70); ax.set_xticks(range(0, 71, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 71, 10)])
    ax.set_ylim(yref - 0.6, ypos[0] + 0.6)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_xlabel("답변 정확도", color=MUTED)
    save(fig, "fig4_3_answer.png")


def main():
    OUT.mkdir(exist_ok=True)
    fig1_1_hier_table()
    fig1_2_rag()
    fig3_1_chunk_split()
    fig3_2_system()
    fig3_3_header_path()
    fig3_3_cell_sentence()
    fig4_1_recall()
    fig4_2_header_ablation()
    fig4_3_answer()


if __name__ == "__main__":
    main()
