"""랩미팅 슬라이드 3: 검색 결과를 LLM 에 넘기는 방식(LangChain 청크 대 셀 문장 RAG) + 실제 문항.
문항 95c8e255e1bb4b79a7187c3f16c78b80, PREREG-2026-10-12-cellform.md 의 [가]·[다] 입력·출력 그대로.
  .venv/bin/python results/cellform_20261012/fig_delivery.py  →  results/cellform_20261012/fig_delivery.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
LIGHT, GRAYF, LINE, RED, GREEN = "#e3eefb", "#f0efec", "#9a9893", "#c0392b", "#1e7b3c"
FONT = "/mnt/c/Windows/Fonts/H2GTRM.TTF"
MONO = "DejaVu Sans Mono"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(), "font.size": 10,
                     "axes.unicode_minus": False, "text.color": INK})

fig = plt.figure(figsize=(13.33, 6.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.33); ax.set_ylim(0, 6.6); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))


def step(x, y, w, h, title, sub, blue):
    box(x, y, w, h, LIGHT if blue else "white", BLUE if blue else LINE, 1.4 if blue else 1.0)
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", fontsize=11, weight="bold", color=BLUE if blue else INK)
    ax.text(x + w / 2, y + h / 2 - 0.12, sub, ha="center", va="center", fontsize=9, color=MUTED, linespacing=1.5)


def arrow(x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.1, color=color))


# 질문
ax.text(0.3, 6.38, "질문:", fontsize=12, weight="bold", va="center")
ax.text(0.95, 6.38, "When does Softwood lumber – board feet reach the largest value?", family="DejaVu Sans", fontsize=11.5, weight="bold", va="center")
ax.text(0.3, 6.08, "정답: 2003      검색 범위: 질문의 문서(표 여러 개)      두 방법 모두 LLM 입력 약 1,080토큰으로 같은 길이", fontsize=10, color=MUTED, va="center")

lanes = [  # (y, 이름, 파랑?, 단계들, 입력 줄들, 답)
    (3.05, "LangChain 청크", False,
     [("① 자르기", "표를 markdown 으로\n이어 붙여 1,000자씩\n(겹침 200자)"),
      ("② 벡터 저장소", "청크마다 임베딩\n(bge-base)\n→ 메모리 벡터 저장소"),
      ("③ 검색", "질문 임베딩과\n코사인 유사도\n상위 청크 4개")],
     ["|  | 2006 | 2005 | 2004 | 2003 | 2002 |",
      "| Softwood lumber – board feet | 7,871 | 8,650 | 8,890 | 8,981 | 8,623 |",
      "…  (청크 4개를 빈 줄로 이어 붙임)"],
     ("2004", "8,890 을 가장 크다고 읽음", False)),
    (0.2, "셀 문장 RAG (본 방법)", True,
     [("① 셀 문장", "셀마다 한 줄\n행 경로 > 열 경로: 값"),
      ("② 벡터 저장소", "셀 문장마다 임베딩\n(bge-base) → 메모리\n벡터 저장소 + BM25"),
      ("③ 검색", "코사인 0.7 + BM25 0.3\n점수 순으로\nLangChain 입력과\n같은 토큰 수까지")],
     ["Softwood lumber – board feet > 2004: 8,890",
      "Softwood lumber – board feet > 2005: 8,650",
      "Softwood lumber – board feet > 2003: 8,981",
      "Softwood lumber – board feet > 2006: 7,871",
      "…  (셀 문장을 줄바꿈으로 이어 붙임)"],
     ("2003", "값 옆의 해를 바로 읽음", True)),
]
LH, SW, IW = 2.5, 1.62, 4.15
for y, name, blue, steps, lines, (ans, why, ok) in lanes:
    ax.text(0.3, y + LH + 0.12, name, fontsize=12, weight="bold", color=BLUE if blue else INK, va="bottom")
    x = 0.3
    for t, s_ in steps:
        step(x, y + 0.45, SW, 1.6, t, s_, blue)
        arrow(x + SW + 0.02, y + 1.25, x + SW + 0.2, y + 1.25)
        x += SW + 0.22
    box(x, y, IW, LH, GRAYF if not blue else "#f4f8fe", BLUE if blue else LINE, 1.0)
    ax.text(x + 0.12, y + LH - 0.22, "④ LLM 에 들어가는 입력 (Context)", fontsize=10, weight="bold", va="center", color=BLUE if blue else INK)
    for i, ln in enumerate(lines):
        ax.text(x + 0.12, y + LH - 0.62 - i * 0.36, ln, fontsize=6.5 if not blue else 7.6, va="center",
                **({} if ln.startswith("…") else {"family": MONO}),
                color=MUTED if ln.startswith("…") else INK, weight="bold" if ("2003" in ln and blue) else "normal")
    if not blue:
        ax.text(x + 0.12, y + 0.3, "8,981 이 몇 번째 칸인지 세어 윗줄의 해(2003)와 맞춰야 함", fontsize=8.5, color=RED, va="center")
    arrow(x + IW + 0.02, y + LH / 2, x + IW + 0.22, y + LH / 2)
    ax_x = 11.98
    box(ax_x, y + 0.65, 1.0, 1.25, "white", GREEN if ok else RED, 1.6)
    ax.text(ax_x + 0.5, y + 1.42, ans, ha="center", va="center", fontsize=17, weight="bold", color=GREEN if ok else RED)
    ax.text(ax_x + 0.5, y + 0.95, "맞음" if ok else "틀림", ha="center", va="center", fontsize=10, color=GREEN if ok else RED)
    ax.text(ax_x + 0.5, y + 0.42, why, ha="center", va="center", fontsize=7.6, color=MUTED)

# ⑤ 공통 LLM 단계 (두 줄에 걸침)
cx, cw = 10.21, 1.52
box(cx, 0.2, cw, 5.35, "white", INK, 1.2)
ax.text(cx + cw / 2, 5.25, "⑤ LLM (공통)", ha="center", va="center", fontsize=11, weight="bold")
ax.text(cx + cw / 2, 2.88,
        "지시문\nLangChain 튜토리얼\n원문 그대로\n\n+ Question\n+ Context\n\n↓\n\nQwen3-8B 4비트\n한 번 호출\n\n두 방법 같음",
        ha="center", va="center", fontsize=8.8, linespacing=1.45)
for y in (3.05, 0.2):
    arrow(cx + cw + 0.02, y + LH / 2, 11.96, y + LH / 2)

out = Path(__file__).with_name("fig_delivery.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out)
