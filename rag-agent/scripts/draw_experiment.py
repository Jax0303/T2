#!/usr/bin/env python3
"""Draw the preprocessing-ladder x retrieval experiment as a diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

KO = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

def box(ax, x, y, w, h, text, fc, ec="#333", fs=11, bold=False, tc="#111"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontproperties=KO, fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=3)

def arrow(ax, x1, y1, x2, y2, color="#444", style="-|>", lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=16, linewidth=lw, color=color, zorder=1))

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

# ---- title ----
ax.text(7, 8.6, "표-RAG 전처리 실험 : 표를 잘 \"검색\"되게 만들기",
        ha="center", fontproperties=KO, fontsize=17, fontweight="bold")
ax.text(7, 8.15, "(query → 맞는 표 찾기.  표를 그냥 넣지 말고 내 코드로 전처리한 뒤 인덱싱)",
        ha="center", fontproperties=KO, fontsize=10.5, color="#555")

# ---- context strip: 2-stage RAG ----
box(ax, 0.4, 7.0, 3.0, 0.7, "① 표 찾기 (retrieval)\n← 지금 이 실험", "#d6ebff", ec="#2b7", fs=10.5, bold=True)
arrow(ax, 3.4, 7.35, 4.1, 7.35, color="#888")
box(ax, 4.1, 7.0, 3.0, 0.7, "② 표 안에서 답 추출\n(다음 단계)", "#eee", ec="#bbb", fs=10.5)
ax.text(11.6, 7.35, "LLM 모델은 무관 — 검색/추출 품질이 목표",
        ha="center", fontproperties=KO, fontsize=9.5, color="#777", style="italic")

# ---- main pipeline boxes ----
y = 4.7; h = 1.0
box(ax, 0.3, y, 1.8, h, "원본 표\n(raw table)", "#fff2cc", ec="#d6a000", fs=10.5, bold=True)
box(ax, 5.7, y, 2.0, h, "인덱스\n(검색용 텍스트)", "#e8e8e8", ec="#999", fs=10.5)
box(ax, 8.2, y, 2.3, h, "검색기 (retriever)\nBM25  /  Dense", "#ffe0e0", ec="#d55", fs=10.5, bold=True)
box(ax, 10.9, y, 2.8, h, "정답 표 순위 측정\nR@1 / R@5 / R@10", "#e0ffe0", ec="#3a3", fs=10.5, bold=True)

# preprocessing ladder (the "내 코드")
lx, lw_ = 2.5, 2.9
box(ax, lx, y-0.05, lw_, h+0.1, "", "#f0f7ff", ec="#2b7bd6", fs=10)
ax.text(lx+lw_/2, y+h+0.28, "전처리 사다리  (내 코드)", ha="center",
        fontproperties=KO, fontsize=10.5, fontweight="bold", color="#1a5fb4")
ladder = [
    ("C0", "생 표만 (지금 상태)"),
    ("C1", "+ 제목/캡션 메타데이터"),
    ("C2", "+ 칼럼 스키마 설명"),
    ("C3", "+ 합성질문 (표가 답할 질문 자동생성)"),
]
for i, (c, d) in enumerate(ladder):
    yy = y + h - 0.27 - i*0.245
    ax.text(lx+0.12, yy, c, fontproperties=KO, fontsize=9.5, fontweight="bold", color="#1a5fb4")
    ax.text(lx+0.55, yy, d, fontproperties=KO, fontsize=8.6, color="#333")

# arrows along pipeline
arrow(ax, 2.1, y+h/2, 2.5, y+h/2)
arrow(ax, lx+lw_, y+h/2, 5.7, y+h/2)
arrow(ax, 7.7, y+h/2, 8.2, y+h/2)
arrow(ax, 10.5, y+h/2, 10.9, y+h/2)

# query feeding into retriever
box(ax, 8.3, 2.9, 2.1, 0.75, "사용자 질문\n(query)", "#fff", ec="#888", fs=10)
arrow(ax, 9.35, 3.65, 9.35, y, color="#d55")

# ---- complexity axis ----
ax.text(0.3, 2.35, "복잡도 축 (단순 → 복잡):", fontproperties=KO, fontsize=10.5, fontweight="bold")
box(ax, 0.3, 1.5, 3.1, 0.7, "flat 표  (OpenWikiTable)\n단순 · 평평한 칼럼", "#eaf6ff", ec="#4a90d9", fs=9.8, bold=True)
arrow(ax, 3.5, 1.85, 4.2, 1.85, color="#888")
box(ax, 4.2, 1.5, 3.1, 0.7, "hier 표  (HiTab)\n복잡 · 계층형 헤더", "#fbeaff", ec="#a04ad9", fs=9.8, bold=True)
ax.text(7.6, 1.85, "← 교수님: \"단순한 표부터\"", fontproperties=KO, fontsize=10, color="#c0392b")

# ---- result callout ----
box(ax, 8.6, 0.9, 5.1, 1.45,
    "결과 (flat, BM25) — 깨끗함, 유의\n"
    "그냥 넣기  C0 : R@1 = 0.18\n"
    "전처리    C3 : R@1 = 0.68   ▲ 거의 4배\n"
    "→ Dense 검색에서도 재현되는지 실험 중",
    "#fffbe6", ec="#e0a800", fs=10, bold=False, tc="#5a4500")

plt.tight_layout()
plt.savefig("/home/user/T2/experiment_diagram.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("saved")
