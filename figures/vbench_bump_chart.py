"""
vbench_bump_chart.py — redesigned bump chart (proportional layout)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import numpy as np
from pathlib import Path

_HERE     = Path(__file__).parent
LATEX_DIR = _HERE.parent / "latex"
OUT_PDF   = str(_HERE / "vbench_bump_chart.pdf")
OUT_PNG   = str(_HERE / "vbench_bump_chart.png")

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ["Veo 3.1", "Kling", "LTX-2", "Wan A14B", "HunyuanVideo", "Wan 1.3B"]
HUMAN    = {"Veo 3.1": 1, "Kling": 2, "Wan A14B": 3, "LTX-2": 4, "HunyuanVideo": 5, "Wan 1.3B": 6}
WORLDJEN = {"Veo 3.1": 1, "Kling": 2, "Wan A14B": 3, "LTX-2": 4, "HunyuanVideo": 5, "Wan 1.3B": 6}
VBENCH   = {"Veo 3.1": 1, "HunyuanVideo": 2, "Wan A14B": 3, "Kling": 4, "LTX-2": 5, "Wan 1.3B": 6}

COLORS = {
    "Veo 3.1":      "#D97706",   # rich amber
    "Kling":        "#2563EB",   # vivid royal blue
    "LTX-2":        "#059669",   # vivid emerald
    "Wan A14B":     "#7C3AED",   # vivid violet
    "HunyuanVideo": "#DC2626",   # vivid crimson
    "Wan 1.3B":     "#64748B",   # slate (bottom model)
}

# ── Layout ────────────────────────────────────────────────────────────────────
N     = 6
FIG_W = 10
FIG_H = 6.5

# Column edges  [left, right] — all 6 columns span 0.02 → 0.98
COL = {
    "rank":  (0.02,  0.08),   # narrow rank badge
    "model": (0.08,  0.26),   # model names
    "hum":   (0.26,  0.46),   # Human Eval
    "wj":    (0.46,  0.66),   # WorldJen BT rating
    "vb":    (0.66,  0.86),   # VBench Quality
    "delta": (0.86,  0.98),   # Δ rank badge
}
def cx(col):  return (COL[col][0] + COL[col][1]) / 2

# Node x positions (centre of each data column)
X_HUM   = cx("hum")
X_WJ    = cx("wj")
X_VB    = cx("vb")

# Row y-positions
HDR_TOP  = 0.96
HDR_BOT  = 0.88   # header/content divider
CONT_BOT = 0.13   # bottom of content rows
STRIP_Y  = 0.065  # centre of stats strip

Y_TOP = 0.84
Y_BOT = 0.17
_STEP = (Y_TOP - Y_BOT) / (N - 1)
def ry(rank): return Y_TOP - (rank - 1) * _STEP

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Outer panel — clean white card with a soft shadow border ──────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (0.015, 0.015), 0.97, 0.97,
    boxstyle="round,pad=0.008", linewidth=1.5,
    edgecolor="#CBD5E1", facecolor="white",
    transform=ax.transAxes, zorder=0))

# ── Solid rich header band (base — dark slate) ────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (0.015, HDR_BOT), 0.97, HDR_TOP - HDR_BOT,
    boxstyle="square,pad=0", linewidth=0,
    facecolor="#1E293B",
    transform=ax.transAxes, zorder=1))

# Solid vivid header cells + light content tints for the three data columns
DATA_COLS = [
    ("hum", "#059669", "#ECFDF5", "Human\nEval"),       # emerald / mint
    ("wj",  "#1D4ED8", "#EFF6FF", "WorldJen\nBT"),      # royal blue / sky
    ("vb",  "#B91C1C", "#FEF2F2", "VBench\nQuality"),   # crimson / blush
]
for key, hdr_col, tint_col, label in DATA_COLS:
    x0, x1 = COL[key]
    # Solid vivid header cell
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, HDR_BOT), x1 - x0, HDR_TOP - HDR_BOT,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=hdr_col,
        transform=ax.transAxes, zorder=2))
    # Light content tint
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, CONT_BOT), x1 - x0, HDR_BOT - CONT_BOT,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=tint_col,
        transform=ax.transAxes, zorder=1))
    # White column header text
    ax.text(cx(key), (HDR_TOP + HDR_BOT) / 2, label,
            color="white", fontsize=9.5, fontweight="bold",
            ha="center", va="center",
            transform=ax.transAxes, zorder=5, linespacing=1.3)

# Header text for rank, model and delta columns (white on dark slate)
for key, label in [
    ("rank",  "#"),
    ("model", "Model"),
    ("delta", "VB − Human\nΔ"),
]:
    ax.text(cx(key), (HDR_TOP + HDR_BOT) / 2, label,
            color="#94A3B8", fontsize=9.5, fontweight="bold",
            ha="center", va="center",
            transform=ax.transAxes, zorder=5, linespacing=1.3)

# ── Full-width header/content divider ─────────────────────────────────────────
ax.plot([0.015, 0.985], [HDR_BOT, HDR_BOT],
        color="#334155", lw=1.6, transform=ax.transAxes, zorder=3)

# ── Column dividers ────────────────────────────────────────────────────────────
for key in ["model", "hum", "wj", "vb", "delta"]:
    xd = COL[key][0]
    ax.plot([xd, xd], [CONT_BOT, HDR_TOP],
            color="#CBD5E1", lw=0.7, transform=ax.transAxes, zorder=3)

# ── Subtle alternating row shading ────────────────────────────────────────────
for r in range(1, N + 1):
    y = ry(r)
    if r % 2 == 0:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.015, y - _STEP * 0.45), 0.97, _STEP * 0.9,
            boxstyle="square,pad=0", linewidth=0,
            facecolor="#F1F5F9", alpha=0.8,
            transform=ax.transAxes, zorder=0))
    # Rank badge
    ax.text(cx("rank"), y, f"#{r}", color="#94A3B8", fontsize=10,
            fontweight="bold", ha="center", va="center",
            transform=ax.transAxes, zorder=4)

# ── Helpers ───────────────────────────────────────────────────────────────────
def bezier(x0, y0, x1, y1, color, lw=2.2, alpha=0.88):
    mid = (x0 + x1) / 2
    verts = [(x0, y0), (mid, y0), (mid, y1), (x1, y1)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ax.add_patch(mpatches.PathPatch(
        MPath(verts, codes),
        facecolor="none", edgecolor=color,
        lw=lw, alpha=alpha, zorder=3, transform=ax.transAxes))

NODE_R = 0.019
def node(x, y, color, label):
    ax.add_patch(plt.Circle((x, y), NODE_R + 0.006, color="white",
                             zorder=5, transform=ax.transAxes, clip_on=False))
    ax.add_patch(plt.Circle((x, y), NODE_R, color=color,
                             zorder=6, transform=ax.transAxes, clip_on=False))
    ax.text(x, y, str(label), color="white", fontsize=8.5, fontweight="bold",
            ha="center", va="center", zorder=7, transform=ax.transAxes)

# ── Draw per model ────────────────────────────────────────────────────────────
for model in MODELS:
    c   = COLORS[model]
    rh  = HUMAN[model]
    rwj = WORLDJEN[model]
    rvb = VBENCH[model]
    yh, ywj, yvb = ry(rh), ry(rwj), ry(rvb)

    # Model label
    ax.text(cx("model"), yh, model, color=c, fontsize=9, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=5)

    # Bezier curves — tier-crossing models get a bolder stroke
    lw = 3.2 if abs(rvb - rh) >= 2 else 2.2
    bezier(X_HUM, yh, X_WJ, ywj, c, lw=lw)
    bezier(X_WJ,  ywj, X_VB, yvb, c, lw=lw)

    # Nodes
    node(X_HUM, yh,  c, rh)
    node(X_WJ,  ywj, c, rwj)
    node(X_VB,  yvb, c, rvb)

    # Delta badge — solid fill pill
    delta = rvb - rh
    if delta == 0:
        bcol, txt = "#059669", "= 0"
    elif abs(delta) == 1:
        bcol = "#D97706"
        txt  = ("▲" if delta < 0 else "▼") + str(abs(delta))
    else:
        bcol = "#B91C1C"
        txt  = ("▲" if delta < 0 else "▼") + str(abs(delta))

    bx0, bx1 = COL["delta"]
    bw = (bx1 - bx0) * 0.78
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx("delta") - bw/2, yh - 0.024), bw, 0.048,
        boxstyle="round,pad=0.006", linewidth=0,
        facecolor=bcol, alpha=0.15,
        transform=ax.transAxes, zorder=4))
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx("delta") - bw/2, yh - 0.024), bw, 0.048,
        boxstyle="round,pad=0.006", linewidth=1.2,
        edgecolor=bcol, facecolor="none", alpha=0.6,
        transform=ax.transAxes, zorder=4))
    ax.text(cx("delta"), yh, txt, color=bcol, fontsize=9.5, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=5)

# ── Tier boundary lines ────────────────────────────────────────────────────────
# Tier 1/2 boundary: between rank 2 and rank 3
# Tier 2/3 boundary: between rank 5 and rank 6
TIER_COL = "#F59E0B"   # warm amber — stands out without being jarring
for r_above, tier_label in [(2, "── Tier boundary ──"), (5, "── Tier boundary ──")]:
    y_boundary = (ry(r_above) + ry(r_above + 1)) / 2
    # Subtle full-width band
    ax.add_patch(mpatches.FancyBboxPatch(
        (COL["hum"][0], y_boundary - 0.008), COL["vb"][1] - COL["hum"][0], 0.016,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=TIER_COL, alpha=0.10,
        transform=ax.transAxes, zorder=7))
    # Dashed centre line
    ax.plot([COL["hum"][0], COL["vb"][1]], [y_boundary, y_boundary],
            color=TIER_COL, lw=1.4, linestyle="--", dashes=(6, 3),
            alpha=0.75, transform=ax.transAxes, zorder=8)
    # Floating label in the model column
    ax.text(cx("model"), y_boundary, tier_label,
            color=TIER_COL, fontsize=6.5, fontweight="bold",
            ha="center", va="center", style="italic",
            transform=ax.transAxes, zorder=9)

# ── Bottom content border line ─────────────────────────────────────────────────
ax.plot([0.015, 0.985], [CONT_BOT, CONT_BOT],
        color="#CBD5E1", lw=1.0, transform=ax.transAxes, zorder=3)

# ── Summary stats strip ───────────────────────────────────────────────────────
ax.text(0.13, STRIP_Y, "vs. Human BT\nranking:",
        color="#64748B", fontsize=7.5, ha="center", va="center",
        transform=ax.transAxes, linespacing=1.4)

for xc, label, stat, hdr_col, txt_col in [
    (0.43, "WorldJen BT",
     "ρ = 1.00 (tier concordance)  ·  15 / 15 pairs correct",
     "#1D4ED8", "#1D4ED8"),
    (0.75, "VBench Quality",
     "ρ = +0.60 (p = 0.21, n.s.)  ·  11 / 15 pairs correct",
     "#B91C1C", "#B91C1C"),
]:
    ax.text(xc, STRIP_Y + 0.013, label,
            color=hdr_col, fontsize=8.5, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(xc, STRIP_Y - 0.011, stat,
            color=txt_col, fontsize=7.6,
            ha="center", va="center", transform=ax.transAxes)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#059669", alpha=0.9, label="Δ = 0  (agrees with human)"),
    mpatches.Patch(facecolor="#D97706", alpha=0.9, label="|Δ| = 1  (within-tier shift)"),
    mpatches.Patch(facecolor="#B91C1C", alpha=0.9, label="|Δ| ≥ 2  (tier-boundary crossing)"),
]
ax.legend(handles=legend_items, loc="upper center",
          bbox_to_anchor=(0.5, 1.03), ncol=3,
          framealpha=0.97, edgecolor="#CBD5E1",
          fontsize=8.5, facecolor="white", labelcolor="#1E293B")

plt.tight_layout(pad=0.2)
for out in [OUT_PDF, OUT_PNG,
            str(LATEX_DIR / "vbench_bump_chart.pdf"),
            str(LATEX_DIR / "vbench_bump_chart.png")]:
    plt.savefig(out, bbox_inches="tight", pad_inches=0.06, dpi=150, facecolor="white")
plt.close()
print(f"Saved → {OUT_PDF}  (+ latex/ copies)")
