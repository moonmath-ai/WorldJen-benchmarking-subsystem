"""
generate_gemma4_comparison.py
─────────────────────────────
Generates gemma4_comparison.pdf — a two-panel figure for the A6 ablation:

  Panel (a): Rank bump chart — Gemma4 BT vs Gemini BT vs Human BT
  Panel (b): PHAS score comparison — Gemma4 vs Gemini (bar chart)

Data sourced from:
  Gemma4  : data/results/summaries/summary_report_gemma4.json
  Gemini  : data/results/summaries/summary_report_unified.json
  Human BT: report_bt_anon.txt (2,696 votes, 7 annotators)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

HERE      = Path(__file__).parent

# ── Data ──────────────────────────────────────────────────────────────────────

MODEL_SHORT = {
    "fal-ai_veo3.1_fast":                         "Veo 3.1 Fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video":  "Kling 2.6 Pro",
    "fal-ai_wan_v2.2-a14b_text-to-video":         "Wan 2.2 A14B",
    "fal-ai_ltx-2_text-to-video":                 "LTX-2",
    "fal-ai_hunyuan-video-v1.5_text-to-video":    "HunyuanVideo",
    "wan2.1-1.3b":                                "Wan 2.1 1.3B",
}

# Human BT ranks/BT rating (from report_bt_anon.txt, 2,696 votes, ρ=1.000 with Gemini)
HUMAN_BT = {
    "fal-ai_veo3.1_fast":                        1614.2,
    "fal-ai_kling-video_v2.6_pro_text-to-video": 1571.8,
    "fal-ai_wan_v2.2-a14b_text-to-video":        1517.8,
    "fal-ai_ltx-2_text-to-video":               1479.1,
    "fal-ai_hunyuan-video-v1.5_text-to-video":  1461.7,
    "wan2.1-1.3b":                               1355.4,
}

with open(HERE.parent.parent / "gemini_VLM" / "summary_report_unified.json") as f:
    gemini_data = json.load(f)
with open(HERE.parent.parent / "Gemma4VLM" / "results" / "summary_report_gemma4.json") as f:
    gemma4_data = json.load(f)

MODELS = list(MODEL_SHORT.keys())

gemini_bt  = {m: gemini_data["bt_ratings"][m]  for m in MODELS}
gemma4_bt  = {m: gemma4_data["bt_rating"][m]       for m in MODELS}
gemini_phas = {m: gemini_data["phas_scores"][m]  for m in MODELS}
gemma4_phas = {m: gemma4_data["phas_scores"][m]  for m in MODELS}

def bt_to_ranks(bt_dict):
    sorted_models = sorted(bt_dict, key=lambda m: -bt_dict[m])
    return {m: r+1 for r, m in enumerate(sorted_models)}

human_ranks  = bt_to_ranks(HUMAN_BT)
gemini_ranks = bt_to_ranks(gemini_bt)
gemma4_ranks = bt_to_ranks(gemma4_bt)

# ── Colours ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "fal-ai_veo3.1_fast":                        "#6366f1",
    "fal-ai_kling-video_v2.6_pro_text-to-video": "#f59e0b",
    "fal-ai_wan_v2.2-a14b_text-to-video":        "#10b981",
    "fal-ai_ltx-2_text-to-video":               "#ef4444",
    "fal-ai_hunyuan-video-v1.5_text-to-video":  "#3b82f6",
    "wan2.1-1.3b":                               "#8b5cf6",
}
COLS = {
    "gemma4": "#059669",
    "gemini": "#4f46e5",
    "human":  "#dc2626",
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Panel (a): Bump chart ─────────────────────────────────────────────────────
xs = [0, 1, 2]   # Gemma4, Gemini, Human
labels = ["Gemma 4\n(open-source)", "Gemini Flash\n(closed-source)", "Human BT\n(ground truth)"]

for m in MODELS:
    ys = [gemma4_ranks[m], gemini_ranks[m], human_ranks[m]]
    c  = MODEL_COLORS[m]
    ax1.plot(xs, ys, "o-", color=c, lw=2, ms=9, zorder=3)
    # label at Human end (right side)
    ax1.text(2.08, human_ranks[m], MODEL_SHORT[m], va="center", fontsize=8.5,
             color=c, fontweight="bold")

ax1.set_xticks(xs)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_yticks([1, 2, 3, 4, 5, 6])
ax1.set_yticklabels(["#1", "#2", "#3", "#4", "#5", "#6"], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlim(-0.3, 2.85)
ax1.set_title("(a) BT rating rank comparison", fontsize=11, fontweight="bold", pad=8)
ax1.set_ylabel("Rank", fontsize=10)
ax1.grid(axis="y", alpha=0.25, ls="--")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Annotate discordances (Gemma4 vs Human)
for m in MODELS:
    if gemma4_ranks[m] != human_ranks[m]:
        ax1.annotate("", xy=(0, gemma4_ranks[m]), xytext=(2, human_ranks[m]),
                     arrowprops=dict(arrowstyle="<->", color=MODEL_COLORS[m],
                                     lw=0.8, ls="dotted"), zorder=2)

# ── Panel (b): PHAS bar chart ─────────────────────────────────────────────────
# Order by human rank
ordered = sorted(MODELS, key=lambda m: human_ranks[m])
x = np.arange(len(ordered))
w = 0.35

bars_g4 = ax2.bar(x - w/2, [gemma4_phas[m] for m in ordered], w,
                   label="Gemma 4 (open-source)",   color=COLS["gemma4"], alpha=0.85)
bars_ge = ax2.bar(x + w/2, [gemini_phas[m] for m in ordered], w,
                   label="Gemini Flash (closed-source)", color=COLS["gemini"], alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels([MODEL_SHORT[m] for m in ordered], rotation=30, ha="right", fontsize=8.5)
ax2.set_ylabel("PHAS score", fontsize=10)
ax2.set_ylim(3.4, 4.45)
ax2.set_title("(b) PHAS score comparison", fontsize=11, fontweight="bold", pad=8)
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(axis="y", alpha=0.25, ls="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Add value labels on bars
for bar in list(bars_g4) + list(bars_ge):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
             ha="center", va="bottom", fontsize=6.5)

plt.suptitle(
    "A6: Open-source (Gemma 4) vs. closed-source (Gemini Flash) vs. Human BT\n"
    r"$\hat{\rho}$(Gemma4, Human)$=0.771$   $\hat{\rho}$(Gemini, Human)$=1.000$   "
    r"$\hat{\rho}$(PHAS)$=0.943$",
    fontsize=10, y=1.02
)
plt.tight_layout()

OUT = HERE / "gemma4_comparison.pdf"
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
