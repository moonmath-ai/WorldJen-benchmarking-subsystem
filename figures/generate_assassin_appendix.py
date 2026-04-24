"""
generate_assassin_appendix.py

Produces two appendix figures for prompt_1102 (the "assassin" case study):

  assassin_frame_grid.pdf   — 2×3 grid of one representative frame per model
  assassin_vlm_scores.pdf   — heatmap of per-question Inertial Consistency
                              Likert scores across all six models

Run:  python generate_assassin_appendix.py
"""

import os
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT         = pathlib.Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
VLM_DIR     = DATA_ROOT / "results/gemini_vlm"
VID_ROOT    = pathlib.Path(
    DATA_ROOT / "videos")
PROMPT_ID   = "prompt_1102"

# ── Model registry ────────────────────────────────────────────────────────────
# (model_key, short_label, vlm_json_stem)
MODEL_ORDER = [
    ("fal-ai_veo3.1_fast",
     "Veo 3.1",
     "fal-ai_veo3.1_fast_prompt_1102"),
    ("fal-ai_kling-video_v2.6_pro_text-to-video",
     "Kling v2.6",
     "fal-ai_kling-video_v2.6_pro_text-to-video_prompt_1102"),
    ("fal-ai_ltx-2_text-to-video",
     "LTX-2",
     "fal-ai_ltx-2_text-to-video_prompt_1102"),
    ("fal-ai_wan_v2.2-a14b_text-to-video",
     "Wan A14B",
     "fal-ai_wan_v2.2-a14b_text-to-video_prompt_1102"),
    ("fal-ai_hunyuan-video-v1.5_text-to-video",
     "Hunyuan",
     "fal-ai_hunyuan-video-v1.5_text-to-video_prompt_1102"),
    ("wan2.1-1.3b",
     "Wan 1.3B",
     "wan2.1-1.3b_prompt_1102"),
]

DIM_KEY = "inertial_consistency"

# Short question labels (Q1–Q10)
Q_LABELS = [
    "Q1: Horse deceleration realism",
    "Q2: Assassin hair inertia (stop)",
    "Q3: Mud/grass projectile physics",
    "Q4: Rider upper-body forward lurch",
    "Q5: Speed ramp-down continuity",
    "Q6: Force consistency (friction/weight)",
    "Q7: Horse haunches-down posture",
    "Q8: Mane/tail post-halt motion",
    "Q9: Secondary objects (cloak/bags)",
    "Q10: Overall deceleration coherence",
]

# RdYlGn colour map (1=red → 3=yellow → 5=green)
CMAP = LinearSegmentedColormap.from_list(
    "rdylgn5", ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"])


# ── Helper: extract N frames evenly spaced across the full video ──────────────
# Timestamps chosen to capture the full deceleration arc for inertial consistency:
#   t=10% (full gallop) → 25% → 40% → 55% → 70% → 85% (post-halt settling)
SAMPLE_POINTS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]

def get_frames(vid_path: pathlib.Path,
               points: list = SAMPLE_POINTS) -> list:
    """Return one RGB frame per timestamp fraction in `points`."""
    blank = np.full((180, 320, 3), 200, dtype=np.uint8)
    if not vid_path.exists():
        return [blank] * len(points)
    cap = cv2.VideoCapture(str(vid_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for t in points:
        idx = max(0, min(int(total * t), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else blank)
    cap.release()
    return frames


# ── Helper: load per-question IC scores for one model ─────────────────────────
def load_ic_scores(json_stem: str) -> list[int] | None:
    p = VLM_DIR / f"{json_stem}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    entries = d.get("results", {}).get(DIM_KEY, [])
    return [e["score"] for e in entries]


# ══════════════════════════════════════════════════════════════════════════════
# Figure A — 6×6 grid: 6 models (columns) × 6 consecutive frames (rows)
# ══════════════════════════════════════════════════════════════════════════════
def make_frame_grid():
    N_FRAMES  = len(SAMPLE_POINTS)   # rows (one per timestamp)
    N_MODELS  = 6                     # columns

    # Collect frames: frames_by_model[model_idx] = list of N_FRAMES RGB arrays
    frames_by_model = []
    ic_avgs = []
    for mk, short, jstem in MODEL_ORDER:
        vid  = VID_ROOT / mk / f"{PROMPT_ID}.mp4"
        frames_by_model.append(get_frames(vid))
        ic_scores = load_ic_scores(jstem)
        ic_avgs.append(np.mean(ic_scores) if ic_scores else 0.0)

    # ── Figure: 6 rows × 6 cols, with a header row above ─────────────────────
    # Use gridspec: row 0 = model-name headers, rows 1-6 = frames
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(15.0, 12.0))
    gs  = GridSpec(N_FRAMES + 1, N_MODELS,
                   figure=fig,
                   height_ratios=[0.38] + [1.0] * N_FRAMES,
                   hspace=0.04, wspace=0.03,
                   left=0.05, right=0.99, top=0.94, bottom=0.02)

    model_labels = [m[1] for m in MODEL_ORDER]

    # ── Header row: model name + IC score ─────────────────────────────────────
    for mi, (short, ic_avg) in enumerate(zip(model_labels, ic_avgs)):
        ax_hdr = fig.add_subplot(gs[0, mi])
        ax_hdr.axis("off")
        badge_col = CMAP((ic_avg - 1) / 4)
        for spine in ax_hdr.spines.values():
            spine.set_visible(False)
        ax_hdr.patch.set_visible(True)
        ax_hdr.set_facecolor(badge_col)
        # Model name (top)
        ax_hdr.text(0.5, 0.68, short,
                    ha="center", va="center", fontsize=11.0, fontweight="bold",
                    color="white", transform=ax_hdr.transAxes)
        # IC score (bottom) — large and prominent
        ax_hdr.text(0.5, 0.22, f"IC = {ic_avg:.1f} / 5",
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white", transform=ax_hdr.transAxes,
                    bbox=dict(facecolor="black", alpha=0.25,
                              edgecolor="none", boxstyle="round,pad=0.3"))

    # ── Frame rows ────────────────────────────────────────────────────────────
    row_labels = [f"t≈{int(p*100)}%" for p in SAMPLE_POINTS]
    for fi in range(N_FRAMES):
        for mi in range(N_MODELS):
            ax = fig.add_subplot(gs[fi + 1, mi])
            ax.axis("off")
            frame = frames_by_model[mi][fi]
            ax.imshow(frame, aspect="auto")

            # Timestamp label on left edge of first column only
            if mi == 0:
                ax.text(-0.06, 0.5, row_labels[fi],
                        ha="right", va="center", fontsize=7.5,
                        color="#333333", transform=ax.transAxes)

    fig.suptitle(
        "Prompt 1102 — Frames sampled across full video duration per model  "
        "(rows = timestamps;  header colour = IC score)",
        fontsize=10, y=0.975)

    fig.savefig(OUT / "assassin_frame_grid.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved → assassin_frame_grid.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure B — per-question Inertial Consistency score heatmap
# ══════════════════════════════════════════════════════════════════════════════
def make_vlm_score_table():
    # Collect score matrix  (10 questions × 6 models)
    model_labels = [m[1] for m in MODEL_ORDER]
    score_matrix = []
    for _, _, jstem in MODEL_ORDER:
        sc = load_ic_scores(jstem)
        score_matrix.append(sc if sc else [np.nan] * 10)
    score_matrix = np.array(score_matrix, dtype=float).T   # (10, 6)

    # Per-model averages
    avg_row = np.nanmean(score_matrix, axis=0)   # (6,)

    # ── Layout axes (all in figure-fraction coords) ───────────────────────────
    # stripe | q-labels | heatmap | avg-row | colorbar
    LS, LB_s, LW_s, LH_s = 0.01, 0.14, 0.022, 0.78   # dim-colour stripe
    LL, LB,   LW,   LH   = 0.034, 0.14, 0.24,  0.78   # question labels
    HL, HB,   HW,   HH   = 0.278, 0.14, 0.68,  0.78   # main heatmap
    AL, AB,   AW,   AH   = 0.278, 0.07, 0.68,  0.055  # per-model avg row
    CL, CB,   CW,   CH   = 0.278, 0.01, 0.68,  0.025  # colorbar

    fig = plt.figure(figsize=(13.0, 7.5))
    ax_stripe = fig.add_axes([LS, LB_s, LW_s, LH_s])
    ax_lbl    = fig.add_axes([LL, LB,   LW,   LH])
    ax_heat   = fig.add_axes([HL, HB,   HW,   HH])
    ax_avg    = fig.add_axes([AL, AB,   AW,   AH])
    ax_cb     = fig.add_axes([CL, CB,   CW,   CH])

    N_Q = 10

    # ── Dimension colour stripe ───────────────────────────────────────────────
    ax_stripe.set_xlim(0, 1); ax_stripe.set_ylim(-0.5, N_Q - 0.5)
    ax_stripe.axis("off")
    dim_col = "#4E5FA8"   # same as C["llm"] in framework figures
    ax_stripe.add_patch(plt.Rectangle((0, -0.5), 1, N_Q,
                                      facecolor=dim_col, edgecolor="none"))
    ax_stripe.text(0.5, N_Q / 2 - 0.5, "Inertial\nConsistency",
                   ha="center", va="center", fontsize=8.5,
                   color="white", fontweight="bold", rotation=90)

    # ── Question labels ───────────────────────────────────────────────────────
    ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(-0.5, N_Q - 0.5)
    ax_lbl.axis("off")
    for qi, lbl in enumerate(reversed(Q_LABELS)):
        ax_lbl.text(0.02, qi, lbl,
                    ha="left", va="center", fontsize=8.0, color="#1A1A1A")

    # ── Main heatmap ──────────────────────────────────────────────────────────
    im = ax_heat.imshow(score_matrix, aspect="auto",
                        cmap=CMAP, vmin=1, vmax=5,
                        origin="upper")
    ax_heat.set_xticks(range(len(model_labels)))
    ax_heat.set_xticklabels(model_labels, fontsize=9.5, fontweight="bold")
    ax_heat.xaxis.tick_top()
    ax_heat.set_yticks([])
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Annotate cells
    for qi in range(N_Q):
        for mi in range(len(model_labels)):
            v = score_matrix[qi, mi]
            if not np.isnan(v):
                tc = "white" if v <= 2 or v >= 4.5 else "#1A1A1A"
                ax_heat.text(mi, qi, f"{v:.0f}",
                             ha="center", va="center",
                             fontsize=9.0, color=tc, fontweight="bold")

    # ── Per-model average row ─────────────────────────────────────────────────
    avg_mat = avg_row.reshape(1, -1)
    ax_avg.imshow(avg_mat, aspect="auto",
                  cmap=CMAP, vmin=1, vmax=5, origin="upper")
    ax_avg.set_xticks([])
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg"], fontsize=8.5, color="#555555")
    ax_avg.tick_params(length=0)
    for spine in ax_avg.spines.values():
        spine.set_visible(False)

    for mi, av in enumerate(avg_row):
        tc = "white" if av <= 2 or av >= 4.5 else "#1A1A1A"
        ax_avg.text(mi, 0, f"{av:.2f}",
                    ha="center", va="center",
                    fontsize=9.0, color=tc, fontweight="bold")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_ticks([1, 2, 3, 4, 5])
    cb.set_ticklabels(["1 (poor)", "2", "3", "4", "5 (excellent)"])
    cb.ax.tick_params(labelsize=8.0)

    fig.suptitle(
        "Inertial Consistency — Per-Question VLM Likert Scores  "
        f"(Prompt 1102 · 10 questions × 6 models)",
        fontsize=11, y=0.995)

    fig.savefig(OUT / "assassin_vlm_scores.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved → assassin_vlm_scores.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_frame_grid()
    make_vlm_score_table()
    print("All done.")
