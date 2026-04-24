"""
generate_figures.py — produce publication-quality figures for the WorldJen paper.

Outputs (all written to the same directory as this script):
  coverage_hist.pdf          — per-prompt dimension-coverage distribution before/after enhancement
  enhancement_comparison.pdf — per-dimension suitability before/after bar chart
  heatmap.pdf                — model × dimension Likert score heatmap with BT rating sidebar

Usage:
  python generate_figures.py

Dependencies: matplotlib, numpy  (pip install matplotlib numpy)
"""

import json
import csv
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent
DATA_ROOT    = pathlib.Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
JUDGED_PATH  = DATA_ROOT / "prompts/prompts_judged_full.jsonl"
RESCORED_PATH= DATA_ROOT / "prompts/prompts_enhanced_full.jsonl"
COMPARISON_CSV = DATA_ROOT / "prompts/suitability_comparison.csv"

OUT_DIR   = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Dimension map: group → [dim_key, …] ──────────────────────────────────────
GROUP_DIMS = {
    "motion_stability": [
        "subject_consistency",
        "scene_consistency",
        "motion_smoothness",
        "temporal_flickering",
        "inertial_consistency",
    ],
    "logic_physics": [
        "physical_mechanics",
        "object_permanence",
        "human_fidelity",
        "dynamic_degree",
    ],
    "instruction_adherence": [
        "semantic_adherence",
        "spatial_relationship",
        "semantic_drift",
    ],
    "aesthetic_quality": [
        "composition_framing",
        "lighting_volumetric",
        "color_harmony",
        "structural_gestalt",
    ],
}
ALL_DIMS = [d for dims in GROUP_DIMS.values() for d in dims]   # 16 total
THRESHOLD = 6   # suitability > 6  →  dimension "covered"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract per-prompt coverage count from a JSONL file
# ─────────────────────────────────────────────────────────────────────────────
def load_coverage_counts(jsonl_path: pathlib.Path) -> list[int]:
    counts = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n = 0
            for group, dims in GROUP_DIMS.items():
                if group not in rec:
                    continue
                gdata = rec[group]
                if not gdata:
                    continue
                for dim in dims:
                    key = f"{dim}_suitability"
                    val = gdata.get(key)
                    if val is not None and val > THRESHOLD:
                        n += 1
            counts.append(n)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Coverage histogram
# ─────────────────────────────────────────────────────────────────────────────
def make_coverage_hist():
    print("Loading judged.jsonl …")
    before = load_coverage_counts(JUDGED_PATH)
    print("Loading rescored.jsonl …")
    after  = load_coverage_counts(RESCORED_PATH)

    print(f"  Before → mean={np.mean(before):.2f}, median={np.median(before):.0f}")
    print(f"  After  → mean={np.mean(after):.2f},  median={np.median(after):.0f}")

    bins = np.arange(-0.5, 17.5, 1)   # 0..16 integer bins

    fig, ax = plt.subplots(figsize=(7, 3.6))

    # shaded background: before
    ax.hist(before, bins=bins, density=True,
            color="#4C72B0", alpha=0.45, label="Before enhancement",
            edgecolor="white", linewidth=0.4)
    # solid outline: after
    ax.hist(after,  bins=bins, density=True,
            color="#DD8452", alpha=0.60, label="After enhancement",
            edgecolor="white", linewidth=0.4)

    ax.axvline(np.median(before), color="#4C72B0", linestyle="--",
               linewidth=1.1, alpha=0.85)
    ax.axvline(np.median(after),  color="#DD8452", linestyle="--",
               linewidth=1.1, alpha=0.85)

    ax.set_xlabel("Number of dimensions with suitability $> 6$  (out of 16)")
    ax.set_ylabel("Proportion of prompts")
    ax.set_xlim(-0.5, 16.5)
    ax.set_xticks(range(0, 17))
    ax.legend(framealpha=0.9)

    # annotation: median lines
    ax.text(np.median(before) - 0.15, ax.get_ylim()[1] * 0.92,
            f"med={int(np.median(before))}", ha="right", fontsize=8,
            color="#4C72B0")
    ax.text(np.median(after)  + 0.15, ax.get_ylim()[1] * 0.92,
            f"med={int(np.median(after))}", ha="left", fontsize=8,
            color="#DD8452")

    out = OUT_DIR / "coverage_hist.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-dimension enhancement bar chart
# ─────────────────────────────────────────────────────────────────────────────
DIM_LABELS = {
    "subject_consistency":  "Subject Consist.",
    "scene_consistency":    "Scene Consist.",
    "motion_smoothness":    "Motion Smooth.",
    "temporal_flickering":  "Temporal Flicker.",
    "inertial_consistency": "Inertial Consist.",
    "physical_mechanics":   "Physical Mech.",
    "object_permanence":    "Object Perm.",
    "human_fidelity":       "Human Fidelity",
    "dynamic_degree":       "Dynamic Degree",
    "semantic_adherence":   "Semantic Adher.",
    "spatial_relationship": "Spatial Relat.",
    "semantic_drift":       "Semantic Drift",
    "composition_framing":  "Composition",
    "lighting_volumetric":  "Lighting & Vol.",
    "color_harmony":        "Color Harmony",
    "structural_gestalt":   "Struct. Gestalt",
}

GROUP_COLORS = {
    "motion_stability":     "#4C72B0",
    "logic_physics":        "#C44E52",
    "instruction_adherence":"#55A868",
    "aesthetic_quality":    "#8172B2",
}

GROUP_DISPLAY = {
    "motion_stability":     "A: Motion & Stability",
    "logic_physics":        "B: Logic & Physics",
    "instruction_adherence":"C: Instruction Adherence",
    "aesthetic_quality":    "D: Aesthetic Quality",
}


def parse_comparison_csv(csv_path: pathlib.Path):
    """Return list of (group, dim, before, after) for the Suitability block."""
    rows = []
    in_suit_block = False
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            joined = ",".join(row).strip()
            if joined.startswith("# Suitability"):
                in_suit_block = True
                continue
            if joined.startswith("#") and in_suit_block:
                break          # hit next comment block → stop
            if not in_suit_block:
                continue
            if row[0] == "group" and row[1] == "dimension":
                continue       # header row
            if len(row) < 5:
                continue
            group, dim, enh, unenh = row[0], row[1], float(row[2]), float(row[3])
            rows.append((group, dim, unenh, enh))   # (group, dim, before, after)
    return rows


def make_enhancement_comparison():
    data = parse_comparison_csv(COMPARISON_CSV)

    # Preserve group ordering
    ordered = []
    for group, dims in GROUP_DIMS.items():
        for dim in dims:
            match = [(g, d, b, a) for (g, d, b, a) in data if g == group and d == dim]
            if match:
                ordered.append(match[0])

    n = len(ordered)
    labels    = [DIM_LABELS.get(d, d) for (_, d, _, _) in ordered]
    before    = [b for (_, _, b, _) in ordered]
    after_    = [a for (_, _, _, a) in ordered]
    colors    = [GROUP_COLORS[g] for (g, _, _, _) in ordered]

    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.5))

    bars_b = ax.bar(x - bar_w/2, before, bar_w,
                    color=[c + "88" for c in colors],   # semi-transparent via alpha
                    label="Before", edgecolor="white", linewidth=0.5)
    bars_a = ax.bar(x + bar_w/2, after_,  bar_w,
                    color=colors, alpha=0.92,
                    label="After",  edgecolor="white", linewidth=0.5)

    # Use alpha properly instead of hex suffix for matplotlib compatibility
    for bar, c in zip(bars_b, colors):
        bar.set_facecolor(c)
        bar.set_alpha(0.40)
    for bar, c in zip(bars_a, colors):
        bar.set_facecolor(c)
        bar.set_alpha(0.90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Suitability score (1–10)")
    ax.set_ylim(0, 10.5)
    ax.set_yticks(range(0, 11, 2))

    # Group separator lines
    group_sizes = [len(dims) for dims in GROUP_DIMS.values()]
    boundaries = np.cumsum(group_sizes)[:-1] - 0.5
    for b in boundaries:
        ax.axvline(b, color="grey", linewidth=0.6, linestyle=":", alpha=0.7)

    # Group labels above bars
    cum = 0
    for g, dims in GROUP_DIMS.items():
        mid = cum + len(dims) / 2 - 0.5
        ax.text(mid, 10.2, GROUP_DISPLAY[g], ha="center", fontsize=8,
                color=GROUP_COLORS[g], fontweight="bold")
        cum += len(dims)

    # Legend
    before_patch = mpatches.Patch(color="grey", alpha=0.40, label="Before enhancement")
    after_patch  = mpatches.Patch(color="grey", alpha=0.90, label="After enhancement")
    ax.legend(handles=[before_patch, after_patch], loc="lower right", framealpha=0.9)

    out = OUT_DIR / "enhancement_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Model × Dimension heatmap
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_JSON = DATA_ROOT / "results/summaries/summary_report_unified.json"

# Canonical dimension order (grouped by evaluation group)
DIM_ORDER = [
    # Group A: Motion & Stability
    "subject_consistency",
    "scene_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "inertial_consistency",
    # Group B: Logic & Physics
    "physical_mechanics",
    "object_permanence",
    "human_fidelity",
    "dynamic_degree",
    # Group C: Instruction Adherence
    "semantic_adherence",
    "spatial_relationship",
    "semantic_drift",
    # Group D: Aesthetic Quality
    "composition_framing",
    "lighting_volumetric",
    "color_harmony",
    "structural_gestalt",
]

DIM_LABELS_HEAT = {
    "subject_consistency":  "Subject Consistency",
    "scene_consistency":    "Scene Consistency",
    "motion_smoothness":    "Motion Smoothness",
    "temporal_flickering":  "Temporal Flickering",
    "inertial_consistency": "Inertial Consistency",
    "physical_mechanics":   "Physical Mechanics",
    "object_permanence":    "Object Permanence",
    "human_fidelity":       "Human Fidelity",
    "dynamic_degree":       "Dynamic Degree",
    "semantic_adherence":   "Semantic Adherence",
    "spatial_relationship": "Spatial Relationship",
    "semantic_drift":       "Semantic Drift",
    "composition_framing":  "Composition & Framing",
    "lighting_volumetric":  "Lighting & Volumetric",
    "color_harmony":        "Color Harmony",
    "structural_gestalt":   "Structural Gestalt",
}

# Model display names & canonical sort (by VLM BT rating descending)
MODEL_KEY_TO_SHORT = {
    "fal-ai_veo3.1_fast":                       "Veo 3.1 Fast",
    "fal-ai_ltx-2_text-to-video":               "LTX-2",
    "fal-ai_kling-video_v2.6_pro_text-to-video":"Kling v2.6 Pro",
    "wan2.1-1.3b":                               "Wan 2.1 1.3B",
    "fal-ai_wan_v2.2-a14b_text-to-video":        "Wan v2.2 A14B",
    "fal-ai_hunyuan-video-v1.5_text-to-video":   "Hunyuan v1.5",
}

# Group boundaries (row indices where a new group starts, 0-based)
GROUP_BOUNDARIES = [0, 5, 9, 12, 16]
GROUP_LABELS = [
    "A: Motion & Stability",
    "B: Logic & Physics",
    "C: Instruction Adherence",
    "D: Aesthetic Quality",
]
GROUP_COLORS_HEAT = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]


def make_heatmap():
    with open(SUMMARY_JSON) as f:
        report = json.load(f)

    mds  = report["model_dimension_stats"]   # {dim: {model_key: score}}
    elos = report["bt_ratings"]             # {model_key: bt}

    # Sort models by BT rating descending
    model_keys = sorted(elos.keys(), key=lambda m: elos[m], reverse=True)
    model_labels = [MODEL_KEY_TO_SHORT.get(k, k) for k in model_keys]
    n_models = len(model_keys)
    n_dims   = len(DIM_ORDER)

    # Build matrix: rows = dimensions (top to bottom), cols = models (left to right)
    import numpy as np
    mat = np.zeros((n_dims, n_models))
    for di, dim in enumerate(DIM_ORDER):
        for mi, mk in enumerate(model_keys):
            mat[di, mi] = mds[dim][mk]

    # ── Figure layout ─────────────────────────────────────────────────────────
    # ax_lbl  : combined group-colour stripe + dimension name labels (left panel)
    #           All text lives here; ax_heat has NO y-tick labels.
    # ax_heat : pure heatmap cells + model-name x-ticks on top
    # ax_bt  : BT rating vertical bars at the bottom, x-aligned with heatmap columns
    # ax_cb   : colorbar (below BT rating panel)
    #
    # L     B     W     H    (figure fractions, figure = 13 × 8.5 inches)
    LL, LB, LW, LH = 0.01, 0.19, 0.235, 0.76   # left label panel
    HL, HB, HW, HH = 0.25, 0.19,  0.72, 0.76   # heatmap (wider — no right BT rating col)
    EL, EB, EW, EH = 0.25, 0.08,  0.72, 0.09   # BT rating vertical bars (bottom)
    CL, CB, CW, CH = 0.25, 0.02,  0.72, 0.022  # colorbar

    fig = plt.figure(figsize=(13.0, 8.5))
    ax_lbl  = fig.add_axes([LL, LB, LW, LH])
    ax_heat = fig.add_axes([HL, HB, HW, HH])
    ax_bt  = fig.add_axes([EL, EB, EW, EH])
    ax_cb   = fig.add_axes([CL, CB, CW, CH])

    # ── Left label panel: colour stripe + dimension names ─────────────────────
    # y axis mirrors heatmap: 0 = top row, n_dims-1 = bottom row
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(n_dims - 0.5, -0.5)
    ax_lbl.axis("off")

    STRIPE_W = 0.09   # fraction of ax_lbl width occupied by the colour stripe

    for i, (start, end, lbl, col) in enumerate(zip(
            GROUP_BOUNDARIES[:-1], GROUP_BOUNDARIES[1:],
            GROUP_LABELS, GROUP_COLORS_HEAT)):
        n_rows = end - start
        # Colour stripe rectangle
        rect = plt.Rectangle((0, start - 0.5), STRIPE_W, n_rows,
                               facecolor=col, alpha=0.90, zorder=2,
                               clip_on=False)
        ax_lbl.add_patch(rect)
        # Group letter centred in the stripe
        mid = (start + end - 1) / 2
        letter = lbl.split(":")[0]
        ax_lbl.text(STRIPE_W / 2, mid, letter,
                    ha="center", va="center", fontsize=8.0,
                    color="white", fontweight="bold", zorder=3)
        # White separator between groups
        if start > 0:
            ax_lbl.axhline(start - 0.5, color="white", linewidth=1.8, zorder=4)

    # Dimension name labels — right-aligned, inside the panel
    for di, dim in enumerate(DIM_ORDER):
        ax_lbl.text(0.99, di, DIM_LABELS_HEAT[dim],
                    ha="right", va="center", fontsize=8.2,
                    color="#1A1A1A", zorder=3)

    # Thin vertical rule separating panel from heatmap
    ax_lbl.axvline(1.0, color="#CCCCCC", linewidth=0.8, zorder=1)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    cmap = plt.cm.RdYlGn
    vmin, vmax = 1.0, 5.0

    im = ax_heat.imshow(mat, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation="nearest")

    # Cell value annotations
    for di in range(n_dims):
        for mi in range(n_models):
            val = mat[di, mi]
            txt_col = "white" if (val < 2.2 or val > 4.7) else "#222222"
            ax_heat.text(mi, di, f"{val:.2f}",
                         ha="center", va="center",
                         fontsize=7.8, color=txt_col, fontweight="bold")

    # Model name x-ticks on top — NO y-tick labels
    ax_heat.set_xticks(range(n_models))
    ax_heat.set_xticklabels(model_labels, rotation=0, ha="center", fontsize=9.2)
    ax_heat.xaxis.set_ticks_position("top")
    ax_heat.xaxis.set_label_position("top")
    ax_heat.set_yticks([])
    ax_heat.tick_params(top=True, bottom=False, left=False, right=False, length=0)

    # Thick white group separator lines
    for start in GROUP_BOUNDARIES[1:-1]:
        ax_heat.axhline(start - 0.5, color="white", linewidth=2.5)

    # Fine cell grid
    for x in np.arange(-0.5, n_models, 1):
        ax_heat.axvline(x, color="white", linewidth=0.7)
    for y in np.arange(-0.5, n_dims, 1):
        ax_heat.axhline(y, color="white", linewidth=0.5)

    ax_heat.set_title("VLM Evaluation: Model × Dimension Likert Scores  (1–5 scale)",
                      fontsize=10.5, pad=28)

    # ── BT rating bottom panel (vertical bars, x-aligned with heatmap columns) ─────
    bt_vals = [elos[mk] for mk in model_keys]
    bt_norm = [(e - min(bt_vals)) / max(max(bt_vals) - min(bt_vals), 1e-9)
                for e in bt_vals]
    bar_col  = [plt.cm.Greens(0.35 + 0.55 * n) for n in bt_norm]

    ax_bt.bar(range(n_models), bt_vals, color=bar_col,
               width=0.65, edgecolor="white", linewidth=0.5)
    ax_bt.set_xticks([])          # model names already shown on heatmap top
    ylo = max(1000, min(bt_vals) - 50)
    ax_bt.set_ylim(ylo, max(bt_vals) + 50)
    ax_bt.set_yticks(np.round(np.linspace(ylo, max(bt_vals), 4), -1).astype(int))
    ax_bt.tick_params(left=True, bottom=False, labelsize=6.5)
    ax_bt.set_ylabel("VLM BT rating", fontsize=7.5, labelpad=3)
    ax_bt.set_xlim(-0.5, n_models - 0.5)
    for spine in ["top", "right", "bottom"]:
        ax_bt.spines[spine].set_visible(False)
    ax_bt.spines["left"].set_color("#AAAAAA")
    ax_bt.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    for i, ev in enumerate(bt_vals):
        ax_bt.text(i, ev + (max(bt_vals) - min(bt_vals)) * 0.02,
                    f"{int(ev)}", ha="center", va="bottom",
                    fontsize=6.8, color="#333333")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_label("Likert score (1–5)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_ticks([1.0, 2.0, 3.0, 4.0, 5.0])

    out = OUT_DIR / "heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    out_png = OUT_DIR / "heatmap.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {out}  +  {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Coverage histogram ===")
    make_coverage_hist()
    print("=== Enhancement comparison ===")
    make_enhancement_comparison()
    print("=== Model × Dimension heatmap ===")
    make_heatmap()
    print("Done.")
