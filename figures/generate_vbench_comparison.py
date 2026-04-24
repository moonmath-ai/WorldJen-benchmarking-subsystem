"""
generate_vbench_comparison.py

Creates a figure showing VBench vs WorldJen normalized scores (0-1)
for each model on the 5 overlapping dimensions.
"""
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
VB_FILE = DATA_ROOT / "vbench/vbench_summary.json"
WJ_FILE = DATA_ROOT / "results/summaries/summary_report_unified.json"
OUT     = Path(__file__).parent / "vbench_worldjen_comparison.pdf"

with open(VB_FILE) as f:
    vbd = json.load(f)
with open(WJ_FILE) as f:
    wjd = json.load(f)

ORDER = [
    'fal-ai_veo3.1_fast',
    'fal-ai_kling-video_v2.6_pro_text-to-video',
    'fal-ai_wan_v2.2-a14b_text-to-video',
    'fal-ai_ltx-2_text-to-video',
    'fal-ai_hunyuan-video-v1.5_text-to-video',
    'wan2.1-1.3b',
]
MODEL_LABELS = ['Veo 3.1', 'Kling', 'Wan A14B', 'LTX-2', 'HunyuanVideo', 'Wan 1.3B']

# WJ BT rating ranks (1-best, 6-worst)
WJ_BT_RANKS = [1, 2, 3, 4, 5, 6]

PAIRS = [
    ('Subject\nConsistency',  'subject_consistency',   'v1', 'subject_consistency'),
    ('Scene\nConsistency',    'background_consistency', 'v1', 'scene_consistency'),
    ('Motion\nSmoothness',    'motion_smoothness',      'v1', 'motion_smoothness'),
    ('Dynamic\nDegree',       'dynamic_degree',         'v1', 'dynamic_degree'),
    ('Human\nQuality',        'Human_Anatomy',          'v2', 'human_fidelity'),
]
N_DIMS = len(PAIRS)
N_MODELS = len(ORDER)

# ── Collect data ─────────────────────────────────────────────────────────────
vb_scores = np.zeros((N_DIMS, N_MODELS))
wj_scores = np.zeros((N_DIMS, N_MODELS))

for di, (label, vb_key, src, wj_key) in enumerate(PAIRS):
    for mi, mk in enumerate(ORDER):
        if src == 'v1':
            vb = vbd['models'][mk]['vbench'][vb_key]
        else:
            vb = vbd['models'][mk].get('vbench2', {}).get(vb_key, np.nan)
        wj_raw = wjd['model_dimension_stats'][wj_key].get(mk)
        wj_n   = (wj_raw - 1) / 4 if wj_raw is not None else np.nan
        vb_scores[di, mi] = vb
        wj_scores[di, mi] = wj_n

vb_mean = vb_scores.mean(axis=0)
wj_mean = wj_scores.mean(axis=0)

# ── Colours ──────────────────────────────────────────────────────────────────
VB_COLOR = "#4C72B0"    # blue for VBench
WJ_COLOR = "#DD8452"    # orange for WorldJen

# ── Figure: two panels ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Panel A: per-dimension heatmap comparison ────────────────────────────────
ax = axes[0]
dim_labels = [p[0] for p in PAIRS]

# Stack: VBench row then WorldJen row per dimension — show as grouped bars
x = np.arange(N_MODELS)
width = 0.35

for di, dlabel in enumerate(dim_labels):
    offset_y = (N_DIMS - 1 - di) * 1.0   # vertical offset per dim

# Instead: simple grouped bar chart per model, averaged across dims
x = np.arange(N_MODELS)
w = 0.35
bars_vb = ax.bar(x - w/2, vb_mean, w, label='VBench (5 overlapping dims)', color=VB_COLOR, alpha=0.85)
bars_wj = ax.bar(x + w/2, wj_mean, w, label='WorldJen (5 overlapping dims)', color=WJ_COLOR, alpha=0.85)

# Annotate bars
for bar in bars_vb:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5, color=VB_COLOR)
for bar in bars_wj:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5, color=WJ_COLOR)

# Add WorldJen BT rating rank annotation above each model pair
for mi, (rank, mvb, mwj) in enumerate(zip(WJ_BT_RANKS, vb_mean, wj_mean)):
    ax.text(mi, max(mvb, mwj) + 0.025, f'WJ rank: {rank}', ha='center', va='bottom',
            fontsize=7, color='#333333', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=9)
ax.set_ylim(0.6, 1.05)
ax.set_ylabel('Normalised score (0–1 scale)', fontsize=10)
ax.set_title('(a) Average score on overlapping dimensions', fontsize=10, pad=10)
ax.legend(fontsize=8.5, loc='lower right')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# ── Panel B: per-dimension breakdown ─────────────────────────────────────────
ax2 = axes[1]

# Show per-dimension scores as grouped mini-charts
colors_dim = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

for di, (dlabel, vb_key, src, wj_key) in enumerate(PAIRS):
    y_vb = vb_scores[di]
    y_wj = wj_scores[di]
    # Scatter: model index on x for VBench, dashed for WorldJen
    color = colors_dim[di]
    ax2.plot(x, y_vb, 'o-', color=color, lw=1.5, ms=6, label=f'{dlabel.replace(chr(10)," ")} (VB)')
    ax2.plot(x, y_wj, 's--', color=color, lw=1.5, ms=6, alpha=0.6)

ax2.set_xticks(x)
ax2.set_xticklabels(MODEL_LABELS, fontsize=9)
ax2.set_ylim(0.5, 1.05)
ax2.set_ylabel('Normalised score (0–1 scale)', fontsize=10)
ax2.set_title('(b) Per-dimension: VBench (solid) vs WorldJen (dashed)', fontsize=10, pad=10)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)

# Custom legend for dimensions
legend_elements = [
    Patch(facecolor=colors_dim[di], label=PAIRS[di][0].replace('\n',' '))
    for di in range(N_DIMS)
]
legend_elements += [
    plt.Line2D([0],[0], color='black', lw=1.5, marker='o', label='VBench'),
    plt.Line2D([0],[0], color='black', lw=1.5, marker='s', linestyle='--', alpha=0.6, label='WorldJen'),
]
ax2.legend(handles=legend_elements, fontsize=7.5, loc='lower left', ncol=2)

fig.suptitle(
    'VBench vs WorldJen — normalised scores on 5 shared dimensions\n'
    r'VBench–WorldJen overlap rank correlation: $\hat{\rho}=-0.54$ ($p=0.27$, $n=6$)',
    fontsize=11, y=1.01)

plt.tight_layout()
plt.savefig(OUT, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved → {OUT}")
