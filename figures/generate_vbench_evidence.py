"""
generate_vbench_evidence.py
Creates vbench_evidence.pdf — three-panel figure showing WorldJen vs VBench
evaluation quality using updated human BT ground truth (2,696 votes, ρ=1.000).

Panels:
  (a) Human rank correlation bar chart (Spearman ρ)
  (b) Pairwise concordance bar chart (correct pairs out of 15)
  (c) Discrimination power: normalized score ranges on shared dimensions
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
LATEX_DIR = HERE.parent / 'latex'
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
VB_FILE   = DATA_ROOT / "vbench/vbench_summary.json"
WJ_FILE   = DATA_ROOT / "results/summaries/summary_report_unified.json"
OUT_PDF   = str(HERE / "vbench_evidence.pdf")
OUT_PNG   = str(HERE / "vbench_evidence.png")

ORDER = [
    'fal-ai_veo3.1_fast',
    'fal-ai_kling-video_v2.6_pro_text-to-video',
    'fal-ai_wan_v2.2-a14b_text-to-video',
    'fal-ai_ltx-2_text-to-video',
    'fal-ai_hunyuan-video-v1.5_text-to-video',
    'wan2.1-1.3b',
]

# Human BT ranks (from report_bt_anon.txt — 2,696 votes, ρ=1.000 with VLM)
HUMAN_RANK   = {m: r for r, m in enumerate(ORDER, 1)}
WORLDJEN_RANK = {m: r for r, m in enumerate(ORDER, 1)}
VBENCH_RANK   = {
    'fal-ai_veo3.1_fast':                         1,
    'fal-ai_kling-video_v2.6_pro_text-to-video':  4,
    'fal-ai_wan_v2.2-a14b_text-to-video':         3,
    'fal-ai_ltx-2_text-to-video':                 5,
    'fal-ai_hunyuan-video-v1.5_text-to-video':    2,
    'wan2.1-1.3b':                                6,
}

# ── Shared dims: label, vb_source ('vbench'|'vbench2'), VBench key, VBench max-possible, WJ key
SHARED_DIMS = [
    ("Subject\nConsistency",  "vbench",  "subject_consistency",    1.0,  "subject_consistency"),
    ("Scene /\nBackground",   "vbench",  "background_consistency", 1.0,  "scene_consistency"),
    ("Motion\nSmoothness",    "vbench",  "motion_smoothness",      1.0,  "motion_smoothness"),
    ("Dynamic\nDegree",       "vbench",  "dynamic_degree",         1.0,  "dynamic_degree"),
    ("Human\nAnatomy",        "vbench2", "Human_Anatomy",          1.0,  "human_fidelity"),
]

vbd = json.load(open(VB_FILE))
wjd = json.load(open(WJ_FILE))

# Compute pairwise concordance
from itertools import combinations
def concordant(rank_dict):
    models = list(rank_dict.keys())
    c = 0
    for a, b in combinations(models, 2):
        if (rank_dict[a] - rank_dict[b]) * (HUMAN_RANK[a] - HUMAN_RANK[b]) > 0:
            c += 1
    return c

wj_conc = concordant(WORLDJEN_RANK)   # 15/15
vb_conc = concordant(VBENCH_RANK)     # 11/15

# ── Normalized score ranges ────────────────────────────────────────────────────
vb_ranges, wj_ranges = [], []
dim_labels = []

for label, vb_src, vb_key, vb_max, wj_key in SHARED_DIMS:
    if vb_src == 'vbench2':
        vb_vals = [vbd['models'][m].get('vbench2', {}).get(vb_key, np.nan) for m in ORDER]
    else:
        vb_vals = [vbd['models'][m]['vbench'].get(vb_key, np.nan) for m in ORDER]
    wj_raw  = [wjd['model_dimension_stats'].get(wj_key, {}).get(m) for m in ORDER]
    wj_vals = [(v - 1) / 4 if v is not None else np.nan for v in wj_raw]

    vb_vals = [v for v in vb_vals if not np.isnan(v)]
    wj_vals = [v for v in wj_vals if not np.isnan(v)]

    vb_ranges.append((max(vb_vals) - min(vb_vals)) / vb_max)
    wj_ranges.append(max(wj_vals) - min(wj_vals))
    dim_labels.append(label)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.patch.set_facecolor('white')

WJ_COLOR = "#059669"   # green
VB_COLOR = "#dc2626"   # red
THRESHOLD = 0.886

# ── Panel (a): Spearman ρ ────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#f8fafc')
ax.spines[['top', 'right']].set_visible(False)

methods = ['WorldJen', 'VBench']
rhos    = [1.000, 0.600]
colors  = [WJ_COLOR, VB_COLOR]
bars    = ax.bar(methods, rhos, color=colors, alpha=0.85, width=0.45,
                 edgecolor=['#047857', '#b91c1c'], linewidth=1.4)

for bar, rho in zip(bars, rhos):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'ρ = {rho:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=bar.get_edgecolor())

ax.axhline(THRESHOLD, linestyle='--', color='#6b7280', linewidth=1.4, zorder=0)
ax.text(1.52, THRESHOLD + 0.01, f'p<0.05 threshold\n(ρ={THRESHOLD})', ha='right',
        va='bottom', fontsize=7.5, color='#6b7280')

ax.set_ylim(0, 1.28)
ax.set_ylabel('Spearman ρ (vs Human BT)', fontsize=10)
ax.set_title('(a) Human rank correlation', fontsize=10, fontweight='bold', pad=8)

# Significance annotation — placed above the ρ label with enough gap
ax.text(0, rhos[0] + 0.10, '★ p < 0.001', ha='center', fontsize=9, color=WJ_COLOR, fontweight='bold')
ax.text(1, rhos[1] + 0.10, 'n.s.  p = 0.21', ha='center', fontsize=9, color=VB_COLOR)

ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── Panel (b): Pairwise concordance ─────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('#f8fafc')
ax.spines[['top', 'right']].set_visible(False)

methods2 = ['WorldJen', 'VBench', 'Chance']
concs    = [wj_conc, vb_conc, 7.5]
colors2  = [WJ_COLOR, VB_COLOR, '#94a3b8']
bars2    = ax.bar(methods2, concs, color=colors2, alpha=0.85, width=0.45,
                  edgecolor=['#047857', '#b91c1c', '#64748b'], linewidth=1.4)

for bar, c in zip(bars2, concs):
    label = f'{int(c)}/15' if c != 7.5 else '7.5/15'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=bar.get_edgecolor())

ax.set_ylim(0, 18)
ax.set_ylabel('Pairs ordered correctly (of 15)', fontsize=10)
ax.set_title('(b) Pairwise concordance', fontsize=10, fontweight='bold', pad=8)
ax.axhline(7.5, linestyle=':', color='#94a3b8', linewidth=1.2, zorder=0)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── Panel (c): Discrimination power ─────────────────────────────────────────
ax = axes[2]
ax.set_facecolor('#f8fafc')
ax.spines[['top', 'right']].set_visible(False)

x = np.arange(len(dim_labels))
w = 0.35
b1 = ax.bar(x - w/2, vb_ranges, w, label='VBench', color=VB_COLOR, alpha=0.80,
            edgecolor='#b91c1c', linewidth=1.2)
b2 = ax.bar(x + w/2, wj_ranges, w, label='WorldJen', color=WJ_COLOR, alpha=0.80,
            edgecolor='#047857', linewidth=1.2)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=6.5, color='#b91c1c')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=6.5, color='#047857')

ax.set_xticks(x)
ax.set_xticklabels(dim_labels, fontsize=8)
ax.set_ylabel('Normalised score range (max–min)', fontsize=9)
ax.set_title('(c) Discrimination power\n(5 shared dimensions)', fontsize=10, fontweight='bold', pad=8)
ax.legend(fontsize=9, loc='upper right')
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

fig.suptitle(
    'WorldJen vs VBench: evaluation quality against human BT ground truth\n'
    r'(Human BT: 2,696 votes, 7 annotators; WorldJen $\hat{\rho}=1.00$, VBench $\hat{\rho}=0.60$)',
    fontsize=10, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(LATEX_DIR / "vbench_evidence.pdf", dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(LATEX_DIR / "vbench_evidence.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {OUT_PDF}  (+ latex/ copies)")
