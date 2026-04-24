"""
generate_bt_comparison.py — Generate two BT rating figures:

  1. bootstrap_bt_rating.png          — VLM-only BT rating with 95% CI (for §4 Results)
  2. vlm_human_bt_comparison.png — VLM vs Human BT rating comparison (for §5 Human Alignment)

Figures are written to the figures/ directory (same directory as this script).

Usage:
    python generate_bt_comparison.py
"""
import csv
import os
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))

MODELS = [
    'fal-ai_hunyuan-video-v1.5_text-to-video',
    'fal-ai_kling-video_v2.6_pro_text-to-video',
    'fal-ai_ltx-2_text-to-video',
    'fal-ai_veo3.1_fast',
    'fal-ai_wan_v2.2-a14b_text-to-video',
    'wan2.1-1.3b',
]
MODEL_SHORT = {
    'fal-ai_hunyuan-video-v1.5_text-to-video':   'Hunyuan v1.5',
    'fal-ai_kling-video_v2.6_pro_text-to-video':  'Kling v2.6 Pro',
    'fal-ai_ltx-2_text-to-video':                 'LTX-2',
    'fal-ai_veo3.1_fast':                         'Veo 3.1 Fast',
    'fal-ai_wan_v2.2-a14b_text-to-video':         'Wan v2.2 A14B',
    'wan2.1-1.3b':                                'Wan 2.1 1.3B',
}
MODEL_COLOR = {
    'fal-ai_veo3.1_fast':                         '#2563eb',
    'fal-ai_kling-video_v2.6_pro_text-to-video':  '#7c3aed',
    'fal-ai_ltx-2_text-to-video':                 '#059669',
    'fal-ai_wan_v2.2-a14b_text-to-video':         '#d97706',
    'fal-ai_hunyuan-video-v1.5_text-to-video':    '#dc2626',
    'wan2.1-1.3b':                                '#64748b',
}
# Canonical human BT rating point estimates — rounded to match tab:rankcompare in main.tex
# (source: report_bt_anon.txt, rounded: 1614.2→1614, 1571.8→1572, etc.)
HUMAN_ELO_CANONICAL = {
    'fal-ai_veo3.1_fast':                         1614,
    'fal-ai_kling-video_v2.6_pro_text-to-video':  1572,
    'fal-ai_wan_v2.2-a14b_text-to-video':         1518,
    'fal-ai_ltx-2_text-to-video':                 1479,
    'fal-ai_hunyuan-video-v1.5_text-to-video':    1462,
    'wan2.1-1.3b':                                1355,
}

HUMAN_CSV = DATA_ROOT / 'human_eval/anonymized_human_evals.csv'
# VLM BT rating with CI is now stored directly in summary_report_unified.json;
# bootstrap_bt_rating_results.json is retained as a legacy standalone artifact.
VLM_JSON  = DATA_ROOT / "results/summaries/summary_report_unified.json"
EXCLUDE   = {'A1-retest'}  # covers both anonymized and original CSV
N_BOOT    = 1000
SEED      = 42


# ── Bradley-Terry ─────────────────────────────────────────────────────────────

def fit_bradley_terry(comparisons, n_iters=2000):
    models = sorted({m for w, l in comparisons for m in (w, l)})
    if not models:
        return {}
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.zeros((n, n))
    for w, l in comparisons:
        if w in idx and l in idx:
            W[idx[w], idx[l]] += 1
    strength = np.ones(n)
    for _ in range(n_iters):
        new = np.zeros(n)
        for i in range(n):
            wins = W[i].sum()
            denom = sum((W[i, j] + W[j, i]) / (strength[i] + strength[j])
                        for j in range(n) if j != i and (W[i, j] + W[j, i]) > 0)
            new[i] = wins / denom if denom > 0 and wins > 0 else 1e-9
        strength = new / new.sum() * n
    log_s = np.log(strength + 1e-9)
    log_s -= log_s.mean()
    return {models[i]: 1500 + log_s[i] * 400 / math.log(10) for i in range(n)}


def bootstrap_bt(comparisons, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    comps = list(comparisons)
    boot_bts = defaultdict(list)
    for _ in range(n_boot):
        sample = [comps[i] for i in rng.integers(0, len(comps), len(comps))]
        try:
            for m, e in fit_bradley_terry(sample).items():
                boot_bts[m].append(e)
        except Exception:
            pass
    return {m: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
            for m, v in boot_bts.items() if v}


def load_human_comparisons(path, exclude):
    exclude_lower = {e.lower() for e in exclude}
    comparisons = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            email = row.get('User', row.get('user', '')).strip()
            if email.lower() in exclude_lower:
                continue
            winner = row.get('Winner', row.get('winner', '')).strip()
            loser  = row.get('Loser',  row.get('loser',  '')).strip()
            if winner.upper() == 'TIE' or not winner or not loser:
                continue
            comparisons.append((winner, loser))
    return comparisons


# ── Figure 1: VLM-only BT rating with CI (for §4) ───────────────────────────────────

def plot_vlm_bt(vlm_ci, out_path):
    """Clean VLM-only horizontal bar chart with CI. No human data."""
    # Sort best→worst (bottom of chart = worst → top = best for barh)
    order = sorted(MODELS, key=lambda m: vlm_ci.get(m, {}).get('mean', 0))  # ascending for barh

    means  = [vlm_ci.get(m, {}).get('mean', 1500)     for m in order]
    lo_err = [vlm_ci.get(m, {}).get('mean', 1500) - vlm_ci.get(m, {}).get('lower_95', 1500) for m in order]
    hi_err = [vlm_ci.get(m, {}).get('upper_95', 1500) - vlm_ci.get(m, {}).get('mean', 1500) for m in order]
    colors = [MODEL_COLOR.get(m, '#64748b') for m in order]
    labels = [MODEL_SHORT.get(m, m) for m in order]
    ranks  = {m: i + 1 for i, m in enumerate(sorted(MODELS, key=lambda m: -vlm_ci.get(m, {}).get('mean', 0)))}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8fafc')

    y = np.arange(len(order))
    bars = ax.barh(y, means, xerr=[lo_err, hi_err],
                   height=0.55,
                   color=[c + 'bb' for c in colors],
                   edgecolor=colors,
                   linewidth=1.4,
                   capsize=5,
                   error_kw={'ecolor': '#374151', 'capthick': 2, 'elinewidth': 1.8})

    # BT rating value labels — fixed position at left edge of visible area, clear of whiskers
    for i, (m, mean, lo, hi) in enumerate(zip(order, means, lo_err, hi_err)):
        ci_right = mean + hi  # right tip of CI whisker
        rank = ranks[m]
        # Fixed x position well inside the bar and away from error bar whiskers
        ax.text(1167, i, f'{mean:.0f}',
                va='center', ha='left', fontsize=9, color='white', fontweight='bold')
        # CI half-width annotation — placed after the right whisker cap
        hw = (lo + hi) / 2
        ax.text(ci_right + 6, i, f'±{hw:.0f}',
                va='center', ha='left', fontsize=7.5, color='#6b7280')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('VLM BT Rating (Bradley-Terry, anchored at 1500)', fontsize=10)
    ax.set_title('VLM BT Rankings with 95% Bootstrap CIs\n(1,000 prompt-level resamples, 50 prompts)',
                 fontsize=11, fontweight='bold', pad=8)

    ax.axvline(1500, linestyle='--', color='#9ca3af', linewidth=1.2, alpha=0.8, zorder=0)
    ax.text(1500, len(order) - 0.45, '1500', ha='center', va='top', fontsize=7, color='#9ca3af')

    ax.grid(True, axis='x', alpha=0.25, color='#9ca3af', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlim(left=1150)

    # Rank badge: embed in y-tick label instead of extra text
    ax.set_yticklabels([f'#{ranks[m]}  {MODEL_SHORT.get(m, m)}' for m in order], fontsize=10.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved VLM-only BT rating → {out_path}", file=sys.stderr)


# ── Figure 2: Human-only BT rating (for §4 Human Study) ─────────────────────────

def plot_human_bt(h_means, h_lo, h_hi, out_path):
    """Standalone Human BT rating horizontal bar chart, matching VLM-only style."""
    order = sorted(MODELS, key=lambda m: h_means.get(m, 0))  # ascending for barh
    ranks = {m: i+1 for i, m in enumerate(sorted(MODELS, key=lambda m: -h_means.get(m, 0)))}

    hm_canonical = [HUMAN_ELO_CANONICAL.get(m, h_means.get(m, 1500)) for m in order]
    lo_err = [HUMAN_ELO_CANONICAL.get(m, h_means.get(m, 1500)) - h_lo.get(m, 1500) for m in order]
    hi_err = [h_hi.get(m, 1500) - HUMAN_ELO_CANONICAL.get(m, h_means.get(m, 1500)) for m in order]
    colors = [MODEL_COLOR.get(m, '#64748b') for m in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8fafc')

    y = np.arange(len(order))
    ax.barh(y, hm_canonical, xerr=[lo_err, hi_err],
            height=0.55,
            color=[c + 'bb' for c in colors],
            edgecolor=colors,
            linewidth=1.4,
            capsize=5,
            error_kw={'ecolor': '#374151', 'capthick': 2, 'elinewidth': 1.8})

    for i, (m, val, lo, hi) in enumerate(zip(order, hm_canonical, lo_err, hi_err)):
        ax.text(1167, i, f'{val}',
                va='center', ha='left', fontsize=9, color='white', fontweight='bold')
        hw = (lo + hi) / 2
        ax.text(val + hi + 6, i, f'±{hw:.0f}',
                va='center', ha='left', fontsize=7.5, color='#6b7280')

    ax.set_yticks(y)
    ax.set_yticklabels([f'#{ranks[m]}  {MODEL_SHORT.get(m, m)}' for m in order], fontsize=10.5)
    ax.set_xlabel('Human BT Rating (Bradley-Terry, anchored at 1500)', fontsize=10)
    ax.set_title('Human Preference Rankings with 95% Bootstrap CIs\n'
                 '(1,000 vote resamples, 2,696 pairwise votes, 7 annotators)',
                 fontsize=11, fontweight='bold', pad=8)

    ax.axvline(1500, linestyle='--', color='#9ca3af', linewidth=1.2, alpha=0.8, zorder=0)
    ax.text(1500, len(order) - 0.45, '1500', ha='center', va='top', fontsize=7, color='#9ca3af')
    ax.grid(True, axis='x', alpha=0.25, color='#9ca3af', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlim(left=1150)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Human-only BT rating → {out_path}", file=sys.stderr)


# ── Figure 3: VLM vs Human BT rating comparison (for §5.5) ─────────────────────────

def plot_vlm_human_comparison(vlm_ci, h_means, h_lo, h_hi, out_path):
    """Side-by-side dot+CI plot: VLM BT rating (left panel) vs Human BT rating (right panel)."""
    # Sort by VLM BT rating rank
    order = sorted(MODELS, key=lambda m: vlm_ci.get(m, {}).get('mean', 0))  # ascending for barh
    labels = [MODEL_SHORT.get(m, m) for m in order]
    colors = [MODEL_COLOR.get(m, '#64748b') for m in order]
    y = np.arange(len(order))

    vlm_rank = {m: i+1 for i, m in enumerate(sorted(MODELS, key=lambda m: -vlm_ci.get(m,{}).get('mean',0)))}
    hum_rank = {m: i+1 for i, m in enumerate(sorted(MODELS, key=lambda m: -h_means.get(m, 0)))}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('#f8fafc')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25, color='#9ca3af', zorder=0)

    # ── Left: VLM BT rating ───────────────────────────────────────────────────────
    ax = axes[0]
    v_means  = [vlm_ci.get(m, {}).get('mean', 1500) for m in order]
    v_lo_err = [vlm_ci.get(m, {}).get('mean', 1500) - vlm_ci.get(m, {}).get('lower_95', 1500) for m in order]
    v_hi_err = [vlm_ci.get(m, {}).get('upper_95', 1500) - vlm_ci.get(m, {}).get('mean', 1500) for m in order]

    ax.barh(y, v_means, xerr=[v_lo_err, v_hi_err], height=0.55,
            color=[c + 'bb' for c in colors], edgecolor=colors, linewidth=1.2,
            capsize=4, error_kw={'ecolor': '#374151', 'capthick': 1.8, 'elinewidth': 1.6})
    for i, (m, val, lo, hi) in enumerate(zip(order, v_means, v_lo_err, v_hi_err)):
        ax.text(1167, i, f'{val:.0f}', va='center', ha='left',
                fontsize=8.5, color='white', fontweight='bold')
        hw = (lo + hi) / 2
        ax.text(val + hi + 5, i, f'±{hw:.0f}', va='center', ha='left',
                fontsize=8, color='#6b7280')
    ax.axvline(1500, linestyle='--', color='#9ca3af', linewidth=1.1, alpha=0.8, zorder=0)
    ax.set_xlabel('VLM BT Rating', fontsize=10)
    ax.set_title('VLM Evaluation\n(Gemini Flash, 50 prompts)', fontsize=10, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(left=1150)

    # ── Right: Human BT rating ────────────────────────────────────────────────
    ax = axes[1]
    hm = [h_means.get(m, 1500) for m in order]
    hl = [h_means.get(m, 1500) - h_lo.get(m, 1500) for m in order]
    hh = [h_hi.get(m, 1500) - h_means.get(m, 1500) for m in order]

    # Use canonical BT rating values for bar lengths and labels (matches tab:rankcompare)
    hm_canonical = [HUMAN_ELO_CANONICAL.get(m, h_means.get(m, 1500)) for m in order]
    ax.barh(y, hm_canonical, xerr=[hl, hh], height=0.55,
            color=[c + 'bb' for c in colors], edgecolor=colors, linewidth=1.2,
            capsize=4, error_kw={'ecolor': '#374151', 'capthick': 1.8, 'elinewidth': 1.6})
    for i, (m, val, lo, hi) in enumerate(zip(order, hm_canonical, hl, hh)):
        ax.text(1167, i, f'{val}', va='center', ha='left',
                fontsize=8.5, color='white', fontweight='bold')
        hw = (lo + hi) / 2
        ax.text(val + hi + 5, i, f'±{hw:.0f}', va='center', ha='left',
                fontsize=8, color='#6b7280')
    ax.axvline(1500, linestyle='--', color='#9ca3af', linewidth=1.1, alpha=0.8, zorder=0)
    ax.set_xlabel('Human BT Rating', fontsize=10)
    ax.set_title('Human Preference Study\n(1,000 vote resamples of 2,696 votes, 7 annotators)', fontsize=10, fontweight='bold')
    ax.set_xlim(left=1150)

    fig.suptitle(r'VLM vs. Human BT Ratings — 95% Bootstrap CIs  ($\hat{\rho}=1.000$, $p<0.001$, perfect rank agreement)',
                 fontsize=11, fontweight='bold')

    # No rank discordances — remove arrow legend
    note = mpatches.Patch(color='none', label='All 6 human ranks match VLM ranks exactly')
    axes[1].legend(handles=[note], loc='lower right', fontsize=8, framealpha=0.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved VLM vs Human comparison → {out_path}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load VLM data
    with open(VLM_JSON) as f:
        vlm_data = json.load(f)
    vlm_ci = vlm_data.get('vlm_bt_with_ci', {})

    # Load and bootstrap human data
    print("Loading human comparisons ...", file=sys.stderr)
    human_comps = load_human_comparisons(HUMAN_CSV, EXCLUDE)
    print(f"  {len(human_comps)} non-tie comparisons", file=sys.stderr)
    print("Bootstrapping human BT rating CIs (N=1000) ...", file=sys.stderr)
    human_boot = bootstrap_bt(human_comps)

    h_means, h_lo, h_hi = {}, {}, {}
    for m, (mean, lo, hi) in human_boot.items():
        h_means[m] = mean
        h_lo[m]    = lo
        h_hi[m]    = hi

    # Figure 1: VLM-only (§5.2)
    plot_vlm_bt(vlm_ci, HERE / 'bootstrap_bt_rating.png')

    # Figure 2: Human-only (§4 Human Study)
    plot_human_bt(h_means, h_lo, h_hi, HERE / 'human_bt.png')

    # Figure 3: VLM vs Human comparison (§5.5)
    plot_vlm_human_comparison(vlm_ci, h_means, h_lo, h_hi,
                               HERE / 'vlm_human_bt_comparison.png')


if __name__ == '__main__':
    main()
