#!/usr/bin/env python3
"""
Given a min suitability score k: count prompts in each bin (bin_i = exactly i dimensions with suitability > k),
then plot: Y = number of prompts per bin, X = bins (0, 1, ..., n).
Input: judge JSONL (rescored.jsonl / judged_review_fixed.jsonl). Null suitability = 0.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from collections import Counter, defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_FILE = str(_REPO_ROOT / 'data' / 'prompts' / 'prompts_50_rescored.jsonl')
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / 'data' / 'stats')

# All 16 dimensions (4 groups)
DIMENSIONS = {
    'motion_stability': [
        ('subject_consistency',), ('scene_consistency',), ('motion_smoothness',),
        ('temporal_flickering',), ('inertial_consistency',),
    ],
    'logic_physics': [
        ('physical_mechanics',), ('object_permanence',), ('human_fidelity',), ('dynamic_degree',),
    ],
    'instruction_adherence': [
        ('semantic_adherence',), ('spatial_relationship',), ('semantic_drift',),
    ],
    'aesthetic_quality': [
        ('composition_framing',), ('lighting_volumetric',), ('color_harmony',), ('structural_gestalt',),
    ],
}
# Flat (group, dim_key) for all dimensions
DIMENSION_LIST = [(g, d[0]) for g, dims in DIMENSIONS.items() for d in dims]
NUM_DIMENSIONS = len(DIMENSION_LIST)


def score_or_zero(entry: dict, group_name: str, dim_key: str, suffix: str) -> float:
    g = entry.get(group_name)
    if not isinstance(g, dict):
        return 0.0
    v = g.get(f'{dim_key}_{suffix}')
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def count_dims_above_k(entry: dict, k: float) -> int:
    """Number of dimensions with suitability > k (null = 0)."""
    n = 0
    for group_name, dim_key in DIMENSION_LIST:
        if score_or_zero(entry, group_name, dim_key, 'suitability') > k:
            n += 1
    return n


def load_data(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# Short labels for radar axes (one per dimension)
DIMENSION_LABELS = [
    'subj', 'scene', 'motion', 'temp', 'inert',
    'phys', 'obj perm', 'human', 'dyn',
    'sem', 'spat', 'drift',
    'comp', 'light', 'color', 'gestalt',
]


def proportion_above_k_per_dimension(prompts: list[dict], k: float) -> list[float]:
    """For each dimension, proportion of prompts in list that have suitability > k."""
    if not prompts:
        return [0.0] * NUM_DIMENSIONS
    n_prompts = len(prompts)
    counts = [0] * NUM_DIMENSIONS
    for entry in prompts:
        for i, (group_name, dim_key) in enumerate(DIMENSION_LIST):
            if score_or_zero(entry, group_name, dim_key, 'suitability') > k:
                counts[i] += 1
    return [c / n_prompts for c in counts]


def plot_radar_top4_bins(data_by_bin: dict[int, list[dict]], k: float, output_dir: Path, k_label: str):
    """Draw 4 radar charts: one per bin with the 4 highest prompt counts (most populous bins)."""
    n_dims = NUM_DIMENSIONS
    # Top 4 bins by number of prompts (most populous first)
    top4_bins = sorted(
        [n for n in range(n_dims + 1) if data_by_bin.get(n)],
        key=lambda n: len(data_by_bin[n]),
        reverse=True,
    )[:4]
    if not top4_bins:
        return
    # Angles for 16 axes (radar)
    theta = np.linspace(0, 2 * np.pi, n_dims, endpoint=False)
    theta_closed = np.concatenate([theta, [theta[0]]])
    labels = DIMENSION_LABELS
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='polar'), figsize=(12, 12))
    axes = axes.flatten()
    for idx, bin_n in enumerate(top4_bins):
        if idx >= 4:
            break
        prompts = data_by_bin[bin_n]
        proportions = proportion_above_k_per_dimension(prompts, k)
        values = np.array(proportions + [proportions[0]])
        ax = axes[idx]
        ax.plot(theta_closed, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(theta_closed, values, alpha=0.25, color='steelblue')
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1'], size=7)
        ax.set_title(f'Bin {bin_n} (n={len(prompts)} prompts)', fontsize=11, fontweight='bold')
    plt.suptitle(f'Dimension contribution (prop. with suitability > k), top 4 bins by prompt count (k = {k})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = output_dir / f'radar_top4_bins_k{k_label}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot: prompts per bin (bin_i = exactly i dimensions with suitability > k).'
    )
    parser.add_argument('--input-file', '-i', default=DEFAULT_INPUT_FILE, help='Judge output JSONL')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--suitability-threshold', '-k', type=float, default=6.0, metavar='K',
                        help='Threshold k: dimensions with suitability > k (default 6)')
    args = parser.parse_args()

    data = load_data(args.input_file)
    k = args.suitability_threshold
    n_dims = NUM_DIMENSIONS

    # Bin counts and group prompts by bin
    counts = [count_dims_above_k(entry, k) for entry in data]
    prompt_per_bin = Counter(counts)
    bins = list(range(n_dims + 1))  # 0, 1, ..., n_dims
    y = [prompt_per_bin.get(n, 0) for n in bins]
    data_by_bin = defaultdict(list)
    for entry, c in zip(data, counts):
        data_by_bin[c].append(entry)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bins, y, color='steelblue', edgecolor='black')
    ax.set_xlabel('Bin (number of dimensions with suitability > k)', fontsize=11)
    ax.set_ylabel('Number of prompts', fontsize=11)
    ax.set_title(f'Prompts per bin (k = {k}); bin_i = exactly i dimensions > k', fontsize=12, fontweight='bold')
    ax.set_xticks(bins)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'suitability_bins_k{int(k) if k == int(k) else k}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    k_label = int(k) if k == int(k) else k
    plot_radar_top4_bins(data_by_bin, k, output_dir, str(k_label))
    print(f"Total prompts: {len(data)}; threshold k = {k}")
    for n in bins:
        print(f"  Bin {n} (exactly {n} dims > k): {y[n]} prompts")


if __name__ == '__main__':
    main()
