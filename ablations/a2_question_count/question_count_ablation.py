"""
Question Count Ablation Study
==============================
Tests whether 10 questions per dimension is necessary, or if fewer suffice.
For each dimension, subsamples N={1,2,3,5,7,10} questions and measures:
  - Score variance across random subsets (stability proxy)
  - Rank correlation of model ordering at different N vs N=10 (gold standard)
  - How much the BT rating changes at each N

Usage:
    python question_count_ablation.py
    python question_count_ablation.py \
        --results-dir /path/to/data/results/gemini_vlm \
        --output /path/to/data/results/summaries/question_count_ablation_results_unified.json
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", _REPO_ROOT / "data"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DIMENSIONS = [
    'subject_consistency', 'scene_consistency', 'motion_smoothness', 'temporal_flickering',
    'inertial_consistency', 'physical_mechanics', 'object_permanence', 'human_fidelity',
    'dynamic_degree', 'semantic_adherence', 'spatial_relationship', 'semantic_drift',
    'composition_framing', 'lighting_volumetric', 'color_harmony', 'structural_gestalt'
]
QUESTION_COUNTS = [1, 2, 3, 5, 7, 10]
N_BOOTSTRAP = 200  # random subsets per (model, prompt, dim, N)
SEED = 42


def load_results(results_dir: Path) -> dict:
    """Load all result JSONs into nested dict: [model][prompt_id][dim] = [scores]"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fpath in results_dir.glob('*.json'):
        stem = fpath.stem  # e.g. fal-ai_veo3.1_fast_prompt_0075
        # Parse model and prompt_id
        for pid_candidate in ['prompt_' + stem.split('_prompt_')[-1]] if '_prompt_' in stem else []:
            model = stem.replace('_' + pid_candidate, '')
            try:
                with open(fpath) as f:
                    obj = json.load(f)
                for dim in DIMENSIONS:
                    if dim in obj.get('results', {}):
                        scores = [float(item['score']) for item in obj['results'][dim] if 'score' in item]
                        if scores:
                            data[model][pid_candidate][dim] = scores
            except Exception as e:
                print(f"Warning: Could not load {fpath}: {e}")
    return data


def compute_model_avg(data: dict, n_questions: int, seed_offset: int = 0) -> dict[str, dict[str, float]]:
    """
    For each model, compute average score per dimension using only n_questions randomly sampled.
    Returns {model: {dim: avg_score}}
    """
    rng = random.Random(SEED + seed_offset)
    results = defaultdict(lambda: defaultdict(list))
    for model, prompts in data.items():
        for pid, dims in prompts.items():
            for dim, scores in dims.items():
                if len(scores) < n_questions:
                    sample = scores  # use all if fewer available
                else:
                    sample = rng.sample(scores, n_questions)
                results[model][dim].append(np.mean(sample))
    # Average across prompts
    return {
        model: {dim: np.mean(vals) for dim, vals in dim_data.items()}
        for model, dim_data in results.items()
    }


def model_overall_avg(dim_scores: dict[str, float]) -> float:
    return np.mean(list(dim_scores.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=_DATA_ROOT / 'results/gemini_vlm', type=Path)
    parser.add_argument('--output', default=str(_DATA_ROOT / 'results/summaries/question_count_ablation_results_unified.json'))
    args = parser.parse_args()

    print("Loading result files...")
    data = load_results(args.results_dir)
    models = sorted(data.keys())
    print(f"Models: {models}")
    print(f"Prompts per model: {[len(data[m]) for m in models]}")

    # Gold standard: use all 10 questions
    gold_scores = compute_model_avg(data, 10)
    gold_overall = {m: model_overall_avg(gold_scores[m]) for m in models}
    gold_rank = {m: r for r, m in enumerate(sorted(models, key=lambda x: -gold_overall[x]))}

    results_by_n = {}
    print("\nRunning ablation across question counts...")

    for n in QUESTION_COUNTS:
        bootstrap_scores = []
        for b in range(N_BOOTSTRAP):
            scores_n = compute_model_avg(data, n, seed_offset=b)
            overall_n = {m: model_overall_avg(scores_n[m]) for m in models if m in scores_n}
            bootstrap_scores.append(overall_n)

        # Mean and std of overall score per model
        model_means = {}
        model_stds = {}
        for m in models:
            vals = [s[m] for s in bootstrap_scores if m in s]
            model_means[m] = np.mean(vals)
            model_stds[m] = np.std(vals)

        # Rank correlation vs gold
        models_sorted_gold = sorted(models, key=lambda x: -gold_overall.get(x, 0))
        models_sorted_n = sorted(models, key=lambda x: -model_means.get(x, 0))
        gold_ranks = [models_sorted_gold.index(m) for m in models]
        n_ranks = [models_sorted_n.index(m) for m in models]
        rho, pval = spearmanr(gold_ranks, n_ranks)

        # Mean score variance across bootstrap runs
        avg_variance = np.mean([model_stds[m] ** 2 for m in models])

        results_by_n[n] = {
            'mean_scores': {m: round(float(model_means[m]), 4) for m in models},
            'std_scores': {m: round(float(model_stds[m]), 4) for m in models},
            'rank_correlation_vs_10q': round(float(rho), 4),
            'rank_correlation_pvalue': round(float(pval), 4),
            'avg_score_variance': round(float(avg_variance), 6),
            'n_bootstrap': N_BOOTSTRAP
        }

        print(f"  N={n:2d}: rank_rho_vs_10q={rho:.4f}  avg_variance={avg_variance:.6f}  "
              f"model_rank_order={[m.split('_')[1][:3] if '_' in m else m[:4] for m in models_sorted_n]}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Question Count Ablation: Score Stability vs N Questions", fontsize=13, fontweight='bold')

    # Left: Rank correlation vs N
    ns = QUESTION_COUNTS
    rhos = [results_by_n[n]['rank_correlation_vs_10q'] for n in ns]
    axes[0].plot(ns, rhos, 'o-', color='#38bdf8', linewidth=2, markersize=7)
    axes[0].axhline(1.0, linestyle='--', color='#10b981', alpha=0.5, label='Perfect (N=10)')
    axes[0].axhline(0.9, linestyle=':', color='#f59e0b', alpha=0.7, label='ρ=0.90 threshold')
    axes[0].set_xlabel('Number of Questions per Dimension')
    axes[0].set_ylabel("Spearman ρ vs 10-question baseline")
    axes[0].set_title("Model Rank Stability")
    axes[0].set_xticks(ns)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_ylim(0, 1.05)

    # Right: Score variance vs N
    variances = [results_by_n[n]['avg_score_variance'] for n in ns]
    axes[1].plot(ns, variances, 'o-', color='#818cf8', linewidth=2, markersize=7)
    axes[1].set_xlabel('Number of Questions per Dimension')
    axes[1].set_ylabel('Average Score Variance (bootstrap)')
    axes[1].set_title("Score Variance vs Question Count")
    axes[1].set_xticks(ns)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = Path(args.output).parent / 'question_count_ablation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        'gold_overall_scores': {m: round(float(gold_overall[m]), 4) for m in models},
        'ablation': results_by_n
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {args.output}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Question Count Ablation Summary ──")
    print(f"{'N':>4}  {'Rank ρ vs 10q':>15}  {'p-value':>10}  {'Avg Variance':>14}")
    print("-" * 50)
    for n in QUESTION_COUNTS:
        r = results_by_n[n]
        print(f"{n:>4}  {r['rank_correlation_vs_10q']:>15.4f}  "
              f"{r['rank_correlation_pvalue']:>10.4f}  {r['avg_score_variance']:>14.6f}")


if __name__ == '__main__':
    main()
