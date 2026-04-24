"""
Bootstrap Confidence Intervals for VLM BT rating Scores
====================================================
Computes 95% bootstrap confidence intervals for each model's BT rating by
resampling (prompt, model_pair) matchups with replacement.

Also computes prompt-level Spearman ρ between VLM BT rating and human BT rating
with proper confidence intervals via bootstrap.

Usage:
    python bootstrap_bt_rating.py

    # Override defaults:
    python bootstrap_bt_rating.py \
        --results-dir data/results/gemini_vlm \
        --human-eval data/human_eval/anonymized_human_evals.csv \
        --output data/results/summaries/bootstrap_bt_rating_results.json
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", _REPO_ROOT / "data"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SEED = 42
N_BOOTSTRAP = 1000
MODELS = [
    'fal-ai_hunyuan-video-v1.5_text-to-video',
    'fal-ai_kling-video_v2.6_pro_text-to-video',
    'fal-ai_ltx-2_text-to-video',
    'fal-ai_veo3.1_fast',
    'fal-ai_wan_v2.2-a14b_text-to-video',
    'wan2.1-1.3b'
]
MODEL_SHORT = {
    'fal-ai_hunyuan-video-v1.5_text-to-video': 'Hunyuan v1.5',
    'fal-ai_kling-video_v2.6_pro_text-to-video': 'Kling v2.6 Pro',
    'fal-ai_ltx-2_text-to-video': 'LTX-2',
    'fal-ai_veo3.1_fast': 'Veo 3.1 Fast',
    'fal-ai_wan_v2.2-a14b_text-to-video': 'Wan v2.2 A14B',
    'wan2.1-1.3b': 'Wan 2.1 1.3B'
}


def compute_bradley_terry(matchups: list[tuple[str, str]], n_iters: int = 500) -> dict[str, float]:
    """
    matchups: list of (winner, loser) tuples (no ties)
    Returns {model: bt}
    """
    models = sorted(set(m for pair in matchups for m in pair))
    if len(models) < 2:
        return {m: 1500.0 for m in models}
    model_idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.zeros((n, n))
    for winner, loser in matchups:
        W[model_idx[winner], model_idx[loser]] += 1

    strength = np.ones(n)
    for _ in range(n_iters):
        new_s = np.zeros(n)
        for i in range(n):
            wins_i = W[i].sum()
            denom = sum((W[i, j] + W[j, i]) / (strength[i] + strength[j])
                        for j in range(n) if j != i and (W[i, j] + W[j, i]) > 0)
            new_s[i] = wins_i / denom if denom > 0 and wins_i > 0 else 1e-6
        strength = new_s / new_s.sum() * n

    log_s = np.log(strength + 1e-9)
    log_s -= log_s.mean()
    return {models[i]: 1500 + log_s[i] * 400 / math.log(10) for i in range(n)}


def load_vlm_matchups(results_dir: Path) -> list[dict]:
    """
    For each (prompt, model_pair), create a matchup based on VLM average score.
    Returns list of {'prompt_id', 'winner', 'loser', 'model_a', 'model_b', 'score_diff'}
    """
    prompt_model_scores = {}  # {(prompt_id, model): avg_score}

    for fpath in results_dir.glob('*.json'):
        stem = fpath.stem
        if '_prompt_' not in stem:
            continue
        parts = stem.split('_prompt_')
        model = parts[0]
        pid = 'prompt_' + parts[1]
        try:
            with open(fpath) as f:
                obj = json.load(f)
            dim_avgs = []
            for dim, questions in obj.get('results', {}).items():
                scores = [q['score'] for q in questions if 'score' in q]
                if scores:
                    dim_avgs.append(np.mean(scores))
            if dim_avgs:
                prompt_model_scores[(pid, model)] = np.mean(dim_avgs)
        except Exception as e:
            print(f"Warning: {fpath}: {e}")

    # Generate pairwise matchups
    prompts = sorted({pid for pid, _ in prompt_model_scores})
    matchups = []
    for pid in prompts:
        models_this = [m for m in MODELS if (pid, m) in prompt_model_scores]
        for i in range(len(models_this)):
            for j in range(i + 1, len(models_this)):
                ma, mb = models_this[i], models_this[j]
                sa = prompt_model_scores[(pid, ma)]
                sb = prompt_model_scores[(pid, mb)]
                if abs(sa - sb) < 0.01:  # treat near-ties as ties
                    continue
                winner = ma if sa > sb else mb
                loser = mb if sa > sb else ma
                matchups.append({'prompt_id': pid, 'winner': winner, 'loser': loser,
                                  'model_a': ma, 'model_b': mb, 'score_diff': abs(sa - sb)})
    return matchups


def bootstrap_bt_rating(matchups: list[dict], n_bootstrap: int = N_BOOTSTRAP) -> dict[str, dict]:
    """
    Returns {model: {'mean': float, 'lower_95': float, 'upper_95': float, 'std': float}}
    """
    rng = random.Random(SEED)
    bt_samples = defaultdict(list)

    # Group matchups by prompt for prompt-level bootstrap
    by_prompt = defaultdict(list)
    for m in matchups:
        by_prompt[m['prompt_id']].append(m)
    prompts = list(by_prompt.keys())

    print(f"Bootstrap BT rating: {n_bootstrap} iterations over {len(matchups)} matchups ({len(prompts)} prompts)...")

    for b in range(n_bootstrap):
        # Resample prompts with replacement (prompt-level bootstrap)
        sampled_prompts = rng.choices(prompts, k=len(prompts))
        sampled_matchups = [(m['winner'], m['loser'])
                            for p in sampled_prompts
                            for m in by_prompt[p]]
        if len(sampled_matchups) < 2:
            continue
        bt = compute_bradley_terry(sampled_matchups)
        for model, e in bt.items():
            bt_samples[model].append(e)

        if (b + 1) % 100 == 0:
            print(f"  {b+1}/{n_bootstrap}")

    results = {}
    for model in MODELS:
        samples = bt_samples.get(model, [])
        if not samples:
            results[model] = {'mean': 1500.0, 'lower_95': 1500.0, 'upper_95': 1500.0, 'std': 0.0}
            continue
        arr = np.array(samples)
        results[model] = {
            'mean': round(float(np.mean(arr)), 1),
            'lower_95': round(float(np.percentile(arr, 2.5)), 1),
            'upper_95': round(float(np.percentile(arr, 97.5)), 1),
            'std': round(float(np.std(arr)), 1),
            'n_samples': len(samples)
        }
    return results


def bootstrap_rank_correlation(vlm_bts: dict, human_bts: dict,
                                n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """
    Bootstrap CI for rank-level Spearman ρ between VLM and human BT ratings.
    """
    models = sorted(set(vlm_bts) & set(human_bts))
    if len(models) < 3:
        return {'rho': float('nan'), 'lower_95': float('nan'), 'upper_95': float('nan')}

    rho_obs, p_obs = spearmanr(
        [vlm_bts[m] for m in models],
        [human_bts[m] for m in models]
    )

    rng = random.Random(SEED)
    rho_samples = []
    for _ in range(n_bootstrap):
        sample = rng.choices(models, k=len(models))
        v = [vlm_bts[m] for m in sample]
        h = [human_bts[m] for m in sample]
        try:
            r, _ = spearmanr(v, h)
            if not np.isnan(r):
                rho_samples.append(r)
        except Exception:
            pass

    arr = np.array(rho_samples) if rho_samples else np.array([rho_obs])
    return {
        'rho_observed': round(float(rho_obs), 4),
        'p_value': round(float(p_obs), 4),
        'lower_95_bootstrap': round(float(np.percentile(arr, 2.5)), 4),
        'upper_95_bootstrap': round(float(np.percentile(arr, 97.5)), 4),
        'n_models': len(models)
    }


def prompt_count_stability(matchups: list[dict], n_bootstrap: int = 500) -> dict:
    """
    For each N in {5, 10, 15, 20}, subsample N prompts (n_bootstrap draws each),
    compute BT rating, and compare rank order to the full-20-prompt BT rating.
    Returns per-N stats: fraction of draws with perfect rank, mean Spearman ρ, 95% CI.
    """
    rng = random.Random(SEED + 999)

    by_prompt = defaultdict(list)
    for m in matchups:
        by_prompt[m['prompt_id']].append(m)
    prompts = sorted(by_prompt.keys())
    N_TOTAL = len(prompts)

    # Gold BT rating from all prompts
    all_matchup_pairs = [(m['winner'], m['loser']) for m in matchups]
    gold_bt = compute_bradley_terry(all_matchup_pairs)
    gold_rank = {m: r for r, (m, _) in enumerate(
        sorted(gold_bt.items(), key=lambda x: -x[1]))}

    stability = {}
    for n in [10, 20, 30, 40, 50]:
        rho_samples = []
        perfect_count = 0
        ci_bts = defaultdict(list)

        for _ in range(n_bootstrap):
            sampled = rng.sample(prompts, min(n, N_TOTAL))
            sub_matchups = [(m['winner'], m['loser'])
                            for p in sampled for m in by_prompt[p]]
            if len(sub_matchups) < 2:
                continue
            bt_n = compute_bradley_terry(sub_matchups)
            for model, e in bt_n.items():
                ci_bts[model].append(e)

            rank_n = {m: r for r, (m, _) in enumerate(
                sorted(bt_n.items(), key=lambda x: -x[1]))}

            shared = [m for m in MODELS if m in gold_rank and m in rank_n]
            gv = [gold_rank[m] for m in shared]
            nv = [rank_n[m] for m in shared]
            if len(shared) >= 2:
                rho, _ = spearmanr(gv, nv)
                if not np.isnan(rho):
                    rho_samples.append(rho)
                    if list(gv) == list(nv):
                        perfect_count += 1

        arr = np.array(rho_samples) if rho_samples else np.array([0.0])
        bt_ci_n = {}
        for model in MODELS:
            s = ci_bts.get(model, [])
            if s:
                bt_ci_n[model] = {
                    'mean': round(float(np.mean(s)), 1),
                    'lower_95': round(float(np.percentile(s, 2.5)), 1),
                    'upper_95': round(float(np.percentile(s, 97.5)), 1),
                    'ci_half_width': round(float((np.percentile(s, 97.5) - np.percentile(s, 2.5)) / 2), 1)
                }
        stability[n] = {
            'n_prompts': n,
            'n_draws': len(rho_samples),
            'mean_spearman_rho': round(float(np.mean(arr)), 4),
            'lower_95_rho': round(float(np.percentile(arr, 2.5)), 4),
            'upper_95_rho': round(float(np.percentile(arr, 97.5)), 4),
            'pct_perfect_rank': round(100.0 * perfect_count / max(len(rho_samples), 1), 1),
            'bt_ci': bt_ci_n
        }
        print(f"  N={n:2d}: mean_ρ={np.mean(arr):.4f}  "
              f"CI=[{np.percentile(arr,2.5):.4f},{np.percentile(arr,97.5):.4f}]  "
              f"perfect_rank={100.0*perfect_count/max(len(rho_samples),1):.1f}%")
    return stability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=_DATA_ROOT / 'results/gemini_vlm', type=Path)
    parser.add_argument('--human-eval', default=str(_DATA_ROOT / 'human_eval/anonymized_human_evals.csv'))
    parser.add_argument('--output', default=str(_DATA_ROOT / 'results/summaries/bootstrap_bt_rating_results.json'))
    args = parser.parse_args()

    # ── Load VLM matchups ──────────────────────────────────────────────────
    matchups = load_vlm_matchups(args.results_dir)
    print(f"Total VLM matchups: {len(matchups)}")

    # ── Bootstrap BT rating ─────────────────────────────────────────────────────
    bt_ci = bootstrap_bt_rating(matchups)

    print("\n── VLM BT rating with 95% Bootstrap CI ──")
    print(f"{'Model':<35} {'BT rating':>6}  {'95% CI':>18}  {'± (half-width)':>15}")
    print("-" * 80)
    for m in sorted(MODELS, key=lambda x: -bt_ci.get(x, {}).get('mean', 0)):
        r = bt_ci.get(m, {})
        hw = (r.get('upper_95', 0) - r.get('lower_95', 0)) / 2
        short = MODEL_SHORT.get(m, m)
        print(f"  {short:<33} {r.get('mean', 0):>6.1f}  "
              f"[{r.get('lower_95', 0):>6.1f}, {r.get('upper_95', 0):>6.1f}]  ±{hw:>6.1f}")

    # ── Load human BT rating from existing analysis or CSV ───────────────────────
    human_bts = {}
    human_eval_path = Path(args.human_eval)
    if human_eval_path.exists():
        df = pd.read_csv(human_eval_path)
        df.columns = [c.lower() for c in df.columns]   # normalise to lowercase
        if 'tie' in df.columns:
            df['tie'] = df['tie'].astype(str).str.lower().isin(['true', '1'])
            non_tie = df[~df['tie']]
        else:
            non_tie = df
        matchups_h = [(row['winner'], row['loser']) for _, row in non_tie.iterrows()]
        human_bts = compute_bradley_terry(matchups_h)
        print("\n── Human BT rating (from CSV) ──")
        for m, bt in sorted(human_bts.items(), key=lambda x: -x[1]):
            print(f"  {MODEL_SHORT.get(m, m):<35} {bt:>6.1f}")

    # ── Rank correlation ───────────────────────────────────────────────────
    vlm_point_bts = {m: bt_ci[m]['mean'] for m in MODELS if m in bt_ci}
    corr_result = {}
    if human_bts:
        corr_result = bootstrap_rank_correlation(vlm_point_bts, human_bts)
        print(f"\n── VLM vs Human Rank Correlation ──")
        print(f"  Spearman ρ = {corr_result['rho_observed']:.4f}  "
              f"p = {corr_result['p_value']:.4f}  (n={corr_result['n_models']} models)")
        print(f"  95% Bootstrap CI: [{corr_result['lower_95_bootstrap']:.4f}, "
              f"{corr_result['upper_95_bootstrap']:.4f}]")
        if corr_result['p_value'] < 0.05:
            print("  ✅ Significant (p<0.05) — NOTE: n=6 makes this rare; prefer pair-level correlation")
        else:
            print("  ⚠️  Not significant at model level (expected with n=6 — use pair-level instead)")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = sorted(MODELS, key=lambda x: -bt_ci.get(x, {}).get('mean', 0))
    y_pos = range(len(model_names))
    means = [bt_ci[m]['mean'] for m in model_names]
    lower = [bt_ci[m]['mean'] - bt_ci[m]['lower_95'] for m in model_names]
    upper = [bt_ci[m]['upper_95'] - bt_ci[m]['mean'] for m in model_names]
    labels = [MODEL_SHORT.get(m, m) for m in model_names]

    bars = ax.barh(y_pos, means, xerr=[lower, upper], capsize=4,
                   color=['#38bdf8' if 'veo' in m else '#818cf8' if 'kling' in m
                          else '#10b981' if 'ltx' in m else '#f59e0b' if 'wan_v2' in m
                          else '#f87171' if 'hunyuan' in m else '#94a3b8'
                          for m in model_names],
                   error_kw={'ecolor': '#64748b', 'capthick': 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('VLM BT Rating')
    ax.set_title('VLM BT Rankings with 95% Bootstrap Confidence Intervals\n'
                 f'(N={N_BOOTSTRAP} bootstrap resamples, prompt-level)')
    ax.axvline(1500, linestyle='--', color='#334155', alpha=0.5, linewidth=1)
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    plot_path = Path(args.output).parent / 'bootstrap_bt_rating.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")

    # ── A3: Prompt count stability ─────────────────────────────────────────
    print("\n── A3: BT rating Stability vs. Number of Prompts (500 draws each) ──")
    stability = prompt_count_stability(matchups, n_bootstrap=500)

    print(f"\n{'N':>4}  {'Mean ρ':>8}  {'95% CI':>18}  {'Perfect rank %':>16}")
    print("-" * 55)
    for n, s in stability.items():
        print(f"{n:>4}  {s['mean_spearman_rho']:>8.4f}  "
              f"[{s['lower_95_rho']:>6.4f},{s['upper_95_rho']:>6.4f}]  "
              f"{s['pct_perfect_rank']:>14.1f}%")

    # Plot stability
    ns_list = [10, 20, 30, 40, 50]
    mean_rhos = [stability[n]['mean_spearman_rho'] for n in ns_list]
    lower_rhos = [stability[n]['lower_95_rho'] for n in ns_list]
    upper_rhos = [stability[n]['upper_95_rho'] for n in ns_list]
    perfect_pcts = [stability[n]['pct_perfect_rank'] for n in ns_list]

    # BT rating CI half-widths per model per N
    avg_hw = []
    for n in ns_list:
        hws = [stability[n]['bt_ci'].get(m, {}).get('ci_half_width', 0) for m in MODELS]
        avg_hw.append(np.mean([h for h in hws if h > 0]))

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("A3: BT Ranking Stability vs. Number of Prompts", fontsize=13, fontweight='bold')

    axes2[0].plot(ns_list, mean_rhos, 'o-', color='#38bdf8', linewidth=2, markersize=8)
    axes2[0].fill_between(ns_list, lower_rhos, upper_rhos, alpha=0.2, color='#38bdf8')
    axes2[0].axhline(0.9, linestyle=':', color='#f59e0b', alpha=0.7, label='ρ=0.90')
    axes2[0].set_xlabel('Number of Prompts')
    axes2[0].set_ylabel('Mean Spearman ρ vs. N=50 gold')
    axes2[0].set_title('Rank Correlation vs. Full Set')
    axes2[0].set_xticks(ns_list)
    axes2[0].legend(fontsize=8)
    axes2[0].grid(True, alpha=0.2)
    axes2[0].set_ylim(0, 1.05)

    axes2[1].bar(ns_list, perfect_pcts, color='#818cf8', width=2)
    axes2[1].set_xlabel('Number of Prompts')
    axes2[1].set_ylabel('% draws with perfect rank order')
    axes2[1].set_title('Perfect Rank Recovery Rate')
    axes2[1].set_xticks(ns_list)
    axes2[1].grid(True, axis='y', alpha=0.2)
    axes2[1].set_ylim(0, 105)

    axes2[2].plot(ns_list, avg_hw, 'o-', color='#10b981', linewidth=2, markersize=8)
    axes2[2].set_xlabel('Number of Prompts')
    axes2[2].set_ylabel('Avg BT Rating 95% CI half-width (pts)')
    axes2[2].set_title('BT Rating Confidence Interval Width')
    axes2[2].set_xticks(ns_list)
    axes2[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plot2_path = Path(args.output).parent / 'prompt_count_stability.png'
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot2_path}")

    # ── Save ───────────────────────────────────────────────────────────────
    out = {
        'n_matchups': len(matchups),
        'n_bootstrap': N_BOOTSTRAP,
        'vlm_bt_with_ci': bt_ci,
        'human_bt': {m: round(float(v), 1) for m, v in human_bts.items()},
        'rank_correlation': corr_result,
        'prompt_count_stability': stability
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == '__main__':
    main()
