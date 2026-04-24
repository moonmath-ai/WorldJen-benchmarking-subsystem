"""
A4 Ablation: Claude vs Gemini VLM Cross-Validation
Computes per-model average scores, BT rating rankings, and Spearman rank correlations
between the two auditors on the 20 validation prompts.
"""

import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
from collections import defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
CLAUDE_DIR  = DATA_ROOT / "results/claude_vlm"
GEMINI_DIR  = DATA_ROOT / "results/gemini_vlm"
VQA_FILE    = DATA_ROOT / "prompts/vqa_questions_50prompts.jsonl"
OUTPUT_JSON = DATA_ROOT / "results/summaries/a4_ablation_results.json"

MODELS = [
    "fal-ai_veo3.1_fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "wan2.1-1.3b",
]

MODEL_SHORT = {
    "fal-ai_veo3.1_fast": "Veo 3.1",
    "fal-ai_kling-video_v2.6_pro_text-to-video": "Kling v2.6",
    "fal-ai_ltx-2_text-to-video": "LTX-2",
    "fal-ai_wan_v2.2-a14b_text-to-video": "Wan v2.2 14B",
    "fal-ai_hunyuan-video-v1.5_text-to-video": "HunyuanVideo",
    "wan2.1-1.3b": "Wan 1.3B",
}

DIMENSIONS = [
    "subject_consistency", "scene_consistency", "motion_smoothness",
    "temporal_flickering", "inertial_consistency", "physical_mechanics",
    "object_permanence", "human_fidelity", "dynamic_degree",
    "semantic_adherence", "spatial_relationship", "semantic_drift",
    "composition_framing", "lighting_volumetric", "color_harmony",
    "structural_gestalt",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def load_null_dims():
    """Return {prompt_id: set(null_dimensions)} from VQA file.
    VQA records store questions nested under rec["vqa"][dim].
    """
    null_map = {}
    with open(VQA_FILE) as f:
        for line in f:
            rec = json.loads(line)
            pid = str(rec["prompt_id"])
            vqa = rec.get("vqa", {})
            null_set = set()
            for dim in DIMENSIONS:
                if vqa.get(dim) is None or vqa.get(dim) == []:
                    null_set.add(dim)
            null_map[pid] = null_set
    return null_map


def get_dim_avg(results_dict, dim):
    """Return mean score for a dimension, or None if absent/empty."""
    qs = results_dict.get(dim)
    if not qs:
        return None
    scores = []
    for q in qs:
        if isinstance(q, dict) and "score" in q:
            try:
                scores.append(float(q["score"]))
            except (TypeError, ValueError):
                pass
    return float(np.mean(scores)) if scores else None


def load_results(directory, model, restrict_pids=None):
    """
    Load all result files for a model → {prompt_id: {dim: avg_score}}.
    If restrict_pids is given, only include those prompt IDs.
    """
    data = {}
    for fname in os.listdir(directory):
        if fname.startswith(model) and fname.endswith(".json"):
            stem = fname[len(model)+1:-5]   # strip model_ prefix and .json
            pid = stem.replace("prompt_", "")
            if restrict_pids is not None and pid not in restrict_pids:
                continue
            with open(os.path.join(directory, fname)) as f:
                d = json.load(f)
            results = d.get("results", {})
            dim_avgs = {}
            for dim in DIMENSIONS:
                avg = get_dim_avg(results, dim)
                if avg is not None:
                    dim_avgs[dim] = avg
            data[pid] = dim_avgs
    return data


def get_claude_prompt_ids(model):
    """Return the set of prompt IDs present in Claude's results for this model."""
    pids = set()
    for fname in os.listdir(CLAUDE_DIR):
        if fname.startswith(model) and fname.endswith(".json"):
            stem = fname[len(model)+1:-5]
            pids.add(stem.replace("prompt_", ""))
    return pids


def model_avg_scores(results_by_pid, null_map):
    """
    Per-model, per-dimension average (skip null dims for each prompt).
    Returns {dim: mean_across_prompts}.
    """
    dim_scores = defaultdict(list)
    for pid, dim_avgs in results_by_pid.items():
        null_dims = null_map.get(pid, set())
        for dim, score in dim_avgs.items():
            if dim not in null_dims:
                dim_scores[dim].append(score)
    return {dim: float(np.mean(v)) for dim, v in dim_scores.items() if v}


def overall_avg(dim_avgs):
    vals = list(dim_avgs.values())
    return float(np.mean(vals)) if vals else None


# ── Bradley-Terry BT rating (canonical MM algorithm, matching unified_analyzer.py) ──

def _bt_raw_strengths(model_scores, models, idx):
    """Return raw log-p (zero-mean) for given model_scores dict."""
    n = len(models)
    W = np.zeros((n, n))
    all_pids = set.union(*[set(v.keys()) for v in model_scores.values()])
    for pid in all_pids:
        per_model = {}
        for m in models:
            if pid in model_scores[m]:
                s = overall_avg(model_scores[m][pid])
                if s is not None:
                    per_model[m] = s
        ms = list(per_model.keys())
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                mi, mj = ms[i], ms[j]
                si, sj = per_model[mi], per_model[mj]
                if si > sj:
                    W[idx[mi], idx[mj]] += 1
                elif sj > si:
                    W[idx[mj], idx[mi]] += 1
                else:
                    W[idx[mi], idx[mj]] += 0.5
                    W[idx[mj], idx[mi]] += 0.5

    p = np.ones(n)
    for _ in range(1000):
        p_new = np.zeros(n)
        for i in range(n):
            denom = sum(
                (W[i, j] + W[j, i]) / (p[i] + p[j])
                for j in range(n) if i != j
            )
            if denom > 0:
                p_new[i] = W[i, :].sum() / denom
        p_new = np.maximum(p_new, 1e-10)
        p_new /= p_new.mean()
        if np.max(np.abs(p_new - p)) < 1e-8:
            break
        p = p_new

    log_p = np.log(p)
    log_p -= log_p.mean()
    return {models[i]: log_p[i] for i in range(n)}


def compute_bt_rating(model_scores, base=1500):
    """
    Bradley-Terry MLE via the Minorization-Maximization (MM) algorithm.
    Exact floating-point ties receive +0.5 to each model (not dropped).
    All C(n,2) prompt-level pairwise comparisons are used.

    Returns {model: bt_score} on the same 1500-centred scale as the
    canonical analysis (unified_analyzer.py), enabling direct comparison
    of BT rating gap magnitudes across auditors.
    """
    models = list(model_scores.keys())
    idx = {m: i for i, m in enumerate(models)}
    raw = _bt_raw_strengths(model_scores, models, idx)
    return {m: base + 400 * raw[m] / np.log(10) for m in models}


def bootstrap_bt_rating_ci(model_scores, n_boot=1000, seed=42, base=1500):
    """
    Bootstrap 95% CI for BT rating on the 1500-centred scale.
    Returns {model: {"lo95": float, "hi95": float}}.
    """
    models = list(model_scores.keys())
    idx = {m: i for i, m in enumerate(models)}
    rng = np.random.default_rng(seed)
    all_pids = sorted(set.union(*[set(v.keys()) for v in model_scores.values()]))
    prompts = np.array(all_pids)
    samples = {m: [] for m in models}

    for b in range(n_boot):
        boot_pids = rng.choice(prompts, size=len(prompts), replace=True)
        boot_scores = {
            m: {f"b{b}_{i}": model_scores[m][pid]
                for i, pid in enumerate(boot_pids)
                if pid in model_scores[m]}
            for m in models
        }
        try:
            raw_b = _bt_raw_strengths(boot_scores, models, idx)
            for m in models:
                samples[m].append(base + 400 * raw_b[m] / np.log(10))
        except Exception:
            pass

    result = {}
    for m in models:
        arr = np.array(samples[m])
        result[m] = {
            "lo95": round(float(np.percentile(arr,  2.5)), 1),
            "hi95": round(float(np.percentile(arr, 97.5)), 1),
        }
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None,
                    help="Override output JSON path (default: data/results/summaries/a4_ablation_results.json)")
    args = ap.parse_args()
    if args.output:
        global OUTPUT_JSON
        OUTPUT_JSON = Path(args.output)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    null_map = load_null_dims()

    print("Loading results...")
    # Restrict Gemini to the same 20 validation prompts evaluated by Claude
    # so both auditors are compared on identical content.
    validation_pids = get_claude_prompt_ids(MODELS[0])
    print(f"  Validation prompt set: {len(validation_pids)} prompts")
    claude_by_model  = {}
    gemini_by_model  = {}
    for model in MODELS:
        claude_by_model[model]  = load_results(CLAUDE_DIR, model)
        gemini_by_model[model]  = load_results(GEMINI_DIR, model,
                                               restrict_pids=validation_pids)
        nc = len(claude_by_model[model])
        ng = len(gemini_by_model[model])
        print(f"  {MODEL_SHORT[model]:20s}  Claude={nc}  Gemini={ng}")

    # ── per-model dimension averages ─────────────────────────────────────────
    print("\n=== Per-model overall average scores ===")
    claude_model_dim  = {}
    gemini_model_dim  = {}
    claude_overall    = {}
    gemini_overall    = {}

    for model in MODELS:
        cd = model_avg_scores(claude_by_model[model], null_map)
        gd = model_avg_scores(gemini_by_model[model], null_map)
        claude_model_dim[model] = cd
        gemini_model_dim[model] = gd
        claude_overall[model]   = overall_avg(cd)
        gemini_overall[model]   = overall_avg(gd)

    print(f"\n{'Model':22s}  {'Claude':>8}  {'Gemini':>8}  {'Delta':>8}")
    print("-" * 55)
    for m in MODELS:
        c = claude_overall[m]
        g = gemini_overall[m]
        d = c - g if (c is not None and g is not None) else None
        print(f"{MODEL_SHORT[m]:22s}  {c:8.3f}  {g:8.3f}  {d:+8.3f}")

    # ── BT rating rankings ────────────────────────────────────────────────────────
    print("\nComputing BT rating rankings...")
    claude_bt = compute_bt_rating(claude_by_model)
    gemini_bt = compute_bt_rating(gemini_by_model)

    print("Bootstrapping Claude BT rating CIs (1000 resamples)…")
    claude_ci = bootstrap_bt_rating_ci(claude_by_model)
    print("Bootstrapping Gemini BT rating CIs (1000 resamples)…")
    gemini_ci = bootstrap_bt_rating_ci(gemini_by_model)

    print(f"\n{'Model':22s}  {'Claude BT rating':>10}  {'[lo,hi]':>14}  {'Gemini BT rating':>10}  {'[lo,hi]':>14}")
    print("-" * 80)
    for m in sorted(MODELS, key=lambda x: -claude_bt[x]):
        cc = claude_ci[m]; gc = gemini_ci[m]
        print(f"{MODEL_SHORT[m]:22s}  {claude_bt[m]:10.1f}  [{cc['lo95']:.0f},{cc['hi95']:.0f}]"
              f"  {gemini_bt[m]:10.1f}  [{gc['lo95']:.0f},{gc['hi95']:.0f}]")

    # ── Spearman rank correlation (model-level) ───────────────────────────────
    c_ranks = [claude_bt[m] for m in MODELS]
    g_ranks = [gemini_bt[m] for m in MODELS]
    rho_bt, p_bt = spearmanr(c_ranks, g_ranks)
    print(f"\nSpearman ρ (BT rating rankings):  ρ = {rho_bt:.4f},  p = {p_bt:.4f}")

    c_avg = [claude_overall[m] for m in MODELS]
    g_avg = [gemini_overall[m] for m in MODELS]
    rho_avg, p_avg = spearmanr(c_avg, g_avg)
    print(f"Spearman ρ (avg scores):    ρ = {rho_avg:.4f},  p = {p_avg:.4f}")

    # ── per-dimension Spearman (model avg score across models per dim) ─────────
    print("\n=== Per-dimension Spearman ρ (6 model scores per dim) ===")
    dim_rho = {}
    for dim in DIMENSIONS:
        c_vals = [claude_model_dim[m].get(dim) for m in MODELS]
        g_vals = [gemini_model_dim[m].get(dim) for m in MODELS]
        valid = [(c, g) for c, g in zip(c_vals, g_vals) if c is not None and g is not None]
        if len(valid) < 3:
            dim_rho[dim] = None
            continue
        cv, gv = zip(*valid)
        rho, p = spearmanr(cv, gv)
        dim_rho[dim] = {"rho": float(rho), "p": float(p)}
        print(f"  {dim:30s}  ρ={rho:+.3f}  p={p:.3f}")

    # ── absolute score deltas per dimension ──────────────────────────────────
    print("\n=== Mean score delta Claude - Gemini per dimension (averaged over all models) ===")
    dim_delta = {}
    for dim in DIMENSIONS:
        deltas = []
        for m in MODELS:
            c = claude_model_dim[m].get(dim)
            g = gemini_model_dim[m].get(dim)
            if c is not None and g is not None:
                deltas.append(c - g)
        if deltas:
            d = float(np.mean(deltas))
            dim_delta[dim] = d
            print(f"  {dim:30s}  Δ = {d:+.3f}")

    # ── rank ordering comparison ─────────────────────────────────────────────
    claude_rank = sorted(MODELS, key=lambda x: -claude_bt[x])
    gemini_rank = sorted(MODELS, key=lambda x: -gemini_bt[x])
    print("\n=== Model rank ordering ===")
    print(f"{'Rank':>4}  {'Claude':25s}  {'Gemini':25s}")
    for i, (cm, gm) in enumerate(zip(claude_rank, gemini_rank)):
        match = "✓" if cm == gm else "✗"
        print(f"  {i+1:2d}  {MODEL_SHORT[cm]:25s}  {MODEL_SHORT[gm]:25s}  {match}")

    # ── save results ─────────────────────────────────────────────────────────
    out = {
        "summary": {
            "spearman_rho_bt": float(rho_bt),
            "spearman_p_bt": float(p_bt),
            "spearman_rho_avg_scores": float(rho_avg),
            "spearman_p_avg_scores": float(p_avg),
            "n_prompts": 20,
            "n_models": 6,
        },
        "model_scores": {
            MODEL_SHORT[m]: {
                "claude_overall": claude_overall[m],
                "gemini_overall": gemini_overall[m],
                "delta_overall": claude_overall[m] - gemini_overall[m] if claude_overall[m] else None,
                "claude_bt": claude_bt[m],
                "claude_bt_lo95": claude_ci[m]["lo95"],
                "claude_bt_hi95": claude_ci[m]["hi95"],
                "gemini_bt": gemini_bt[m],
                "gemini_bt_lo95": gemini_ci[m]["lo95"],
                "gemini_bt_hi95": gemini_ci[m]["hi95"],
                "claude_rank": claude_rank.index(m) + 1,
                "gemini_rank": gemini_rank.index(m) + 1,
            } for m in MODELS
        },
        "dimension_analysis": {
            dim: {
                "spearman": dim_rho.get(dim),
                "mean_delta_claude_minus_gemini": dim_delta.get(dim),
                "claude_by_model": {MODEL_SHORT[m]: claude_model_dim[m].get(dim) for m in MODELS},
                "gemini_by_model": {MODEL_SHORT[m]: gemini_model_dim[m].get(dim) for m in MODELS},
            } for dim in DIMENSIONS
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
