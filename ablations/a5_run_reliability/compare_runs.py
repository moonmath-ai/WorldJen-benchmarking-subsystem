"""
compare_runs.py
Compare Run1 (results/) vs Run2 (results2/) of the Gemini VLM evaluator.

Outputs:
  1. Per-dimension mean score comparison (both runs) with paired t-test
  2. Per-model BT_rating comparison (both runs) with Spearman rank correlation
  3. Overall score correlation (Pearson r, scatter stats)
  4. Summary table saved to data/results/summaries/a5_ablation_results.json
"""

import argparse
import os
import json, os, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
RUN1 = DATA_ROOT / "results/gemini_vlm"
RUN2 = DATA_ROOT / "results/gemini_vlm_run2"

_ap = argparse.ArgumentParser()
_ap.add_argument("--output", default=None,
                 help="Override output JSON path (default: data/results/summaries/a5_ablation_results.json)")
_args, _ = _ap.parse_known_args()
OUT = Path(_args.output) if _args.output else DATA_ROOT / "results/summaries/a5_ablation_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

DIMS = [
    "subject_consistency","scene_consistency","motion_smoothness",
    "temporal_flickering","inertial_consistency","physical_mechanics",
    "object_permanence","human_fidelity","dynamic_degree",
    "semantic_adherence","spatial_relationship","semantic_drift",
    "composition_framing","lighting_volumetric","color_harmony","structural_gestalt",
]

MODELS = [
    "fal-ai_veo3.1_fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "wan2.1-1.3b",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_run(run_dir: Path):
    """Return dict: (model, prompt_id, dim) -> mean score"""
    data = {}
    for fpath in sorted(run_dir.glob("*.json")):
        with open(fpath) as f:
            d = json.load(f)
        model = d.get("model","")
        pid   = d.get("prompt_id","")
        for dim, qlist in d.get("results", {}).items():
            if not isinstance(qlist, list): continue
            scores = [q["score"] for q in qlist if isinstance(q.get("score"), (int,float))]
            if scores:
                data[(model, pid, dim)] = np.mean(scores)
    return data

def model_dim_means(data):
    """Return dict: model -> dim -> mean across prompts"""
    acc = defaultdict(lambda: defaultdict(list))
    for (model, pid, dim), score in data.items():
        acc[model][dim].append(score)
    return {m: {d: np.mean(vs) for d,vs in dims.items()} for m,dims in acc.items()}

def model_overall(mdm):
    """Overall mean across all dims for each model"""
    return {m: np.mean(list(dims.values())) for m,dims in mdm.items()}

def bt_rank(scores_dict):
    """Rank models by overall score (1=best)"""
    ranked = sorted(scores_dict.items(), key=lambda x: -x[1])
    return {m: i+1 for i,(m,_) in enumerate(ranked)}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading Run 1...")
r1 = load_run(RUN1)
print(f"  {len(r1)} (model, prompt, dim) entries")

print("Loading Run 2...")
r2 = load_run(RUN2)
print(f"  {len(r2)} (model, prompt, dim) entries")

# ── Matched pairs for statistical tests ───────────────────────────────────────
common_keys = sorted(set(r1) & set(r2))
print(f"\nMatched (model, prompt, dim) pairs: {len(common_keys)}")

scores1 = np.array([r1[k] for k in common_keys])
scores2 = np.array([r2[k] for k in common_keys])

# ── 1. Overall correlation ─────────────────────────────────────────────────────
pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
mae = np.mean(np.abs(scores1 - scores2))
rmse = np.sqrt(np.mean((scores1 - scores2)**2))

print("\n── Overall Run1 vs Run2 ──────────────────────────────────────")
print(f"  Pearson  r = {pearson_r:.4f}  (p={pearson_p:.2e})")
print(f"  Spearman r = {spearman_r:.4f}  (p={spearman_p:.2e})")
print(f"  MAE        = {mae:.4f}")
print(f"  RMSE       = {rmse:.4f}")
print(f"  Mean Run1  = {scores1.mean():.4f}  std={scores1.std():.4f}")
print(f"  Mean Run2  = {scores2.mean():.4f}  std={scores2.std():.4f}")

# ── 2. Per-dimension analysis ─────────────────────────────────────────────────
print("\n── Per-Dimension Comparison ──────────────────────────────────")
print(f"{'Dimension':<26} {'Run1':>6} {'Run2':>6} {'Δ':>7} {'t':>7} {'p':>9} {'sig':>4}")
print("─"*72)

dim_results = {}
mdm1 = model_dim_means(r1)
mdm2 = model_dim_means(r2)

for dim in DIMS:
    # Collect matched pairs for this dim
    dim_keys = [k for k in common_keys if k[2]==dim]
    if not dim_keys: continue
    s1 = np.array([r1[k] for k in dim_keys])
    s2 = np.array([r2[k] for k in dim_keys])
    t_stat, t_p = stats.ttest_rel(s1, s2)
    delta = s2.mean() - s1.mean()
    sig = "***" if t_p<0.001 else "**" if t_p<0.01 else "*" if t_p<0.05 else "ns"
    print(f"{dim:<26} {s1.mean():>6.3f} {s2.mean():>6.3f} {delta:>+7.3f} {t_stat:>7.3f} {t_p:>9.4f} {sig:>4}")
    dim_results[dim] = {
        "run1_mean": float(s1.mean()), "run2_mean": float(s2.mean()),
        "delta": float(delta), "t_stat": float(t_stat), "p_value": float(t_p),
        "significant": sig != "ns", "n_pairs": len(dim_keys),
    }

# ── 3. Per-model analysis ─────────────────────────────────────────────────────
print("\n── Per-Model Comparison ──────────────────────────────────────")
print(f"{'Model':<45} {'Run1':>6} {'Run2':>6} {'Δ':>7} {'t':>7} {'p':>9} {'sig':>4}")
print("─"*84)

model_results = {}
for model in MODELS:
    model_keys = [k for k in common_keys if k[0]==model]
    if not model_keys: continue
    s1 = np.array([r1[k] for k in model_keys])
    s2 = np.array([r2[k] for k in model_keys])
    t_stat, t_p = stats.ttest_rel(s1, s2)
    delta = s2.mean() - s1.mean()
    sig = "***" if t_p<0.001 else "**" if t_p<0.01 else "*" if t_p<0.05 else "ns"
    short = model.replace("fal-ai_","").replace("_text-to-video","")
    print(f"{short:<45} {s1.mean():>6.3f} {s2.mean():>6.3f} {delta:>+7.3f} {t_stat:>7.3f} {t_p:>9.4f} {sig:>4}")
    model_results[model] = {
        "run1_mean": float(s1.mean()), "run2_mean": float(s2.mean()),
        "delta": float(delta), "t_stat": float(t_stat), "p_value": float(t_p),
        "significant": sig != "ns", "n_pairs": len(model_keys),
    }

# ── 4. Rank stability ─────────────────────────────────────────────────────────
overall1 = {m: model_results[m]["run1_mean"] for m in MODELS if m in model_results}
overall2 = {m: model_results[m]["run2_mean"] for m in MODELS if m in model_results}
rank1 = bt_rank(overall1)
rank2 = bt_rank(overall2)

print("\n── Model Rankings ────────────────────────────────────────────")
print(f"{'Model':<45} {'Rank1':>6} {'Rank2':>6} {'ΔRank':>7}")
print("─"*66)
for model in sorted(rank1, key=lambda m: rank1[m]):
    short = model.replace("fal-ai_","").replace("_text-to-video","")
    dr = rank2[model] - rank1[model]
    print(f"{short:<45} {rank1[model]:>6} {rank2[model]:>6} {dr:>+7}")

# Rank correlation
ranks1_arr = np.array([rank1[m] for m in MODELS if m in rank1])
ranks2_arr = np.array([rank2[m] for m in MODELS if m in rank2])
rank_sr, rank_sp = stats.spearmanr(ranks1_arr, ranks2_arr)
print(f"\n  Rank Spearman r = {rank_sr:.4f}  (p={rank_sp:.4f})")

# ── 5. ICC (intraclass correlation) estimate via two-way mixed ────────────────
# Simple ICC(3,1) approximation: r = (MSb - MSw) / (MSb + MSw)
n = len(scores1)
grand_mean = (scores1 + scores2).mean() / 1
row_means  = (scores1 + scores2) / 2
MSb = 2 * np.var(row_means, ddof=1)          # between-target variance
residuals  = np.concatenate([scores1 - row_means, scores2 - row_means])
MSw = np.var(residuals, ddof=1)
icc = (MSb - MSw) / (MSb + MSw) if (MSb + MSw) > 0 else 0
print(f"\n  ICC(3,1) estimate = {icc:.4f}  (>0.75 = excellent, >0.60 = good)")

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "overall": {
        "n_matched_pairs": len(common_keys),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "mae": float(mae), "rmse": float(rmse),
        "icc_3_1": float(icc),
        "run1_mean": float(scores1.mean()), "run1_std": float(scores1.std()),
        "run2_mean": float(scores2.mean()), "run2_std": float(scores2.std()),
    },
    "rank_stability": {
        "spearman_r": float(rank_sr), "spearman_p": float(rank_sp),
        "run1_ranks": rank1, "run2_ranks": rank2,
    },
    "per_dimension": dim_results,
    "per_model": model_results,
}

with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary → {OUT}")
