"""
compare_a1.py
─────────────
A1 ablation: Enhanced vs Unenhanced prompt comparison.

Loads enhanced VLM results for the 20 validation prompts from results/,
and unenhanced VLM results from results_ablation_a1/ (which already
contains only the 20 validation prompt files), then computes:
  - Bradley-Terry BT rating per condition
  - Per-model average Likert score per condition
  - Per-dimension Δ (Enhanced − Unenhanced)
  - Spearman ρ between enhanced and unenhanced BT rating rankings

Writes: data/results/summaries/a1_enhancement_ablation_results.json

Usage:
    cd /path/to/WorldJen-benchmarking-subsystem/ablations/a1_prompt_enhancement
    python3 compare_a1.py
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
ENHANCED_DIR = DATA_ROOT / "results/gemini_vlm"
UNENH_DIR    = DATA_ROOT / "results/ablation_a1"
PROMPTS_FILE = DATA_ROOT / "prompts/prompts_50.jsonl"
OUTPUT_JSON  = DATA_ROOT / "results/summaries/a1_enhancement_ablation_results.json"

MODELS = [
    "fal-ai_veo3.1_fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "wan2.1-1.3b",
]
MODEL_LABELS = {
    "fal-ai_veo3.1_fast":                        "Veo 3.1 Fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video": "Kling v2.6 Pro",
    "fal-ai_ltx-2_text-to-video":                "LTX-2",
    "fal-ai_wan_v2.2-a14b_text-to-video":        "Wan v2.2 A14B",
    "fal-ai_hunyuan-video-v1.5_text-to-video":   "Hunyuan v1.5",
    "wan2.1-1.3b":                               "Wan 2.1 1.3B",
}
DIMENSIONS = [
    "subject_consistency", "scene_consistency", "motion_smoothness",
    "temporal_flickering", "inertial_consistency", "physical_mechanics",
    "object_permanence", "human_fidelity", "dynamic_degree",
    "semantic_adherence", "spatial_relationship", "semantic_drift",
    "composition_framing", "lighting_volumetric", "color_harmony",
    "structural_gestalt",
]


def get_validation_prompt_ids():
    """Return list of validation-tagged prompt IDs from prompts_50.jsonl."""
    pids = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("prompt_set") == "validation":
                pids.append(rec["prompt_id"])
    return sorted(pids)


def load_results(results_dir, filter_pids=None):
    """
    Load VLM results from results_dir.
    filter_pids: if provided, only load files for those prompt IDs.
    Returns {model: {prompt_id: {dim: mean_score}}}
    """
    pid_set = set(filter_pids) if filter_pids is not None else None
    data = defaultdict(dict)
    for fp in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(fp) as f:
                d = json.load(f)
            model     = d["model"]
            prompt_id = d["prompt_id"]
            if pid_set is not None and prompt_id not in pid_set:
                continue
            dim_scores = {}
            for dim, queries in d["results"].items():
                if queries is None:
                    continue
                scores = []
                for q in queries:
                    if q and q.get("score") is not None:
                        try:
                            scores.append(float(q["score"]))
                        except (ValueError, TypeError):
                            pass
                if scores:
                    dim_scores[dim] = float(np.mean(scores))
            data[model][prompt_id] = dim_scores
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e}")
    return dict(data)


def overall_avg(dim_scores):
    vals = list(dim_scores.values())
    return float(np.mean(vals)) if vals else None


def bradley_terry_bt_rating(model_data, base=1500):
    """BT rating via MM algorithm from per-(model,prompt) average scores."""
    models = list(model_data.keys())
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    W = np.zeros((n, n))

    all_pids = set()
    for m in models:
        all_pids |= set(model_data[m].keys())

    for pid in all_pids:
        scores = {}
        for m in models:
            if pid in model_data.get(m, {}):
                avg = overall_avg(model_data[m][pid])
                if avg is not None:
                    scores[m] = avg
        ms = list(scores.keys())
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                ma, mb = ms[a], ms[b]
                if scores[ma] > scores[mb]:
                    W[idx[ma], idx[mb]] += 1
                elif scores[mb] > scores[ma]:
                    W[idx[mb], idx[ma]] += 1
                else:
                    W[idx[ma], idx[mb]] += 0.5
                    W[idx[mb], idx[ma]] += 0.5

    p = np.ones(n)
    for _ in range(1000):
        p_new = np.zeros(n)
        for i in range(n):
            denom = sum(
                (W[i, j] + W[j, i]) / (p[i] + p[j])
                for j in range(n) if i != j and p[i] + p[j] > 0
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
    bt = {models[i]: float(round(base + 400 * log_p[i] / np.log(10), 1))
           for i in range(n)}
    return dict(sorted(bt.items(), key=lambda x: -x[1]))


def model_avg_score(model_data, model):
    all_scores = []
    for ds in model_data.get(model, {}).values():
        all_scores.extend(ds.values())
    return float(np.mean(all_scores)) if all_scores else None


def dim_avg(model_data, model, dim):
    scores = [ds[dim] for ds in model_data.get(model, {}).values() if dim in ds]
    return float(np.mean(scores)) if scores else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None,
                    help="Override output JSON path (default: data/results/summaries/a1_enhancement_ablation_results.json)")
    args = ap.parse_args()
    if args.output:
        global OUTPUT_JSON
        OUTPUT_JSON = Path(args.output)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading validation prompt IDs …")
    val_pids = get_validation_prompt_ids()
    print(f"  {len(val_pids)} validation prompts")

    print("Loading enhanced results (results/, validation prompts only) …")
    enh_data = load_results(ENHANCED_DIR, filter_pids=val_pids)
    for m in MODELS:
        print(f"  {MODEL_LABELS[m]:22s}: {len(enh_data.get(m, {})):2d} prompts")

    print("Loading unenhanced results (results_ablation_a1/) …")
    unh_data = load_results(UNENH_DIR)   # all files there are already val prompts
    for m in MODELS:
        print(f"  {MODEL_LABELS[m]:22s}: {len(unh_data.get(m, {})):2d} prompts")

    # Find common prompt IDs across both conditions and all models
    common_pids = set(val_pids)
    for m in MODELS:
        common_pids &= (set(enh_data.get(m, {}).keys()) & set(unh_data.get(m, {}).keys()))
    print(f"\nCommon prompts in both conditions: {len(common_pids)}")

    # Trim to common set
    for m in MODELS:
        enh_data[m] = {p: v for p, v in enh_data.get(m, {}).items() if p in common_pids}
        unh_data[m] = {p: v for p, v in unh_data.get(m, {}).items() if p in common_pids}

    print("Computing Bradley-Terry BT rating …")
    bt_enh = bradley_terry_bt_rating(enh_data)
    bt_unh = bradley_terry_bt_rating(unh_data)

    enh_ranks = {m: r for r, m in enumerate(bt_enh.keys(), 1)}
    unh_ranks = {m: r for r, m in enumerate(bt_unh.keys(), 1)}
    rho, pval = spearmanr(
        [enh_ranks[m] for m in MODELS],
        [unh_ranks[m] for m in MODELS])
    print(f"  Spearman ρ = {rho:.4f}  p = {pval:.4f}")

    print(f"\n{'Model':22s}  {'Enh BT rating':>8}  {'Unh BT rating':>8}  {'ΔELO':>7}  {'Enh Avg':>8}  {'Unh Avg':>8}  {'ΔAvg':>7}")
    print("-" * 85)
    model_results = {}
    for m in MODELS:
        e_bt = bt_enh.get(m)
        u_bt = bt_unh.get(m)
        d_bt = e_bt - u_bt
        e_avg = model_avg_score(enh_data, m)
        u_avg = model_avg_score(unh_data, m)
        d_avg = e_avg - u_avg
        lbl = MODEL_LABELS[m]
        print(f"{lbl:22s}  {e_bt:8.1f}  {u_bt:8.1f}  {d_bt:+7.1f}  {e_avg:8.3f}  {u_avg:8.3f}  {d_avg:+7.3f}")
        model_results[m] = {
            "label":                lbl,
            "bt_enhanced":         e_bt,
            "bt_unenhanced":       u_bt,
            "delta_bt":            round(d_bt, 1),
            "rank_enhanced":        enh_ranks[m],
            "rank_unenhanced":      unh_ranks[m],
            "avg_score_enhanced":   round(e_avg, 4),
            "avg_score_unenhanced": round(u_avg, 4),
            "delta_avg_score":      round(d_avg, 4),
            "n_prompts_enhanced":   len(enh_data.get(m, {})),
            "n_prompts_unenhanced": len(unh_data.get(m, {})),
        }

    # Per-dimension delta
    print("\n=== Per-dimension Δ (Enhanced − Unenhanced) ===")
    dim_results = {}
    for dim in DIMENSIONS:
        pool = []
        per_model = {}
        for m in MODELS:
            e = dim_avg(enh_data, m, dim)
            u = dim_avg(unh_data, m, dim)
            if e is not None and u is not None:
                delta = e - u
                per_model[m] = round(delta, 4)
                pool.append(delta)
        pool_delta = round(float(np.mean(pool)), 4) if pool else None
        dim_results[dim] = {"pooled_delta": pool_delta, "per_model": per_model}
        if pool_delta is not None:
            print(f"  {dim:30s}  Δ={pool_delta:+.3f}")

    all_enh = [v for m in MODELS for ds in enh_data.get(m, {}).values() for v in ds.values()]
    all_unh = [v for m in MODELS for ds in unh_data.get(m, {}).values() for v in ds.values()]
    ov_enh = float(np.mean(all_enh))
    ov_unh = float(np.mean(all_unh))
    print(f"\nOverall: Enh={ov_enh:.4f}  Unh={ov_unh:.4f}  Δ={ov_enh-ov_unh:+.4f}")

    out = {
        "experiment":   "A1: Enhanced vs Unenhanced prompt comparison",
        "description":  "BT rating and average Likert scores for enhanced vs unenhanced prompts on 20 validation prompts",
        "data_sources": {
            "enhanced_results":   "data/results/gemini_vlm/",
            "unenhanced_results": "data/results/ablation_a1/",
            "enhanced_prompts":   "data/prompts/prompts_50.jsonl",
            "unenhanced_prompts": "data/prompts/prompts_ablation_a1_validation20.jsonl",
            "unenhanced_vqa":     "data/prompts/vqa_questions_ablation_a1.jsonl",
            "unenhanced_videos":  "data/videos_ablation_a1/",
        },
        "summary": {
            "n_prompts":              len(common_pids),
            "n_models":               len(MODELS),
            "n_videos_per_condition": len(MODELS) * len(common_pids),
            "prompt_ids":             sorted(common_pids),
            "spearman_rho_bt":       float(round(rho, 4)),
            "spearman_p_bt":         float(round(pval, 4)),
            "overall_avg_enhanced":   round(ov_enh, 4),
            "overall_avg_unenhanced": round(ov_unh, 4),
            "overall_delta_avg":      round(ov_enh - ov_unh, 4),
        },
        "model_results":     model_results,
        "dimension_results": dim_results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
