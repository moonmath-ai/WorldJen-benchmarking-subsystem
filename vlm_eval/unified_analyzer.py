"""
unified_analyzer.py

Loads VLM results from the single official results directory (results/),
tags each record as "validation" or "calibration" using prompts_50.jsonl,
then computes per-model/per-dimension scores, Bradley-Terry rating (with
bootstrap 95% CI), and PHAS.

Key handling:
  - Null dimensions (e.g. human_fidelity=None for non-human prompts) are
    skipped in all averages.
  - BT rating: computed over all 50 prompts for maximum BT stability.
  - Bootstrap CI: 1,000 prompt-level resamples; stored as vlm_bt_with_ci
    in summary_report_unified.json for figure generation.
  - PHAS: computed on validation set (20 prompts) only to avoid in-sample
    bias (PHAS weights were calibrated on the 30 calibration prompts).
  - Bradley-Terry MLE is used for BT rating (order-independent).
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
RESULTS_DIR  = DATA_ROOT / "results/gemini_vlm"
PROMPTS_FILE = DATA_ROOT / "prompts/prompts_50.jsonl"
OUTPUT_DIR   = DATA_ROOT / "results/summaries"
PLOTS_DIR    = OUTPUT_DIR / "plots_unified"
OUTPUT_JSON  = OUTPUT_DIR / "summary_report_unified.json"
OUTPUT_MD    = OUTPUT_DIR / "results_summary_unified.md"

# ── PHAS weights (16 dimensions) ───────────────────────────────────────────
# Calibrated via non-negative ridge logistic regression on 618 human pairwise
# annotations (979 total; 5 annotators). Source: human_eval_analysis.json
# CV accuracy: 70.4%, validation accuracy: 68.1%.
# Note: physics dims (inertial, physical_mechanics, object_permanence) receive
# near-zero empirical weight — human annotators prioritise composition and
# semantic adherence over physical plausibility in short video clips.
PHAS_WEIGHTS = {
    # Non-negative ridge logistic regression on 1653 calibration annotations
    # (2696 total; 7 annotators, A1-retest excluded). Source: human_eval_analysis.json
    # CV accuracy: 64.4%, validation accuracy: 62.6%.
    "subject_consistency": 0.0907,
    "scene_consistency":   0.0000,
    "motion_smoothness":   0.0743,
    "temporal_flickering": 0.0144,
    "inertial_consistency":0.0542,
    "physical_mechanics":  0.0575,
    "object_permanence":   0.0000,
    "human_fidelity":      0.0000,
    "dynamic_degree":      0.0000,
    "semantic_adherence":  0.2638,
    "spatial_relationship":0.0000,
    "semantic_drift":      0.1261,
    "composition_framing": 0.1304,
    "lighting_volumetric": 0.0534,
    "color_harmony":       0.0816,
    "structural_gestalt":  0.0535,
}
assert abs(sum(PHAS_WEIGHTS.values()) - 1.0) < 1e-3, "PHAS weights must sum to 1"

ALL_DIMS = list(PHAS_WEIGHTS.keys())

# Pretty display names
MODEL_LABELS = {
    "fal-ai_veo3.1_fast":                        "Veo 3.1",
    "fal-ai_kling-video_v2.6_pro_text-to-video": "Kling",
    "fal-ai_wan_v2.2-a14b_text-to-video":        "Wan A14B",
    "fal-ai_ltx-2_text-to-video":                "LTX-2",
    "fal-ai_hunyuan-video-v1.5_text-to-video":   "Hunyuan",
    "wan2.1-1.3b":                               "Wan 1.3B",
}

# ── Helpers ────────────────────────────────────────────────────────────────

def load_prompt_set_map():
    """Return {prompt_id: prompt_set} from prompts_50.jsonl."""
    mapping = {}
    with open(PROMPTS_FILE) as f:
        for line in f:
            rec = json.loads(line.strip())
            if "prompt_id" in rec and "prompt_set" in rec:
                mapping[rec["prompt_id"]] = rec["prompt_set"]
    return mapping


def load_all_results():
    """Load every JSON result file from the official results directory."""
    prompt_set_map = load_prompt_set_map()
    records = []
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
            model     = data["model"]
            prompt_id = data["prompt_id"]
            source    = prompt_set_map.get(prompt_id, "unknown")
            for dim, queries in data["results"].items():
                if queries is None:          # null suitability — skip
                    continue
                scores = []
                for q in queries:
                    if q and q.get("score") is not None:
                        try:
                            scores.append(int(q["score"]))
                        except (ValueError, TypeError):
                            pass
                if scores:
                    records.append({
                        "model":      model,
                        "prompt_id":  prompt_id,
                        "dimension":  dim,
                        "mean_score": np.mean(scores),
                        "variance":   np.var(scores, ddof=1) if len(scores) > 1 else 0.0,
                        "n_questions":len(scores),
                        "source":     source,
                    })
        except Exception as e:
            print(f"  [WARN] skipping {fp.name}: {e}")
    return pd.DataFrame(records)


def _bt_strengths(df, base=1500):
    """
    Core Bradley-Terry MLE.  Returns float BT rating values (not rounded).
    For each prompt_id we aggregate all dimension scores into a single
    mean per model, then run all C(n,2) pairwise comparisons.
    """
    models = sorted(df["model"].unique())
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    W = np.zeros((n, n))
    for pid, grp in df.groupby("prompt_id"):
        per_model = grp.groupby("model")["mean_score"].mean()
        m_list = list(per_model.index)
        for a in range(len(m_list)):
            for b in range(a + 1, len(m_list)):
                ma, mb = m_list[a], m_list[b]
                sa, sb = per_model[ma], per_model[mb]
                if sa > sb:
                    W[idx[ma], idx[mb]] += 1
                elif sb > sa:
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
    return {models[i]: base + 400 * log_p[i] / np.log(10) for i in range(n)}


def bradley_terry_bt(df, base=1500):
    """
    Order-independent Bradley-Terry MLE (deterministic point estimate).
    Returns a dict {model: bt} with integer-rounded BT rating values.
    """
    raw = _bt_strengths(df, base)
    bt = {m: int(round(v)) for m, v in raw.items()}
    return dict(sorted(bt.items(), key=lambda x: x[1], reverse=True))


def bootstrap_bradley_terry_bt(df, n_boot=1000, seed=42, base=1500):
    """
    Bootstrap 95% CI for VLM BT BT rating by resampling prompts with replacement.

    Returns:
        dict  {model: {"mean": float, "lower_95": float, "upper_95": float,
                       "std": float}}
    The deterministic point estimate is NOT the bootstrap mean; use
    bradley_terry_bt() for the canonical BT rating value.
    """
    rng = np.random.default_rng(seed)
    prompts = np.array(df["prompt_id"].unique())
    models  = sorted(df["model"].unique())
    samples = {m: [] for m in models}

    print(f"  Bootstrapping VLM BT rating CI ({n_boot} prompt-level resamples) …",
          flush=True)
    for b in range(n_boot):
        boot_ids = rng.choice(prompts, size=len(prompts), replace=True)
        dfs = []
        for i, pid in enumerate(boot_ids):
            sub = df[df["prompt_id"] == pid].copy()
            sub["prompt_id"] = f"b{b}_{i}"
            dfs.append(sub)
        boot_df = pd.concat(dfs, ignore_index=True)
        try:
            raw = _bt_strengths(boot_df, base)
            for m in models:
                if m in raw:
                    samples[m].append(raw[m])
        except Exception:
            pass

    result = {}
    for m in models:
        arr = np.array(samples[m])
        result[m] = {
            "mean":      round(float(np.mean(arr)),   1),
            "lower_95":  round(float(np.percentile(arr,  2.5)), 1),
            "upper_95":  round(float(np.percentile(arr, 97.5)), 1),
            "std":       round(float(np.std(arr)),    1),
        }
    return result


def compute_phas(df):
    """
    Per-model PHAS.  For each (model, prompt_id) we compute a weighted
    score using only dimensions that are present (null dims excluded by
    load_all_results).  Weight is renormalised to available dims.
    """
    records = []
    for (model, pid), grp in df.groupby(["model", "prompt_id"]):
        scores   = grp.set_index("dimension")["mean_score"].to_dict()
        variances= grp.set_index("dimension")["variance"].to_dict()

        avail = {d: w for d, w in PHAS_WEIGHTS.items() if d in scores}
        if not avail:
            continue
        total_w = sum(avail.values())
        base = sum(scores[d] * w for d, w in avail.items()) / total_w

        mean_var = np.mean([variances.get(d, 0) for d in avail])
        penalty  = 1 - min(0.3, mean_var * 0.05)
        phas = base * penalty

        records.append({"model": model, "prompt_id": pid, "phas": phas})

    phas_df = pd.DataFrame(records)
    model_phas = phas_df.groupby("model")["phas"].mean().sort_values(ascending=False)
    return model_phas, phas_df


def generate_plots(model_stats, bt, model_phas):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── Heatmap ──────────────────────────────────────────────────────────
    # Reorder columns by BT rating rank
    bt_order = list(bt.keys())
    stat_cols  = [m for m in bt_order if m in model_stats.index]
    plot_stats = model_stats.loc[stat_cols].T   # dims × models

    # Use pretty labels for model columns
    plot_stats.columns = [MODEL_LABELS.get(m, m) for m in plot_stats.columns]

    # Reorder dimensions vertically by group
    DIM_GROUPS = [
        # Group A: Temporal/Spatial Coherence
        ("subject_consistency", "Subject Consistency"),
        ("scene_consistency",   "Scene Consistency"),
        ("motion_smoothness",   "Motion Smoothness"),
        ("temporal_flickering", "Temporal Flickering"),
        ("inertial_consistency","Inertial Consistency"),
        # Group B: Physical Plausibility
        ("physical_mechanics",  "Physical Mechanics"),
        ("object_permanence",   "Object Permanence"),
        ("human_fidelity",      "Human Fidelity"),
        ("dynamic_degree",      "Dynamic Degree"),
        # Group C: Semantic Fidelity
        ("semantic_adherence",  "Semantic Adherence"),
        ("spatial_relationship","Spatial Relationship"),
        ("semantic_drift",      "Semantic Drift"),
        # Group D: Aesthetic Quality
        ("composition_framing", "Composition & Framing"),
        ("lighting_volumetric", "Lighting & Volumetric"),
        ("color_harmony",       "Color Harmony"),
        ("structural_gestalt",  "Structural Gestalt"),
    ]
    GROUP_LABELS = [
        (0, 5,  "A: Temporal /\nSpatial"),
        (5, 9,  "B: Physical\nPlausibility"),
        (9, 12, "C: Semantic\nFidelity"),
        (12, 16,"D: Aesthetic\nQuality"),
    ]
    ordered_keys   = [k for k, _ in DIM_GROUPS]
    ordered_labels = [lbl for _, lbl in DIM_GROUPS]

    # Select only dims present in plot_stats
    present_keys   = [k for k in ordered_keys if k in plot_stats.index]
    present_labels = [ordered_labels[ordered_keys.index(k)] for k in present_keys]
    plot_stats = plot_stats.loc[present_keys]
    plot_stats.index = present_labels

    fig, ax = plt.subplots(figsize=(13, 9))
    # RdYlGn: red=low (worst), yellow=mid, green=high (best)
    sns.heatmap(
        plot_stats, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5, linewidths=0.4, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Per-model per-dimension Likert scores (all 50 prompts)", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Add group separators and labels on the left
    group_boundaries = [0, 5, 9, 12, 16]
    for b in group_boundaries[1:-1]:
        ax.axhline(b, color="black", linewidth=1.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_dimension_heatmap.png", dpi=150)
    plt.close()

    # ── Overall bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    labels   = [MODEL_LABELS.get(m, m) for m in stat_cols]
    avg      = [model_stats.loc[m].mean() for m in stat_cols]
    phas_vals= [model_phas.get(m, 0) for m in stat_cols]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, avg,       w, label="Avg Score", color="#4C72B0")
    ax.bar(x + w/2, phas_vals, w, label="PHAS",      color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(1, 5); ax.set_ylabel("Score (1–5)")
    ax.set_title("Model comparison: Avg Score (50 prompts) vs PHAS (validation 20)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overall_comparison.png", dpi=150)
    plt.close()

    # ── Per-model profile bars ────────────────────────────────────────
    for model in stat_cols:
        label = MODEL_LABELS.get(model, model)
        row   = model_stats.loc[model].sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(row))
        row.plot(kind="barh", color=colors, ax=ax)
        ax.set_xlim(1, 5); ax.set_xlabel("Score (1–5)")
        ax.set_title(f"Performance profile: {label} (50 prompts)")
        plt.tight_layout()
        safe = model.replace("/", "_").replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"profile_{safe}.png", dpi=150)
        plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


def write_markdown(bt, model_phas, model_stats, df):
    n_prompts = df["prompt_id"].nunique()
    n_models  = df["model"].nunique()
    n_answers = len(df)

    lines = [
        "# WorldJen Unified Analysis — 50 Prompts",
        "",
        f"**Prompts:** {n_prompts} &nbsp;|&nbsp; **Models:** {n_models} &nbsp;|&nbsp;"
        f" **Dimension-level records:** {n_answers}",
        "",
        "## BT rating Leaderboard (Bradley-Terry MLE)",
        "",
        "| Rank | Model | BT rating | PHAS | Avg Score |",
        "|------|-------|-----|------|-----------|",
    ]
    for rank, (model, bt_val) in enumerate(bt.items(), 1):
        label = MODEL_LABELS.get(model, model)
        phas  = model_phas.get(model, float("nan"))
        avg   = model_stats.loc[model].mean() if model in model_stats.index else float("nan")
        lines.append(f"| {rank} | {label} | {bt_val} | {phas:.3f} | {avg:.3f} |")

    lines += [
        "",
        "## Per-dimension scores (mean across all applicable prompts)",
        "",
    ]
    # Pivot: models as columns, dims as rows
    pivot = model_stats.T.copy()
    pivot.columns = [MODEL_LABELS.get(m, m) for m in pivot.columns]
    lines.append("| Dimension | " + " | ".join(pivot.columns) + " |")
    lines.append("|-----------|" + "|".join(["------"] * len(pivot.columns)) + "|")
    for dim, row in pivot.iterrows():
        vals = " | ".join(f"{v:.2f}" if not np.isnan(v) else "—" for v in row)
        lines.append(f"| {dim} | {vals} |")

    lines += ["", "---", f"*Generated by unified_analyzer.py*", ""]

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown summary saved to {OUTPUT_MD}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None,
                    help="Override output JSON path (default: data/results/summaries/summary_report_unified.json)")
    args = ap.parse_args()
    if args.output:
        global OUTPUT_JSON, OUTPUT_MD, PLOTS_DIR
        OUTPUT_JSON = Path(args.output)
        OUTPUT_MD   = OUTPUT_JSON.with_suffix(".md")
        PLOTS_DIR   = OUTPUT_JSON.parent / "plots_unified"
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results …")
    df = load_all_results()
    print(f"  {len(df)} dimension-level records from "
          f"{df['prompt_id'].nunique()} prompts × {df['model'].nunique()} models")

    val_count  = df[df["source"] == "validation"]["prompt_id"].nunique()
    cal_count  = df[df["source"] == "calibration"]["prompt_id"].nunique()
    print(f"  Validation set: {val_count} prompts | Calibration set: {cal_count} prompts")

    print("\nComputing per-model per-dimension means (all 50 prompts) …")
    model_stats = df.groupby(["model", "dimension"])["mean_score"].mean().unstack()

    # BT rating: all 50 prompts for maximum BT stability
    print("Computing Bradley-Terry BT rating (all 50 prompts) …")
    bt = bradley_terry_bt(df)
    print("Computing bootstrap 95% CI for VLM BT rating …")
    bt_ci = bootstrap_bradley_terry_bt(df)

    # PHAS: validation set only — weights were calibrated on calibration_30,
    # so computing PHAS on calibration_30 would be in-sample evaluation.
    df_val = df[df["source"] == "validation"]
    print(f"Computing PHAS (validation set only: {df_val['prompt_id'].nunique()} prompts) …")
    model_phas, phas_df = compute_phas(df_val)

    print("\n── BT rating Leaderboard ──────────────────────────")
    for rank, (model, val) in enumerate(bt.items(), 1):
        label = MODEL_LABELS.get(model, model)
        phas  = model_phas.get(model, float("nan"))
        avg   = model_stats.loc[model].mean() if model in model_stats.index else float("nan")
        ci    = bt_ci.get(model, {})
        lo, hi = ci.get("lower_95", "?"), ci.get("upper_95", "?")
        print(f"  {rank}. {label:<12}  BT rating {val:>5} [{lo}, {hi}]  PHAS {phas:.3f}  Avg {avg:.3f}")

    print("\n── Dimension averages (all models) ──────────")
    dim_avgs = df.groupby("dimension")["mean_score"].mean().sort_values()
    for dim, val in dim_avgs.items():
        print(f"  {dim:<28} {val:.3f}")

    print("\nGenerating plots …")
    generate_plots(model_stats, bt, model_phas)

    write_markdown(bt, model_phas, model_stats, df)

    # Save full JSON
    # Variance penalty per model
    model_variances = df.groupby(["model", "dimension"])["variance"].mean().unstack()
    penalty_table = {}
    for model in df["model"].unique():
        if model not in model_variances.index:
            continue
        dim_vars = model_variances.loc[model].dropna()
        avail = [d for d in PHAS_WEIGHTS if d in dim_vars.index]
        mean_var = float(dim_vars[avail].mean())
        penalty_raw = 0.05 * mean_var
        lam = 1 - min(0.30, penalty_raw)
        penalty_table[model] = {
            "mean_variance":  round(mean_var, 4),
            "penalty_raw":    round(penalty_raw, 4),
            "lambda":         round(lam, 4),
            "penalty_pct":    round((1 - lam) * 100, 2),
        }

    # Build vlm_bt_with_ci in the format expected by generate_bt_comparison.py:
    # {model: {"mean": float, "lower_95": float, "upper_95": float}}
    vlm_bt_with_ci = {}
    for m, pt in bt.items():
        ci = bt_ci.get(m, {})
        vlm_bt_with_ci[m] = {
            "mean":      float(pt),          # deterministic point estimate
            "lower_95":  ci.get("lower_95", float(pt)),
            "upper_95":  ci.get("upper_95", float(pt)),
            "std":       ci.get("std", 0.0),
        }

    summary = {
        "meta": {
            "n_prompts":        int(df["prompt_id"].nunique()),
            "n_validation":     int(val_count),
            "n_calibration":    int(cal_count),
            "n_models":         int(df["model"].nunique()),
            "n_records":        int(len(df)),
            "bt_source":       "all 50 prompts",
            "phas_source":      "validation 20 prompts only (calibration_30 used for weight fitting)",
            "bootstrap_n":      1000,
            "bootstrap_unit":   "prompt-level resamples",
        },
        "bt_ratings":             bt,
        "vlm_bt_with_ci":         vlm_bt_with_ci,
        "bt_bootstrap_details":   bt_ci,
        "phas_scores":             model_phas.to_dict(),
        "avg_scores":              model_stats.mean(axis=1).to_dict(),
        "variance_penalty":        penalty_table,
        "model_dimension_stats":   model_stats.to_dict(),
        "phas_weights_used":       PHAS_WEIGHTS,
        "video_level_phas":        phas_df.to_dict(orient="records"),
        "dimension_averages":      dim_avgs.to_dict(),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
