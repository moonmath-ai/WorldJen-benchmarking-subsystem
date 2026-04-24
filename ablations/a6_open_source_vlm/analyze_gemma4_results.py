"""
analyze_gemma4_results.py — Analyze WorldJen Gemma 4 evaluation results.

Reuses the same scoring logic as unified_analyzer.py (Gemini run) so outputs
are directly comparable. Produces:
  - results/summary_report_gemma4.json  (machine-readable, same schema)
  - results/results_summary_gemma4.md   (human-readable markdown)
  - results/plots/                       (per-dimension heatmap, BT rating CI, PHAS)

Usage:
    python analyze_gemma4_results.py
    python analyze_gemma4_results.py --results-dir /custom/path
"""

import os
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
RESULTS_DIR  = DATA_ROOT / "results/gemma4_vlm"
PROMPTS_FILE = DATA_ROOT / "prompts/prompts_50.jsonl"
OUTPUT_DIR   = RESULTS_DIR
PLOTS_DIR    = OUTPUT_DIR / "plots"

# ── PHAS weights (copied from unified_analyzer.py) ─────────────────────────────
PHAS_WEIGHTS = {
    # Non-negative ridge logistic regression on 2105 calibration annotations
    # (3448 total; 8 annotators). Source: calibrate_phas.py
    # CV accuracy: 65.8%, validation accuracy: 64.6%.
    "subject_consistency":  0.1025,
    "scene_consistency":    0.0000,
    "motion_smoothness":    0.0770,
    "temporal_flickering":  0.0398,
    "inertial_consistency": 0.0572,
    "physical_mechanics":   0.0638,
    "object_permanence":    0.0176,
    "human_fidelity":       0.0000,
    "dynamic_degree":       0.0000,
    "semantic_adherence":   0.2250,
    "spatial_relationship": 0.0000,
    "semantic_drift":       0.0914,
    "composition_framing":  0.1277,
    "lighting_volumetric":  0.0543,
    "color_harmony":        0.0828,
    "structural_gestalt":   0.0611,
}

DIMENSIONS = list(PHAS_WEIGHTS.keys())
SEED = 42
N_BOOT = 1000


# ── Bradley-Terry BT rating ─────────────────────────────────────────────────────────

def compute_bt_rating(scores_by_model_prompt: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Convert per-model per-prompt scores into BT rating (1000-centred).
    All prompt-level pairwise comparisons are derived from scores.
    """
    models = sorted(scores_by_model_prompt.keys())
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # Build wins matrix from score comparisons
    wins = np.zeros((n, n))
    prompts = set()
    for m in models:
        prompts |= set(scores_by_model_prompt[m].keys())

    for pid in prompts:
        scored = [(m, scores_by_model_prompt[m][pid])
                  for m in models if pid in scores_by_model_prompt[m]]
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                mi, si = scored[i]
                mj, sj = scored[j]
                if si > sj:
                    wins[idx[mi], idx[mj]] += 1
                elif sj > si:
                    wins[idx[mj], idx[mi]] += 1
                else:
                    wins[idx[mi], idx[mj]] += 0.5
                    wins[idx[mj], idx[mi]] += 0.5

    # MLE for BT parameters
    def neg_log_likelihood(params):
        nll = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and (wins[i, j] + wins[j, i]) > 0:
                    total = wins[i, j] + wins[j, i]
                    if wins[i, j] > 0:
                        nll -= wins[i, j] * (params[i] - np.logaddexp(params[i], params[j]))
                    if wins[j, i] > 0:
                        nll -= wins[j, i] * (params[j] - np.logaddexp(params[i], params[j]))
        return nll

    result = minimize(
        neg_log_likelihood,
        x0=np.zeros(n),
        method="L-BFGS-B",
        options={"maxiter": 2000}
    )
    params = result.x - result.x.mean()
    # Convert to BT rating scale (centred at 1500, 400/ln10 factor)
    bt = {models[i]: 1500 + 400 / np.log(10) * params[i] for i in range(n)}
    return bt


def bootstrap_bt_rating(scores_by_model_prompt: dict[str, dict[str, float]],
                  n_boot: int = N_BOOT) -> dict[str, tuple[float, float, float]]:
    """Bootstrap CIs on BT rating. Returns {model: (mean, lo95, hi95)}."""
    rng = np.random.default_rng(SEED)
    all_prompts = list({pid for m in scores_by_model_prompt for pid in scores_by_model_prompt[m]})
    boot_bts = {m: [] for m in scores_by_model_prompt}

    for _ in range(n_boot):
        sampled = rng.choice(all_prompts, size=len(all_prompts), replace=True)
        sub = {m: {pid: scores_by_model_prompt[m][pid]
                   for pid in sampled if pid in scores_by_model_prompt[m]}
               for m in scores_by_model_prompt}
        try:
            elos = compute_bt_rating(sub)
            for m, e in elos.items():
                boot_bts[m].append(e)
        except Exception:
            pass

    result = {}
    for m, vals in boot_bts.items():
        if vals:
            arr = np.array(vals)
            result[m] = (float(arr.mean()), float(np.percentile(arr, 2.5)),
                         float(np.percentile(arr, 97.5)))
        else:
            result[m] = (1500.0, 1500.0, 1500.0)
    return result


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    records = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if "results" in data and "model" in data:
                records.append(data)
        except Exception as e:
            print(f"  WARNING: could not load {f.name}: {e}")
    return records


def load_prompt_sets() -> dict[str, str]:
    mapping = {}
    with open(PROMPTS_FILE) as f:
        for line in f:
            e = json.loads(line)
            mapping[e["prompt_id"]] = e.get("prompt_set", "unknown")
    return mapping


def aggregate_dim_score(dim_result: list | None) -> float | None:
    """Average question scores within a dimension."""
    if not dim_result:
        return None
    scores = [q["score"] for q in dim_result if isinstance(q.get("score"), (int, float))]
    return float(np.mean(scores)) if scores else None


# ── Report generation ──────────────────────────────────────────────────────────

def compute_summary(records: list[dict], prompt_sets: dict[str, str]) -> dict:
    models = sorted(set(r["model"] for r in records))

    # Per-model, per-prompt, per-dimension scores
    scores = {m: {} for m in models}
    dim_scores = {m: {d: [] for d in DIMENSIONS} for m in models}

    for rec in records:
        m = rec["model"]
        pid = rec["prompt_id"]
        if m not in scores:
            continue
        prompt_dim_scores = {}
        for dim in DIMENSIONS:
            s = aggregate_dim_score(rec["results"].get(dim))
            if s is not None:
                prompt_dim_scores[dim] = s
                dim_scores[m][dim].append(s)
        if prompt_dim_scores:
            scores[m][pid] = np.mean(list(prompt_dim_scores.values()))

    # Per-model mean per dimension
    mean_dim = {m: {d: float(np.mean(v)) if v else None
                    for d, v in dim_scores[m].items()}
                for m in models}

    # PHAS score (weighted combination)
    phas_scores = {}
    for m in models:
        w_sum = num = 0.0
        for d, w in PHAS_WEIGHTS.items():
            v = mean_dim[m].get(d)
            if v is not None:
                w_sum += w * v
                num += w
        phas_scores[m] = float(w_sum / num) if num > 0 else None

    # BT rating: deterministic MLE as point estimate (matching unified_analyzer.py);
    # bootstrap provides 95% CIs only.
    bt_rating = compute_bt_rating(scores)
    bt_rating_ci = bootstrap_bt_rating(scores)
    ranked = sorted(models, key=lambda m: bt_rating[m], reverse=True)

    return {
        "models": models,
        "ranked": ranked,
        "bt_rating": bt_rating,
        "bt_rating_ci": {m: {"point_estimate": bt_rating[m],
                           "boot_mean": bt_rating_ci[m][0],
                           "lo95": bt_rating_ci[m][1],
                           "hi95": bt_rating_ci[m][2]} for m in models},
        "phas_scores": phas_scores,
        "mean_dim": mean_dim,
        "scores_by_model_prompt": scores,
        "n_records": len(records),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

MODEL_LABELS = {
    "fal-ai_veo3.1_fast":                         "Veo 3.1",
    "fal-ai_kling-video_v2.6_pro_text-to-video":  "Kling 2.6",
    "fal-ai_wan_v2.2-a14b_text-to-video":         "Wan v2.2",
    "fal-ai_ltx-2_text-to-video":                 "LTX-2",
    "fal-ai_hunyuan-video-v1.5_text-to-video":    "HunyuanVideo",
    "wan2.1-1.3b":                                "Wan 2.1",
}

DIM_LABELS = {
    "subject_consistency":  "Subject Consistency",
    "scene_consistency":    "Scene Consistency",
    "motion_smoothness":    "Motion Smoothness",
    "temporal_flickering":  "Temporal Flickering",
    "inertial_consistency": "Inertial Consistency",
    "physical_mechanics":   "Physical Mechanics",
    "object_permanence":    "Object Permanence",
    "human_fidelity":       "Human Fidelity",
    "dynamic_degree":       "Dynamic Degree",
    "semantic_adherence":   "Semantic Adherence",
    "spatial_relationship": "Spatial Relationship",
    "semantic_drift":       "Semantic Drift",
    "composition_framing":  "Composition & Framing",
    "lighting_volumetric":  "Lighting & Volumetric",
    "color_harmony":        "Color Harmony",
    "structural_gestalt":   "Structural Gestalt",
}


def plot_bt(summary: dict, out: Path):
    models  = summary["ranked"]
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    means   = [summary["bt_rating_ci"][m]["point_estimate"] for m in models]
    lo      = [summary["bt_rating_ci"][m]["lo95"] for m in models]
    hi      = [summary["bt_rating_ci"][m]["hi95"] for m in models]
    errs    = [[m - l for m, l in zip(means, lo)],
               [h - m for m, h in zip(means, hi)]]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(models)))
    y = np.arange(len(models))
    ax.barh(y, means, xerr=errs, color=colors, height=0.6,
            error_kw=dict(ecolor="black", capsize=4, lw=1.5))
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{i+1} {l}" for i, l in enumerate(labels)], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Gemma 4 BT rating", fontsize=11)
    ax.set_title("WorldJen — Gemma 4 31B IT BT rating Rankings\n(1000 bootstrap resamples, 95% CI)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved BT rating plot → {out}")


def plot_heatmap(summary: dict, out: Path):
    models = summary["ranked"]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    dim_labels = [DIM_LABELS.get(d, d) for d in DIMENSIONS]

    data = np.full((len(DIMENSIONS), len(models)), np.nan)
    for j, m in enumerate(models):
        for i, d in enumerate(DIMENSIONS):
            v = summary["mean_dim"][m].get(d)
            if v is not None:
                data[i, j] = v

    fig, ax = plt.subplots(figsize=(max(8, len(models)*1.5), max(7, len(DIMENSIONS)*0.5)))
    sns.heatmap(
        data, ax=ax,
        xticklabels=labels,
        yticklabels=dim_labels,
        annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5, linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Mean Score (1–5)"},
    )
    ax.set_title("Gemma 4 VLM Evaluation — Per-Dimension Scores", fontsize=13, pad=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap → {out}")


# ── Markdown report ────────────────────────────────────────────────────────────

def write_markdown(summary: dict, out: Path):
    lines = [
        "# WorldJen — Gemma 4 31B IT Evaluation Report",
        "",
        f"**Records loaded**: {summary['n_records']}",
        "",
        "## BT rating Rankings",
        "",
        "| Rank | Model | BT rating | 95% CI Lo | 95% CI Hi | PHAS |",
        "|------|-------|-----|-----------|-----------|------|",
    ]
    for rank, m in enumerate(summary["ranked"], 1):
        label = MODEL_LABELS.get(m, m)
        bt   = summary["bt_rating_ci"][m]["point_estimate"]
        lo    = summary["bt_rating_ci"][m]["lo95"]
        hi    = summary["bt_rating_ci"][m]["hi95"]
        phas  = summary["phas_scores"].get(m)
        phas_str = f"{phas:.3f}" if phas is not None else "N/A"
        lines.append(f"| {rank} | {label} | {bt:.1f} | {lo:.1f} | {hi:.1f} | {phas_str} |")

    lines += ["", "## Per-Dimension Mean Scores", ""]
    header = "| Dimension |" + "".join(f" {MODEL_LABELS.get(m, m)} |" for m in summary["ranked"])
    lines.append(header)
    lines.append("|" + "-----------|" * (len(summary["ranked"]) + 1))
    for d in DIMENSIONS:
        row = f"| {DIM_LABELS.get(d, d)} |"
        for m in summary["ranked"]:
            v = summary["mean_dim"][m].get(d)
            row += f" {v:.2f} |" if v is not None else " N/A |"
        lines.append(row)

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved markdown → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    PLOTS_DIR_OUT = results_dir / "plots"
    PLOTS_DIR_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    records = load_results(results_dir)
    if not records:
        print("No result JSON files found. Run the evaluator first.")
        sys.exit(1)
    print(f"  {len(records)} records from "
          f"{len(set(r['model'] for r in records))} models × "
          f"{len(set(r['prompt_id'] for r in records))} prompts")

    prompt_sets = load_prompt_sets()

    print("Computing summary statistics ...")
    summary = compute_summary(records, prompt_sets)

    print("\nELO Rankings:")
    for rank, m in enumerate(summary["ranked"], 1):
        ci = summary["bt_rating_ci"][m]
        print(f"  #{rank} {MODEL_LABELS.get(m, m):20s}  BT rating={ci['point_estimate']:.1f}  "
              f"[{ci['lo95']:.1f}, {ci['hi95']:.1f}]")

    print("\nGenerating plots ...")
    plot_bt(summary, PLOTS_DIR_OUT / "gemma4_bt_rating.png")
    plot_heatmap(summary, PLOTS_DIR_OUT / "gemma4_heatmap.png")

    # Save JSON
    json_out = results_dir / "summary_report_gemma4.json"
    # Remove non-serialisable numpy floats
    def clean(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        return obj

    with open(json_out, "w") as f:
        json.dump(clean(summary), f, indent=2)
    print(f"  Saved JSON → {json_out}")

    write_markdown(summary, results_dir / "results_summary_gemma4.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
