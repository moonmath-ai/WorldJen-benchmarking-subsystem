"""
phas_annotator_stability.py
────────────────────────────
Leave-one-annotator-out (LOAO) stability analysis for PHAS dimension weights.

Addresses reviewer concern W3:
  "The paper should investigate whether the 5 zero-weight dimensions remain zero
   under bootstrap resampling of annotators (leave-one-annotator-out). If the
   zero-weight set is unstable across annotator subsets, the current weights
   are unreliable."

Method:
  1. Fit PHAS weights on ALL calibration annotations (baseline).
  2. For each annotator, remove their calibration rows and refit.
  3. Report: which dimensions are zero in each run, Spearman ρ vs full-data
     weights, validation accuracy per run.
  4. Save a heatmap figure: phas_loao_stability.png

Usage:
    cd /path/to/WorldJen-benchmarking-subsystem/human_eval
    python3 phas_annotator_stability.py
    python3 phas_annotator_stability.py --csv anonymized_human_evals.csv
"""

import os
import argparse
import csv
import glob
import json
import warnings
from collections import defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
OUTPNG    = HERE / "phas_loao_stability.png"
OUTJSON   = HERE / "phas_loao_stability.json"
RES_DIRS  = [str(DATA_ROOT / "results/gemini_vlm")]
EXCLUDE_IDS = {"A1-retest"}
ZERO_THRESH = 1e-4   # treat weight ≤ this as "zero"

DIMS = [
    "subject_consistency", "scene_consistency", "motion_smoothness",
    "temporal_flickering", "inertial_consistency", "physical_mechanics",
    "object_permanence", "human_fidelity", "dynamic_degree",
    "semantic_adherence", "spatial_relationship", "semantic_drift",
    "composition_framing", "lighting_volumetric", "color_harmony",
    "structural_gestalt",
]
PRETTY = {
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

# ── Score loader ───────────────────────────────────────────────────────────────
_score_cache: dict = {}

def get_scores(model: str, pid: str) -> dict:
    key = (model, pid)
    if key in _score_cache:
        return _score_cache[key]
    for d in RES_DIRS:
        fps = glob.glob(f"{d}/{model}_{pid}.json")
        if fps:
            with open(fps[0]) as f:
                data = json.load(f)
            out = {}
            for dim in DIMS:
                qs = data["results"].get(dim, [])
                if isinstance(qs, list) and qs:
                    scores = [float(q["score"]) for q in qs if "score" in q]
                    if scores:
                        out[dim] = float(np.mean(scores))
            _score_cache[key] = out
            return out
    _score_cache[key] = {}
    return {}


def build_matrix(subset: list):
    """Return (X, y, w) arrays for logistic regression."""
    X, y, w = [], [], []
    for r in subset:
        pid, ma, mb = r["Prompt ID"], r["Model A"], r["Model B"]
        sa = get_scores(ma, pid)
        sb = get_scores(mb, pid)
        X.append([sa.get(d, 0.0) - sb.get(d, 0.0) for d in DIMS])
        y.append(1 if r["Winner"] == ma else 0)
        w.append(float(r["Weight"]))
    return np.array(X), np.array(y), np.array(w)


# ── Non-negative ridge logistic regression ────────────────────────────────────
def _nll_ridge(beta, X, y, w, C):
    logits = np.clip(X @ beta, -30, 30)
    prob   = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)
    ll     = -np.sum(w * (y * np.log(prob) + (1 - y) * np.log(1 - prob)))
    return ll + (1 / (2 * C)) * np.sum(beta ** 2)


def _grad_nll_ridge(beta, X, y, w, C):
    logits = np.clip(X @ beta, -30, 30)
    prob   = 1 / (1 + np.exp(-logits))
    return X.T @ (w * (prob - y)) + beta / C


def nonneg_ridge_cv(X, y, w, Cs=(0.001, 0.01, 0.1, 1, 10), n_splits=5):
    """5-fold CV over C; refit on full data. Returns (best_C, weights, cv_acc)."""
    n   = len(y)
    rng = np.random.default_rng(42)
    fold_idx = np.array_split(rng.permutation(n), n_splits)
    bounds   = [(0, None)] * len(DIMS)

    best_C, best_acc = Cs[0], -1.0
    for C in Cs:
        accs = []
        for fi in range(n_splits):
            val_i   = fold_idx[fi]
            train_i = np.concatenate([fold_idx[j] for j in range(n_splits) if j != fi])
            res = minimize(_nll_ridge, np.ones(len(DIMS)) * 0.1,
                           args=(X[train_i], y[train_i], w[train_i], C),
                           jac=_grad_nll_ridge, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10})
            preds = (1 / (1 + np.exp(-X[val_i] @ res.x)) >= 0.5).astype(int)
            accs.append(float(np.mean(preds == y[val_i])))
        if np.mean(accs) > best_acc:
            best_acc, best_C = float(np.mean(accs)), C

    res = minimize(_nll_ridge, np.ones(len(DIMS)) * 0.1,
                   args=(X, y, w, best_C),
                   jac=_grad_nll_ridge, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10})
    raw = res.x
    norm = raw / raw.sum() if raw.sum() > 0 else raw
    return best_C, norm, best_acc


def val_accuracy(beta, Xv, yv, wv):
    preds = (1 / (1 + np.exp(-Xv @ beta)) >= 0.5).astype(int)
    return float(np.mean(preds == yv))


# ── Load CSV ───────────────────────────────────────────────────────────────────
def load_csv(path: Path, exclude: set) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = r.get("User", r.get("user", "")).strip()
            if uid in exclude or uid.lower() in {e.lower() for e in exclude}:
                continue
            rows.append(r)
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=HERE / "anonymized_human_evals.csv")
    args = parser.parse_args()

    rows = load_csv(args.csv, EXCLUDE_IDS)
    cal_rows = [r for r in rows if r["Source"] == "calibration"]
    val_rows = [r for r in rows if r["Source"] == "validation"]

    print(f"Loaded {len(rows)} rows  |  cal={len(cal_rows)}  val={len(val_rows)}")
    print("Pre-loading VLM score cache …")
    Xc, yc, wc = build_matrix(cal_rows)
    Xv, yv, wv = build_matrix(val_rows)
    print(f"  Calibration matrix: {Xc.shape}  |  Validation matrix: {Xv.shape}")

    # ── Baseline: all annotators ──────────────────────────────────────────────
    print("\n[Baseline] Fitting on ALL calibration annotations …")
    base_C, base_w, base_cv = nonneg_ridge_cv(Xc, yc, wc)
    base_val = val_accuracy(base_w, Xv, yv, wv)
    base_zero = {DIMS[i] for i, w in enumerate(base_w) if w <= ZERO_THRESH}

    print(f"  C={base_C}  CV acc={base_cv:.3f}  Val acc={base_val:.3f}")
    print(f"  Zero-weight dims ({len(base_zero)}): {', '.join(sorted(base_zero))}")

    # ── Leave-one-annotator-out ───────────────────────────────────────────────
    annotators = sorted({r["User"] for r in cal_rows})
    print(f"\nRunning LOAO over {len(annotators)} annotators: {annotators}")

    results = {}
    for ann in annotators:
        sub_cal  = [r for r in cal_rows if r["User"] != ann]
        n_dropped = len(cal_rows) - len(sub_cal)
        print(f"\n  LOO drop {ann} ({n_dropped} cal rows dropped, {len(sub_cal)} remain)")

        if len(sub_cal) < 50:
            print(f"    SKIP: too few rows after dropping {ann}")
            continue

        Xs, ys, ws = build_matrix(sub_cal)
        C_s, w_s, cv_s = nonneg_ridge_cv(Xs, ys, ws)
        val_s = val_accuracy(w_s, Xv, yv, wv)
        zero_s = {DIMS[i] for i, wi in enumerate(w_s) if wi <= ZERO_THRESH}
        rho, pval = spearmanr(base_w, w_s)

        n_entered = len(zero_s - base_zero)   # newly zero in LOO
        n_exited  = len(base_zero - zero_s)    # zero in baseline but non-zero in LOO

        print(f"    C={C_s}  CV={cv_s:.3f}  Val={val_s:.3f}  "
              f"ρ(base,LOO)={rho:.3f}(p={pval:.3f})  "
              f"zero_dims={len(zero_s)}  entered={n_entered}  exited={n_exited}")
        if n_entered:
            print(f"    ⚠ newly zero in this LOO run: {zero_s - base_zero}")
        if n_exited:
            print(f"    ⚠ recovered non-zero: {base_zero - zero_s}")

        results[ann] = {
            "n_cal":        len(sub_cal),
            "n_dropped":    n_dropped,
            "best_C":       C_s,
            "cv_acc":       round(cv_s, 4),
            "val_acc":      round(val_s, 4),
            "spearman_rho": round(rho, 4),
            "spearman_p":   round(pval, 4),
            "n_zero":       len(zero_s),
            "zero_dims":    sorted(zero_s),
            "weights":      {d: round(float(w_s[i]), 5) for i, d in enumerate(DIMS)},
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  LOAO STABILITY SUMMARY")
    print("═" * 80)
    header = f"  {'Ann':<12} {'N_cal':>6}  {'Val%':>6}  {'ρ(base)':>8}  {'#zero':>5}  {'Δzero':>6}"
    print(header)
    print("  " + "─" * 70)
    for ann, r in results.items():
        delta = r["n_zero"] - len(base_zero)
        sign  = "+" if delta > 0 else ""
        flag  = "  ← unstable" if abs(delta) > 0 else ""
        print(f"  {ann:<12} {r['n_cal']:>6}  {r['val_acc']*100:>5.1f}%  "
              f"{r['spearman_rho']:>8.3f}  {r['n_zero']:>5}  {sign}{delta:>5}{flag}")
    print()
    print(f"  BASELINE (all annotators): Val={base_val*100:.1f}%  #zero={len(base_zero)}")
    print(f"  Stable zero dims (zero in ALL runs): ", end="")
    always_zero = base_zero.copy()
    for r in results.values():
        always_zero &= set(r["zero_dims"])
    print(", ".join(sorted(always_zero)) if always_zero else "(none)")
    print(f"  Unstable dims (zero in SOME but not ALL runs): ", end="")
    sometimes_zero = set()
    for r in results.values():
        sometimes_zero |= (set(r["zero_dims"]) ^ base_zero)
    print(", ".join(sorted(sometimes_zero)) if sometimes_zero else "(none)")

    # ── Figure: heatmap of weights per run ────────────────────────────────────
    run_labels  = ["All"] + [f"−{a}" for a in results.keys()]
    weight_mat  = np.vstack([base_w] + [
        np.array([results[a]["weights"][d] for d in DIMS])
        for a in results.keys()
    ])
    zero_mat    = weight_mat <= ZERO_THRESH

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [3, 1]})

    # Left: weight heatmap
    ax = axes[0]
    im = ax.imshow(weight_mat.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=weight_mat.max())
    # Overlay zero cells
    for r_idx in range(weight_mat.shape[0]):
        for d_idx in range(len(DIMS)):
            if zero_mat[r_idx, d_idx]:
                ax.add_patch(mpatches.Rectangle(
                    (r_idx - 0.5, d_idx - 0.5), 1, 1,
                    linewidth=0, facecolor="#2196F3", alpha=0.55))
    ax.set_xticks(range(len(run_labels)))
    ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(DIMS)))
    ax.set_yticklabels([PRETTY[d] for d in DIMS], fontsize=9)
    ax.set_title("PHAS dimension weights across annotator subsets\n"
                 "(blue = zero weight ≤1e-4)", fontsize=10)
    plt.colorbar(im, ax=ax, label="Normalised weight")

    # Right: zero-count bar
    ax2 = axes[1]
    zero_counts = zero_mat.sum(axis=0)  # per dimension: how many runs is it zero
    n_runs = weight_mat.shape[0]
    colors  = ["#2196F3" if c == n_runs else
               ("#FF9800" if c >= n_runs // 2 else "#4CAF50")
               for c in zero_counts]
    ax2.barh(range(len(DIMS)), zero_counts, color=colors, edgecolor="white")
    ax2.set_xlim(0, n_runs)
    ax2.set_xticks(range(n_runs + 1))
    ax2.set_yticks(range(len(DIMS)))
    ax2.set_yticklabels([PRETTY[d] for d in DIMS], fontsize=9)
    ax2.set_xlabel("# runs with zero weight", fontsize=9)
    ax2.set_title(f"Zero-weight frequency\n(out of {n_runs} runs)", fontsize=10)
    ax2.axvline(n_runs, color="gray", linestyle="--", alpha=0.5)
    blue_patch  = mpatches.Patch(color="#2196F3", label="Zero in ALL runs (stable)")
    orange_patch= mpatches.Patch(color="#FF9800", label=f"Zero in ≥{n_runs//2} runs")
    green_patch = mpatches.Patch(color="#4CAF50", label="Zero in <50% runs")
    ax2.legend(handles=[blue_patch, orange_patch, green_patch],
               loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPNG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {OUTPNG}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "baseline": {
            "n_cal":     len(cal_rows),
            "n_val":     len(val_rows),
            "best_C":    base_C,
            "cv_acc":    round(base_cv, 4),
            "val_acc":   round(base_val, 4),
            "n_zero":    len(base_zero),
            "zero_dims": sorted(base_zero),
            "weights":   {d: round(float(base_w[i]), 5) for i, d in enumerate(DIMS)},
        },
        "loao": results,
        "stability": {
            "always_zero":    sorted(always_zero),
            "sometimes_zero": sorted(sometimes_zero),
            "n_runs":         len(run_labels),
        },
    }
    with open(OUTJSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {OUTJSON}")


if __name__ == "__main__":
    main()
