"""
calibrate_phas.py
─────────────────
Runs three PHAS weight calibration methods on the human evaluation CSV
and compares them against hand-tuned weights:

  1. Unconstrained ridge logistic regression (sklearn, 5-fold CV)
  2. Non-negative constrained ridge logistic regression (scipy.optimize)
  3. Hand-tuned baseline

Outputs:
  - human_eval_analysis.json  (full structured results, updated in-place)
  - phas_weights_comparison.png  (bar chart of all four weight sets)

Usage:
  cd /path/to/WorldJen-benchmarking-subsystem/human_eval

  # Anonymized dataset (default — for public release)
  python3 calibrate_phas.py

  # Original dataset (private)
  python3 calibrate_phas.py --csv "human evals - Human Evaluations.csv"
"""

import os
import argparse
import csv, json, glob, warnings, datetime
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Calibrate PHAS weights from human eval CSV.")
_parser.add_argument("--csv", type=Path,
                     default=DATA_ROOT / "human_eval/anonymized_human_evals.csv",
                     help="Path to CSV (anonymized or original). "
                          "Default: data/human_eval/anonymized_human_evals.csv")
_parser.add_argument("--output", type=Path, default=None,
                     help="Override output JSON path (default: data/human_eval/human_eval_analysis.json)")
_args = _parser.parse_args()
CSV    = _args.csv

# Exclude the self-consistency retest pass (both anonymized and original forms)
EXCLUDE_IDS = {"A1-retest"}
OUTJSON= _args.output if _args.output else DATA_ROOT / "human_eval/human_eval_analysis.json"
OUTPNG = (OUTJSON.parent / "phas_weights_comparison.png") if _args.output else DATA_ROOT / "human_eval/phas_weights_comparison.png"
OUTJSON.parent.mkdir(parents=True, exist_ok=True)
RES_DIRS = [
    # path configured via DATA_ROOT
]
PROMPTS_FILE = str(DATA_ROOT / "prompts/prompts_50.jsonl")

# ── Dimension definitions ─────────────────────────────────────────────────────
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
HAND = {
    "subject_consistency": 0.07, "scene_consistency": 0.06,
    "motion_smoothness": 0.05,   "temporal_flickering": 0.04,
    "inertial_consistency": 0.09,"physical_mechanics": 0.09,
    "object_permanence": 0.09,   "human_fidelity": 0.07,
    "dynamic_degree": 0.04,      "semantic_adherence": 0.09,
    "spatial_relationship": 0.06,"semantic_drift": 0.06,
    "composition_framing": 0.05, "lighting_volumetric": 0.05,
    "color_harmony": 0.04,       "structural_gestalt": 0.05,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_scores(model, pid):
    """Return {dim: mean_score} for a (model, prompt_id) pair."""
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
            return out
    return {}


def build_matrix(subset):
    """Build (X, y, w) from a list of annotation rows."""
    X, y, w = [], [], []
    for r in subset:
        pid, ma, mb = r["Prompt ID"], r["Model A"], r["Model B"]
        sa = get_scores(ma, pid)
        sb = get_scores(mb, pid)
        X.append([sa.get(d, 0.0) - sb.get(d, 0.0) for d in DIMS])
        y.append(1 if r["Winner"] == ma else 0)
        w.append(float(r["Weight"]))
    return np.array(X), np.array(y), np.array(w)


def neg_log_likelihood_ridge(beta, X, y, w, C):
    """Weighted binary cross-entropy + L2 penalty (for scipy minimise)."""
    logits = X @ beta
    # clip for stability
    logits = np.clip(logits, -30, 30)
    prob   = 1 / (1 + np.exp(-logits))
    prob   = np.clip(prob, 1e-9, 1 - 1e-9)
    ll     = -np.sum(w * (y * np.log(prob) + (1 - y) * np.log(1 - prob)))
    reg    = (1 / (2 * C)) * np.sum(beta ** 2)
    return ll + reg


def grad_neg_ll_ridge(beta, X, y, w, C):
    logits = np.clip(X @ beta, -30, 30)
    prob   = 1 / (1 + np.exp(-logits))
    residual = w * (prob - y)
    return X.T @ residual + beta / C


def nonneg_elasticnet_cv(X, y, w, Cs, l1_ratios, n_splits=5):
    """
    Cross-validate C and l1_ratio for non-negative constrained elastic net
    logistic regression.

    With beta >= 0, the L1 term reduces to l1_ratio/C * sum(beta), which is
    fully differentiable, so L-BFGS-B applies directly.

    Objective:
        NLL(beta) + (l1_ratio/C) * sum(beta) + (1-l1_ratio)/(2C) * ||beta||^2
    """
    n = len(y)
    rng = np.random.default_rng(42)
    fold_idx = np.array_split(rng.permutation(n), n_splits)

    def obj(beta, X, y, w, C, l1_ratio):
        logits = np.clip(X @ beta, -30, 30)
        prob   = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)
        ll     = -np.sum(w * (y * np.log(prob) + (1 - y) * np.log(1 - prob)))
        reg    = (l1_ratio / C) * beta.sum() + (1 - l1_ratio) / (2 * C) * (beta ** 2).sum()
        return ll + reg

    def grad(beta, X, y, w, C, l1_ratio):
        logits   = np.clip(X @ beta, -30, 30)
        prob     = 1 / (1 + np.exp(-logits))
        residual = w * (prob - y)
        l1_grad  = np.full(len(beta), l1_ratio / C)
        l2_grad  = (1 - l1_ratio) / C * beta
        return X.T @ residual + l1_grad + l2_grad

    best_C, best_l1, best_acc = None, None, -1.0
    for C in Cs:
        for l1_ratio in l1_ratios:
            fold_accs = []
            for fi in range(n_splits):
                val_idx   = fold_idx[fi]
                train_idx = np.concatenate([fold_idx[j] for j in range(n_splits) if j != fi])
                Xtr, ytr, wtr = X[train_idx], y[train_idx], w[train_idx]
                Xva, yva      = X[val_idx],   y[val_idx]
                res = minimize(obj, x0=np.ones(len(DIMS)) * 0.1,
                               args=(Xtr, ytr, wtr, C, l1_ratio),
                               jac=grad, method="L-BFGS-B",
                               bounds=[(0, None)] * len(DIMS),
                               options={"maxiter": 2000, "ftol": 1e-10})
                preds = (1 / (1 + np.exp(-Xva @ res.x)) >= 0.5).astype(int)
                fold_accs.append(float(np.mean(preds == yva)))
            mean_acc = float(np.mean(fold_accs))
            if mean_acc > best_acc:
                best_acc, best_C, best_l1 = mean_acc, C, l1_ratio

    # Refit on full calibration data with best hyperparameters
    res = minimize(obj, x0=np.ones(len(DIMS)) * 0.1,
                   args=(X, y, w, best_C, best_l1),
                   jac=grad, method="L-BFGS-B",
                   bounds=[(0, None)] * len(DIMS),
                   options={"maxiter": 2000, "ftol": 1e-10})
    return best_C, best_l1, res.x, best_acc


def nonneg_ridge_cv(X, y, w, Cs, n_splits=5):
    """
    Cross-validate C for non-negative constrained ridge logistic regression.
    Returns (best_C, fitted_beta).
    """
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    fold_idx = np.array_split(rng.permutation(n), n_splits)

    best_C, best_acc = None, -1.0
    for C in Cs:
        fold_accs = []
        for fi in range(n_splits):
            val_idx   = fold_idx[fi]
            train_idx = np.concatenate([fold_idx[j] for j in range(n_splits) if j != fi])
            Xtr, ytr, wtr = X[train_idx], y[train_idx], w[train_idx]
            Xva, yva, wva = X[val_idx],   y[val_idx],   w[val_idx]

            bounds = [(0, None)] * len(DIMS)   # non-negativity constraint
            res = minimize(
                neg_log_likelihood_ridge, x0=np.ones(len(DIMS)) * 0.1,
                args=(Xtr, ytr, wtr, C),
                jac=grad_neg_ll_ridge, method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10},
            )
            beta = res.x
            preds = (1 / (1 + np.exp(-Xva @ beta)) >= 0.5).astype(int)
            fold_accs.append(float(np.mean(preds == yva)))

        mean_acc = float(np.mean(fold_accs))
        if mean_acc > best_acc:
            best_acc, best_C = mean_acc, C

    # Refit on full data with best C
    bounds = [(0, None)] * len(DIMS)
    res = minimize(
        neg_log_likelihood_ridge, x0=np.ones(len(DIMS)) * 0.1,
        args=(X, y, w, best_C),
        jac=grad_neg_ll_ridge, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10},
    )
    return best_C, res.x, best_acc


def bradley_terry(wins, n_iter=1000):
    ms = list({m for pair in wins for m in pair})
    s  = {m: 1.0 for m in ms}
    for _ in range(n_iter):
        ns = {}
        for m in ms:
            num  = sum(wins.get((m, o), 0) for o in ms if o != m)
            denom= sum(
                (wins.get((m,o),0) + wins.get((o,m),0)) / (s[m] + s[o])
                for o in ms if o != m and s[m] + s[o] > 0
            )
            ns[m] = num / denom if denom > 0 else s[m]
        tot = sum(ns.values())
        s   = {m: v / tot * len(ms) for m, v in ns.items()}
    return s


def short(m):
    return m.replace("fal-ai_", "").replace("_text-to-video", "")


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading annotations from {CSV.name} …")
rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        uid = r.get("User", "").strip()
        if uid in EXCLUDE_IDS or uid.lower() in {e.lower() for e in EXCLUDE_IDS}:
            continue
        rows.append(r)
print(f"  Loaded {len(rows)} rows (excluded: {EXCLUDE_IDS})")

cal_rows = [r for r in rows if r["Source"] == "calibration"]
val_rows = [r for r in rows if r["Source"] == "validation"]
print(f"  Calibration: {len(cal_rows)}   Validation: {len(val_rows)}")

Xc, yc, wc = build_matrix(cal_rows)
Xv, yv, wv = build_matrix(val_rows)

# ── Human Bradley-Terry ───────────────────────────────────────────────────────
print("Computing Human Bradley-Terry …")
wins = defaultdict(float)
for r in rows:
    wins[(r["Winner"], r["Loser"])] += 1.0  # unweighted: direction only
bt_s = bradley_terry(wins)
human_rank = {m: rk for rk, (m, _) in enumerate(
    sorted(bt_s.items(), key=lambda x: -x[1]), 1)}

# Bootstrap CIs
rng_boot = np.random.default_rng(42)
bt_boot  = defaultdict(list)
for _ in range(1000):
    sample = [rows[i] for i in rng_boot.integers(0, len(rows), len(rows))]
    bw = defaultdict(float)
    for r in sample:
        bw[(r["Winner"], r["Loser"])] += 1.0  # unweighted: direction only
    try:
        bs = bradley_terry(bw, n_iter=200)
        for m, v in bs.items():
            bt_boot[m].append(v)
    except Exception:
        pass
bt_ci = {m: (float(np.percentile(bt_boot[m], 2.5)),
             float(np.percentile(bt_boot[m], 97.5)))
         for m in bt_s if bt_boot[m]}

# ── VLM BT rating comparison ────────────────────────────────────────────────────────
with open(DATA_ROOT / "results/summaries/summary_report_unified.json") as f:
    rep = json.load(f)
vlm_bt  = rep["bt_ratings"]
vlm_rank = {m: rk for rk, (m, _) in enumerate(
    sorted(vlm_bt.items(), key=lambda x: -x[1]), 1)}
MODELS = list(vlm_bt.keys())
rho_vlm, p_vlm = spearmanr(
    [vlm_rank[m] for m in MODELS],
    [human_rank.get(m, 6) for m in MODELS])

# ── Method 1: Unconstrained ridge (sklearn) ───────────────────────────────────
print("Method 1: Unconstrained ridge (sklearn) …")
clf = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1, 10, 100], cv=5, penalty="l2",
    solver="lbfgs", max_iter=2000, random_state=42,
)
clf.fit(Xc, yc, sample_weight=wc)
cv_acc_unc = cross_val_score(clf, Xc, yc, cv=5, scoring="accuracy")
val_acc_unc = clf.score(Xv, yv, sample_weight=wv)
raw_unc = clf.coef_[0]
pos_unc = np.maximum(raw_unc, 0)
w_unc   = pos_unc / pos_unc.sum() if pos_unc.sum() > 0 else pos_unc

# ── Method 2: Non-negative constrained ridge (scipy) ─────────────────────────
print("Method 2: Non-negative constrained ridge (scipy) …")
best_C_nn, beta_nn, cv_acc_nn_mean = nonneg_ridge_cv(
    Xc, yc, wc, Cs=[0.001, 0.01, 0.1, 1, 10, 100])
# Validation accuracy
preds_v = (1 / (1 + np.exp(-Xv @ beta_nn)) >= 0.5).astype(int)
val_acc_nn = float(np.mean(preds_v == yv))
# Normalise to sum=1
w_nn = beta_nn / beta_nn.sum() if beta_nn.sum() > 0 else beta_nn

# ── Method 3: Non-negative Elastic Net (scipy) ───────────────────────────────
print("Method 3: Non-negative elastic net (scipy) …")
best_C_en, best_l1_en, beta_en, cv_acc_en_mean = nonneg_elasticnet_cv(
    Xc, yc, wc,
    Cs=[0.001, 0.01, 0.1, 1, 10, 100],
    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9])
preds_v_en = (1 / (1 + np.exp(-Xv @ beta_en)) >= 0.5).astype(int)
val_acc_en = float(np.mean(preds_v_en == yv))
w_en = beta_en / beta_en.sum() if beta_en.sum() > 0 else beta_en

# ── Inter-annotator ───────────────────────────────────────────────────────────
pair_votes = defaultdict(list)
for r in rows:
    key = (r["Prompt ID"], tuple(sorted([r["Model A"], r["Model B"]])))
    pair_votes[key].append((r["User"], r["Winner"]))
shared = {k: v for k, v in pair_votes.items()
          if len({u for u, _ in v}) >= 2}
agree  = sum(1 for v in shared.values() if len({w for _, w in v}) == 1)

# ── Print comparison table ────────────────────────────────────────────────────
print(f"\n{'Dimension':<26} {'Unconstrained':>14} {'Non-neg ridge':>14} {'Elastic Net':>12} {'Hand-tuned':>11}")
print("─" * 82)
for i, dim in enumerate(DIMS):
    flag = " ◀ was neg" if raw_unc[i] < 0 else ""
    print(f"  {PRETTY[dim]:<24} {w_unc[i]:>14.4f} {w_nn[i]:>14.4f} {w_en[i]:>12.4f} {HAND[dim]:>11.4f}{flag}")
print("─" * 82)
print(f"  {'SUM':<24} {w_unc.sum():>14.4f} {w_nn.sum():>14.4f} {w_en.sum():>12.4f} {sum(HAND.values()):>11.4f}")
print(f"\nUnconstrained  — CV acc: {cv_acc_unc.mean():.3f} ± {cv_acc_unc.std():.3f}  | val acc: {val_acc_unc:.3f}")
print(f"Non-neg ridge  — CV acc: {cv_acc_nn_mean:.3f}                | val acc: {val_acc_nn:.3f}")
print(f"Elastic Net    — CV acc: {cv_acc_en_mean:.3f}  (C={best_C_en}, l1_ratio={best_l1_en}) | val acc: {val_acc_en:.3f}")
print(f"\nZero weights — ridge: {sum(beta_nn<=1e-6)}/16  elastic net: {sum(beta_en<=1e-6)}/16")
print(f"Negative raw coefs (unconstrained): {[PRETTY[DIMS[i]] for i in range(len(DIMS)) if raw_unc[i] < 0]}")

# ── Save figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(DIMS))
bw = 0.20
labels = [PRETTY[d].replace(" & ", "\n& ").replace(" and ", "\n& ") for d in DIMS]
ax.bar(x - 1.5*bw, w_unc,                   bw, label="Unconstrained ridge (clipped)",   color="#38bdf8", alpha=0.85)
ax.bar(x - 0.5*bw, w_nn,                    bw, label=f"Non-neg ridge (val {val_acc_nn:.3f})",              color="#10b981", alpha=0.85)
ax.bar(x + 0.5*bw, w_en,                    bw, label=f"Elastic Net l1={best_l1_en} (val {val_acc_en:.3f})", color="#a78bfa", alpha=0.85)
ax.bar(x + 1.5*bw, [HAND[d] for d in DIMS], bw, label="Hand-tuned",                      color="#f59e0b", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("PHAS weight"); ax.set_title("PHAS weight calibration comparison")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPNG, dpi=150, bbox_inches="tight")
print(f"\nSaved figure → {OUTPNG}")

# ── Update JSON ───────────────────────────────────────────────────────────────
result = {
    "generated": datetime.date.today().isoformat(),
    "csv_source": str(CSV.name),
    "summary": {
        "total_annotations": len(rows),
        "annotators": {
            u: {"count": len(rs),
                "calibration": sum(1 for r in rs if r["Source"] == "calibration"),
                "validation":  sum(1 for r in rs if r["Source"] == "validation")}
            for u, rs in {u: [r for r in rows if r["User"] == u]
                          for u in set(r["User"] for r in rows)}.items()
        },
        "prompts_covered":    len(set(r["Prompt ID"] for r in rows)),
        "unique_pairs_judged":len(set((r["Prompt ID"], r["Model A"], r["Model B"]) for r in rows)),
        "by_source":          dict(Counter(r["Source"] for r in rows)),
        "confidence_breakdown": dict(Counter(r["Confidence Label"] for r in rows)),
        "inter_annotator": {
            "shared_pairs":    len(shared),
            "exact_agreement": agree,
            "agreement_pct":   round(100 * agree / len(shared), 1) if shared else None,
            "note": "Based on 750 shared pairs across 8 annotators"
        },
    },
    "human_bradley_terry": {
        short(m): {
            "bt_strength": round(bt_s[m], 4),
            "rank":        human_rank[m],
            "bt_ci_95":    [round(bt_ci[m][0], 4), round(bt_ci[m][1], 4)] if m in bt_ci else None,
        }
        for m in MODELS
    },
    "vlm_vs_human": {
        "spearman_rho":    round(rho_vlm, 4),
        "p_value":         round(p_vlm,   4),
        "significant_p005": bool(p_vlm < 0.05),
        "rank_comparison": {
            short(m): {
                "vlm_rank":   vlm_rank[m],
                "human_rank": human_rank.get(m),
                "delta":      human_rank.get(m, 0) - vlm_rank[m],
            }
            for m in MODELS
        },
    },
    "phas_calibration": {
        "note": (
            f"Final — {len(cal_rows)} calibration annotations from 7 annotators "
            f"({len(rows)} total, A1-retest excluded). "
            f"CV accuracy {cv_acc_nn_mean*100:.1f}%, validation accuracy {val_acc_nn*100:.1f}%."
        ),
        "n_calibration": len(cal_rows),
        "n_validation":  len(val_rows),
        "unconstrained_ridge": {
            "best_C":       float(clf.C_[0]),
            "cv_accuracy":  round(float(cv_acc_unc.mean()), 4),
            "cv_std":       round(float(cv_acc_unc.std()),  4),
            "val_accuracy": round(float(val_acc_unc), 4),
            "n_negative_raw_coefs": int(sum(raw_unc < 0)),
            "weights": {dim: round(float(w_unc[i]), 4) for i, dim in enumerate(DIMS)},
        },
        "nonneg_ridge": {
            "best_C":       float(best_C_nn),
            "cv_accuracy":  round(float(cv_acc_nn_mean), 4),
            "val_accuracy": round(float(val_acc_nn), 4),
            "n_zero_weights": int(sum(beta_nn <= 1e-6)),
            "weights": {dim: round(float(w_nn[i]), 4) for i, dim in enumerate(DIMS)},
        },
        "nonneg_elasticnet": {
            "best_C":        float(best_C_en),
            "best_l1_ratio": float(best_l1_en),
            "cv_accuracy":   round(float(cv_acc_en_mean), 4),
            "val_accuracy":  round(float(val_acc_en), 4),
            "n_zero_weights": int(sum(beta_en <= 1e-6)),
            "weights": {dim: round(float(w_en[i]), 4) for i, dim in enumerate(DIMS)},
        },
        "hand_tuned": {
            "weights": HAND,
        },
        "recommended": (
            "nonneg_elasticnet" if val_acc_en >= val_acc_nn else "nonneg_ridge"
        ),
        "recommendation_reason": (
            "Elastic net (L1+L2 with non-negativity) blends sparsity and stability. "
            "With beta>=0 the L1 term is differentiable, so zeros are driven by genuine "
            "signal absence rather than boundary artifacts. "
            "Chosen when val accuracy >= nonneg_ridge; otherwise nonneg_ridge is used."
        ),
    },
}

with open(OUTJSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"Updated JSON → {OUTJSON}")
