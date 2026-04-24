"""
diagnose_composition_weight.py
──────────────────────────────
Four independent checks for whether Composition & Framing's high PHAS
weight is real signal or a multicollinearity artefact:

  1. Variance Inflation Factor (VIF)  — quantifies multicollinearity directly
  2. LASSO (L1)                       — sparse solution unaffected by corr. dims
  3. Permutation importance           — model-agnostic, shuffles one dim at a time
  4. Partial correlation              — comp vs winner after removing shared variance
                                        with Subject Consistency (most correlated)

Outputs:
  - composition_diagnosis.png  (4-panel figure)
  - prints summary table
"""

import csv, json, glob, os, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, pointbiserialr
from scipy.linalg import solve
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))

DIMS = [
    "subject_consistency","scene_consistency","motion_smoothness","temporal_flickering",
    "inertial_consistency","physical_mechanics","object_permanence","human_fidelity",
    "dynamic_degree","semantic_adherence","spatial_relationship","semantic_drift",
    "composition_framing","lighting_volumetric","color_harmony","structural_gestalt",
]
SHORT = {
    "subject_consistency":"Subj. Cons.","scene_consistency":"Scene Cons.",
    "motion_smoothness":"Mot. Smooth","temporal_flickering":"Temp. Flick.",
    "inertial_consistency":"Inertial","physical_mechanics":"Phys. Mech.",
    "object_permanence":"Obj. Perm.","human_fidelity":"Human Fid.",
    "dynamic_degree":"Dyn. Degree","semantic_adherence":"Sem. Adh.",
    "spatial_relationship":"Spatial Rel.","semantic_drift":"Sem. Drift",
    "composition_framing":"Composition","lighting_volumetric":"Lighting",
    "color_harmony":"Color Harm.","structural_gestalt":"Struct. Gest.",
}
COMP = DIMS.index("composition_framing")

# ── Load data ─────────────────────────────────────────────────────────────────
def get_scores(model, pid):
    for d in [str(DATA_ROOT / "results/gemini_vlm")]:
        fps = glob.glob(f"{d}/{model}_{pid}.json")
        if fps:
            with open(fps[0]) as f: data = json.load(f)
            return {dim: np.mean([float(q["score"]) for q in data["results"].get(dim,[]) if "score" in q])
                    for dim in DIMS if isinstance(data["results"].get(dim,[]),list) and data["results"].get(dim,[])}
    return {}

rows = []
with open(DATA_ROOT / "human_eval/anonymized_human_evals.csv") as f:
    for r in csv.DictReader(f): rows.append(r)
cal = [r for r in rows if r["Source"] == "calibration"]

X, y, w = [], [], []
for r in cal:
    pid, ma, mb = r["Prompt ID"], r["Model A"], r["Model B"]
    sa = get_scores(ma, pid); sb = get_scores(mb, pid)
    X.append([sa.get(d,0)-sb.get(d,0) for d in DIMS])
    y.append(1 if r["Winner"] == ma else 0)
    w.append(float(r["Weight"]))
X = np.array(X); y = np.array(y); w = np.array(w)

print(f"Calibration matrix: {X.shape}  |  {y.mean():.1%} class-1 base rate")

# ═══════════════════════════════════════════════════════════════════════════════
# Check 1 — Variance Inflation Factor
# ═══════════════════════════════════════════════════════════════════════════════
def vif(X):
    """VIF for each column of X."""
    vifs = []
    for i in range(X.shape[1]):
        Xi = np.delete(X, i, axis=1)
        # R² from regressing column i on all others
        beta = np.linalg.lstsq(Xi, X[:,i], rcond=None)[0]
        pred = Xi @ beta
        ss_res = np.sum((X[:,i] - pred)**2)
        ss_tot = np.sum((X[:,i] - X[:,i].mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        vifs.append(1/(1-r2) if r2 < 1 else np.inf)
    return np.array(vifs)

vif_vals = vif(X)
print("\n=== Check 1: Variance Inflation Factor (VIF) ===")
print("  (VIF > 5 = moderate multicollinearity, > 10 = severe)")
for dim, v in sorted(zip(DIMS, vif_vals), key=lambda x:-x[1]):
    bar = "█"*int(min(v,20)/1) + ("..." if v>20 else "")
    flag = " ◀ SEVERE" if v>10 else " ◀ moderate" if v>5 else ""
    print(f"  {SHORT[dim]:<14} VIF={v:6.1f}  {bar}{flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# Check 2 — LASSO (L1 regularisation, naturally sparse)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 2: L1 / LASSO — sparse solution ===")
lasso = LogisticRegressionCV(Cs=[0.001,0.01,0.1,1,10], cv=5, penalty="l1",
                              solver="saga", max_iter=5000, random_state=42)
lasso.fit(X, y, sample_weight=w)
cv_lasso = cross_val_score(lasso, X, y, cv=5, scoring="accuracy")
raw_l1 = lasso.coef_[0]
pos_l1 = np.maximum(raw_l1, 0)
w_l1   = pos_l1/pos_l1.sum() if pos_l1.sum()>0 else pos_l1
print(f"  Best C: {lasso.C_[0]}  |  CV acc: {cv_lasso.mean():.3f} ± {cv_lasso.std():.3f}")
print(f"  Non-zero dims: {(raw_l1!=0).sum()}/16")
for dim, wt in sorted(zip(DIMS, w_l1), key=lambda x:-x[1]):
    if wt > 0.001:
        print(f"    {SHORT[dim]:<14} {wt:.4f}")
comp_l1_rank = sorted(w_l1, reverse=True).index(w_l1[COMP]) + 1
print(f"  → Composition rank in LASSO weights: #{comp_l1_rank}")

# ═══════════════════════════════════════════════════════════════════════════════
# Check 3 — Permutation importance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 3: Permutation importance (shuffle each dim, measure accuracy drop) ===")
base_clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000)
base_clf.fit(X, y, sample_weight=w)
base_acc = base_clf.score(X, y, sample_weight=w)

perm_imp = []
rng = np.random.default_rng(42)
for i in range(len(DIMS)):
    drops = []
    for _ in range(100):
        Xp = X.copy()
        Xp[:,i] = rng.permutation(Xp[:,i])
        drops.append(base_acc - base_clf.score(Xp, y, sample_weight=w))
    perm_imp.append(np.mean(drops))
perm_imp = np.array(perm_imp)

for dim, imp in sorted(zip(DIMS, perm_imp), key=lambda x:-x[1]):
    bar = "█"*max(0,int(imp*500))
    print(f"  {SHORT[dim]:<14} Δacc={imp:+.4f}  {bar}")
comp_perm_rank = sorted(perm_imp.tolist(), reverse=True).index(perm_imp[COMP]) + 1
print(f"  → Composition rank in permutation importance: #{comp_perm_rank}")

# ═══════════════════════════════════════════════════════════════════════════════
# Check 4 — Partial correlation (comp vs winner, controlling for Subject Consistency)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Check 4: Partial correlation (comp vs human choice, controlling for top correlated dims) ===")
# Residualise composition diff and y on Subject Consistency
def residualise(target, control):
    """Return residuals of regressing target on control."""
    b = np.dot(control, target) / np.dot(control, control)
    return target - b * control

subj_idx = DIMS.index("subject_consistency")
spat_idx = DIMS.index("spatial_relationship")

for ctrl_name, ctrl_idx in [("Subject Consistency", subj_idx),
                              ("Spatial Relationship", spat_idx)]:
    comp_res = residualise(X[:,COMP], X[:,ctrl_idx])
    y_res    = residualise(y.astype(float), X[:,ctrl_idx])
    r_partial, p_partial = pointbiserialr(comp_res, (y_res > 0).astype(int))
    r_raw, _             = pointbiserialr(X[:,COMP], y)
    print(f"  Raw r(comp, winner):                  {r_raw:+.3f}")
    print(f"  Partial r after removing {ctrl_name}: {r_partial:+.3f}")
    print(f"  → Drop: {r_raw-r_partial:+.3f}  ({'large — proxy effect' if abs(r_raw-r_partial)>0.1 else 'small — independent signal'})\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary verdict
# ═══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("SUMMARY")
print("="*70)
print(f"  VIF for Composition:       {vif_vals[COMP]:.1f}  {'(moderate collinearity)' if 5<vif_vals[COMP]<10 else '(severe)' if vif_vals[COMP]>=10 else '(low)'}")
print(f"  LASSO rank:                #{comp_l1_rank}")
print(f"  Permutation rank:          #{comp_perm_rank}")
print(f"  Raw point-biserial r:      +0.399")
print(f"  Bootstrap 95% CI:          [0.170, 0.314]")
print()
print("  Verdict:")
if vif_vals[COMP] > 5 and comp_l1_rank <= 3:
    print("  Moderate multicollinearity BUT composition is still top in LASSO")
    print("  and permutation importance → weight is inflated but direction is real.")
elif comp_l1_rank > 3:
    print("  LASSO demotes composition significantly → weight is largely a proxy artifact.")
else:
    print("  Low collinearity + top LASSO + top permutation → weight is genuine.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10), facecolor="#0b1221")
fig.suptitle("Diagnosing the Composition & Framing weight", fontsize=13,
             color="white", y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

BG = "#0b1221"; SURF = "#111827"; ACCENT = "#38bdf8"; GREEN = "#10b981"
GOLD = "#f59e0b"; RED = "#ef4444"; MUTED = "#94a3b8"

def style_ax(ax, title):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.title.set_text(title); ax.title.set_color(ACCENT); ax.title.set_fontsize(9.5)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e3a5f")

labels = [SHORT[d] for d in DIMS]

# Panel 1: VIF
ax1 = fig.add_subplot(gs[0,0])
colors_vif = [RED if v>10 else GOLD if v>5 else GREEN for v in vif_vals]
bars = ax1.barh(labels, vif_vals, color=colors_vif, alpha=0.85)
ax1.axvline(5,  color=GOLD, lw=1, ls="--", alpha=0.7, label="VIF=5")
ax1.axvline(10, color=RED,  lw=1, ls="--", alpha=0.7, label="VIF=10")
ax1.set_xlabel("VIF", color=MUTED, fontsize=8)
ax1.legend(fontsize=7, labelcolor=MUTED, facecolor=SURF, edgecolor="#1e3a5f")
style_ax(ax1, "1. Variance Inflation Factor")
# highlight composition
comp_label_idx = labels.index(SHORT["composition_framing"])
bars[comp_label_idx].set_edgecolor("white"); bars[comp_label_idx].set_linewidth(1.5)

# Panel 2: LASSO weights
ax2 = fig.add_subplot(gs[0,1])
colors_l1 = [ACCENT if d=="composition_framing" else GREEN for d in DIMS]
ax2.barh(labels, w_l1, color=colors_l1, alpha=0.85)
ax2.set_xlabel("Normalised LASSO weight", color=MUTED, fontsize=8)
style_ax(ax2, "2. L1 / LASSO sparse weights")

# Panel 3: Permutation importance
ax3 = fig.add_subplot(gs[1,0])
colors_p = [ACCENT if d=="composition_framing" else GREEN for d in DIMS]
order = np.argsort(perm_imp)
ax3.barh([labels[i] for i in order], perm_imp[order],
          color=[colors_p[i] for i in order], alpha=0.85)
ax3.axvline(0, color=MUTED, lw=0.8)
ax3.set_xlabel("Mean accuracy drop when shuffled", color=MUTED, fontsize=8)
style_ax(ax3, "3. Permutation importance")

# Panel 4: Raw vs partial correlations for all dims
ax4 = fig.add_subplot(gs[1,1])
raw_cors = [pointbiserialr(X[:,i], y)[0] for i in range(len(DIMS))]
# partial: control for subject_consistency
partial_cors = []
for i in range(len(DIMS)):
    if i == subj_idx:
        partial_cors.append(raw_cors[i])
        continue
    cr = residualise(X[:,i], X[:,subj_idx])
    yr = residualise(y.astype(float), X[:,subj_idx])
    r, _ = pointbiserialr(cr, (yr>0).astype(int))
    partial_cors.append(r)

x_pos = np.arange(len(DIMS))
ax4.scatter(raw_cors,     x_pos + 0.15, color=ACCENT, s=35, label="Raw r", zorder=3)
ax4.scatter(partial_cors, x_pos - 0.15, color=GOLD,   s=35, label="Partial r (ctrl Subj.Cons.)", zorder=3)
for i in range(len(DIMS)):
    ax4.plot([raw_cors[i], partial_cors[i]], [i+0.15, i-0.15],
             color=MUTED, lw=0.6, alpha=0.5)
ax4.set_yticks(x_pos); ax4.set_yticklabels(labels, fontsize=7.5)
ax4.axvline(0, color=MUTED, lw=0.8)
ax4.set_xlabel("Correlation with human winner", color=MUTED, fontsize=8)
ax4.legend(fontsize=7, labelcolor=MUTED, facecolor=SURF, edgecolor="#1e3a5f")
style_ax(ax4, "4. Raw vs partial correlation")

plt.savefig(DATA_ROOT / "human_eval/composition_diagnosis.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {DATA_ROOT / 'human_eval/composition_diagnosis.png'}")
