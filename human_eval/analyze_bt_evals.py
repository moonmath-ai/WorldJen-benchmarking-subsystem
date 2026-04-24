"""
analyze_bt_evals.py — Comprehensive analysis of WorldJen human pairwise (BT) evaluation.

Sections:
  1. Dataset overview & annotator contributions
  2. Bradley-Terry model fit & BT rating rankings with bootstrap CIs
  3. Inter-annotator agreement (Krippendorff α, exact agreement per pair)
  4. Annotator-level bias & self-consistency check
  5. Coverage analysis (pair-level heatmap)
  6. VLM BT rating vs Human BT rank correlation
  7. Win-rate table per model

Usage:
    # Anonymized dataset (default — for public release)
    python analyze_bt_evals.py
    python analyze_bt_evals.py --csv anonymized_human_evals.csv

    # Original dataset (private)
    python analyze_bt_evals.py --csv "human evals - Human Evaluations.csv"

    # Exclude specific annotator IDs (anonymized) or emails (original)
    python analyze_bt_evals.py --exclude A1-retest
    python analyze_bt_evals.py --exclude abc@gmail.com
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))

EXCLUDE_DEFAULT = {"A1-retest"}   # duplicate self-consistency pass (both anonymized and original forms)

MODEL_SHORT = {
    "fal-ai_hunyuan-video-v1.5_text-to-video":   "HunyuanVideo",
    "fal-ai_kling-video_v2.6_pro_text-to-video":  "Kling 2.6 Pro",
    "fal-ai_ltx-2_text-to-video":                 "LTX-2",
    "fal-ai_veo3.1_fast":                         "Veo 3.1 Fast",
    "fal-ai_wan_v2.2-a14b_text-to-video":         "Wan 2.2 A14B",
    "wan2.1-1.3b":                                "Wan 2.1 1.3B",
}

VLM_JSON_DEFAULT = DATA_ROOT / "results/gemini_vlm/summary_report_unified.json"


# ── Report builder ────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self._lines = []

    def h1(self, title):
        self._lines += ["", "═" * 80, f"  {title}", "═" * 80]

    def h2(self, title):
        self._lines += ["", f"── {title} " + "─" * max(0, 74 - len(title))]

    def row(self, text=""):
        self._lines.append("  " + text)

    def table_header(self, cols, widths):
        self._lines.append("  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
        self._lines.append("  " + "  ".join("─" * w for w in widths))

    def table_row(self, vals, widths, fmts=None):
        parts = []
        for i, (v, w) in enumerate(zip(vals, widths)):
            fmt = fmts[i] if fmts else "<"
            parts.append(f"{v:{fmt}{w}}")
        self._lines.append("  " + "  ".join(parts))

    def blank(self):
        self._lines.append("")

    def render(self):
        return "\n".join(self._lines)


# ── Bradley-Terry ─────────────────────────────────────────────────────────────

def fit_bradley_terry(comparisons, n_iters=2000):
    """MLE Bradley-Terry. comparisons = list of (winner, loser) strings."""
    models = sorted({m for w, l in comparisons for m in (w, l)})
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.zeros((n, n))
    for w, l in comparisons:
        W[idx[w], idx[l]] += 1
    strength = np.ones(n)
    for _ in range(n_iters):
        new = np.zeros(n)
        for i in range(n):
            wins = W[i].sum()
            denom = sum((W[i, j] + W[j, i]) / (strength[i] + strength[j])
                        for j in range(n) if j != i and (W[i, j] + W[j, i]) > 0)
            new[i] = wins / denom if denom > 0 and wins > 0 else 1e-9
        strength = new / new.sum() * n
    log_s = np.log(strength + 1e-9)
    log_s -= log_s.mean()
    bt = {models[i]: 1500 + log_s[i] * 400 / math.log(10) for i in range(n)}
    return bt


def bootstrap_bt(comparisons, n_boot=1000, seed=42):
    """Bootstrap 95% CI for BT rating scores."""
    rng = np.random.default_rng(seed)
    comps = list(comparisons)
    boot_bts = defaultdict(list)
    for _ in range(n_boot):
        sample = [comps[i] for i in rng.integers(0, len(comps), len(comps))]
        try:
            bt = fit_bradley_terry(sample)
            for m, e in bt.items():
                boot_bts[m].append(e)
        except Exception:
            pass
    cis = {}
    for m, vals in boot_bts.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        cis[m] = (lo, hi)
    return cis


# ── IAA helpers ───────────────────────────────────────────────────────────────

def normalise_pair_key(pid, ma, mb):
    """Canonical key: sort model names so A<B always."""
    a, b = sorted([ma, mb])
    return f"{pid}|{a}|{b}"


def exact_agreement(votes1, votes2):
    """Given two dicts {pair_key: winner_model}, return exact match fraction."""
    shared = set(votes1) & set(votes2)
    if not shared:
        return float("nan"), 0
    agree = sum(1 for k in shared if votes1[k] == votes2[k])
    return agree / len(shared), len(shared)


def krippendorff_nominal(ratings_matrix):
    """
    Krippendorff α nominal for binary preference data.
    ratings_matrix: np.ndarray shape (n_raters, n_units), NaN for missing.
    Values: 0 = model A wins, 1 = model B wins (ties excluded).
    """
    m, n = ratings_matrix.shape
    # Observed disagreement
    Do_num, Do_den = 0.0, 0
    for j in range(n):
        col = ratings_matrix[:, j]
        vals = col[~np.isnan(col)]
        k = len(vals)
        if k < 2:
            continue
        for v1, v2 in combinations(vals, 2):
            Do_num += float(v1 != v2)
            Do_den += 1
    if Do_den == 0:
        return float("nan")
    Do = Do_num / Do_den

    # Expected disagreement
    all_vals = ratings_matrix[~np.isnan(ratings_matrix)]
    if len(all_vals) < 2:
        return float("nan")
    p = np.mean(all_vals)  # fraction of "B wins"
    De = 2 * p * (1 - p)   # nominal: expected disagreement for binary
    if De == 0:
        return 1.0
    return 1.0 - Do / De


# ── Load data ─────────────────────────────────────────────────────────────────

def load_csv(path, exclude_ids):
    """Load CSV from either the anonymized (A1-A7) or original (email) dataset.

    The 'User' column may contain either an opaque annotator ID (anonymized CSV)
    or an email address (original CSV).  Both are stored in the 'email' key for
    backward-compatibility; 'annotator' is a short display label derived from it.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = r.get("User", r.get("user", "")).strip()
            uid_lower = uid.lower()
            if uid_lower in {e.lower() for e in exclude_ids} or uid in exclude_ids:
                continue
            # Display label: for emails use the local part; for anon IDs use as-is
            display = uid.split("@")[0] if "@" in uid else uid
            rows.append({
                "email":      uid,       # kept as 'email' for backward-compat
                "annotator":  display,   # short display name
                "prompt_id":  r.get("Prompt ID", r.get("prompt_id", "")).strip(),
                "model_a":    r.get("Model A", r.get("model_a", "")).strip(),
                "model_b":    r.get("Model B", r.get("model_b", "")).strip(),
                "winner":     r.get("Winner", r.get("winner", "")).strip(),
                "loser":      r.get("Loser",  r.get("loser",  "")).strip(),
                "weight":     float(r.get("Weight", r.get("weight", 1)) or 1),
                "confidence": r.get("Confidence Label", r.get("confidence_label", "")).strip(),
                "source":     r.get("Source", r.get("source", "")).strip(),
            })
    return rows


# ── Sections ──────────────────────────────────────────────────────────────────

def section_overview(R, rows):
    R.h1("1. DATASET OVERVIEW")

    annotators = sorted({r["email"] for r in rows})
    prompts    = sorted({r["prompt_id"] for r in rows})
    models     = sorted({m for r in rows for m in (r["model_a"], r["model_b"])})

    R.h2("Global stats")
    R.row(f"Total comparisons : {len(rows)}")
    R.row(f"Annotators        : {len(annotators)}")
    R.row(f"Prompts covered   : {len(prompts)}")
    R.row(f"Models            : {len(models)}")
    ties = sum(1 for r in rows if r["winner"].upper() == "TIE")
    R.row(f"Tie rate          : {ties}/{len(rows)} = {100*ties/len(rows):.1f}%")

    R.h2("Per-annotator contribution")
    R.table_header(["Annotator", "N", "%", "Confidence breakdown"], [32, 5, 6, 50])
    for ann in annotators:
        ann_rows = [r for r in rows if r["email"] == ann]
        pct = 100 * len(ann_rows) / len(rows)
        conf_counts = defaultdict(int)
        for r in ann_rows:
            conf_counts[r["confidence"]] += 1
        conf_str = "  ".join(f"{k}:{v}" for k, v in sorted(conf_counts.items()))
        R.table_row([ann, len(ann_rows), f"{pct:.1f}%", conf_str], [32, 5, 6, 50])

    R.h2("Per-source breakdown")
    source_counts = defaultdict(int)
    for r in rows:
        source_counts[r["source"]] += 1
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        R.row(f"  {src or '(none)':<20}: {cnt}")

    R.h2("Pair coverage: (prompt × model_a × model_b) unique pairs")
    pair_set = {normalise_pair_key(r["prompt_id"], r["model_a"], r["model_b"]) for r in rows}
    n_models = len(models)
    total_possible = len(prompts) * (n_models * (n_models - 1) // 2)
    R.row(f"Unique pairs seen : {len(pair_set)}")
    R.row(f"Max possible      : {total_possible}  ({len(prompts)} prompts × C({n_models},2)={n_models*(n_models-1)//2} model-pairs)")
    R.row(f"Coverage          : {100*len(pair_set)/total_possible:.1f}%")

    # Multi-annotator pairs
    pair_ann = defaultdict(set)
    for r in rows:
        pk = normalise_pair_key(r["prompt_id"], r["model_a"], r["model_b"])
        pair_ann[pk].add(r["email"])
    multi = sum(1 for v in pair_ann.values() if len(v) >= 2)
    R.row(f"Pairs rated by ≥2 annotators: {multi}  ({100*multi/len(pair_set):.1f}%)")


def section_bt(R, rows):
    R.h1("2. BRADLEY-TERRY MODEL — BT rating RANKINGS")

    non_tie = [(r["winner"], r["loser"]) for r in rows
               if r["winner"].upper() != "TIE" and r["winner"] and r["loser"]]

    bt  = fit_bradley_terry(non_tie)
    cis  = bootstrap_bt(non_tie)
    ranked = sorted(bt.items(), key=lambda x: -x[1])

    R.h2("Overall ranking (all annotators, bootstrap 95% CI, N=1000)")
    R.table_header(["Rank", "Model", "BT rating", "95% CI", "Win rate", "W", "L"], [5, 22, 7, 18, 9, 5, 5])

    model_wins   = defaultdict(int)
    model_losses = defaultdict(int)
    for winner, loser in non_tie:
        model_wins[winner]   += 1
        model_losses[loser]  += 1

    for rank, (model, e) in enumerate(ranked, 1):
        lo, hi = cis.get(model, (float("nan"), float("nan")))
        w = model_wins[model];  l = model_losses[model]
        wr = f"{100*w/(w+l):.1f}%" if (w + l) > 0 else "—"
        short = MODEL_SHORT.get(model, model)
        R.table_row([f"#{rank}", short, f"{e:.1f}", f"[{lo:.1f}, {hi:.1f}]", wr, w, l],
                    [5, 22, 7, 18, 9, 5, 5])

    R.blank()
    R.row("Note: overlapping CIs indicate statistically indistinguishable models.")

    R.h2("Per-annotator BT ranking (subset BT rating fitted on each annotator's votes alone)")
    ann_list = sorted({r["email"] for r in rows})
    shorts = [MODEL_SHORT.get(m, m)[:10] for m, _ in ranked]
    widths = [22] + [11] * len(ranked)
    R.table_header(["Annotator"] + [f"#{i+1} {s}" for i, s in enumerate(shorts)], widths)
    for ann in ann_list:
        ann_rows = [(r["winner"], r["loser"]) for r in rows
                    if r["email"] == ann and r["winner"].upper() != "TIE"]
        if len(ann_rows) < 5:
            R.row(f"  {ann.split('@')[0]:<20} (only {len(ann_rows)} votes — skipped)")
            continue
        try:
            ann_bt = fit_bradley_terry(ann_rows)
            ann_ranked = sorted(ann_bt, key=lambda m: -ann_bt[m])
            rank_str = [str(ann_ranked.index(m) + 1) if m in ann_ranked else "—" for m, _ in ranked]
            R.table_row([ann.split("@")[0]] + [f"#{r}" for r in rank_str], widths)
        except Exception:
            R.row(f"  {ann.split('@')[0]:<20} (BT fit failed)")

    R.h2("Rank stability: how many annotators agree on top-2 and bottom-1")
    top2_agree = 0; bot1_agree = 0; total_ann = 0
    global_top2 = {m for m, _ in ranked[:2]}
    global_bot1 = {ranked[-1][0]}
    for ann in ann_list:
        ann_rows = [(r["winner"], r["loser"]) for r in rows
                    if r["email"] == ann and r["winner"].upper() != "TIE"]
        if len(ann_rows) < 10:
            continue
        try:
            ann_bt = fit_bradley_terry(ann_rows)
            ann_ranked = sorted(ann_bt, key=lambda m: -ann_bt[m])
            ann_top2 = set(ann_ranked[:2])
            ann_bot1 = {ann_ranked[-1]}
            if ann_top2 == global_top2: top2_agree += 1
            if ann_bot1 == global_bot1: bot1_agree += 1
            total_ann += 1
        except Exception:
            pass
    if total_ann:
        R.row(f"Annotators agreeing on top-2 models  : {top2_agree}/{total_ann} = {100*top2_agree/total_ann:.0f}%")
        R.row(f"Annotators agreeing on bottom-1 model: {bot1_agree}/{total_ann} = {100*bot1_agree/total_ann:.0f}%")


def section_iaa(R, rows):
    R.h1("3. INTER-ANNOTATOR AGREEMENT")

    ann_list = sorted({r["email"] for r in rows})

    # Build canonical vote dict per annotator
    def ann_votes(email):
        votes = {}
        for r in rows:
            if r["email"] != email or r["winner"].upper() == "TIE":
                continue
            pk = normalise_pair_key(r["prompt_id"], r["model_a"], r["model_b"])
            # Canonical winner: whichever of model_a/model_b won
            votes[pk] = r["winner"]
        return votes

    vote_dicts = {ann: ann_votes(ann) for ann in ann_list}

    R.h2("Pairwise exact agreement (shared pairs only, ties excluded)")
    R.table_header(["Pair", "Shared N", "Exact agree%", "Note"], [36, 9, 13, 30])
    all_exact = []
    pair_rows_data = []
    for a1, a2 in combinations(ann_list, 2):
        frac, n = exact_agreement(vote_dicts[a1], vote_dicts[a2])
        if n < 3:
            continue
        pct = frac * 100
        all_exact.append(frac)
        note = "✓ good" if pct >= 70 else ("⚠ moderate" if pct >= 55 else "✗ low")
        pair_label = f"{a1.split('@')[0]} ∩ {a2.split('@')[0]}"
        pair_rows_data.append((pair_label, n, pct, note))
        R.table_row([pair_label, n, f"{pct:.1f}%", note], [36, 9, 13, 30])

    R.blank()
    if all_exact:
        R.row(f"Mean exact agreement : {100*np.mean(all_exact):.1f}%  "
              f"(range {100*min(all_exact):.1f}%–{100*max(all_exact):.1f}%)")

    R.h2("Krippendorff α (nominal) on all shared pairs")
    # Build reliability matrix: rows=annotators, cols=unique pair keys
    all_pairs = sorted({normalise_pair_key(r["prompt_id"], r["model_a"], r["model_b"])
                        for r in rows if r["winner"].upper() != "TIE"})
    pair_idx = {p: i for i, p in enumerate(all_pairs)}
    matrix = np.full((len(ann_list), len(all_pairs)), np.nan)
    ann_idx = {a: i for i, a in enumerate(ann_list)}
    for r in rows:
        if r["winner"].upper() == "TIE":
            continue
        pk  = normalise_pair_key(r["prompt_id"], r["model_a"], r["model_b"])
        ann = r["email"]
        if ann not in ann_idx or pk not in pair_idx:
            continue
        # Encode: 0 = first model (alphabetically) wins, 1 = second wins
        a, b = sorted([r["model_a"], r["model_b"]])
        val = 0.0 if r["winner"] == a else 1.0
        matrix[ann_idx[ann], pair_idx[pk]] = val

    alpha = krippendorff_nominal(matrix)
    n_multi = sum(1 for j in range(matrix.shape[1])
                  if np.sum(~np.isnan(matrix[:, j])) >= 2)
    R.row(f"Krippendorff α (nominal)  : {alpha:.4f}  (over {n_multi} pairs rated by ≥2 annotators)")
    if not np.isnan(alpha):
        if   alpha >= 0.80: tier = "Excellent"
        elif alpha >= 0.67: tier = "Acceptable"
        elif alpha >= 0.40: tier = "Tentative / Moderate"
        elif alpha >= 0.20: tier = "Fair"
        else:               tier = "Poor (below publishable threshold)"
        R.row(f"Tier                      : {tier}")
    R.row("Note: pairwise preference is binary — nominal α is appropriate (not ordinal/interval).")

    R.h2("Per-model IAA: do annotators agree on WHO wins each matchup?")
    R.row("For each model-pair, % of shared votes that agree on winner.")
    model_pairs = list(combinations(sorted({m for r in rows for m in (r["model_a"], r["model_b"])}), 2))
    R.table_header(["Model A", "Model B", "Total votes", "Multi-ann pairs", "Mean agree%"], [20, 20, 12, 16, 12])
    for ma, mb in model_pairs:
        relevant = [r for r in rows if set([r["model_a"], r["model_b"]]) == {ma, mb}
                    and r["winner"].upper() != "TIE"]
        if not relevant:
            continue
        pair_ann_votes = defaultdict(dict)
        for r in relevant:
            pk = normalise_pair_key(r["prompt_id"], r["model_a"], r["model_b"])
            pair_ann_votes[pk][r["email"]] = r["winner"]
        multi_pairs = {pk: v for pk, v in pair_ann_votes.items() if len(v) >= 2}
        if not multi_pairs:
            R.table_row([MODEL_SHORT.get(ma, ma)[:18], MODEL_SHORT.get(mb, mb)[:18],
                         len(relevant), 0, "—"], [20, 20, 12, 16, 12])
            continue
        agrees = []
        for pk, ann_v in multi_pairs.items():
            anns = list(ann_v.keys())
            for a1, a2 in combinations(anns, 2):
                agrees.append(int(ann_v[a1] == ann_v[a2]))
        mean_agree = 100 * np.mean(agrees) if agrees else float("nan")
        R.table_row([MODEL_SHORT.get(ma, ma)[:18], MODEL_SHORT.get(mb, mb)[:18],
                     len(relevant), len(multi_pairs), f"{mean_agree:.1f}%"],
                    [20, 20, 12, 16, 12])


def section_bias(R, rows):
    R.h1("4. ANNOTATOR BIAS & SELF-CONSISTENCY")

    ann_list = sorted({r["email"] for r in rows})
    models   = sorted({m for r in rows for m in (r["model_a"], r["model_b"])})

    R.h2("Per-annotator win-rate per model (bias check)")
    R.row("High variance across annotators for the same model → strong rater bias.")
    shorts = [MODEL_SHORT.get(m, m)[:11] for m in models]
    widths = [22] + [12] * len(models)
    R.table_header(["Annotator"] + shorts, widths)

    model_wr_by_ann = {}
    for ann in ann_list:
        ann_rows = [r for r in rows if r["email"] == ann and r["winner"].upper() != "TIE"]
        if len(ann_rows) < 5:
            continue
        wr = {}
        for model in models:
            w = sum(1 for r in ann_rows if r["winner"] == model)
            l = sum(1 for r in ann_rows if r["loser"]  == model)
            wr[model] = w / (w + l) if (w + l) > 0 else float("nan")
        model_wr_by_ann[ann] = wr
        vals = [f"{wr[m]*100:.0f}%" if not np.isnan(wr[m]) else "—" for m in models]
        R.table_row([ann.split("@")[0]] + vals, widths)

    R.blank()
    R.row("Std of win-rate across annotators per model (high = disagreement):")
    widths2 = [22] + [12] * len(models)
    R.table_header([""] + shorts, widths2)
    stds = []
    for model in models:
        vals = [model_wr_by_ann[ann][model] for ann in model_wr_by_ann
                if model in model_wr_by_ann[ann] and not np.isnan(model_wr_by_ann[ann][model])]
        stds.append(f"{np.std(vals)*100:.1f}%" if len(vals) >= 2 else "—")
    R.table_row(["Std across annotators"] + stds, widths2)

    R.h2("Confidence distribution per annotator")
    R.table_header(["Annotator", "N", "Much better%", "Clearly%", "Slightly%"], [28, 5, 13, 9, 10])
    for ann in ann_list:
        ann_rows = [r for r in rows if r["email"] == ann]
        if not ann_rows:
            continue
        n = len(ann_rows)
        conf = defaultdict(int)
        for r in ann_rows:
            conf[r["confidence"]] += 1
        much    = 100 * conf.get("Much better", 0) / n
        clearly = 100 * conf.get("Clearly better", 0) / n
        slight  = 100 * conf.get("Slightly better", 0) / n
        R.table_row([ann.split("@")[0], n, f"{much:.1f}%", f"{clearly:.1f}%", f"{slight:.1f}%"],
                    [28, 5, 13, 9, 10])


def section_coverage(R, rows):
    R.h1("5. COVERAGE ANALYSIS")

    models  = sorted({m for r in rows for m in (r["model_a"], r["model_b"])})
    prompts = sorted({r["prompt_id"] for r in rows})
    model_pairs = list(combinations(models, 2))
    shorts = [MODEL_SHORT.get(m, m)[:11] for m in models]

    R.h2("Votes per (model_a × model_b) pair — total across all prompts")
    widths = [20] + [11] * len(models)
    R.table_header([""] + shorts, widths)
    for ma in models:
        cells = []
        for mb in models:
            if ma == mb:
                cells.append("—")
            else:
                cnt = sum(1 for r in rows
                          if set([r["model_a"], r["model_b"]]) == {ma, mb})
                cells.append(str(cnt))
        R.table_row([MODEL_SHORT.get(ma, ma)[:18]] + cells, widths)

    R.h2("Prompts with ≥1 vote per model-pair")
    R.table_header(["Model A", "Model B", "Prompts covered", "Votes total", "Votes/prompt"], [20, 20, 16, 12, 13])
    for ma, mb in model_pairs:
        relevant = [r for r in rows if set([r["model_a"], r["model_b"]]) == {ma, mb}]
        prompts_covered = len({r["prompt_id"] for r in relevant})
        vpp = len(relevant) / prompts_covered if prompts_covered else 0
        R.table_row([MODEL_SHORT.get(ma, ma)[:18], MODEL_SHORT.get(mb, mb)[:18],
                     prompts_covered, len(relevant), f"{vpp:.1f}"],
                    [20, 20, 16, 12, 13])


def section_vlm(R, rows, vlm_path):
    R.h1("6. VLM BT rating vs HUMAN BT RANK CORRELATION")

    if not Path(vlm_path).exists():
        R.row(f"⚠  VLM summary not found at {vlm_path} — skipping.")
        return

    with open(vlm_path) as f:
        vlm = json.load(f)

    vlm_bt = vlm.get("bt_ratings", {})
    vlm_avg = vlm.get("avg_scores", {})

    non_tie = [(r["winner"], r["loser"]) for r in rows
               if r["winner"].upper() != "TIE" and r["winner"] and r["loser"]]
    human_bt = fit_bradley_terry(non_tie)

    def short(m):
        return (m.replace("fal-ai_", "")
                 .replace("_text-to-video", "")
                 .replace("_fast", ""))

    h_map = {short(m): v for m, v in human_bt.items()}
    v_map = {short(m): v for m, v in vlm_bt.items()}
    common = sorted(set(h_map) & set(v_map))

    if len(common) < 3:
        R.row(f"Not enough model overlap (found {len(common)}) for rank correlation.")
        return

    h_ord = [h_map[k] for k in common]
    v_ord = [v_map[k] for k in common]
    rho, pval = spearmanr(h_ord, v_ord)

    h_ranked = sorted(common, key=lambda k: -h_map[k])
    v_ranked = sorted(common, key=lambda k: -v_map[k])
    h_rank   = {k: i + 1 for i, k in enumerate(h_ranked)}
    v_rank   = {k: i + 1 for i, k in enumerate(v_ranked)}

    R.h2("Model rank comparison: Human BT vs VLM BT rating")
    R.table_header(["Model", "Human BT rating", "Human #", "VLM BT rating", "VLM #", "Δ rank", ""],
                   [22, 10, 8, 9, 6, 7, 2])
    for k in h_ranked:
        delta = h_rank[k] - v_rank[k]
        arrow = "↑" if delta < 0 else ("↓" if delta > 0 else "=")
        R.table_row([MODEL_SHORT.get(k, k)[:20],
                     f"{h_map[k]:.1f}", f"#{h_rank[k]}",
                     f"{v_map[k]:.1f}", f"#{v_rank[k]}",
                     f"{delta:+d}", arrow],
                    [22, 10, 8, 9, 6, 7, 2])

    R.blank()
    sig = "✓ significant" if pval < 0.05 else ("marginal" if pval < 0.10 else "ns")
    R.row(f"Spearman ρ (Human BT vs VLM BT rating) : {rho:.4f}  p={pval:.4f}  n={len(common)} models  [{sig}]")

    # Also compare human BT rating vs VLM avg score rank
    if vlm_avg:
        a_map = {short(m): v for m, v in vlm_avg.items()}
        common_a = sorted(set(h_map) & set(a_map))
        if len(common_a) >= 3:
            h2 = [h_map[k] for k in common_a]
            a2 = [a_map[k] for k in common_a]
            rho2, pval2 = spearmanr(h2, a2)
            R.row(f"Spearman ρ (Human BT vs VLM avg score): {rho2:.4f}  p={pval2:.4f}  n={len(common_a)} models")


def section_winrates(R, rows):
    R.h1("7. DETAILED WIN RATES & HEAD-TO-HEAD")

    models = sorted({m for r in rows for m in (r["model_a"], r["model_b"])})
    non_tie = [r for r in rows if r["winner"].upper() != "TIE"]

    R.h2("Overall win/loss/tie per model")
    R.table_header(["Model", "W", "L", "T", "Total", "Win%", "Win%(no tie)"], [22, 5, 5, 5, 6, 7, 13])
    for model in models:
        w = sum(1 for r in rows if r["winner"] == model)
        l = sum(1 for r in rows if r["loser"]  == model)
        t = sum(1 for r in rows if r["winner"].upper() == "TIE"
                and model in (r["model_a"], r["model_b"]))
        total = w + l + t
        wr_all = 100 * w / total if total else 0
        wr_not = 100 * w / (w + l) if (w + l) else 0
        R.table_row([MODEL_SHORT.get(model, model)[:20],
                     w, l, t, total, f"{wr_all:.1f}%", f"{wr_not:.1f}%"],
                    [22, 5, 5, 5, 6, 7, 13])

    R.h2("Head-to-head win matrix  (row = winner, value = wins vs column)")
    shorts = [MODEL_SHORT.get(m, m)[:10] for m in models]
    widths = [18] + [11] * len(models)
    R.table_header([""] + shorts, widths)
    for ma in models:
        cells = []
        for mb in models:
            if ma == mb:
                cells.append("—")
            else:
                wins = sum(1 for r in non_tie
                           if r["winner"] == ma and r["loser"] == mb)
                total = sum(1 for r in non_tie
                            if set([r["model_a"], r["model_b"]]) == {ma, mb})
                cells.append(f"{wins}/{total}" if total else "0/0")
        R.table_row([MODEL_SHORT.get(ma, ma)[:16]] + cells, widths)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive BT human eval analysis.")
    parser.add_argument("--csv", type=Path,
                        default=DATA_ROOT / "human_eval/anonymized_human_evals.csv",
                        help="Path to CSV (anonymized or original). "
                             "Default: anonymized_human_evals.csv")
    parser.add_argument("--vlm", type=Path, default=VLM_JSON_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Annotator IDs or emails to exclude. "
                             "Default: A1-retest (anonymized)")
    args = parser.parse_args()

    exclude = set(args.exclude) if args.exclude is not None else EXCLUDE_DEFAULT

    rows = load_csv(args.csv, exclude)
    print(f"Loaded {len(rows)} rows (excluded: {exclude})", file=sys.stderr)

    np.random.seed(42)
    R = Report()
    R.h1("WorldJen — Pairwise (BT) Human Evaluation: Full Analysis Report")
    R.row(f"CSV     : {args.csv}")
    R.row(f"VLM     : {args.vlm}")
    R.row(f"Excluded: {exclude}")

    section_overview(R, rows)
    section_bt(R, rows)
    section_iaa(R, rows)
    section_bias(R, rows)
    section_coverage(R, rows)
    section_vlm(R, rows, args.vlm)
    section_winrates(R, rows)

    R.h1("END OF REPORT")

    report = R.render()
    print(report)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"\n[Report saved → {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
