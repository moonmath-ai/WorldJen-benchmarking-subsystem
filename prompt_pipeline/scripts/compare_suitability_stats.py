#!/usr/bin/env python3
"""
Compare average suitability and difficulty scores (all dimensions) between enhanced and unenhanced judge outputs.
Treats null as 0 when computing averages.

Usage:
  python compare_suitability_stats.py
  python compare_suitability_stats.py --enhanced path/to/rescored.jsonl --unenhanced path/to/judged.jsonl --output path/to/stats.csv
"""

import argparse
import json
import sys
from pathlib import Path

# Dimension groups and their suitability keys (from judge schema)
GROUPS = [
    "motion_stability",
    "logic_physics",
    "instruction_adherence",
    "aesthetic_quality",
]

# All known suitability keys per group (order for consistent output)
DIMENSION_KEYS = [
    ("motion_stability", "subject_consistency_suitability"),
    ("motion_stability", "scene_consistency_suitability"),
    ("motion_stability", "motion_smoothness_suitability"),
    ("motion_stability", "temporal_flickering_suitability"),
    ("motion_stability", "inertial_consistency_suitability"),
    ("logic_physics", "physical_mechanics_suitability"),
    ("logic_physics", "object_permanence_suitability"),
    ("logic_physics", "human_fidelity_suitability"),
    ("logic_physics", "dynamic_degree_suitability"),
    ("instruction_adherence", "semantic_adherence_suitability"),
    ("instruction_adherence", "spatial_relationship_suitability"),
    ("instruction_adherence", "semantic_drift_suitability"),
    ("aesthetic_quality", "composition_framing_suitability"),
    ("aesthetic_quality", "lighting_volumetric_suitability"),
    ("aesthetic_quality", "color_harmony_suitability"),
    ("aesthetic_quality", "structural_gestalt_suitability"),
]

# Difficulty keys: same dimensions, _difficulty suffix
DIFFICULTY_KEYS = [(g, k.replace("_suitability", "_difficulty")) for g, k in DIMENSION_KEYS]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def score_or_zero(rec: dict, group: str, key: str) -> float:
    """Get numeric score; treat null/missing as 0."""
    g = rec.get(group)
    if not isinstance(g, dict):
        return 0.0
    v = g.get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def compute_avg_scores(records: list[dict], keys: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Per (group, key) average; null counts as 0."""
    sums = {}
    counts = {}
    for rec in records:
        for group, key in keys:
            v = score_or_zero(rec, group, key)
            k = (group, key)
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {
        k: (sums[k] / counts[k] if counts[k] else 0.0)
        for k in keys
    }


def main():
    ap = argparse.ArgumentParser(description="Compare average suitability: enhanced vs unenhanced")
    ap.add_argument(
        "--enhanced",
        default=None,
        help="Enhanced judge output (rescored.jsonl)",
    )
    ap.add_argument(
        "--unenhanced",
        default=None,
        help="Unenhanced judge output (judged_review_fixed.jsonl)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write results to this path (CSV). If not set, print to stdout only.",
    )
    args = ap.parse_args()

    enhanced_path = Path(args.enhanced)
    unenhanced_path = Path(args.unenhanced)
    if not enhanced_path.exists():
        print(f"Error: enhanced file not found: {enhanced_path}", file=sys.stderr)
        sys.exit(1)
    if not unenhanced_path.exists():
        print(f"Error: unenhanced file not found: {unenhanced_path}", file=sys.stderr)
        sys.exit(1)

    enhanced_recs = load_jsonl(enhanced_path)
    unenhanced_recs = load_jsonl(unenhanced_path)
    print(f"Enhanced:   {len(enhanced_recs)} records from {enhanced_path.name}")
    print(f"Unenhanced: {len(unenhanced_recs)} records from {unenhanced_path.name}")

    enhanced_suit = compute_avg_scores(enhanced_recs, DIMENSION_KEYS)
    unenhanced_suit = compute_avg_scores(unenhanced_recs, DIMENSION_KEYS)
    enhanced_diff = compute_avg_scores(enhanced_recs, DIFFICULTY_KEYS)
    unenhanced_diff = compute_avg_scores(unenhanced_recs, DIFFICULTY_KEYS)

    # Dimension short name for display
    def dim_name(key: str) -> str:
        return key.replace("_suitability", "").replace("_difficulty", "")

    # ---- Suitability ----
    lines_suit = []
    lines_suit.append("group,dimension,enhanced_avg,unenhanced_avg,diff (enhanced - unenhanced)")
    for (group, key) in DIMENSION_KEYS:
        e, u = enhanced_suit[(group, key)], unenhanced_suit[(group, key)]
        dim = dim_name(key)
        lines_suit.append(f"{group},{dim},{e:.3f},{u:.3f},{e - u:+.3f}")

    group_suit_enhanced = {}
    group_suit_unenhanced = {}
    for (group, key) in DIMENSION_KEYS:
        group_suit_enhanced[group] = group_suit_enhanced.get(group, 0.0) + enhanced_suit[(group, key)]
        group_suit_unenhanced[group] = group_suit_unenhanced.get(group, 0.0) + unenhanced_suit[(group, key)]
    n_dims = {g: sum(1 for gr, _ in DIMENSION_KEYS if gr == g) for g in GROUPS}
    for g in GROUPS:
        n = n_dims[g]
        group_suit_enhanced[g] /= n
        group_suit_unenhanced[g] /= n
    overall_suit_enhanced = sum(enhanced_suit[k] for k in DIMENSION_KEYS) / len(DIMENSION_KEYS)
    overall_suit_unenhanced = sum(unenhanced_suit[k] for k in DIMENSION_KEYS) / len(DIMENSION_KEYS)

    # ---- Difficulty ----
    lines_diff = []
    lines_diff.append("group,dimension,enhanced_avg,unenhanced_avg,diff (enhanced - unenhanced)")
    for (group, key) in DIFFICULTY_KEYS:
        e, u = enhanced_diff[(group, key)], unenhanced_diff[(group, key)]
        dim = dim_name(key)
        lines_diff.append(f"{group},{dim},{e:.3f},{u:.3f},{e - u:+.3f}")

    group_diff_enhanced = {}
    group_diff_unenhanced = {}
    for (group, key) in DIFFICULTY_KEYS:
        group_diff_enhanced[group] = group_diff_enhanced.get(group, 0.0) + enhanced_diff[(group, key)]
        group_diff_unenhanced[group] = group_diff_unenhanced.get(group, 0.0) + unenhanced_diff[(group, key)]
    for g in GROUPS:
        n = n_dims[g]
        group_diff_enhanced[g] /= n
        group_diff_unenhanced[g] /= n
    overall_diff_enhanced = sum(enhanced_diff[k] for k in DIFFICULTY_KEYS) / len(DIFFICULTY_KEYS)
    overall_diff_unenhanced = sum(unenhanced_diff[k] for k in DIFFICULTY_KEYS) / len(DIFFICULTY_KEYS)

    # ---- Category comparison (JSON "categories" field: per-label suitability + difficulty) ----
    enhanced_by_id = {r["prompt_id"]: r for r in enhanced_recs if r.get("prompt_id") is not None}
    unenhanced_by_id = {r["prompt_id"]: r for r in unenhanced_recs if r.get("prompt_id") is not None}
    common_ids = set(enhanced_by_id) & set(unenhanced_by_id)
    all_categories = set()
    for rec in enhanced_recs + unenhanced_recs:
        for c in rec.get("categories") or []:
            if c is not None and str(c).strip():
                all_categories.add(str(c).strip())
    category_order = sorted(all_categories)

    category_lines_csv = []
    cat_header = "category,count,suitability_enhanced,suitability_unenhanced,suitability_diff,difficulty_enhanced,difficulty_unenhanced,difficulty_diff"
    category_lines_csv.append(cat_header)
    print("\n--- Category comparison (categories field from JSON) ---")
    print(cat_header)
    for cat in category_order:
        pids = [pid for pid in common_ids if enhanced_by_id[pid].get("categories") and cat in (enhanced_by_id[pid]["categories"] or [])]
        if not pids:
            continue
        recs_e = [enhanced_by_id[pid] for pid in pids]
        recs_u = [unenhanced_by_id[pid] for pid in pids]
        avg_suit_e = compute_avg_scores(recs_e, DIMENSION_KEYS)
        avg_suit_u = compute_avg_scores(recs_u, DIMENSION_KEYS)
        avg_diff_e = compute_avg_scores(recs_e, DIFFICULTY_KEYS)
        avg_diff_u = compute_avg_scores(recs_u, DIFFICULTY_KEYS)
        suit_e = sum(avg_suit_e[k] for k in DIMENSION_KEYS) / len(DIMENSION_KEYS)
        suit_u = sum(avg_suit_u[k] for k in DIMENSION_KEYS) / len(DIMENSION_KEYS)
        diff_e = sum(avg_diff_e[k] for k in DIFFICULTY_KEYS) / len(DIFFICULTY_KEYS)
        diff_u = sum(avg_diff_u[k] for k in DIFFICULTY_KEYS) / len(DIFFICULTY_KEYS)
        line = f"{cat},{len(pids)},{suit_e:.3f},{suit_u:.3f},{suit_e - suit_u:+.3f},{diff_e:.3f},{diff_u:.3f},{diff_e - diff_u:+.3f}"
        print(line)
        category_lines_csv.append(line)

    # Print to stdout
    print("\n--- Per-dimension average SUITABILITY (null = 0) ---")
    print(lines_suit[0])
    for line in lines_suit[1:]:
        print(line)
    print("\n--- By group SUITABILITY ---")
    print("group,enhanced_avg,unenhanced_avg,diff")
    for g in GROUPS:
        e, u = group_suit_enhanced[g], group_suit_unenhanced[g]
        print(f"{g},{e:.3f},{u:.3f},{e - u:+.3f}")
    print(f"\nOverall SUITABILITY: enhanced={overall_suit_enhanced:.3f}, unenhanced={overall_suit_unenhanced:.3f}, diff={overall_suit_enhanced - overall_suit_unenhanced:+.3f}")

    print("\n--- Per-dimension average DIFFICULTY (null = 0) ---")
    print(lines_diff[0])
    for line in lines_diff[1:]:
        print(line)
    print("\n--- By group DIFFICULTY ---")
    print("group,enhanced_avg,unenhanced_avg,diff")
    for g in GROUPS:
        e, u = group_diff_enhanced[g], group_diff_unenhanced[g]
        print(f"{g},{e:.3f},{u:.3f},{e - u:+.3f}")
    print(f"\nOverall DIFFICULTY: enhanced={overall_diff_enhanced:.3f}, unenhanced={overall_diff_unenhanced:.3f}, diff={overall_diff_enhanced - overall_diff_unenhanced:+.3f}")

    # ---- Group comparison (dimension groups: motion_stability, logic_physics, etc.) ----
    print("\n--- Group comparison (dimension groups) ---")
    group_header = "group,suitability_enhanced,suitability_unenhanced,suitability_diff,difficulty_enhanced,difficulty_unenhanced,difficulty_diff"
    print(group_header)
    group_lines = [group_header]
    for g in GROUPS:
        se, su = group_suit_enhanced[g], group_suit_unenhanced[g]
        de, du = group_diff_enhanced[g], group_diff_unenhanced[g]
        line = f"{g},{se:.3f},{su:.3f},{se - su:+.3f},{de:.3f},{du:.3f},{de - du:+.3f}"
        print(line)
        group_lines.append(line)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Suitability (null = 0)\n")
            f.write("\n".join(lines_suit) + "\n")
            f.write("\n# By group SUITABILITY\n")
            f.write("group,enhanced_avg,unenhanced_avg,diff\n")
            for g in GROUPS:
                e, u = group_suit_enhanced[g], group_suit_unenhanced[g]
                f.write(f"{g},{e:.3f},{u:.3f},{e - u:+.3f}\n")
            f.write(f"\n# Overall SUITABILITY: enhanced={overall_suit_enhanced:.3f}, unenhanced={overall_suit_unenhanced:.3f}, diff={overall_suit_enhanced - overall_suit_unenhanced:+.3f}\n")
            f.write("\n# Difficulty (null = 0)\n")
            f.write("\n".join(lines_diff) + "\n")
            f.write("\n# By group DIFFICULTY\n")
            f.write("group,enhanced_avg,unenhanced_avg,diff\n")
            for g in GROUPS:
                e, u = group_diff_enhanced[g], group_diff_unenhanced[g]
                f.write(f"{g},{e:.3f},{u:.3f},{e - u:+.3f}\n")
            f.write(f"\n# Overall DIFFICULTY: enhanced={overall_diff_enhanced:.3f}, unenhanced={overall_diff_unenhanced:.3f}, diff={overall_diff_enhanced - overall_diff_unenhanced:+.3f}\n")
            f.write("\n# Group comparison (dimension groups)\n")
            f.write("\n".join(group_lines) + "\n")
            f.write("\n# Category comparison (categories field from JSON)\n")
            f.write("\n".join(category_lines_csv) + "\n")
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
