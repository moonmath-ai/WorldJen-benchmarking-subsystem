"""
Print a summary table of semantic adherence scores.

Usage:
    python semantic_adherence/generate_report.py

    # Custom results file:
    python semantic_adherence/generate_report.py \
        --results semantic_adherence/results/semantic_adherence_results_50.json
"""

import argparse
import json
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Summarise semantic adherence results.")
    p.add_argument(
        "--results",
        default="semantic_adherence/results/semantic_adherence_results_50.json",
        help="Path to the results JSON produced by compute_semantic_adherence.py",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.results) as f:
        results = json.load(f)

    print("# Semantic Adherence Report (Cosine Similarity via Gemini Embedding)\n")
    header = f"{'Model':<35} | {'Avg':>6} | {'Min':>6} | {'Max':>6} | {'N':>4}"
    print(header)
    print("-" * len(header))

    ranked = []
    for model_name, entries in results.items():
        if not entries:
            print(f"{model_name:<35} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {0:>4}")
            continue
        vals = [e["score"] for e in entries]
        avg = np.mean(vals)
        ranked.append((model_name, avg, np.min(vals), np.max(vals), len(vals)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    for model_name, avg, mn, mx, n in ranked:
        print(f"{model_name:<35} | {avg:>6.4f} | {mn:>6.4f} | {mx:>6.4f} | {n:>4}")

    if ranked:
        print(f"\nBest model: {ranked[0][0]}  (avg = {ranked[0][1]:.4f})")
        print(f"Worst model: {ranked[-1][0]}  (avg = {ranked[-1][1]:.4f})")


if __name__ == "__main__":
    main()
