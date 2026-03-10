import os
import argparse
import numpy as np
import pandas as pd

from load_data import load_proper_elements, load_family_membership
from preprocess import preprocess
from metric import compute_distance_matrix
from hcm import build_linkage, cut_dendrogram, get_families, plot_dendrogram
from significance import compute_qrl, filter_significant
from evaluate import (
    match_clusters_to_families,
    compute_completeness_contamination,
    check_benchmark,
)
from visualize import plot_proper_elements, plot_completeness
from constants import V_CUTOFF_DEFAULT, CLUSTERING_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Asteroid Family Identification")
    parser.add_argument(
        "--vcutoff",
        type=float,
        default=V_CUTOFF_DEFAULT,
        help="Velocity cutoff in m/s (default: 100)",
    )
    parser.add_argument(
        "--skip-qrl",
        action="store_true",
        help="Skip the Monte Carlo QRL significance test (faster)",
    )
    parser.add_argument(
        "--syn", type=str, default="tro_syn.csv", help="Path to tro_syn.csv"
    )
    parser.add_argument(
        "--members", type=str, default="tro_members.csv", help="Path to tro_members.csv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    print("=" * 55)
    print("ASTEROID FAMILY IDENTIFICATION")
    print("Zappalà HCM — Trojan Asteroids")
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Step 1: Loading data")
    df = load_proper_elements(args.syn)
    family_membership = load_family_membership(args.members)
    print()

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("Step 2: Preprocessing")
    df_clean, norm_stats = preprocess(df)
    print()

    # ------------------------------------------------------------------
    # 3. Compute distance matrix
    # ------------------------------------------------------------------
    print("Step 3: Computing Zappalà distances")
    X = df_clean[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    condensed = compute_distance_matrix(X)
    print()

    # ------------------------------------------------------------------
    # 4. Hierarchical clustering
    # ------------------------------------------------------------------
    print("Step 4: Hierarchical clustering")
    Z = build_linkage(condensed)
    labels = cut_dendrogram(Z, v_cutoff=args.vcutoff)
    families = get_families(df_clean, labels)
    print()

    # ------------------------------------------------------------------
    # 5. QRL significance test
    # ------------------------------------------------------------------
    if not args.skip_qrl:
        print("Step 5: QRL significance test")
        qrl_threshold, _ = compute_qrl(X, v_cutoff=args.vcutoff)
        families = filter_significant(families, qrl_threshold)
        print()
    else:
        print("Step 5: Skipping QRL significance test\n")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("Step 6: Evaluating against AstDys ground truth")
    matches = match_clusters_to_families(families, family_membership)
    results = compute_completeness_contamination(families, family_membership, matches)
    passed = check_benchmark(results)
    results.to_csv("results/evaluation.csv", index=False)
    print(f"\n  Full results saved to results/evaluation.csv")
    print()

    # ------------------------------------------------------------------
    # 7. Visualize
    # ------------------------------------------------------------------
    print("Step 7: Generating plots")
    plot_dendrogram(Z, save_path="results/dendrogram.png")
    plot_proper_elements(df_clean, labels, save_path="results/proper_elements.png")
    plot_completeness(results, save_path="results/completeness.png")
    print()

    print("=" * 55)
    print(f"Pipeline complete. Results in results/")
    print("=" * 55)


if __name__ == "__main__":
    main()
