import numpy as np
import pandas as pd

COMPLETENESS_THRESHOLD = 0.95

# The 7 actual Trojan families in AstDys ground truth
TARGET_FAMILIES = [3548, 8060, 624, 9799, 17492, 291316, 222861]


def evaluate(df_clean, labels, family_membership):
    """
    For each known AstDys Trojan family, find the best matching cluster
    and compute completeness and contamination.

    Parameters
    ----------
    df_clean : pd.DataFrame  (indexed by asteroid_number as str)
    labels   : np.ndarray of cluster labels (0 = background)
    family_membership : pd.DataFrame (indexed by asteroid_number as str)

    Returns
    -------
    pd.DataFrame with completeness/contamination per family
    """
    df = df_clean.copy()
    df["cluster"] = labels

    rows = []
    for fam_id in TARGET_FAMILIES:
        # ground truth members of this family that exist in tro_syn
        gt = family_membership[family_membership["family1"] == fam_id]
        gt_ids = set(gt.index.astype(str)) & set(df.index.astype(str))
        n_astdys = len(gt_ids)
        if n_astdys == 0:
            continue

        # find which cluster contains the most ground truth members
        gt_subset = df.loc[df.index.isin(gt_ids)]
        cluster_counts = gt_subset["cluster"].value_counts()
        cluster_counts = cluster_counts[cluster_counts.index != 0]  # ignore background

        if cluster_counts.empty:
            rows.append(
                {
                    "family_id": fam_id,
                    "best_cluster": None,
                    "n_astdys": n_astdys,
                    "n_cluster": 0,
                    "n_overlap": 0,
                    "completeness": 0.0,
                    "contamination": 1.0,
                }
            )
            continue

        best_cluster = cluster_counts.idxmax()
        cluster_ids = set(df[df["cluster"] == best_cluster].index.astype(str))
        n_overlap = len(gt_ids & cluster_ids)
        n_cluster = len(cluster_ids)

        rows.append(
            {
                "family_id": fam_id,
                "best_cluster": best_cluster,
                "n_astdys": n_astdys,
                "n_cluster": n_cluster,
                "n_overlap": n_overlap,
                "completeness": round(n_overlap / n_astdys, 4),
                "contamination": round(1 - n_overlap / n_cluster, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("completeness", ascending=False)


def check_benchmark(results):
    """Print benchmark results and return True if passed."""
    passing = results[results["completeness"] >= COMPLETENESS_THRESHOLD]
    n_pass = len(passing)
    target = len(TARGET_FAMILIES)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(
        f"  Target   : {target} families at >= {COMPLETENESS_THRESHOLD*100:.0f}% completeness"
    )
    print(f"  Achieved : {n_pass} families passing")
    print(f"  Status   : {'PASSED' if n_pass >= target else 'FAILED'}")
    print("=" * 60)
    print(
        results[
            [
                "family_id",
                "n_astdys",
                "n_cluster",
                "n_overlap",
                "completeness",
                "contamination",
            ]
        ].to_string(index=False)
    )
    return n_pass >= target


if __name__ == "__main__":
    from load_data import load_proper_elements, load_family_membership
    from preprocess import preprocess
    from metric import compute_distance_matrix
    from hcm import build_linkage, cut_dendrogram, get_families
    from constants import CLUSTERING_COLS

    df = load_proper_elements()
    family_membership = load_family_membership()
    df_clean = preprocess(df)

    X = df_clean[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    condensed = compute_distance_matrix(X)
    Z = build_linkage(condensed)

    # sweep cutoffs to find best
    best_results = None
    best_passing = -1
    best_vc = 100

    for vc in [50, 75, 100, 125, 150, 175, 200, 250, 300]:
        labels = cut_dendrogram(Z, v_cutoff=vc)
        results = evaluate(df_clean, labels, family_membership)
        n_pass = len(results[results["completeness"] >= COMPLETENESS_THRESHOLD])
        print(f"  v_cutoff={vc:>4} m/s → {n_pass} families passing")
        if n_pass > best_passing:
            best_passing = n_pass
            best_results = results
            best_vc = vc

    print(f"\nBest cutoff: {best_vc} m/s")
    check_benchmark(best_results)
