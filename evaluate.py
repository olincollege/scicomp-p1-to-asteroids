import pandas as pd
import numpy as np
from constants import TARGET_FAMILIES, COMPLETENESS_THRESHOLD


def match_clusters_to_families(families, family_membership):
    """
    Match each discovered cluster to the best-matching AstDys family.

    For each cluster, find which AstDys family has the most overlap.

    Parameters
    ----------
    families : dict
        {family_id -> pd.DataFrame} from hcm.get_families()
        Each DataFrame is indexed by asteroid_number.
    family_membership : pd.DataFrame
        AstDys ground truth from load_data.load_family_membership()
        Indexed by asteroid_number, contains 'family1' column.

    Returns
    -------
    matches : dict
        {cluster_id -> astdys_family_id} best match for each cluster.
    """
    matches = {}
    for cid, members in families.items():
        member_ids = set(members.index.astype(str))
        gt_ids = set(family_membership.index.astype(str))
        overlap = member_ids & gt_ids

        if not overlap:
            matches[cid] = None
            continue

        # find which AstDys family has the most members in this cluster
        gt_subset = family_membership.loc[family_membership.index.isin(overlap)]
        best_family = gt_subset["family1"].value_counts().idxmax()
        matches[cid] = best_family

    return matches


def compute_completeness_contamination(families, family_membership, matches):
    """
    Compute completeness and contamination for each matched family.

    Parameters
    ----------
    families : dict
        {cluster_id -> pd.DataFrame}
    family_membership : pd.DataFrame
        AstDys ground truth.
    matches : dict
        {cluster_id -> astdys_family_id} from match_clusters_to_families()

    Returns
    -------
    results : pd.DataFrame
        One row per matched family with columns:
        cluster_id, astdys_family_id, n_cluster, n_astdys,
        n_overlap, completeness, contamination
    """
    rows = []
    for cid, astdys_fid in matches.items():
        if astdys_fid is None:
            continue

        cluster_ids = set(families[cid].index.astype(str))
        astdys_ids = set(
            family_membership[family_membership["family1"] == astdys_fid].index.astype(
                str
            )
        )

        overlap = cluster_ids & astdys_ids
        n_overlap = len(overlap)
        n_cluster = len(cluster_ids)
        n_astdys = len(astdys_ids)
        completeness = n_overlap / n_astdys if n_astdys > 0 else 0.0
        contamination = 1 - (n_overlap / n_cluster) if n_cluster > 0 else 0.0

        rows.append(
            {
                "cluster_id": cid,
                "astdys_family_id": astdys_fid,
                "n_cluster": n_cluster,
                "n_astdys": n_astdys,
                "n_overlap": n_overlap,
                "completeness": round(completeness, 4),
                "contamination": round(contamination, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("completeness", ascending=False)


def check_benchmark(results):
    """
    Check whether the 95% completeness benchmark is met for 8 families.

    Parameters
    ----------
    results : pd.DataFrame
        Output of compute_completeness_contamination()

    Returns
    -------
    bool
        True if benchmark is passed.
    """
    passing = results[results["completeness"] >= COMPLETENESS_THRESHOLD]
    n_pass = len(passing)
    target = len(TARGET_FAMILIES)  # 8

    print("\n" + "=" * 55)
    print("BENCHMARK RESULTS")
    print("=" * 55)
    print(
        f"  Target   : {target} families at >= "
        f"{COMPLETENESS_THRESHOLD*100:.0f}% completeness"
    )
    print(f"  Achieved : {n_pass} families")
    print(f"  Status   : {'PASSED' if n_pass >= target else 'FAILED'}")
    print("=" * 55)
    print(
        results[
            [
                "astdys_family_id",
                "n_cluster",
                "n_astdys",
                "completeness",
                "contamination",
            ]
        ].to_string(index=False)
    )
    return n_pass >= target


if __name__ == "__main__":
    import pandas as pd
    from load_data import load_proper_elements, load_family_membership
    from preprocess import preprocess
    from metric import compute_distance_matrix
    from hcm import build_linkage, cut_dendrogram, get_families
    from constants import V_CUTOFF_DEFAULT, CLUSTERING_COLS

    df = load_proper_elements()
    family_membership = load_family_membership()
    df_clean, _ = preprocess(df)

    X = df_clean[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    condensed = compute_distance_matrix(X)
    Z = build_linkage(condensed)
    labels = cut_dendrogram(Z, v_cutoff=V_CUTOFF_DEFAULT)
    families = get_families(df_clean, labels)

    matches = match_clusters_to_families(families, family_membership)
    results = compute_completeness_contamination(families, family_membership, matches)
    check_benchmark(results)
