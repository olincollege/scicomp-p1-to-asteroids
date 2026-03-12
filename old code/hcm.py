import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from constants import (
    LINKAGE_METHOD,
    V_CUTOFF_DEFAULT,
    V_CUTOFF_MIN,
    V_CUTOFF_MAX,
    V_CUTOFF_STEP,
    MIN_FAMILY_SIZE,
    FIG_DPI,
)


def build_linkage(condensed_distances):
    """
    Build the hierarchical clustering linkage matrix.

    Parameters
    ----------
    condensed_distances : np.ndarray
        Condensed distance matrix from metric.compute_distance_matrix()

    Returns
    -------
    Z : np.ndarray
        Linkage matrix from scipy — input to fcluster and dendrogram.
    """
    print(f"  [hcm] Building linkage matrix (method='{LINKAGE_METHOD}')...")
    Z = linkage(condensed_distances, method=LINKAGE_METHOD)
    print(f"  [hcm] Linkage matrix built. Shape: {Z.shape}")
    return Z


def cut_dendrogram(Z, v_cutoff=V_CUTOFF_DEFAULT):
    """
    Cut the dendrogram at a given velocity threshold.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from build_linkage()
    v_cutoff : float
        Velocity cutoff in m/s. Clusters with all pairwise distances
        below this threshold are grouped into one family.

    Returns
    -------
    labels : np.ndarray of shape (n,)
        Cluster label for each asteroid. Label 0 = background (singleton).
    """
    raw_labels = fcluster(Z, t=v_cutoff, criterion="distance")

    # relabel small clusters as background (label = 0)
    labels = np.zeros_like(raw_labels)
    unique, counts = np.unique(raw_labels, return_counts=True)
    family_id = 1
    for uid, count in zip(unique, counts):
        if count >= MIN_FAMILY_SIZE:
            labels[raw_labels == uid] = family_id
            family_id += 1

    n_families = family_id - 1
    n_background = np.sum(labels == 0)
    print(
        f"  [hcm] v_cutoff={v_cutoff:.0f} m/s → "
        f"{n_families} families, {n_background:,} background asteroids"
    )
    return labels


def sweep_cutoffs(Z, df):
    """
    Sweep velocity cutoffs and record number of families at each step.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix.
    df : pd.DataFrame
        Preprocessed proper elements DataFrame.

    Returns
    -------
    results : list of dict
        Each dict has keys: v_cutoff, n_families, labels
    """
    cutoffs = np.arange(V_CUTOFF_MIN, V_CUTOFF_MAX + V_CUTOFF_STEP, V_CUTOFF_STEP)
    results = []
    print(f"  [hcm] Sweeping cutoffs {V_CUTOFF_MIN:.0f}–{V_CUTOFF_MAX:.0f} m/s...")
    for vc in cutoffs:
        labels = cut_dendrogram(Z, v_cutoff=vc)
        n_fam = len(np.unique(labels[labels > 0]))
        results.append({"v_cutoff": vc, "n_families": n_fam, "labels": labels})
    return results


def get_families(df, labels):
    """
    Return a dict mapping family_id -> DataFrame of member asteroids.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed proper elements DataFrame.
    labels : np.ndarray
        Cluster labels from cut_dendrogram().

    Returns
    -------
    dict : {family_id (int) -> pd.DataFrame}
    """
    df = df.copy()
    df["family_id"] = labels
    families = {}
    for fid in sorted(np.unique(labels)):
        if fid == 0:
            continue
        families[fid] = df[df["family_id"] == fid]
    return families


def plot_dendrogram(Z, max_display=50, save_path=None):
    """Plot the top of the dendrogram for visual inspection."""
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=max_display,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title("Asteroid Family Dendrogram (Zappalà HCM)")
    ax.set_xlabel("Asteroid cluster")
    ax.set_ylabel("Velocity distance [m/s]")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI)
        print(f"  [hcm] Dendrogram saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess
    from metric import compute_distance_matrix
    from constants import CLUSTERING_COLS

    df = pd.read_csv("tro_syn.csv", dtype={"asteroid_number": str})
    df_clean, _ = preprocess(df)

    X = df_clean[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    condensed = compute_distance_matrix(X)

    Z = build_linkage(condensed)
    labels = cut_dendrogram(Z, v_cutoff=V_CUTOFF_DEFAULT)
    families = get_families(df_clean, labels)

    print(f"\nFound {len(families)} families at {V_CUTOFF_DEFAULT} m/s cutoff")
    for fid, members in list(families.items())[:5]:
        print(f"  Family {fid}: {len(members)} members")
