import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from constants import (
    LINKAGE_METHOD,
    N_MONTE_CARLO,
    QRL_SIGNIFICANCE_PERCENTILE,
    MIN_FAMILY_SIZE,
    CLUSTERING_COLS,
)
from metric import compute_distance_matrix


def _shuffle_elements(X):
    """
    Shuffle each proper element column independently.

    This preserves the marginal distributions of D_deg, e_p, sin_i_p
    but destroys any real correlations (i.e. real families).
    """
    X_shuf = X.copy()
    for col in range(X.shape[1]):
        np.random.shuffle(X_shuf[:, col])
    return X_shuf


def _largest_cluster_size(X, v_cutoff):
    """Run clustering on X and return the largest cluster size found."""
    condensed = compute_distance_matrix(X)
    Z = linkage(condensed, method=LINKAGE_METHOD)
    labels = fcluster(Z, t=v_cutoff, criterion="distance")

    unique, counts = np.unique(labels, return_counts=True)
    family_counts = counts[counts >= MIN_FAMILY_SIZE]

    return int(np.max(family_counts)) if len(family_counts) > 0 else 0


def compute_qrl(X, v_cutoff, n_trials=N_MONTE_CARLO):
    """
    Estimate the quasi-random level (QRL) background distribution.

    Parameters
    ----------
    X : np.ndarray of shape (n, 3)
        Clustering feature matrix [D_deg, e_p, sin_i_p].
    v_cutoff : float
        Velocity cutoff in m/s.
    n_trials : int
        Number of Monte Carlo shuffles.

    Returns
    -------
    threshold : float
        Cluster size at the QRL_SIGNIFICANCE_PERCENTILE of the background.
        Real families with size > threshold are statistically significant.
    background_sizes : list of int
        Largest cluster size found in each Monte Carlo trial.
    """
    print(
        f"  [significance] Running {n_trials} Monte Carlo trials "
        f"at v_cutoff={v_cutoff:.0f} m/s..."
    )
    background_sizes = []

    for i in range(n_trials):
        X_shuf = _shuffle_elements(X)
        max_size = _largest_cluster_size(X_shuf, v_cutoff)
        background_sizes.append(max_size)
        if (i + 1) % 10 == 0:
            print(f"    trial {i+1}/{n_trials} done")

    threshold = float(np.percentile(background_sizes, QRL_SIGNIFICANCE_PERCENTILE))
    print(
        f"  [significance] QRL threshold (p={QRL_SIGNIFICANCE_PERCENTILE}): "
        f"{threshold:.1f} members"
    )
    return threshold, background_sizes


def filter_significant(families, qrl_threshold):
    """
    Keep only families that exceed the QRL significance threshold.

    Parameters
    ----------
    families : dict
        {family_id -> pd.DataFrame} from hcm.get_families()
    qrl_threshold : float
        Minimum cluster size to be considered significant.

    Returns
    -------
    dict
        Filtered families dict containing only significant families.
    """
    significant = {
        fid: members
        for fid, members in families.items()
        if len(members) > qrl_threshold
    }
    print(
        f"  [significance] {len(significant)} / {len(families)} families "
        f"exceed QRL threshold of {qrl_threshold:.1f} members."
    )
    return significant


if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess
    from metric import compute_distance_matrix
    from hcm import build_linkage, cut_dendrogram, get_families
    from constants import V_CUTOFF_DEFAULT

    df = pd.read_csv("tro_syn.csv", dtype={"asteroid_number": str})
    df_clean, _ = preprocess(df)
    X = df_clean[CLUSTERING_COLS].to_numpy(dtype=np.float64)

    condensed = compute_distance_matrix(X)
    Z = build_linkage(condensed)
    labels = cut_dendrogram(Z, v_cutoff=V_CUTOFF_DEFAULT)
    families = get_families(df_clean, labels)

    threshold, bg = compute_qrl(X, v_cutoff=V_CUTOFF_DEFAULT, n_trials=20)
    sig_families = filter_significant(families, threshold)
    print(f"\nSignificant families: {len(sig_families)}")
