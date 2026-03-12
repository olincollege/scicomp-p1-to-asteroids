import numpy as np
from scipy.spatial.distance import pdist, squareform
from constants import K_A, K_E, K_I, GM_AU_DAY, AU_TO_KM, CLUSTERING_COLS

# Jupiter semi-major axis [AU]
A_JUPITER = 5.2044

# Mean motion of Jupiter [rad/day]
N_JUPITER = np.sqrt(GM_AU_DAY / A_JUPITER**3)

# Scale factor n*a in m/s
NA_MS = N_JUPITER * A_JUPITER * AU_TO_KM / 86400.0 * 1000.0

# Default DBSCAN eps [m/s]
EPS = 50.0


def zappala_distance(u, v):
    """Zappalà velocity distance between two asteroids in m/s."""
    da1, e1, s1 = u
    da2, e2, s2 = v
    return NA_MS * np.sqrt(
        K_A * ((da1 - da2) / A_JUPITER) ** 2
        + K_E * (e1 - e2) ** 2
        + K_I * (s1 - s2) ** 2
    )


def compute_distance_matrix(syn):
    """
    Compute full pairwise Zappalà distance matrix.

    Parameters
    ----------
    syn : pd.DataFrame with columns da_AU, e_p, sin_i_p

    Returns
    -------
    D : np.ndarray — square distance matrix in m/s
    X : np.ndarray — feature matrix used
    """
    print("\nStep 3: Computing Zappalà distance matrix")
    X = syn[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    print(f"  Computing distances for {len(X):,} asteroids...")
    D = squareform(pdist(X, metric=zappala_distance))
    print(f"  Done.")
    return D, X
