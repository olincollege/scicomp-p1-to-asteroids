# Importing Files and Libraries
import numpy as np
from scipy.spatial.distance import pdist, squareform
from constants import NA_MS, K_A, K_E, K_I, A_JUPITER, CLUSTERING_COLS


def zappala_distance(u, v):
    """Zappalà et al. (1990) Eq. 1 velocity metric in m/s."""
    da1, e1, s1 = u
    da2, e2, s2 = v
    return NA_MS * np.sqrt(
        K_A * ((da1 - da2) / A_JUPITER) ** 2
        + K_E * (e1 - e2) ** 2
        + K_I * (s1 - s2) ** 2
    )


def compute_distances(syn):
    print("\nStep 3: Computing Zappalà distance matrix")
    X = syn[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    print(f"  Computing {len(X):,} x {len(X):,} distance matrix...")
    D = squareform(pdist(X, metric=zappala_distance))
    print(f"  Done.")
    return D, X
