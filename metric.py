import numpy as np
from scipy.spatial.distance import pdist, squareform
from constants import K_A, K_E, K_I, GM_SUN_AU_DAY, AU_TO_KM

# Jupiter's semi-major axis [AU] — Trojans orbit at ~5.2 AU
A_JUPITER = 5.2044

# Mean motion of Jupiter [rad/day]
N_JUPITER = np.sqrt(GM_SUN_AU_DAY / A_JUPITER**3)

# Scale factor: n * a in km/s  (converts metric output to m/s)
# n [rad/day] * a [AU] * AU_TO_KM [km/AU] / DAY_TO_SEC [s/day]
_NA_KMS = N_JUPITER * A_JUPITER * AU_TO_KM / 86400.0
_NA_MS = _NA_KMS * 1000.0  # convert to m/s


def zappala_distance(ast1, ast2):
    """
    Compute the Zappalà velocity distance between two asteroids.

    Parameters
    ----------
    ast1, ast2 : array-like of shape (3,)
        Each array contains [D_deg, e_p, sin_i_p] for one asteroid.

    Returns
    -------
    float
        Distance in m/s.

    Notes
    -----
    Formula (Zappalà et al. 1990, Eq. 1):
        d = n*a * sqrt(k_a*(dD/D_ref)^2 + k_e*(de)^2 + k_i*(d_sini)^2)

    For Trojans, D_deg / D_ref replaces da/a from the main belt formulation,
    where D_ref is the mean libration amplitude of the two asteroids.
    """
    D1, e1, sini1 = ast1
    D2, e2, sini2 = ast2

    D_ref = 0.5 * (D1 + D2)  # mean libration amplitude [deg]

    # avoid division by zero
    if D_ref == 0:
        D_ref = 1e-10

    dD_term = K_A * ((D1 - D2) / D_ref) ** 2
    de_term = K_E * (e1 - e2) ** 2
    dsini_term = K_I * (sini1 - sini2) ** 2

    return _NA_MS * np.sqrt(dD_term + de_term + dsini_term)


def compute_distance_matrix(X):
    """
    Compute the full pairwise Zappalà distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n, 3)
        Columns: [D_deg, e_p, sin_i_p]

    Returns
    -------
    np.ndarray of shape (n*(n-1)/2,)
        Condensed distance matrix (upper triangle) in m/s.
        Pass directly to scipy.cluster.hierarchy.linkage.
    """
    print(f"  [metric] Computing pairwise distances for {len(X):,} asteroids...")

    def _dist(u, v):
        return zappala_distance(u, v)

    condensed = pdist(X, metric=_dist)
    print(f"  [metric] Done. {len(condensed):,} pairwise distances computed.")
    return condensed


if __name__ == "__main__":
    # quick sanity check
    ast1 = np.array([10.0, 0.10, 0.20])
    ast2 = np.array([10.5, 0.11, 0.21])
    d = zappala_distance(ast1, ast2)
    print(f"Test distance between two similar asteroids: {d:.2f} m/s")

    ast3 = np.array([5.0, 0.05, 0.10])
    d2 = zappala_distance(ast1, ast3)
    print(f"Test distance between two different asteroids: {d2:.2f} m/s")
