# Importing from Libraries and Files
import numpy as np
from sklearn.cluster import DBSCAN
from constants import EPS, MIN_SAMPLES


def cluster(D, eps=EPS):
    """
    Cluster asteroids using DBSCAN on a precomputed Zappalà distance matrix.

    Args:
        D (array): Symmetric pairwise distance matrix in m/s (Zappalà metric).
        eps (float): Neighbourhood radius in m/s. Asteroids within this velocity distance
                     of each other are considered neighbours. Default: EPS (50.0 m/s).

    Returns:
        labels (array): Cluster label for each asteroid. -1 indicates noise (no family).
    """
    print(f"\nStep 4: DBSCAN clustering (eps={eps} m/s, min_samples={MIN_SAMPLES})")
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(D)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"  Found {n_clusters} clusters, {n_noise:,} noise points")
    return labels
