# Importing from Libraries and Files
import numpy as np
from sklearn.cluster import DBSCAN
from constants import EPS, MIN_SAMPLES


def cluster(D, eps=EPS):
    print(f"\nStep 4: DBSCAN clustering (eps={eps} m/s, min_samples={MIN_SAMPLES})")
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(D)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"  Found {n_clusters} clusters, {n_noise:,} noise points")
    return labels
