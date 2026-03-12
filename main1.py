"""
main.py
"""

import matplotlib

matplotlib.use("Agg")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from load import load_data
from preprocess import preprocess
from metric import compute_distance_matrix, EPS
from clustering import cluster
from evaluate import evaluate
from visualize import visualize

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    syn, mem = load_data()
    syn = preprocess(syn)
    D, X = compute_distance_matrix(syn)
    labels = cluster(D, eps=EPS)
    results = evaluate(syn, labels, mem)
    visualize(syn, labels, results)

    results.to_csv("results/evaluation.csv", index=False)
    print("\nDone. Results saved to results/")
