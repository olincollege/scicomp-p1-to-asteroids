# Importing in from CONSTANTS FILE
from constants import CLUSTERING_COLS, D_DEG_MIN, D_DEG_MAX, LYAPUNOV_MAX


def preprocess(syn):
    print("\nStep 2: Preprocessing")
    before = len(syn)
    syn = syn.dropna(subset=CLUSTERING_COLS)
    syn = syn[(syn["D_deg"] >= D_DEG_MIN) & (syn["D_deg"] <= D_DEG_MAX)]
    syn = syn[syn["L"] <= LYAPUNOV_MAX].copy()
    print(f"  Kept {len(syn):,} of {before:,} asteroids after filtering")
    return syn
