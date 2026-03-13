# Importing in from CONSTANTS FILE
from constants import CLUSTERING_COLS, D_DEG_MIN, D_DEG_MAX, LYAPUNOV_MAX


def preprocess(syn):
    """
    Filter and clean the proper elements dataset before clustering.

    Args:
        syn (pd.DataFrame): Raw proper elements DataFrame loaded from tro_syn.csv. Must
                            be in a column

    Returns:
        syn (pd.DataFrame): Filtered DataFrame ready for distance matrix computation.
    """
    print("\nStep 2: Preprocessing")
    before = len(syn)

    # remove asteroids with missing orbital elements
    syn = syn.dropna(subset=CLUSTERING_COLS)

    # remove weakly trapped and near-Lagrange-point Trojans
    syn = syn[(syn["D_deg"] >= D_DEG_MIN) & (syn["D_deg"] <= D_DEG_MAX)]

    # remove dynamically unstable asteroids
    syn = syn[syn["L"] <= LYAPUNOV_MAX].copy()

    print(f"  Kept {len(syn):,} of {before:,} asteroids after filtering")
    return syn
