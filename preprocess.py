# Importing in our needed libraries

import pandas as pd
import numpy as np
from constants import (
    CLUSTERING_COLS,
    D_DEG_MIN,
    D_DEG_MAX,
    LYAPUNOV_MAX,
)


# Dropping in values
def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in clustering columns."""
    before = len(df)
    df = df.dropna(subset=CLUSTERING_COLS).copy()
    after = len(df)
    print(
        f"  [drop_missing] Dropped {before - after:,} rows with NaN values. "
        f"{after:,} remaining."
    )
    return df


# Filter the amplitude
def filter_libration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to valid libration amplitude range.

    Asteroids with D_deg outside [D_DEG_MIN, D_DEG_MAX] are weakly trapped
    Trojans or have unreliable proper elements.

    Args:
        df : pd.DataFrame
        Output of drop_missing()

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    before = len(df)
    df = df[(df["D_deg"] >= D_DEG_MIN) & (df["D_deg"] <= D_DEG_MAX)].copy()
    after = len(df)
    print(
        f"  [filter_libration] Removed {before - after:,} asteroids outside "
        f"D_deg range [{D_DEG_MIN}, {D_DEG_MAX}]. {after:,} remaining."
    )
    return df


def filter_unstable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove chaotic/unstable asteroids with high Lyapunov number.

    The L column in tro_syn indicates orbital stability. High values mean
    the proper elements are unreliable and will pollute clustering results.

    Parameters
    ----------
    df : pd.DataFrame
        Output of filter_libration()

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only stable asteroids.
    """
    before = len(df)
    df = df[df["L"] <= LYAPUNOV_MAX].copy()
    after = len(df)
    print(
        f"  [filter_unstable] Removed {before - after:,} unstable asteroids "
        f"(L > {LYAPUNOV_MAX}). {after:,} remaining."
    )
    return df


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize clustering columns to zero mean and unit variance.

    Without normalization D_deg (range ~1-30 deg) would dominate over
    e_p and sin_i_p (range ~0-0.6) in the distance calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Output of filter_unstable()

    Returns
    -------
    df : pd.DataFrame
        DataFrame with normalized clustering columns added as
        D_deg_norm, e_p_norm, sin_i_p_norm

    stats : dict
        Mean and std for each column — needed to invert normalization later.
    """
    stats = {}
    for col in CLUSTERING_COLS:
        mean = df[col].mean()
        std = df[col].std()
        df[col + "_norm"] = (df[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
        print(f"  [normalize] {col}: mean={mean:.4f}, std={std:.4f}")

    return df, stats


# =============================================================================
# FULL PIPELINE
# =============================================================================


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw proper elements DataFrame from load_data.py

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame ready for clustering.
        Contains original columns plus normalized columns:
        D_deg_norm, e_p_norm, sin_i_p_norm

    stats : dict
        Normalization statistics (mean, std) per clustering column.
    """
    print("Preprocessing...")
    df = drop_missing(df)
    df = filter_libration(df)
    df = filter_unstable(df)
    df, stats = normalize(df)
    print(f"  Preprocessing complete. {len(df):,} asteroids ready for clustering.\n")
    return df, stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # load raw data
    df_raw = pd.read_csv("tro_syn.csv", dtype={"asteroid_number": str})

    # run preprocessing
    df_clean, stats = preprocess(df_raw)

    # save cleaned data
    df_clean.to_csv("tro_syn_clean.csv", index=False)
    print("Saved cleaned data to tro_syn_clean.csv")
    print(df_clean[CLUSTERING_COLS + [c + "_norm" for c in CLUSTERING_COLS]].head(10))
