# Importing needed libraries
import pandas as pd
from constants import CLUSTERING_COLS, D_DEG_MIN, D_DEG_MAX, LYAPUNOV_MAX


def drop_missing(df):
    """Drop rows with missing values in clustering columns."""
    before = len(df)
    df = df.dropna(subset=CLUSTERING_COLS).copy()
    print(f"  [drop_missing] Dropped {before - len(df):,} rows. {len(df):,} remaining.")
    return df


def filter_libration(df):
    """Filter to valid libration amplitude range."""
    before = len(df)
    df = df[(df["D_deg"] >= D_DEG_MIN) & (df["D_deg"] <= D_DEG_MAX)].copy()
    print(
        f"  [filter_libration] Removed {before - len(df):,} asteroids. {len(df):,} remaining."
    )
    return df


def filter_unstable(df):
    """Remove chaotic asteroids with high Lyapunov number."""
    before = len(df)
    df = df[df["L"] <= LYAPUNOV_MAX].copy()
    print(
        f"  [filter_unstable] Removed {before - len(df):,} asteroids. {len(df):,} remaining."
    )
    return df


def preprocess(df):
    """Run the full preprocessing pipeline."""
    print("Preprocessing...")
    df = drop_missing(df)
    df = filter_libration(df)
    df = filter_unstable(df)
    print(f"  Preprocessing complete. {len(df):,} asteroids ready for clustering.\n")
    return df


if __name__ == "__main__":
    df = pd.read_csv("tro_syn.csv", dtype={"asteroid_number": str})
    df = df.set_index("asteroid_number")
    df_clean = preprocess(df)
    print(df_clean[CLUSTERING_COLS].head(10))
