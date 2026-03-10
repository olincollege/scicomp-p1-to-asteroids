import pandas as pd


def load_proper_elements(filepath="tro_syn.csv"):
    """Load Trojan proper elements from tro_syn.csv."""
    df = pd.read_csv(filepath, dtype={"asteroid_number": str})
    print(f"Loaded {len(df):,} asteroids from {filepath}")
    return df


def load_family_membership(filepath="tro_members.csv"):
    """Load AstDys family membership ground truth from tro_members.csv."""
    df = pd.read_csv(filepath, dtype={"asteroid_number": str}, low_memory=False)

    # convert family1 to int, coerce errors to 0
    df["family1"] = pd.to_numeric(df["family1"], errors="coerce").fillna(0).astype(int)

    # keep only asteroids assigned to a family
    df = df[df["family1"] > 0].copy()

    print(
        f"Loaded {len(df):,} family members across "
        f"{df['family1'].nunique()} families from {filepath}"
    )
    return df


if __name__ == "__main__":
    proper_elements = load_proper_elements()
    family_membership = load_family_membership()

    print("\nProper elements sample:")
    print(proper_elements.head())

    print("\nFamily membership sample:")
    print(family_membership.head())
