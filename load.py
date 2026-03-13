# Loading in Pandas Library
import pandas as pd


def load_data(syn_path="tro_syn.csv", members_path="tro_members.csv"):
    """
    Load proper elements and AstDys family membership data from CSV files.

    Args:
        syn_path (str): Path to the proper elements CSV. Default: 'tro_syn.csv'.
        members_path (str): Path to the AstDys membership CSV. Default: 'tro_members.csv'.

    Returns:
        syn (pd.DataFrame): Proper elements indexed by asteroid_number.
        mem (pd.DataFrame): Family membership indexed by asteroid_number.
    """
    print("Step 1: Loading data")

    # load proper elements and index by asteroid number
    syn = pd.read_csv(syn_path, dtype={"asteroid_number": str})
    syn = syn.set_index("asteroid_number")
    syn.index = syn.index.astype(str)
    print(f"  Loaded {len(syn):,} asteroids from {syn_path}")

    # load membership file, drop unclassified asteroid
    mem = pd.read_csv(members_path, dtype={"asteroid_number": str}, low_memory=False)
    mem["family1"] = (
        pd.to_numeric(mem["family1"], errors="coerce").fillna(0).astype(int)
    )
    mem = mem[mem["family1"] > 0].set_index("asteroid_number")
    mem.index = mem.index.astype(str)
    print(
        f"  Loaded {len(mem):,} family members across {mem['family1'].nunique()} families"
    )
    return syn, mem
