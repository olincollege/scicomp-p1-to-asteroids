# Loading in Pandas Library
import pandas as pd


def load_data(syn_path="tro_syn.csv", members_path="tro_members.csv"):
    print("Step 1: Loading data")
    syn = pd.read_csv(syn_path, dtype={"asteroid_number": str})
    syn = syn.set_index("asteroid_number")
    syn.index = syn.index.astype(str)
    print(f"  Loaded {len(syn):,} asteroids from {syn_path}")

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
