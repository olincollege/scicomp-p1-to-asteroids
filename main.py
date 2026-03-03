from src.parser import load_astdys_prop
from src.clustering import run_hcm_search
from config.constants import HIRAYAMA_BENCHMARKS, V_CUTOFF


def main():
    # 1. Ingest Data
    df = load_astdys_prop("data/all.prop")

    # 2. Identify the 'Big Three' Families
    for name, info in HIRAYAMA_BENCHMARKS.items():
        found = run_hcm_search(df, info["seed"], V_CUTOFF)
        print(f"Family: {name} | Members Found: {len(found)}")

        # Note: Compare 'len(found)' to all.fam for 95% completeness check


if __name__ == "__main__":
    main()
