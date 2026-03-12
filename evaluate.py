# Importing in Libraries and Files
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from constants import TARGET_FAMILIES, COMPLETENESS_TARGET


def evaluate(syn, labels, mem):
    print("\nStep 5: Evaluating against AstDys ground truth")
    syn = syn.copy()
    syn["cluster"] = labels

    rows = []
    for fam_id in TARGET_FAMILIES:
        gt_ids = set(mem[mem["family1"] == fam_id].index) & set(syn.index)
        n_astdys = len(gt_ids)
        if n_astdys == 0:
            continue

        gt_sub = syn.loc[syn.index.isin(gt_ids)]
        cc = gt_sub["cluster"].value_counts()
        cc = cc[cc.index != -1]

        if cc.empty:
            rows.append(
                {
                    "family_id": fam_id,
                    "n_astdys": n_astdys,
                    "n_cluster": 0,
                    "n_overlap": 0,
                    "completeness": 0.0,
                    "contamination": 1.0,
                }
            )
            continue

        best = cc.idxmax()
        cluster_ids = set(syn[syn["cluster"] == best].index)
        n_overlap = len(gt_ids & cluster_ids)
        n_cluster = len(cluster_ids)

        rows.append(
            {
                "family_id": fam_id,
                "n_astdys": n_astdys,
                "n_cluster": n_cluster,
                "n_overlap": n_overlap,
                "completeness": round(n_overlap / n_astdys, 4),
                "contamination": round(1 - n_overlap / n_cluster, 4),
            }
        )

    results = pd.DataFrame(rows).sort_values("completeness", ascending=False)

    passing = results[results["completeness"] >= COMPLETENESS_TARGET]
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(
        f"  Target   : {len(TARGET_FAMILIES)} families at >= {COMPLETENESS_TARGET*100:.0f}% completeness"
    )
    print(f"  Achieved : {len(passing)} families passing")
    print(
        f"  Status   : {'PASSED' if len(passing) >= len(TARGET_FAMILIES) else 'FAILED'}"
    )
    print(f"{'='*60}")
    print(results.to_string(index=False))
    return results
