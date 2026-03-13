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
    """
    Evaluate clustering results against the AstDys ground truth membership. For each target
    family, finds the best-matching cluster by maximum asteroid ID overlap, then computes
    completeness and contamination:

    Args:
        syn (pd.DataFrame): Preprocessed proper elements DataFrame, indexed by asteroid_number.
        labels (array): Cluster labels from DBSCAN. -1 indicates noise.
        mem (pd.DataFrame): AstDys membership DataFrame, indexed by asteroid_number. Must contain
                            column 'family1' with integer family IDs.

    Returns:
        results (pd.DataFrame): One row per target family with columns
    """

    print("\nStep 5: Evaluating against AstDys ground truth")
    syn = syn.copy()
    syn["cluster"] = labels

    rows = []
    for fam_id in TARGET_FAMILIES:
        # ground truth: AstDys members of this family that are also in syn
        gt_ids = set(mem[mem["family1"] == fam_id].index) & set(syn.index)
        n_astdys = len(gt_ids)
        if n_astdys == 0:
            continue

        # find which cluster contains the most ground truth members
        gt_sub = syn.loc[syn.index.isin(gt_ids)]
        cc = gt_sub["cluster"].value_counts()
        cc = cc[cc.index != -1]

        # if no cluster contains any ground truth members, family is missed entirely
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

        # best cluster = the one with the most AstDys members inside it
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

    # print benchmark summary
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
