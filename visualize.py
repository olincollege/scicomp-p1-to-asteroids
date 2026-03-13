# Importing in Libraries needed and the files needed
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from constants import COMPLETENESS_TARGET


def visualize(syn, labels, results, save_dir="results"):
    """
    Generate and save diagnostic plots for the clustering results.

    Args:
        syn (pd.DataFrame): Preprocessed proper elements DataFrame, indexed by asteroid_number.
        labels : Cluster labels from DBSCAN.
        results (pd.DataFrame): Evaluation results from evaluate().
        save_dir (str): Directory to save plots into.

    Returns:
        None
    """
    print("\nStep 6: Generating plots")
    os.makedirs(save_dir, exist_ok=True)

    # assign a distinct color to each cluster ID
    family_ids = sorted([f for f in np.unique(labels) if f >= 0])
    cmap = cm.get_cmap("tab20")
    color_map = {
        fid: cmap(i / max(len(family_ids), 1)) for i, fid in enumerate(family_ids)
    }

    # Plot 1: proper element scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bg = labels == -1

    for ax, ycol, ylabel in zip(
        axes, ["e_p", "sin_i_p"], ["Proper Eccentricity $e_p$", r"$\sin(i_p)$"]
    ):
        # background asteroids in grey
        ax.scatter(
            syn.loc[bg, "da_AU"],
            syn.loc[bg, ycol],
            s=0.5,
            c="grey",
            alpha=0.1,
            label="Background",
        )

        # overlay each family cluster in color
        for fid in family_ids[:20]:
            mask = labels == fid
            ax.scatter(
                syn.loc[mask, "da_AU"],
                syn.loc[mask, ycol],
                s=3,
                color=color_map[fid],
                alpha=0.7,
                label=f"Cluster {fid}",
            )
        ax.set_xlabel("da_AU")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Trojan Families: da_AU vs {ylabel}")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/proper_elements.png", dpi=150, bbox_inches="tight")
    print(f"  Saved proper_elements.png")
    plt.close()

    # Plot 2: completeness bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    # color bars by whether they pass the 95% benchmark
    colors = [
        "steelblue" if c >= COMPLETENESS_TARGET else "salmon"
        for c in results["completeness"]
    ]
    ax.bar(range(len(results)), results["completeness"], color=colors)
    ax.axhline(COMPLETENESS_TARGET, color="red", linestyle="--", label="95% threshold")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([str(f) for f in results["family_id"]], rotation=45)
    ax.set_ylabel("Completeness")
    ax.set_ylim(0, 1.05)
    ax.set_title("Family Completeness vs AstDys Ground Truth")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/completeness.png", dpi=150, bbox_inches="tight")
    print(f"  Saved completeness.png")
    plt.close()
