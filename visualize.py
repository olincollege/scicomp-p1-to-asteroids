import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from constants import (
    FIG_DPI,
    SCATTER_SIZE,
    BACKGROUND_ALPHA,
    FAMILY_ALPHA,
    FAMILY_CMAP,
    COMPLETENESS_THRESHOLD,
)


def _get_colors(n_families):
    """Return a list of n_families distinct colors."""
    cmap = cm.get_cmap(FAMILY_CMAP)
    return [cmap(i / max(n_families, 1)) for i in range(n_families)]


def plot_proper_elements(df, labels, save_path=None):
    """
    Plot families in proper element space.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed proper elements with columns D_deg, e_p, sin_i_p.
    labels : np.ndarray
        Cluster labels from hcm.cut_dendrogram(). 0 = background.
    save_path : str or None
        If provided, save the figure to this path.
    """
    family_ids = sorted([f for f in np.unique(labels) if f > 0])
    colors = _get_colors(len(family_ids))
    color_map = {fid: colors[i] for i, fid in enumerate(family_ids)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- plot 1: D_deg vs e_p ---
    ax = axes[0]
    bg = labels == 0
    ax.scatter(
        df.loc[bg, "D_deg"],
        df.loc[bg, "e_p"],
        s=SCATTER_SIZE,
        c="grey",
        alpha=BACKGROUND_ALPHA,
        label="Background",
    )

    for fid in family_ids:
        mask = labels == fid
        ax.scatter(
            df.loc[mask, "D_deg"],
            df.loc[mask, "e_p"],
            s=SCATTER_SIZE * 3,
            color=color_map[fid],
            alpha=FAMILY_ALPHA,
            label=f"Family {fid}",
        )

    ax.set_xlabel("Libration Amplitude D (deg)")
    ax.set_ylabel("Proper Eccentricity $e_p$")
    ax.set_title("Trojan Families: D vs $e_p$")

    # --- plot 2: D_deg vs sin_i_p ---
    ax = axes[1]
    ax.scatter(
        df.loc[bg, "D_deg"],
        df.loc[bg, "sin_i_p"],
        s=SCATTER_SIZE,
        c="grey",
        alpha=BACKGROUND_ALPHA,
        label="Background",
    )

    for fid in family_ids:
        mask = labels == fid
        ax.scatter(
            df.loc[mask, "D_deg"],
            df.loc[mask, "sin_i_p"],
            s=SCATTER_SIZE * 3,
            color=color_map[fid],
            alpha=FAMILY_ALPHA,
            label=f"Family {fid}",
        )

    ax.set_xlabel("Libration Amplitude D (deg)")
    ax.set_ylabel(r"$\sin(i_p)$")
    ax.set_title(r"Trojan Families: D vs $\sin(i_p)$")

    # shared legend (top 10 families only to keep it readable)
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(
        handles[:11],
        lbls[:11],
        loc="lower center",
        ncol=6,
        markerscale=6,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  [visualize] Proper element plot saved to {save_path}")
    plt.show()


def plot_completeness(results, save_path=None):
    """
    Bar chart of completeness per family vs the 95% benchmark line.

    Parameters
    ----------
    results : pd.DataFrame
        Output of evaluate.compute_completeness_contamination()
    save_path : str or None
    """
    results = results.sort_values("completeness", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = [
        "steelblue" if c >= COMPLETENESS_THRESHOLD else "salmon"
        for c in results["completeness"]
    ]

    ax.bar(range(len(results)), results["completeness"], color=colors)
    ax.axhline(
        COMPLETENESS_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{COMPLETENESS_THRESHOLD*100:.0f}% threshold",
    )

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(
        [f"F{int(r)}" for r in results["astdys_family_id"]],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Completeness")
    ax.set_ylim(0, 1.05)
    ax.set_title("Family Completeness vs AstDys Ground Truth")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  [visualize] Completeness plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from load_data import load_proper_elements, load_family_membership
    from preprocess import preprocess
    from metric import compute_distance_matrix
    from hcm import build_linkage, cut_dendrogram, get_families
    from evaluate import match_clusters_to_families, compute_completeness_contamination
    from constants import V_CUTOFF_DEFAULT, CLUSTERING_COLS

    df = load_proper_elements()
    family_membership = load_family_membership()
    df_clean, _ = preprocess(df)

    X = df_clean[CLUSTERING_COLS].to_numpy(dtype=np.float64)
    condensed = compute_distance_matrix(X)
    Z = build_linkage(condensed)
    labels = cut_dendrogram(Z, v_cutoff=V_CUTOFF_DEFAULT)
    families = get_families(df_clean, labels)

    matches = match_clusters_to_families(families, family_membership)
    results = compute_completeness_contamination(families, family_membership, matches)

    plot_proper_elements(df_clean, labels, save_path="results/proper_elements.png")
    plot_completeness(results, save_path="results/completeness.png")
