import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


def plot():
    df = pd.read_csv("CCVAE/training_metrics/steps.csv")

    fig, axes = plt.subplots(
        3, 1,
        figsize=(8, 9),
        sharex=True
    )

    # Reconstruction
    axes[0].plot(df["Step"] / 1000, df["Train Recon"], label="Training")
    axes[0].plot(df["Step"] / 1000, df["Val Recon"], label="Validation")
    axes[0].set_ylabel("Reconstruction loss")

    # Weighted KL
    axes[1].plot(df["Step"] / 1000, df["Train KL"], label="Training")
    axes[1].plot(df["Step"] / 1000, df["Val KL"], label="Validation")
    axes[1].set_ylabel(r"KL divergence")

    # SupCon
    axes[2].plot(df["Step"] / 1000, df["Train SupCon"], label="Training")
    axes[2].plot(df["Step"] / 1000, df["Val SupCon"], label="Validation")
    axes[2].set_ylabel("SupCon loss")
    axes[2].set_xlabel("Training step (in thousands)")

    fig.legend(
        ["Training", "Validation"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False
    )

    # Same ticks across all three plots
    ticks = np.arange(0, 451, 50)
    axes[2].set_xticks(ticks)

    # Only show x-axis labels on bottom plot
    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[2].spines["top"].set_visible(False)

    # Mark end of beta ramp-up
    ramp_end = 66  # 60,000 training steps

    for ax in axes:
        ax.axvline(
            ramp_end,
            linestyle="--",
            linewidth=1,
            color="grey"
        )

    axes[0].text(
        ramp_end + 3,
        axes[0].get_ylim()[1] * 0.9,
        r"$\beta$ ramp-up ends",
        rotation=90,
        va="top",
        color="grey"
    )

    fig.subplots_adjust(hspace=0)
    fig.align_ylabels(axes)
    plt.savefig("training_objectives.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot()