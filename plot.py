import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    axes[0].legend()

    # Weighted KL
    axes[1].plot(df["Step"] / 1000, df["Train β-KL"], label="Training")
    axes[1].plot(df["Step"] / 1000, df["Val β-KL"], label="Validation")
    axes[1].set_ylabel(r"$\beta$ KL divergence")
    axes[1].legend()

    # SupCon
    axes[2].plot(df["Step"] / 1000, df["Train SupCon"], label="Training")
    axes[2].plot(df["Step"] / 1000, df["Val SupCon"], label="Validation")
    axes[2].set_ylabel("SupCon loss")
    axes[2].set_xlabel("Training step (×1000)")
    axes[2].legend()

    # Same ticks across all three plots
    ticks = np.arange(0, 451, 50)
    axes[2].set_xticks(ticks)

    # Only show x-axis labels on bottom plot
    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig("training_objectives.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot()