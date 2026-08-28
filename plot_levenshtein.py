import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot():
    df = pd.read_csv("CCVAE/training_metrics/levenshtein.csv")

    # Levenshtein is calculated every 15,000 training steps
    df["Step"] = np.arange(1, len(df) + 1) * 15000

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        df["Step"] / 1000,
        df["Normalised Levenshtein"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.set_xlabel("Training step (in thousands)")
    ax.set_ylabel("Normalised Levenshtein distance")

    # 15k increments
    ax.set_xticks(np.arange(0, 451, 50))

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    ramp_end = 66

    ax.axvline(
        ramp_end,
        linestyle="--",
        linewidth=1,
        color="grey"
    )

    ax.text(
        ramp_end + 3,
        ax.get_ylim()[1] * 0.9,
        r"$\beta$ ramp-up ends",
        rotation=90,
        va="top",
        color="grey"
    )

    plt.savefig(
        "levenshtein_training.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    plot()