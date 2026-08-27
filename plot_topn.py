import matplotlib.pyplot as plt

def plot():
    k = list(range(1, 11))

    macro = [
        16.85, 24.09, 28.98, 32.93, 36.23,
        39.11, 41.56, 43.85, 46.00, 47.94
    ]

    weighted = [
        29.97, 41.15, 47.30, 52.31, 56.44,
        59.89, 62.77, 65.47, 68.08, 70.37
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(k, macro, marker="o", label="Macro")
    plt.plot(k, weighted, marker="o", label="Weighted")

    plt.xlabel("Top-$k$")
    plt.ylabel("Generation accuracy (%)")
    plt.xticks(k)
    plt.ylim(0, 100)

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("linguistic_coherence.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot()