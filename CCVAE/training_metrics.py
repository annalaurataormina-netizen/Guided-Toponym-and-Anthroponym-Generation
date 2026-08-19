import re
from pathlib import Path

import pandas as pd


def parse_log(input_file):
    input_file = Path(input_file)

    steps = []
    epochs = []
    levenshtein = []

    # ---------------------------------------------------------
    # Step-level metrics
    # ---------------------------------------------------------
    step_pattern = re.compile(
        r"Epoch (\d+)/\d+, Step ([\d,]+), Beta = ([\d.]+), "
        r"Avg validation loss \(full validation set\) = ([\d.]+), "
        r"Avg validation reconstruction loss \(full validation set\) = ([\d.]+), "
        r"Avg validation KL divergence \(full validation set\) = ([\d.]+), "
        r"Avg validation beta-adjusted KL divergence \(full validation set\) = ([\d.]+), "
        r"Avg validation SupCon loss \(full validation set\) = ([\d.]+), "
        r"Avg validation lambda-adjusted SupCon loss \(full validation set\) = ([\d.]+), "
        r"Avg training loss \(last 2000 batches\) = ([\d.]+), "
        r"Avg training reconstruction loss \(last 2000 batches\) = ([\d.]+), "
        r"Avg training beta-adjusted KL divergence \(last 2000 batches\) = ([\d.]+), "
        r"Avg training SupCon loss \(last 2000 batches\) = ([\d.]+), "
        r"Avg training lambda-adjusted SupCon loss \(last 2000 batches\) = ([\d.]+)"
    )

    # ---------------------------------------------------------
    # Epoch-level metrics
    # ---------------------------------------------------------
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+, "
        r"Avg train loss per epoch: ([\d.]+), "
        r"Avg reconstruction loss per epoch: ([\d.]+), "
        r"Avg KL divergence per epoch: ([\d.]+), "
        r"Avg beta-adjusted KL divergence per epoch: ([\d.]+), "
        r"Avg SupCon loss per epoch: ([\d.]+), "
        r"Avg lambda-adjusted SupCon loss per epoch: ([\d.]+)"
    )

    # ---------------------------------------------------------
    # Levenshtein
    # ---------------------------------------------------------
    levenshtein_pattern = re.compile(
        r"Epoch (\d+)/\d+, Step ([\d,]+), "
        r"Avg normalised Levenshtein distance "
        r"\(portion of validation set\) = ([\d.]+)"
    )

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:

            # =========================
            # Step metrics
            # =========================
            match = step_pattern.search(line)

            if match:
                (
                    epoch,
                    step,
                    beta,
                    val_loss,
                    val_recon,
                    val_kl,
                    val_beta_kl,
                    val_supcon,
                    val_lambda_supcon,
                    train_loss,
                    train_recon,
                    train_beta_kl,
                    train_supcon,
                    train_lambda_supcon,
                ) = match.groups()

                steps.append({
                    "Epoch": int(epoch),
                    "Step": int(step.replace(",", "")),
                    "Train Loss": float(train_loss),
                    "Train Recon": float(train_recon),
                    "Train β-KL": float(train_beta_kl),
                    "Train SupCon": float(train_supcon),
                    "Val Loss": float(val_loss),
                    "Val Recon": float(val_recon),
                    "Val β-KL": float(val_beta_kl),
                    "Val SupCon": float(val_supcon),
                })

                continue

            # =========================
            # Epoch metrics
            # =========================
            match = epoch_pattern.search(line)

            if match:
                (
                    epoch,
                    train_loss,
                    train_recon,
                    train_kl,
                    train_beta_kl,
                    train_supcon,
                    train_lambda_supcon,
                ) = match.groups()

                epochs.append({
                    "Epoch": int(epoch),
                    "Train Loss": float(train_loss),
                    "Train Recon": float(train_recon),
                    "Train KL": float(train_kl),
                    "Train SupCon": float(train_supcon),
                })

                continue

            # =========================
            # Levenshtein
            # =========================
            match = levenshtein_pattern.search(line)

            if match:
                epoch, step, distance = match.groups()

                levenshtein.append({
                    "Epoch": int(epoch),
                    "Step": int(step.replace(",", "")),
                    "Normalised Levenshtein": float(distance),
                })

    # Convert to DataFrames
    steps_df = pd.DataFrame(steps)
    epochs_df = pd.DataFrame(epochs)
    levenshtein_df = pd.DataFrame(levenshtein)

    # ---------------------------------------------------------
    # Add validation metrics to epoch table
    #
    # There are multiple validation measurements per epoch.
    # We use the LAST validation measurement recorded in each
    # epoch.
    # ---------------------------------------------------------
    if not steps_df.empty:

        final_validation = (
            steps_df
            .sort_values(["Epoch", "Step"])
            .groupby("Epoch")
            .last()
            .reset_index()
        )

        final_validation = final_validation[
            [
                "Epoch",
                "Val Loss",
                "Val Recon",
                "Val β-KL",
                "Val SupCon",
            ]
        ]

        epochs_df = epochs_df.merge(
            final_validation,
            on="Epoch",
            how="left",
        )

    # ---------------------------------------------------------
    # Reconstruction table
    # ---------------------------------------------------------
    reconstruction_df = epochs_df[
        [
            "Epoch",
            "Train Recon",
            "Val Recon",
        ]
    ].rename(columns={
        "Train Recon": "Train Reconstruction",
        "Val Recon": "Validation Reconstruction",
    })

    # ---------------------------------------------------------
    # Levenshtein table
    #
    # There should normally be one measurement per epoch.
    # Keep the last one if there are multiple.
    # ---------------------------------------------------------
    levenshtein_df = (
        levenshtein_df
        .sort_values(["Epoch", "Step"])
        .groupby("Epoch")
        .last()
        .reset_index()
    )

    levenshtein_df = levenshtein_df[
        [
            "Epoch",
            "Normalised Levenshtein",
        ]
    ]

    return (
        steps_df,
        epochs_df,
        reconstruction_df,
        levenshtein_df,
    )


def save_tables(input_file, output_dir="training_metrics"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        steps_df,
        epochs_df,
        reconstruction_df,
        levenshtein_df,
    ) = parse_log(input_file)

    # CSV files
    steps_df.to_csv(
        output_dir / "steps.csv",
        index=False,
    )

    epochs_df.to_csv(
        output_dir / "epochs.csv",
        index=False,
    )

    reconstruction_df.to_csv(
        output_dir / "reconstruction.csv",
        index=False,
    )

    levenshtein_df.to_csv(
        output_dir / "levenshtein.csv",
        index=False,
    )

    # One Excel workbook containing all four tables
    with pd.ExcelWriter(
        output_dir / "training_metrics.xlsx",
        engine="openpyxl",
    ) as writer:

        steps_df.to_excel(
            writer,
            sheet_name="Steps",
            index=False,
        )

        epochs_df.to_excel(
            writer,
            sheet_name="Epochs",
            index=False,
        )

        reconstruction_df.to_excel(
            writer,
            sheet_name="Reconstruction",
            index=False,
        )

        levenshtein_df.to_excel(
            writer,
            sheet_name="Levenshtein",
            index=False,
        )

    print(f"Saved results to {output_dir}/")
    print(f"Steps:          {len(steps_df)} rows")
    print(f"Epochs:         {len(epochs_df)} rows")
    print(f"Reconstruction: {len(reconstruction_df)} rows")
    print(f"Levenshtein:    {len(levenshtein_df)} rows")


if __name__ == "__main__":
    save_tables(
        input_file="CCVAE/evaluation_results/evaluation_results_lr0.0005.txt",
        output_dir="training_metrics",
    )