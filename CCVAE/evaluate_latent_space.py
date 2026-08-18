import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from ContrastiveVAE.NameDataset import NameDataset
from utils import load_all, normalise


'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def evaluate_latent_space():

    seed = 1996

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Model hyperparameters
    # ---------------------------------------------------------

    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    lr = 0.0005
    epochs = 50
    patience = 10
    beta_max = 0.025
    n_epochs_ramp_up = 5
    temperature = 0.1
    lambda_supcon = 0.75
    culture_embed_dim = 64

    vocab = CharVocab(ALLOWED_CHARS)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    names = load_all(culture=True)

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    # ---------------------------------------------------------
    # Recreate the 80/10/10 train/validation/test split
    # ---------------------------------------------------------

    train_val_names, test_names = train_test_split(
        names_normalised,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    train_names, validation_names = train_test_split(
        train_val_names,
        test_size=0.125,
        random_state=seed,
        shuffle=True
    )

    print()
    print("Dataset split:")
    print(f"Train:      {len(train_names)}")
    print(f"Validation: {len(validation_names)}")
    print(f"Test:       {len(test_names)}")

    # ---------------------------------------------------------
    # Use ONLY the test set for latent-space evaluation
    # ---------------------------------------------------------

    test_dataset = NameDataset(test_names, vocab)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = ConditionalVAE(
        vocab,
        embed_dim,
        hidden_dim_encoder,
        hidden_dim_decoder,
        num_layers_encoder,
        num_layers_decoder,
        latent_dim,
        num_cultures,
        culture_embed_dim
    )

    model_name = (
        f'CCVAE/models/best_model_conditional_supcon_logits_'
        f'bi_'
        f'bs{batch_size}_'
        f'ed{embed_dim}_'
        f'hde{hidden_dim_encoder}_'
        f'hdd{hidden_dim_decoder}_'
        f'nle{num_layers_encoder}_'
        f'nld{num_layers_decoder}_'
        f'ld{latent_dim}_'
        f'lr{lr}adam_'
        f'ep{epochs}es{patience}_'
        f'cd0.25_'
        f'blf0t{beta_max}o{n_epochs_ramp_up}_'
        f'ced{culture_embed_dim}_'
        f't{temperature}_'
        f'l{lambda_supcon}.pt'
    )

    print()
    print(f"Model: {model_name}")

    model.load_state_dict(
        torch.load(
            model_name,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Encode test names into latent space
    # ---------------------------------------------------------

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_latents = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            x, lengths, culture = batch

            x = x.to(device)
            lengths = lengths.to(device)
            culture = culture.to(device)

            _, mu, _ = model.encoder(
                x,
                lengths,
                culture
            )

            all_latents.append(mu.cpu())
            all_labels.append(culture.cpu())

    latents = torch.cat(
        all_latents,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    X = latents.numpy()
    y = labels.numpy()

    print()
    print(f"Test samples encoded: {len(X)}")
    print(f"Cultures represented: {len(np.unique(y))}")

    # ---------------------------------------------------------
    # Evaluation settings
    # ---------------------------------------------------------

    max_samples_per_culture = 1000
    k = 10

    # Number of test samples per culture
    culture_counts = {
        culture: np.sum(y == culture)
        for culture in np.unique(y)
    }

    thresholds = {
        ">=100": 100,
        ">=1000": 1000,
        ">=10000": 10000,
    }

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Evaluate each threshold
    # ---------------------------------------------------------

    for threshold_name, min_samples in thresholds.items():

        valid_cultures = [
            culture
            for culture, count in culture_counts.items()
            if count >= min_samples
        ]

        print()
        print("=" * 70)
        print(
            f"{threshold_name}: "
            f"{len(valid_cultures)} cultures"
        )
        print("=" * 70)

        if len(valid_cultures) < 2:
            print("Not enough cultures for evaluation.")
            continue

        # -----------------------------------------------------
        # Sample up to 1,000 test names per culture
        # -----------------------------------------------------

        sampled_indices = []

        for culture in valid_cultures:

            culture_indices = np.where(
                y == culture
            )[0]

            n = min(
                len(culture_indices),
                max_samples_per_culture
            )

            selected = rng.choice(
                culture_indices,
                size=n,
                replace=False
            )

            sampled_indices.extend(selected)

        sampled_indices = np.array(
            sampled_indices
        )

        X_eval = X[sampled_indices]
        y_eval = y[sampled_indices]

        print(
            f"Samples used: {len(X_eval)}"
        )

        # -----------------------------------------------------
        # Silhouette
        # -----------------------------------------------------

        print()
        print("SILHOUETTE SCORE")

        for metric in ["euclidean", "cosine", "manhattan", "correlation"]:
            silhouette = silhouette_samples(
                X_eval,
                y_eval,
                metric=metric
            )

            overall_silhouette = silhouette.mean()

            per_culture_silhouette = {
                culture: silhouette[y_eval == culture].mean()
                for culture in np.unique(y_eval)
            }

            macro_silhouette = np.mean(
                list(per_culture_silhouette.values())
            )

            weighted_silhouette = np.average(
                list(per_culture_silhouette.values()),
                weights=[
                    np.sum(y_eval == culture)
                    for culture in per_culture_silhouette
                ]
            )

            print(f"{metric}:")
            print(f"  Overall:  {overall_silhouette:.4f}")
            print(f"  Macro:    {macro_silhouette:.4f}")
            print(f"  Weighted: {weighted_silhouette:.4f}")

        # -----------------------------------------------------
        # 10-NN purity
        # -----------------------------------------------------

        for metric in ["euclidean", "cosine", "manhattan", "correlation"]:

            neighbours = NearestNeighbors(
                n_neighbors=k + 1,
                metric=metric
            )

            neighbours.fit(X_eval)

            _, indices = neighbours.kneighbors(
                X_eval
            )

            sample_purities = np.zeros(
                len(X_eval)
            )

            for i in range(len(X_eval)):

                neighbour_labels = y_eval[
                    indices[i][1:]
                ]

                sample_purities[i] = np.mean(
                    neighbour_labels == y_eval[i]
                )

            overall_purity = sample_purities.mean()

            per_culture_purity = {}

            for culture in np.unique(y_eval):

                mask = y_eval == culture

                per_culture_purity[culture] = (
                    sample_purities[mask].mean()
                )

            # Macro: each culture gets equal weight
            macro_purity = np.mean(
                list(
                    per_culture_purity.values()
                )
            )

            # Weighted: each sample gets equal weight
            weighted_purity = np.average(
                [
                    per_culture_purity[culture]
                    for culture in per_culture_purity
                ],
                weights=[
                    np.sum(y_eval == culture)
                    for culture in per_culture_purity
                ]
            )

            print(f"{metric}:")
            print(f"  Overall:  {overall_purity:.4f}")
            print(f"  Macro:    {macro_purity:.4f}")
            print(f"  Weighted: {weighted_purity:.4f}")


if __name__ == "__main__":
    evaluate_latent_space()