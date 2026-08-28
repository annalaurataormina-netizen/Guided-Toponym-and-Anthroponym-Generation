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
    epochs = 100
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

    train_names, temp_names = train_test_split(
        names_normalised,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    validation_names, test_names = train_test_split(
        temp_names,
        test_size=0.5,
        random_state=seed,
        shuffle=True
    )

    print()
    print("Dataset split:")
    print(f"Train:      {len(train_names)}")
    print(f"Validation: {len(validation_names)}")
    print(f"Test:       {len(test_names)}")

    # ---------------------------------------------------------
    # Number of training samples per culture
    # Used for the weighted metrics
    # ---------------------------------------------------------

    train_culture_counts = {
        culture: sum(
            1 for _, train_culture in train_names
            if train_culture == culture
        )
        for culture in set(
            culture for _, culture in train_names
        )
    }

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

    k = 10
    samples_per_culture = 100
    minimum_culture_size = 100

    # Number of test samples per culture
    culture_counts = {
        culture: np.sum(y == culture)
        for culture in np.unique(y)
    }

    # Only include cultures with at least 100 test samples
    valid_cultures = [
        culture
        for culture, count in culture_counts.items()
        if count >= minimum_culture_size
    ]

    print()
    print("=" * 70)
    print(
        f"Languages with at least {minimum_culture_size} test samples: "
        f"{len(valid_cultures)}"
    )
    print("=" * 70)

    if len(valid_cultures) < 2:
        print("Not enough cultures for evaluation.")
        return

    # ---------------------------------------------------------
    # Sample exactly 100 test names per culture
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    sampled_indices = []

    for culture in valid_cultures:

        culture_indices = np.where(
            y == culture
        )[0]

        selected = rng.choice(
            culture_indices,
            size=samples_per_culture,
            replace=False
        )

        sampled_indices.extend(selected)

    sampled_indices = np.array(sampled_indices)

    X_eval = X[sampled_indices]
    y_eval = y[sampled_indices]

    print(
        f"Samples per culture: {samples_per_culture}"
    )

    print(
        f"Total samples used: {len(X_eval)}"
    )

    print(
        f"Cultures used: {len(np.unique(y_eval))}"
    )

    # ---------------------------------------------------------
    # Silhouette score
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SILHOUETTE SCORE")
    print("=" * 70)

    for metric in [
        "euclidean",
        "cosine",
        "manhattan",
        "correlation"
    ]:

        silhouette = silhouette_samples(
            X_eval,
            y_eval,
            metric=metric
        )

        # Overall: every sample has equal weight
        overall_silhouette = silhouette.mean()

        # Per-culture scores
        per_culture_silhouette = {
            culture: silhouette[y_eval == culture].mean()
            for culture in np.unique(y_eval)
        }

        # Macro: every culture has equal weight
        macro_silhouette = np.mean(
            list(per_culture_silhouette.values())
        )

        # Weighted: weight cultures by their training-set size
        weighted_silhouette = np.average(
            list(per_culture_silhouette.values()),
            weights=[
                train_culture_counts.get(culture, 0)
                for culture in per_culture_silhouette
            ]
        )

        print(f"{metric}:")
        print(f"  Overall:  {overall_silhouette:.4f}")
        print(f"  Macro:    {macro_silhouette:.4f}")
        print(f"  Weighted: {weighted_silhouette:.4f}")

    # ---------------------------------------------------------
    # 10-NN purity
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("10-NN PURITY")
    print("=" * 70)

    for metric in [
        "euclidean",
        "cosine",
        "manhattan",
        "correlation"
    ]:

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

            # Exclude the sample itself
            neighbour_labels = y_eval[
                indices[i][1:]
            ]

            sample_purities[i] = np.mean(
                neighbour_labels == y_eval[i]
            )

        # Overall: every sample has equal weight
        overall_purity = sample_purities.mean()

        # Per-culture purity
        per_culture_purity = {}

        for culture in np.unique(y_eval):

            mask = y_eval == culture

            per_culture_purity[culture] = (
                sample_purities[mask].mean()
            )

        # Macro: every culture has equal weight
        macro_purity = np.mean(
            list(per_culture_purity.values())
        )

        # Weighted: weight cultures by their training-set size
        weighted_purity = np.average(
            [
                per_culture_purity[culture]
                for culture in per_culture_purity
            ],
            weights=[
                train_culture_counts.get(culture, 0)
                for culture in per_culture_purity
            ]
        )

        print(f"{metric}:")
        print(f"  Overall:  {overall_purity:.4f}")
        print(f"  Macro:    {macro_purity:.4f}")
        print(f"  Weighted: {weighted_purity:.4f}")


if __name__ == "__main__":
    evaluate_latent_space()