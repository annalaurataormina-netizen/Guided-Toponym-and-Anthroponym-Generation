import json
import random
import statistics

import editdistance
import torch
from sklearn.model_selection import train_test_split

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from ContrastiveVAE.NameDataset import NameDataset
from utils import load_all, normalise, compute_novelty, compute_ngram_coverage


'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def interpolate():

    # ---------------------------------------------------------
    # Reproducibility / device
    # ---------------------------------------------------------

    seed = 1996

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    names = load_all(culture=True)

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    # Same 80/10/10 split used during training
    train_names, temp_names = train_test_split(
        names_normalised,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    val_names, test_names = train_test_split(
        temp_names,
        test_size=0.5,
        random_state=seed,
        shuffle=True
    )

    vocab = CharVocab(ALLOWED_CHARS)

    test_dataset = NameDataset(test_names, vocab)

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

    num_cultures = len(language_to_id)

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

    print(f"Model name: {model_name}")

    model.load_state_dict(
        torch.load(model_name, map_location=device)
    )

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Group test indices by culture
    # ---------------------------------------------------------

    culture_to_indices = {}

    for i, (_, culture) in enumerate(test_names):

        if culture not in culture_to_indices:
            culture_to_indices[culture] = []

        culture_to_indices[culture].append(i)

    # Only cultures with at least two test names can be interpolated
    eligible_cultures = [
        culture
        for culture, indices in culture_to_indices.items()
        if len(indices) >= 2
    ]

    print(f"Test samples: {len(test_names)}")
    print(f"Cultures with >= 2 test samples: {len(eligible_cultures)}")

    # ---------------------------------------------------------
    # Interpolation
    # ---------------------------------------------------------

    n_trajectories = 500

    interpolations = []
    generated = []

    with torch.no_grad():

        for _ in range(n_trajectories):

            # Pick a culture first
            culture = random.choice(eligible_cultures)

            # Pick two different names from that same culture
            i, j = random.sample(
                culture_to_indices[culture],
                2
            )

            trajectory = {
                "source_index": i,
                "destination_index": j,
                "culture": culture,
                "source_name": test_names[i][0],
                "destination_name": test_names[j][0],
                "alphas": [],
                "generated": [],
            }

            x1, length1, label1 = test_dataset[i]
            x2, length2, label2 = test_dataset[j]

            x1 = x1.unsqueeze(0).to(device)
            x2 = x2.unsqueeze(0).to(device)

            length1 = torch.tensor(
                [length1],
                device=device
            )

            length2 = torch.tensor(
                [length2],
                device=device
            )

            label1 = torch.tensor(
                [label1],
                dtype=torch.long,
                device=device
            )

            # Encode both names
            _, mu1, _ = model.encoder(
                x1,
                length1,
                label1
            )

            _, mu2, _ = model.encoder(
                x2,
                length2,
                label1
            )

            # Use ONE fixed culture embedding for the whole trajectory
            culture_embedding = model.decoder.culture_embedding(label1)

            # Interpolate between the two latent representations
            for alpha in torch.linspace(0.1, 0.9, 9):

                z = (
                    (1 - alpha.item()) * mu1
                    + alpha.item() * mu2
                )

                names_generated = model.decoder.generate(
                    z,
                    culture_embedding=culture_embedding,
                    max_len=50
                )

                name = names_generated[0]

                generated.append(name)

                trajectory["alphas"].append(
                    alpha.item()
                )

                trajectory["generated"].append(name)

            interpolations.append(trajectory)

    # ---------------------------------------------------------
    # Endpoint similarity
    # ---------------------------------------------------------

    near_endpoint_count = 0
    near_other_generated_count = 0

    total = 0
    pairs = 0

    threshold = 0.1

    for trajectory in interpolations:

        for i, name in enumerate(
            trajectory["generated"]
        ):

            total += 1

            source_distance = (
                editdistance.eval(
                    name,
                    trajectory["source_name"]
                )
                / max(
                    len(name),
                    len(trajectory["source_name"])
                )
            )

            destination_distance = (
                editdistance.eval(
                    name,
                    trajectory["destination_name"]
                )
                / max(
                    len(name),
                    len(trajectory["destination_name"])
                )
            )

            if (
                source_distance <= threshold
                or destination_distance <= threshold
            ):
                near_endpoint_count += 1

            for j in range(
                i + 1,
                len(trajectory["generated"])
            ):

                other_name = trajectory["generated"][j]

                pairs += 1

                distance = (
                    editdistance.eval(
                        name,
                        other_name
                    )
                    / max(
                        len(name),
                        len(other_name)
                    )
                )

                if distance <= threshold:
                    near_other_generated_count += 1

    # ---------------------------------------------------------
    # Smoothness
    # ---------------------------------------------------------

    smoothness_distances = []

    for trajectory in interpolations:

        names_trajectory = [
            trajectory["source_name"],
            *trajectory["generated"],
            trajectory["destination_name"]
        ]

        for i in range(
            len(names_trajectory) - 1
        ):

            distance = (
                editdistance.eval(
                    names_trajectory[i],
                    names_trajectory[i + 1]
                )
                / max(
                    len(names_trajectory[i]),
                    len(names_trajectory[i + 1])
                )
            )

            smoothness_distances.append(distance)

    # ---------------------------------------------------------
    # Print example trajectories
    # ---------------------------------------------------------

    print()
    print("5 random same-culture trajectories (including endpoints)")
    print()

    for trajectory in random.sample(
        interpolations,
        5
    ):

        names_trajectory = [
            trajectory["source_name"],
            *trajectory["generated"],
            trajectory["destination_name"]
        ]

        print(
            f"Culture {trajectory['culture']}: "
            f"{names_trajectory}"
        )

    # ---------------------------------------------------------
    # Pronounceability
    # ---------------------------------------------------------

    print()

    for n in (2, 3, 4):

        print(
            f"{n}-gram coverage: "
            f"{compute_ngram_coverage(
                generated,
                test_dataset,
                n
            ):.2%}"
        )

    # ---------------------------------------------------------
    # Novelty
    # ---------------------------------------------------------

    print(
        f"Novelty wrt test data: "
        f"{compute_novelty(
            generated,
            test_dataset
        ):.2%}"
    )

    # ---------------------------------------------------------
    # Endpoint similarity
    # ---------------------------------------------------------

    print(
        f"Near endpoints "
        f"(normalised Levenshtein distance <= {threshold}): "
        f"{near_endpoint_count / total:.2%}"
    )

    # ---------------------------------------------------------
    # Diversity
    # ---------------------------------------------------------

    print(
        f"Unique rate (among generated names): "
        f"{len(set(generated)) / len(generated):.2%}"
    )

    print(
        f"Near other generated "
        f"(normalised Levenshtein distance <= {threshold}): "
        f"{near_other_generated_count / pairs:.2%}"
    )

    # ---------------------------------------------------------
    # Smoothness
    # ---------------------------------------------------------

    print(
        f"Step distance mean: "
        f"{statistics.mean(smoothness_distances):.3f}"
    )

    print(
        f"Step distance std deviation: "
        f"{statistics.stdev(smoothness_distances):.3f}"
    )


if __name__ == "__main__":
    interpolate()