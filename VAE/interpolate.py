import random
import statistics

import editdistance
import torch
from sklearn.model_selection import train_test_split

from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from utils import load_all, normalise, compute_novelty, compute_ngram_coverage
from .VAE import VAE

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def interpolate():
    # Set random seed for reproducibility
    random.seed(1996)
    torch.manual_seed(1996)

    # Set device
    device = torch.device('cpu')

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all()

    # List of name_romanised after normalising (i.e., splitting diacritics)
    names_normalised = [normalise(n) for n in names]

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    train_names, _ = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    train_dataset = NameDataset(train_names, vocab)

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5

    # Recreate the model architecture first, then load the weights from the saved model
    model = VAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder,
                latent_dim)
    model_name = f'VAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)

    print(f"Model name: {model_name}")

    # Evaluation mode
    model.eval()

    interpolations = []
    generated = []

    with torch.no_grad():
        for _ in range(500):
            # Randomly sample 2 names from the training dataset
            i, j = random.sample(range(len(train_dataset)), 2)

            trajectory = {
                "source_index": i,
                "destination_index": j,
                "source_name": train_names[i],
                "destination_name": train_names[j],
                "alphas": [],
                "generated": [],
            }

            x1, length1 = train_dataset[i]
            x2, length2 = train_dataset[j]

            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

            length1 = torch.tensor([length1])
            length2 = torch.tensor([length2])

            _, mu1, _ = model.encoder(x1, length1)
            _, mu2, _ = model.encoder(x2, length2)

            for alpha in torch.linspace(0.1, 0.9, 9):
                z = (1 - alpha.item()) * mu1 + alpha.item() * mu2

                name = model.decoder.generate(z)

                generated.append(name)
                trajectory["alphas"].append(alpha.item())
                trajectory["generated"].append(name)

            interpolations.append(trajectory)

    near_endpoint_count = 0
    near_other_generated_count = 0
    total = 0
    pairs = 0

    threshold = 0.1

    for trajectory in interpolations:
        for i, name in enumerate(trajectory["generated"]):
            total += 1
            if (editdistance.eval(name, trajectory["source_name"]) / max(len(name),
                                                                         len(trajectory["source_name"])) <= threshold
                    or
                    editdistance.eval(name, trajectory["destination_name"]) / max(len(name), len(
                        trajectory["destination_name"])) <= threshold):
                near_endpoint_count += 1
            for j in range(i + 1, len(trajectory["generated"])):
                other_name = trajectory["generated"][j]
                pairs += 1
                if editdistance.eval(name, other_name) / max(len(name), len(other_name)) <= threshold:
                    near_other_generated_count += 1

    smoothness_distances = []

    for trajectory in interpolations:
        names = [
            trajectory["source_name"],
            *trajectory["generated"],
            trajectory["destination_name"]
        ]

        for i in range(len(names) - 1):
            smoothness_distances.append(
                editdistance.eval(names[i], names[i + 1]) / max(len(names[i]), len(names[i + 1]))
            )

    print("5 random trajectories (including endpoints)")

    trajectories = random.sample(interpolations, 5)

    for trajectory in trajectories:
        names = [
            trajectory["source_name"],
            *trajectory["generated"],
            trajectory["destination_name"]
        ]
        print(names)

    # Pronounceability
    for n in (2, 3, 4):
        print(f"{n}-gram coverage: {compute_ngram_coverage(generated, train_dataset, n):.2%}")

    # Novelty
    print(f"Unique rate wrt training data: {compute_novelty(generated, train_dataset):.2%}")
    print(f"Near endpoints (normalised Levenshtein distance <= {threshold}): {near_endpoint_count / total:.2%}")

    # Diversity
    print(f"Unique rate (among generated names): {len(set(generated)) / len(generated):.2%}")
    print(f"Near other (normalised Levenshtein distance <= {threshold}): {near_other_generated_count / pairs:.2%}")

    # Smoothness
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
