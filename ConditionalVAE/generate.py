import json
import random
from collections import Counter

import editdistance
import torch
from sklearn.model_selection import train_test_split

from AE.CharVocab import CharVocab
from ContrastiveVAE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from utils import load_all, normalise, compute_novelty, compute_ngram_coverage
from .ConditionalVAE import ConditionalVAE

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def generate():

    # Set random seed for reproducibility
    seed = 1996
    random.seed(seed)
    torch.manual_seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Toponyms and Anthroponyms (list of name_romanised, culture)
    names = load_all(culture=True)

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 32, 0.0015, 100, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 2, 0.5
    culture_embed_dim = 64
    temperature, lambda_supcon = 0.1, 0.75

    model_name = f'ConditionalVAE/models/best_model_supcon_logits_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'

    print(f"Model name: {model_name}")

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    # Recreate model
    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)

    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()

    # Normalise names and map cultures to ids
    names_normalised = [[normalise(name), language_to_id[lang]] for name, lang in names]

    # Training split (same as training)
    train_names, _ = train_test_split( names_normalised, test_size=0.2, random_state=seed, shuffle=True)

    train_dataset = NameDataset(train_names, vocab)

    culture_counts = Counter(label for _, label in train_names)

    min_samples = 1000

    valid_cultures = [culture_id for culture_id, count in culture_counts.items() if count >= min_samples]

    print(f"Total cultures: {num_cultures}")
    print(f"Cultures evaluated: {len(valid_cultures)}")

    n_samples_per_culture = 1000
    generated_by_culture = {}

    with torch.no_grad():

        for culture_id in valid_cultures:

            generated = []

            for _ in range(n_samples_per_culture):

                z = torch.randn(1, latent_dim, device=device)
                label = torch.tensor([culture_id], device=device)
                name = model.decoder.generate(z, label)
                generated.append(name)

            generated_by_culture[culture_id] = generated

    print(f"Generated {len(valid_cultures) * n_samples_per_culture} names")

    generated = [name for names in generated_by_culture.values() for name in names]

    print(f"Model name: {model_name}")

    print(
        f"100 random generated names: "
        f"{random.sample(generated, min(100, len(generated)))}"
    )

    for n in (2, 3, 4):
        print(f"{n}-gram coverage: {compute_ngram_coverage(generated, train_dataset, n):.2%}")

    print(
        f"Exact novelty wrt training data: "
        f"{compute_novelty(generated, train_dataset):.2%}"
    )

    print(f"Unique rate: {len(set(generated)) / len(generated):.2%}")

    threshold = 0.25

    duplicates = 0
    pairs = 0

    for i in range(len(generated)):
        for j in range(i + 1, len(generated)):

            distance = editdistance.eval(generated[i], generated[j]) / max(len(generated[i]), len(generated[j]))

            if distance <= threshold:
                duplicates += 1

            pairs += 1

    print(f"Near duplicates (normalised Levenshtein <= {threshold}): {duplicates / pairs:.2%}")

    print("\nPer-culture diversity:")

    for culture_id, culture_names in generated_by_culture.items():
        unique_rate = len(set(culture_names)) / len(culture_names)
        print(f"Culture {culture_id}: {unique_rate:.2%} unique")


if __name__ == "__main__":
    generate()