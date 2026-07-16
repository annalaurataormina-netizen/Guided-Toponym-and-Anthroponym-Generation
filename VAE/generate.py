import random

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


def generate():
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
    batch_size, embed_dim, hidden_dim, num_layers, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 2, 64, 0.001, 30, 0.005, 5
    # n_cycles, ratio = 6, 0.75

    # Recreate the model architecture first, then load the weights from the saved model
    model = VAE(vocab, embed_dim, hidden_dim, num_layers, latent_dim)
    model_name = f'best_model_bs{batch_size}_ed{embed_dim}_hd{hidden_dim}_nl{num_layers}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)

    print(f"Model name: {model_name}")

    # Evaluation mode
    model.eval()

    generated = []

    with torch.no_grad():
        for _ in range(5000):
            # Tensor of (latent_dim) where each number is sample from N(0,1)
            z = torch.randn(1, latent_dim)
            name = model.decoder.generate(z)
            generated.append(name)

    duplicates = 0
    pairs = 0

    threshold = 0.25

    for i, g in enumerate(generated):
        for j in range(i + 1, len(generated)):
            if editdistance.eval(generated[i], generated[j]) / max(len(generated[i]), generated[j]) <= threshold:
                duplicates += 1
            pairs += 1

    print(f"100 random generated names: {random.sample(generated, 100)}")

    # Pronounceability
    for n in (2, 3, 4):
        print(f"{n}-gram coverage: {compute_ngram_coverage(generated, train_dataset, n):.2%}")

    # Novelty
    print(f"Exact novelty wrt training data: {compute_novelty(generated, train_dataset):.2%}")

    # Diversity
    print(f"Unique rate (among generated names): {len(set(generated)) / len(generated):.2%}")
    print(f"Near other (normalised Levenshtein distance <= {threshold}): {duplicates / pairs:.2%}")


if __name__ == "__main__":
    generate()
