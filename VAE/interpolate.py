import random

import torch
from sklearn.model_selection import train_test_split

from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from AE.utils import load_all, normalise, compute_novelty, compute_ngram_coverage
from .VAE import VAE


def interpolate():
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
    batch_size, embed_dim, hidden_dim, num_layers, latent_dim, lr, epochs, beta = 512, 64, 64, 2, 64, 0.001, 30, 0

    # Recreate the model architecture first, then load the weights from the saved model
    model = VAE(vocab, embed_dim, hidden_dim, num_layers, latent_dim)
    state_dict = torch.load("best_model_bs512_ed64_hd64_nl2_ld64_lr0.001_ep30_b0.pt", map_location=device)
    model.load_state_dict(state_dict)

    # Evaluation mode
    model.eval()

    # Generated names
    generated = []

    with torch.no_grad():
        for _ in range(5_000):
            # Randomly sample 2 names from the training dataset
            i, j = random.sample(range(len(train_dataset)), 2)

            x1, length1 = train_dataset[i]
            x2, length2 = train_dataset[j]

            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

            length1 = torch.tensor([length1])
            length2 = torch.tensor([length2])

            _, mu1, _ = model.encoder(x1, length1)
            _, mu2, _ = model.encoder(x2, length2)

            alpha = torch.rand(1).item()
            z = (1 - alpha) * mu1 + alpha * mu2

            name = model.decoder.generate(z)
            generated.append(name)

    print(f"Generated names: {generated}")

    # Pronounceability
    for n in (2, 3, 4):
        print(f"{n}-gram coverage: {compute_ngram_coverage(generated, train_dataset, n):.2%}")

    # Novelty
    print(
        f"Novelty wrt training data: {compute_novelty(generated, train_dataset)}",
        '''WRITE THIS'''
        f"Levenshtein distance to nearest name in training data: {compute_levenshtein_to_nearest_name_in_training_data}",
    )

    # Diversity
    print(
        f"Duplicate rate (among generated names): {len(set(generated)) / len(generated)}",
        '''WRITE THIS'''
        # Pairwise normalised Levenshtein distance
        # Self-BLEU
        '''WRITE THIS'''
    )

    # Smoothness
    print(
        f"Smoothness: {compute_smoothness(...):.3f}"
    )


if __name__ == "__main__":
    interpolate()
