import random

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
    random.seed(1996)
    torch.manual_seed(1996)

    # Set device
    device = torch.device('cpu')

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all(culture=True)

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    culture_embed_dim = 16

    model_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_ced{culture_embed_dim}.pt'
    checkpoint = torch.load(model_name, map_location=device)
    language_to_id = checkpoint["language_to_id"]
    num_cultures = len(language_to_id)

    # Recreate the model architecture first, then load the weights from the saved model
    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    # List of name_romanised, culture label after normalising (i.e., splitting diacritics)
    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    train_names, _ = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    train_dataset = NameDataset(train_names, vocab)

    print(f"Model name: {model_name}")

    # Evaluation mode
    model.eval()

    generated = []

    with torch.no_grad():
        for _ in range(5000):
            # Tensor of (latent_dim) where each number is sample from N(0,1)
            z = torch.randn(1, latent_dim)
            label = torch.tensor([random.choice(list(language_to_id.values()))])
            name = model.decoder.generate(z, label)
            generated.append(name)

    duplicates = 0
    pairs = 0

    threshold = 0.25

    for i, g in enumerate(generated):
        for j in range(i + 1, len(generated)):
            if editdistance.eval(generated[i], generated[j]) / max(len(generated[i]), len(generated[j])) <= threshold:
                duplicates += 1
            pairs += 1

    print(f"100 random generated names: {random.sample(generated, 100)}")

    # Pronounceability
    for n in (2, 3, 4):
        print(f"{n}-gram coverage: {compute_ngram_coverage(generated, train_names, n):.2%}")

    # Novelty
    print(f"Exact novelty wrt training data: {compute_novelty(generated, train_dataset):.2%}")

    # Diversity
    print(f"Unique rate (among generated names): {len(set(generated)) / len(generated):.2%}")
    print(f"Near other (normalised Levenshtein distance <= {threshold}): {duplicates / pairs:.2%}")


if __name__ == "__main__":
    generate()
