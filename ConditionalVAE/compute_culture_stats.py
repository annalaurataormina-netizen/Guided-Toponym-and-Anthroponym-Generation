import json
from collections import defaultdict

import torch

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from ContrastiveVAE.NameDataset import NameDataset
from utils import load_all, normalise

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def compute_culture_stats(model, dataloader, device):
    culture_latents = defaultdict(list)

    with torch.no_grad():
        for sequences, lengths, labels in dataloader:
            sequences, lengths = sequences.to(device), lengths.cpu()

            _, mu, _ = model.encoder(sequences, lengths, labels)

            for latent, label in zip(mu.cpu(), labels):
                culture_latents[label.item()].append(latent)

    culture_stats = {}

    for culture, latents in culture_latents.items():
        latents = torch.stack(latents)

        culture_stats[culture] = {
            "mean": latents.mean(dim=0),
            "std": latents.std(dim=0) + 1e-6,
            "cov": torch.cov(latents.T)
        }

    return culture_stats


if __name__ == "__main__":
    seed = 1996
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocab(ALLOWED_CHARS)

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    names = load_all(culture=True)

    names_normalised = [[normalise(name), language_to_id[lang]] for name, lang in names]

    dataset = NameDataset(names_normalised, vocab)

    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 32, 0.0015, 100, 0.008, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    temperature, lambda_supcon = 0.1, 0.75
    culture_embed_dim = 64

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)

    model_name = f'ConditionalVAE/models/best_model_supcon_logits_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'

    model.load_state_dict(torch.load(model_name, map_location=device))

    model.to(device)
    model.eval()

    culture_stats = compute_culture_stats(model, dataloader, device)

    torch.save(culture_stats, "ConditionalVAE/culture_stats.pt")

    print(f"Saved statistics for {len(culture_stats)} cultures")
