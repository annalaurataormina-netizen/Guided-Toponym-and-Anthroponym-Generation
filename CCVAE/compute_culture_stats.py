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
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

            _, mu, _ = model.encoder(sequences, lengths, labels)

            for latent, label in zip(mu.cpu(), labels):
                culture_latents[label.item()].append(latent)

    culture_stats = {}

    for culture, latents in culture_latents.items():
        latents = torch.stack(latents)

        mean = latents.mean(dim=0)

        if len(latents) > 1:
            std = latents.std(dim=0, unbiased=False) + 1e-6
            cov = torch.cov(latents.T)
        else:
            std = torch.ones_like(mean)
            cov = torch.eye(latents.shape[1])

        culture_stats[culture] = {
            "mean": mean,
            "std": std,
            "cov": cov
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

    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    lr = lr  # grid search
    epochs = 100
    patience = 10
    beta_max = 0.025
    n_epochs_ramp_up = 5
    temperature = 0.1
    lambda_supcon = 0.75
    culture_embed_dim = 64

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)

    model_name = (f'CCVAE/models/best_model_conditional_supcon_logits_'
                  f'bi_'
                  f'bs{batch_size}_'
                  f'ed{embed_dim}_'
                  f'hde{hidden_dim_encoder}_'
                  f'hdd{hidden_dim_decoder}_'
                  f'nle{num_layers_encoder}_'
                  f'nld{num_layers_decoder}_'
                  f'ld{latent_dim}_'
                  f'lr{lr}adamW_'
                  f'ep{epochs}es{patience}_'
                  f'cd0.25_'
                  f'blf0t{beta_max}o{n_epochs_ramp_up}_'
                  f'ced{culture_embed_dim}_'
                  f't{temperature}_'
                  f'l{lambda_supcon}.pt'
                  )

    model.load_state_dict(torch.load(model_name, map_location=device))

    model.to(device)
    model.eval()

    culture_stats = compute_culture_stats(model, dataloader, device)

    torch.save(culture_stats, f"CCVAE/culture_stats.pt")

    print(f"Saved statistics for {len(culture_stats)} cultures")
