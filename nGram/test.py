import json

import torch
from torch import device

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from nGram.nGram import nGram


def test():

    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 32, 0.0015, 100, 0.005, 5
    culture_embed_dim = 64
    temperature, lambda_supcon = 0.1, 0.75

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    vocab = CharVocab(ALLOWED_CHARS)

    generator_name = f'ConditionalVAE/models/best_model_supcon_logits_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'
    generator = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
    generator.load_state_dict(torch.load(generator_name, map_location=device))

    for n in range(2, 5):
        model = nGram(n)
        model.load()
        culture = "Italian"
        culture_id = language_to_id[culture]
        generated = generator.generate(culture=culture_id, n=50, max_length=50)
        for name in generated:
            print(f"Name: {name}, Log-probability: {model.log_probability((name, culture))}")

if __name__ == "main":
    test()