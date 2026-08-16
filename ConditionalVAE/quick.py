import json

import editdistance
import torch

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from nGram.nGram import nGram
from utils import normalise
from .ConditionalVAE import ConditionalVAE


def quick():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 32, 0.0015, 100, 0.025, 5
    # free_bits = 0.05
    # n_cycles, ratio = 2, 0.5
    culture_embed_dim = 64
    temperature, lambda_supcon = 0.1, 0.50

    model_name = f'ConditionalVAE/models/best_model_supcon_logits_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'

    print(f"Model name: {model_name}")

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    vocab = CharVocab(ALLOWED_CHARS)

    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()

    '''
    with torch.no_grad():
        name = "Johannes"
        source_culture = "German"

        encoded = vocab.encode(normalise(name))

        x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        length = torch.tensor([x.size(1)], dtype=torch.long, device=device)

        source_label = torch.tensor([language_to_id[source_culture]], dtype=torch.long, device=device)
        source_embedding = model.decoder.culture_embedding(source_label)

        z, mu, logvar = model.encoder(x, length, source_label)
        z = mu

        reconstruction = model.decoder.generate(z, source_embedding)
        print(f"Reconstruction ({source_culture}): {reconstruction}")

        for target_culture in ["Chinese", "Italian", "French", "Japanese", "German", "Spanish"]:
            target_label = torch.tensor([language_to_id[target_culture]], dtype=torch.long, device=device)
            target_embedding = model.decoder.culture_embedding(target_label)
            generated = model.decoder.generate(z, target_embedding)
            print(f"{source_culture} -> {target_culture}: {generated}")
        
    '''

    '''
        culture_1 = "Italian"
        culture_2 = "Italian"
        percentage_culture_1 = 50
        culture_id_1 = language_to_id[culture_1]
        culture_id_2 = language_to_id[culture_2]
        target_culture = f'{percentage_culture_1}% {culture_1} {100 - percentage_culture_1}% {culture_2}'
        italian_embedding = model.decoder.culture_embedding(torch.tensor([culture_id_1], device=device))
        german_embedding = model.decoder.culture_embedding(torch.tensor([culture_id_2], device=device))
        target_embedding = (italian_embedding + german_embedding) / 2
        generated = model.decoder.generate(z, culture_embedding=target_embedding)
        print(f"{source_culture} -> {target_culture}: {generated}")

        # print(model.generate(culture_embedding=target_embedding))
    '''

    '''
    #generated = model.generate(culture=language_to_id["Italian"], culture_embedding=None, n=100, max_length=50, temperature=0.4)
    #for name in generated:
    #    print(name)
    '''

    threshold = 0.5

    for t in [0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
        spaces = 0
        capitals = 0
        duplicates = 0
        pairs = 0
        generated = model.generate(culture=language_to_id["Italian"], culture_embedding=None, n=1000, max_length=50, temperature=t)
        for i in range(len(generated)):
            for j in range(i + 1, len(generated)):

                distance = (
                    editdistance.eval(
                        generated[i],
                        generated[j]
                    )
                    / max(
                        len(generated[i]),
                        len(generated[j])
                    )
                )

                if distance <= threshold:
                    duplicates += 1

                pairs += 1

        print(f"temperature: {t}; spaces: {spaces}; capitals: {capitals}; duplicate rate: {duplicates / pairs * 100}%")
    '''
    with torch.no_grad():
        name = "Anna"
        source_culture = "Italian"

        encoded = vocab.encode(normalise(name))

        x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        length = torch.tensor([x.size(1)], dtype=torch.long, device=device)

        source_label = torch.tensor([language_to_id[source_culture]], dtype=torch.long, device=device)
        source_embedding = model.decoder.culture_embedding(source_label)

        z, mu, logvar = model.encoder(x, length, source_label)
        z = mu

        for temperature in [0.5, 0.6, 0.7, 0.8]:

            n = 10

            z_samples = z + temperature * torch.randn(
                n,
                model.latent_dim,
                device=device
            )

            culture_labels = torch.full(
                (n,),
                language_to_id["Italian"],
                dtype=torch.long,
                device=device
            )

            culture_embedding = model.decoder.culture_embedding(culture_labels)

            names = model.decoder.generate(
                z_samples,
                culture_embedding=culture_embedding,
                max_len=50
            )

            print(f"\nTemperature: {temperature}")

            for generated_name in names:
                print(generated_name)
    '''

if __name__ == "__main__":
    quick()
