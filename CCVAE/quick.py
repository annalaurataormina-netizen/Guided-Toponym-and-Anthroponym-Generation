import json

import editdistance
import torch

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from nGram.nGram import nGram
from utils import normalise
from ConditionalVAE.ConditionalVAE import ConditionalVAE


def quick():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model hyperparameters
    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    lr = 0.0005
    epochs = 100
    patience = 10
    beta_max = 0.025
    n_epochs_ramp_up = 5
    temperature = 0.1
    lambda_supcon = 0.75
    culture_embed_dim = 64

    model_name = (f'CCVAE/models/best_model_conditional_supcon_logits_'
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
    generated = model.generate(culture_embedding=target_embedding, temperature=0.6, n=1000)

    ngram2 = nGram(2)
    ngram2.load()
    ngram3 = nGram(3)
    ngram3.load()
    ngram4 = nGram(4)
    ngram4.load()

    for idx, n in enumerate(generated):
        generated[idx] = (n, ngram2.sequence_log_probability((n, culture_1)), ngram3.sequence_log_probability((n, culture_1)), ngram4.sequence_log_probability((n, culture_1)))

    count = 0

    for n in generated:
        if n[1] > -2.5 and n[2] > -2.5 and n[3] > -2.5 :
            print(n[0])
            count += 1

    print(f"Count: {count}")
    '''

    '''
    culture = "French"
    ngram2 = nGram(2)
    ngram2.load()
    ngram3 = nGram(3)
    ngram3.load()
    ngram4 = nGram(4)
    ngram4.load()
    generated = model.generate(culture=language_to_id[culture], temperature=0.6, n=10, ngram2=ngram2, ngram3=ngram3, ngram4=ngram4, cultures=[culture])
    for n in generated:
        print(n)
    '''

    culture1 = "Italian"
    culture2 = "Japanese"
    ngram2 = nGram(2)
    ngram2.load()
    ngram3 = nGram(3)
    ngram3.load()
    ngram4 = nGram(4)
    ngram4.load()
    id1 = language_to_id[culture1]
    id2 = language_to_id[culture2]

    culture_embedding = (0.5 * model.decoder.culture_embedding(torch.tensor(id1, device=device))
            + 0.5 * model.decoder.culture_embedding(torch.tensor(id2, device=device)))

    # generated = model.generate(culture_embedding=culture_embedding, temperature=0.6, n=10, ngram2=ngram2, ngram3=ngram3, ngram4=ngram4, cultures=[culture1, culture2])

    generated = model.generate(culture_embedding=culture_embedding, temperature=0.6, n=10)

    for n in generated:
        print(n)

    '''
    #generated = model.generate(culture=language_to_id["Italian"], culture_embedding=None, n=100, max_length=50, temperature=0.4)
    #for name in generated:
    #    print(name)
    '''

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
