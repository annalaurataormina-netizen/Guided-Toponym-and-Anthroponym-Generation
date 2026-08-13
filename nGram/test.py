import json
from collections import Counter

import torch

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from nGram.nGram import nGram
from utils import load_all


def test():

    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 32, 0.0015, 100, 0.005, 5
    culture_embed_dim = 64
    temperature, lambda_supcon = 0.1, 0.75

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    vocab = CharVocab(ALLOWED_CHARS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator_name = f'ConditionalVAE/models/best_model_supcon_logits_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'
    generator = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim, culture_stats_path=f"culture_stats_{beta_max}.pt")
    generator.load_state_dict(torch.load(generator_name, map_location=device))

    print(f"Generator name: {generator_name}")

    generator.to(device)
    generator.eval()

    names = load_all(culture=True)

    culture_counts = Counter(
        language
        for _, language in names
    )

    generated_for_italian = {}

    for language, language_id in language_to_id.items():

        if culture_counts[language] < 1000:
            pass

        for n in range(2, 4):
            model = nGram(n)
            model.load()
            print(f"N = {n}")

            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                generated = generator.generate(
                    culture=language_id,
                    n=1000,
                    max_length=50,
                    temperature=t,
                )

                if language == "Italian" and n == 2:
                    generated_for_italian[t] = generated

                log_probabilities = [
                    model.sequence_log_probability((name, language))
                    for name in generated
                ]

                num_inf = sum(
                    1 for log_prob in log_probabilities
                    if log_prob == float("-inf")
                )

                valid_log_probabilities = [
                    log_prob
                    for log_prob in log_probabilities
                    if log_prob != float("-inf")
                ]

                percentage_inf = 100 * num_inf / len(log_probabilities)

                if valid_log_probabilities:
                    average_log_probability = (
                            sum(valid_log_probabilities)
                            / len(valid_log_probabilities)
                    )
                else:
                    average_log_probability = float("-inf")

                print(
                    f"Language: {language} | "
                    f"Counts: {culture_counts[language]} | "
                    f"Temperature: {t:.1f} | "
                    f"% -inf: {percentage_inf:.2f}% | "
                    f"Average log-probability: {average_log_probability:.4f} | "
                    f"Valid: {len(valid_log_probabilities)}/1000"
                )

    for temperature, generated in generated_for_italian.items():
        print(f"\nTemperature: {temperature}")
        for name in generated[:100]:
            print(name)


if __name__ == "__main__":
    test()
