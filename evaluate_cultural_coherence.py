import json
import random
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from ContrastiveVAE.NameDataset import NameDataset
from CultureClassifier.CultureClassifier import CultureClassifier
from utils import normalise, load_all

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS (OF GENERATOR AND CLASSIFIER) BELOW SO THAT THE RIGHT MODELS ARE LOADED.
'''


def evaluate(generator, classifier, language_to_id, mapping, train_names, device,
             batch_size, n_per_culture=1000):
    vocab = CharVocab(ALLOWED_CHARS)

    classifier.eval()

    culture_sizes = Counter(label for _, label in train_names)

    results = []

    with torch.no_grad():
        for language, label in sorted(language_to_id.items(), key=lambda x: x[1]):

            old_label = mapping[label]

            train_examples = culture_sizes[label]

            generated = generator.generate(culture=old_label, n=n_per_culture, max_length=50)

            dataset = NameDataset([[name, label] for name in generated], vocab)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            correct = 0
            top5_correct = 0
            total = 0

            for sequences, lengths, _ in dataloader:
                sequences, lengths = sequences.to(device), lengths.cpu()

                logits = classifier(sequences, lengths)

                predictions = logits.argmax(dim=1)
                top5_predictions = torch.topk(logits, k=5, dim=1).indices

                correct += (predictions == label).sum().item()

                top5_correct += (top5_predictions == label).any(dim=1).sum().item()

                total += len(sequences)

            generation_accuracy = correct / total
            top5_generation_accuracy = top5_correct / total

            results.append(
                {
                    "language": language,
                    "train_examples": train_examples,
                    "generation_accuracy": generation_accuracy,
                    "top5_generation_accuracy": top5_generation_accuracy,
                }
            )

    print("\nGeneration accuracy by culture")
    print("--------------------------------")

    for result in results:
        print(
            f"{result['language']}: "
            f"examples={result['train_examples']}, "
            f"accuracy={result['generation_accuracy']:.3f}, "
            f"top5={result['top5_generation_accuracy']:.3f}"
        )

    print("\nAggregated results")

    accuracies = [r["generation_accuracy"] for r in results]
    top5_accuracies = [r["top5_generation_accuracy"] for r in results]

    print(f"Mean generation accuracy: {sum(accuracies) / len(accuracies):.3f}")
    print(f"Mean top-5 generation accuracy: {sum(top5_accuracies) / len(top5_accuracies):.3f}")

    return results


if __name__ == "__main__":
    seed = 1996
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    vocab = CharVocab(ALLOWED_CHARS)

    # Load language mapping (used to train both models, but filtered down for the classifier)
    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    # Before filtering
    num_cultures = len(language_to_id.items())

    names = load_all(culture=True)

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    culture_counts = Counter(label for _, label in names_normalised)

    min_samples = 1000

    names_normalised = [
        [name, label]
        for name, label in names_normalised
        if culture_counts[label] >= min_samples
    ]

    remaining_cultures = sorted(
        set(label for _, label in names_normalised)
    )

    old_to_new = {
        old: new
        for new, old in enumerate(remaining_cultures)
    }

    names_normalised = [
        [name, old_to_new[label]]
        for name, label in names_normalised
    ]

    language_to_id = {
        language: old_to_new[label]
        for language, label in language_to_id.items()
        if label in old_to_new
    }

    new_to_old = {new: old for old, new in old_to_new.items()}

    # After filtering
    num_cultures_filtered = len(language_to_id)

    # Same split logic as training
    labels = [x[1] for x in names_normalised]

    train_names, _ = train_test_split(names_normalised, test_size=0.2, random_state=seed, shuffle=True, stratify=labels)

    # Load generator
    # VAE
    '''
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    generator = VAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder,
                    latent_dim, culture_stats_path="VAE/culture_stats.pt")
    generator_name = f'VAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''

    # ContrastiveVAE2
    '''
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 128, 0.0015, 100, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    temperature, lambda_supcon = 0.1, 0.75
    generator = VAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder,
                    latent_dim, culture_stats_path="ContrastiveVAE2/culture_stats.pt")
    generator_name = f'ContrastiveVAE2/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''

    # ContrastiveVAE
    '''
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 128, 0.0015, 100, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    proj_hidden_dim, proj_output_dim, temperature, lambda_supcon = 256, 128, 0.1, 0.75
    generator = ContrastiveVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                               num_layers_decoder, latent_dim, proj_hidden_dim, proj_output_dim,
                               culture_stats_path="ContrastiveVAE/culture_stats.pt")
    generator_name = f'ContrastiveVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_phd{proj_hidden_dim}_pod{proj_output_dim}_t{temperature}_l{lambda_supcon}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''

    # ConditionalVAE
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 32, 0.0015, 100, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 3, 0.5
    culture_embed_dim = 64
    # temperature, lambda_supcon = 0.1, 0.75
    margin, lambda_triplet = 1.0, 0.75
    generator = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                               num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
    '''
    generator_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_ced{culture_embed_dim}.pt'
    checkpoint = torch.load(generator_name, map_location=device)
    '''
    '''
    generator_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_bcf0t{beta_max}o{n_cycles}w{ratio}_ced{culture_embed_dim}.pt'
    checkpoint = torch.load(generator_name, map_location=device)
    '''
    '''
    generator_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_bcf0t{beta_max}o{n_cycles}w{ratio}_ced{culture_embed_dim}_se.pt'
    checkpoint = torch.load(generator_name, map_location=device)
    '''
    '''
    generator_name = f'ConditionalVAE/models/best_model_supcon_mu_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''
    '''
    generator_name = f'ConditionalVAE/models/best_model_supcon_out_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_t{temperature}_l{lambda_supcon}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''
    '''
    generator_name = f'ConditionalVAE/models/best_model_triplet_mu_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_m{margin}_l{lambda_triplet}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''
    generator_name = f'ConditionalVAE/models/best_model_triplet_out_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_m{margin}_l{lambda_triplet}.pt'
    generator.load_state_dict(torch.load(generator_name, map_location=device))
    '''
    generator_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_ced{culture_embed_dim}_se.pt'
    checkpoint = torch.load(generator_name, map_location=device)
    '''
    '''
    generator_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_ced{culture_embed_dim}_se_cd.pt'
    checkpoint = torch.load(generator_name, map_location=device)
    '''

    print(f"Generator name: {generator_name}")

    generator.to(device)

    generator.eval()

    # Load classifier
    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = 512, 32, 256, 1, 0.0005, 30

    classifier = CultureClassifier(vocab, embed_dim, hidden_dim, num_layers, num_cultures_filtered)

    classifier_name = f'CultureClassifier/models/best_model_bs{batch_size}_ed{embed_dim}_hd{hidden_dim}_nl{num_layers}_lr{lr}_ep{epochs}_ms{min_samples}.pt'

    print(f"Classifier name: {classifier_name}")

    classifier.load_state_dict(torch.load(classifier_name, map_location=device))

    classifier.to(device)
    classifier.eval()

    evaluate(
        generator=generator,
        classifier=classifier,
        language_to_id=language_to_id,
        mapping=new_to_old,
        train_names=train_names,
        device=device,
        batch_size=batch_size,
        n_per_culture=1000,
    )
