import json
import random
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from ConditionalVAE2.ConditionalVAE2 import ConditionalVAE2
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

    with torch.no_grad():

        data = [["Nikos", 3]]

        dataset = NameDataset(data, vocab)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for sequences, lengths, _ in dataloader:
            sequences, lengths = sequences.to(device), lengths.cpu()

            logits = classifier(sequences, lengths)
            top5_predictions = torch.topk(logits, k=5, dim=1).indices
            top5_predictions_old = [
                [mapping[i.item()] for i in row]
                for row in top5_predictions
            ]
            id_to_language = {
                id_: language
                for language, id_ in language_to_id.items()
            }
            top5_languages = [
                [id_to_language[i] for i in row]
                for row in top5_predictions_old
            ]
            print(top5_languages)


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

    new_to_old = {new: old for old, new in old_to_new.items()}

    # After filtering
    num_cultures_filtered = len(new_to_old)

    # Same split logic as training
    labels = [x[1] for x in names_normalised]

    train_names, _ = train_test_split(names_normalised, test_size=0.2, random_state=seed, shuffle=True, stratify=labels)

    # Load generator
    # ConditionalVAE
    # Model hyperparameters
    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    lr = 0.001
    epochs = 50
    patience = 10
    beta_max = 0.025
    n_epochs_ramp_up = 5
    temperature = 0.1
    lambda_supcon = 0.75
    culture_embed_dim = 64
    generator = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                               num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)

    generator_name = (f'CCVAE/models/best_model_conditional_supcon_logits_'
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
    generator.load_state_dict(torch.load(generator_name, map_location=device))


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
