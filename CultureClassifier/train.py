import json
import random
from collections import Counter

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ContrastiveVAE.NameDataset import NameDataset
from CultureClassifier.CultureClassifier import CultureClassifier
from utils import load_all, normalise

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def train():
    # Set seed for reproducibility
    seed = 1996
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ConditionalVAE hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 100, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    culture_embed_dim = 16

    # Classifier hyperparameters
    embed_dim_classifier, hidden_dim, num_layers, lr_classifier, epochs_classifier = 32, 64, 1, 0.001, 30

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (name_romanised, label)
    names = load_all(culture=True)

    # ConditionalVAE
    model_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_ced{culture_embed_dim}.pt'

    print(f"Testing on {model_name}")
    print(f"Embedding dimension: {embed_dim_classifier}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of layers: {num_layers}")
    print(f"Epochs: {epochs_classifier}")
    print(f"Learning rate: {lr_classifier}")

    checkpoint = torch.load(model_name, map_location=device)

    language_to_id = checkpoint["language_to_id"]

    with open("language_to_id.json", "w") as f:
        json.dump(language_to_id, f)

    '''

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    # 80/10/10 split of the dataset into train/validation/test
    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=seed, shuffle=True)
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=seed, shuffle=True)

    train_dataset = NameDataset(train_names, vocab)
    val_dataset = NameDataset(val_names, vocab)
    test_dataset = NameDataset(test_names, vocab)

    # Same seed as the one used to split the dataset into train, validation and test, for consistency
    g = torch.Generator()
    g.manual_seed(seed)

    # Shuffling means that batches are random, which is important when training the model
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = CultureClassifier(vocab, embed_dim_classifier, hidden_dim, num_layers, num_cultures)
    classifier.to(device)

    counts = Counter(label for _, label in train_names)

    criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(classifier.parameters(), lr=lr_classifier)

    classifier.train()

    best_loss = float('inf')

    for epoch in range(epochs_classifier):

        epoch_losses = []

        for batch in train_dataloader:
            sequences, lengths, labels = batch
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

            # Zero out the gradients
            optimiser.zero_grad()

            logits = classifier(sequences, lengths)

            loss = criterion(logits, labels)

            # Backprop (compute gradients, update model params via backpropagation)
            loss.backward()
            optimiser.step()

            epoch_losses.append(loss.item())

        classifier.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                sequences, lengths, labels = batch
                sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

                logits = classifier(sequences, lengths)

                val_loss = criterion(logits, labels)

                val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                classifier_name = f'CultureClassifier/models/best_model_bs{batch_size}_hd{hidden_dim}_lr{lr_classifier}_ep{epochs_classifier}.pt'
                torch.save(classifier.state_dict(), classifier_name)

        print(f"Epoch {epoch + 1}/{epochs_classifier}, ")
        print(f"Avg train loss per epoch: {sum(epoch_losses) / len(epoch_losses):.4f}")

    classifier.eval()

    val_pred_cultures = []
    val_labels = []

    print(f"Number of cultures: {num_cultures}")
    print(f"Random accuracy: {1 / num_cultures}")

    classifier.load_state_dict(
        torch.load(classifier_name, map_location=device)
    )

    with torch.no_grad():
        for batch in val_dataloader:
            sequences, lengths, labels = batch
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

            logits = classifier(sequences, lengths)
            pred_cultures_batch = logits.argmax(dim=-1)
            val_pred_cultures.append(pred_cultures_batch)
            val_labels.append(labels)

    val_pred_cultures = torch.cat(val_pred_cultures)
    val_labels = torch.cat(val_labels)
    val_accuracy = (val_pred_cultures == val_labels).float().mean()

    # Convert tensors to CPU numpy arrays
    val_pred_cultures_np = val_pred_cultures.cpu().numpy()
    val_labels_np = val_labels.cpu().numpy()

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(
        val_labels_np,
        val_pred_cultures_np
    )

    # Macro F1 (each culture weighted equally)
    macro_f1 = f1_score(
        val_labels_np,
        val_pred_cultures_np,
        average="macro"
    )

    # Weighted F1
    weighted_f1 = f1_score(
        val_labels_np,
        val_pred_cultures_np,
        average="weighted"
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(
        val_labels_np,
        val_pred_cultures_np
    )

    # Per-culture recall / precision / F1
    report = classification_report(
        val_labels_np,
        val_pred_cultures_np,
        zero_division=0
    )

    print("VALIDATION")
    print(f"Accuracy: {val_accuracy.item():.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(f"Classification report:\n{report}")
    '''
    '''
    test_pred_cultures = []
    test_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            sequences, lengths, labels = batch
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

            logits = classifier(sequences, lengths)
            pred_cultures_batch = logits.argmax(dim=-1)
            test_pred_cultures.append(pred_cultures_batch)
            test_labels.append(labels_batch)

    test_pred_cultures = torch.cat(test_pred_cultures)
    test_labels = torch.cat(test_labels)
    test_accuracy = (test_pred_cultures == test_labels).float().mean()

    # Convert tensors to CPU numpy arrays
    test_pred_cultures_np = test_pred_cultures.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(
        test_labels_np,
        test_pred_cultures_np
    )

    # Macro F1 (each culture weighted equally)
    macro_f1 = f1_score(
        test_labels_np,
        test_pred_cultures_np,
        average="macro"
    )

    # Weighted F1
    weighted_f1 = f1_score(
        test_labels_np,
        test_pred_cultures_np,
        average="weighted"
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(
        test_labels_np,
        test_pred_cultures_np
    )

    # Per-culture recall / precision / F1
    report = classification_report(
        test_labels_np,
        test_pred_cultures_np,
        zero_division=0
    )

    print("TEST")
    print(f"Accuracy: {test_accuracy.item():.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(f"Classification report:\n{report}")
    '''


if __name__ == "__main__":
    train()
