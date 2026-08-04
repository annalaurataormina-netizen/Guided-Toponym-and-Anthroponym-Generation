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

    # Classifier hyperparameters
    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = 512, 32, 256, 1, 0.0001, 30

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (name_romanised, label)
    names = load_all(culture=True)

    print(f"Embedding dimension: {embed_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of layers: {num_layers}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    culture_counts = Counter(label for _, label in names_normalised)
    min_samples = 10
    names_normalised = [
        x for x in names_normalised
        if culture_counts[x[1]] >= min_samples
    ]

    # Re-index remaining cultures
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

    num_cultures = len(language_to_id)

    # 80/10/10 split of the dataset into train/validation/test
    labels = [x[1] for x in names_normalised]
    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=seed, shuffle=True, stratify=labels)
    temp_labels = [x[1] for x in temp_names]
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=seed, shuffle=True, stratify=temp_labels)

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

    classifier = CultureClassifier(vocab, embed_dim, hidden_dim, num_layers, num_cultures)
    classifier.to(device)

    counts = Counter(label for _, label in train_names)
    class_weights = torch.tensor(
        [
            len(train_names) / (num_cultures * counts[i])
            for i in range(num_cultures)
        ],
        dtype=torch.float,
        device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimiser = torch.optim.Adam(classifier.parameters(), lr=lr)

    best_loss = float('inf')

    patience = 5
    epochs_without_improvement = 0

    for epoch in range(epochs):

        classifier.train()

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
                classifier_name = f'CultureClassifier/models/best_model_bs{batch_size}_ed{embed_dim}_hd{hidden_dim}_nl{num_layers}_lr{lr}_ep{epochs}.pt'
                torch.save(classifier.state_dict(), classifier_name)

            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

        print(f"Epoch {epoch + 1}/{epochs}, ")
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
