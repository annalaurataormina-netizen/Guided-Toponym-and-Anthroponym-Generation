import random

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ContrastiveVAE.NameDataset import NameDataset
from CultureClassifier.CultureClassifier import CultureClassifier
from CultureClassifier.LatentDataset import LatentDataset
from CultureClassifier.LatentExtractor import LatentExtractor
from VAE.VAE import VAE
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

    # ContrastiveVAE hyperparameters
    '''
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    proj_hidden_dim, proj_output_dim, temperature, lambda_supcon = 128, 64, 0.07, 0.05
    '''

    # VAE hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5

    # Classifier hyperparameters
    hidden_dim, lr_classifier, epochs_classifier = 128, 0.001, 10

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (name_romanised, label)
    names = load_all(culture=True)

    # Create mapping (language_code -> integer)
    language_to_id = {
        lang: i for i, lang in enumerate(sorted(set(n[1] for n in names)))
    }

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

    # Recreate the model architecture first, then load the weights from the saved model
    # ContrastiveVAE
    '''
    model = ContrastiveVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, proj_hidden_dim, proj_output_dim)
    model_name = f'ContrastiveVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_phd{proj_hidden_dim}_pod{proj_output_dim}_t{temperature}_l{lambda_supcon}.pt'
    '''

    # VAE
    model = VAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder,
                latent_dim)
    model_name = f'VAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'

    model.to(device)
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)

    encoder = model.encoder
    encoder.to(device)
    encoder.eval()

    extractor = LatentExtractor(encoder)

    latent_vectors, cultures = extractor.extract(train_dataset, batch_size, device)
    train_latentdataset = LatentDataset(latent_vectors, cultures)

    latent_vectors, cultures = extractor.extract(val_dataset, batch_size, device)
    val_latentdataset = LatentDataset(latent_vectors, cultures)

    latent_vectors, cultures = extractor.extract(test_dataset, batch_size, device)
    test_latentdataset = LatentDataset(latent_vectors, cultures)

    # Same seed as the one used to split the dataset into train, validation and test, for consistency
    g = torch.Generator()
    g.manual_seed(seed)

    # Shuffling means that batches are random, which is important when training the model
    train_dataloader = DataLoader(train_latentdataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_latentdataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_latentdataset, batch_size=batch_size, shuffle=False)

    number_of_cultures = len(language_to_id)

    # Compute class weights to reduce imbalance bias
    class_counts = torch.bincount(
        train_latentdataset.cultures,
        minlength=number_of_cultures
    )

    class_weights = torch.zeros(number_of_cultures)

    mask = class_counts > 0
    class_weights[mask] = 1.0 / torch.sqrt(class_counts[mask].float())

    # Normalise so the average non-zero weight is 1
    class_weights /= class_weights[mask].mean()

    class_weights = class_weights.to(device)

    classifier = CultureClassifier(latent_dim, hidden_dim, number_of_cultures)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=lr_classifier)

    classifier.train()

    best_loss = float('inf')

    for epoch in range(epochs_classifier):

        epoch_losses = []

        for batch in train_dataloader:
            latent_vectors, labels = batch
            latent_vectors, labels = latent_vectors.to(device), labels.to(device)

            # Zero out the gradients
            optimiser.zero_grad()

            logits = classifier(latent_vectors)

            loss = criterion(logits, labels)

            # Backprop (compute gradients, update model params via backpropagation)
            loss.backward()
            optimiser.step()

            epoch_losses.append(loss.item())

        classifier.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                latent_vectors, labels_batch = batch
                latent_vectors, labels_batch = latent_vectors.to(device), labels_batch.to(device)

                logits = classifier(latent_vectors)

                val_loss = criterion(logits, labels_batch)

                val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                classifier_name = f'CultureClassifier/models/best_model_bs{batch_size}_hd{hidden_dim}_lr{lr_classifier}_ep{epochs_classifier}.pt'
                torch.save(classifier.state_dict(), classifier_name)

        print(
            f"Epoch {epoch + 1}/{epochs_classifier}, "
            f"Avg train loss per epoch: {sum(epoch_losses) / len(epoch_losses):.4f}"
        )

    classifier.eval()

    val_pred_cultures = []
    val_labels = []

    print(
        f"Number of cultures: {number_of_cultures}, "
        f"Random accuracy: {1 / number_of_cultures}"
    )

    classifier.load_state_dict(
        torch.load(classifier_name, map_location=device)
    )

    with torch.no_grad():
        for batch in val_dataloader:
            latent_vectors, labels_batch = batch
            latent_vectors, labels_batch = latent_vectors.to(device), labels_batch.to(device)

            logits = classifier(latent_vectors)
            pred_cultures_batch = logits.argmax(dim=-1)
            val_pred_cultures.append(pred_cultures_batch)
            val_labels.append(labels_batch)

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

    print(
        "VALIDATION"
        f"Accuracy: {val_accuracy.item():.4f}, "
        f"Balanced accuracy: {balanced_acc:.4f}, "
        f"Macro F1: {macro_f1:.4f}, "
        f"Weighted F1: {weighted_f1:.4f}, "
        f"Confusion matrix:\n{conf_matrix}"
        f"Classification report:\n{report}"
    )

    '''
    test_pred_cultures = []
    test_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            latent_vectors, labels_batch = batch
            latent_vectors, labels_batch = latent_vectors.to(device), labels_batch.to(device)

            logits = classifier(latent_vectors)
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
    
    print(
        "TEST"
        f"Accuracy: {test_accuracy.item():.4f}, "
        f"Balanced accuracy: {balanced_acc:.4f}, "
        f"Macro F1: {macro_f1:.4f}, "
        f"Weighted F1: {weighted_f1:.4f}, "
        f"Confusion matrix:\n{conf_matrix}"
        f"Classification report:\n{report}"
    )
    '''


if __name__ == "__main__":
    train()
