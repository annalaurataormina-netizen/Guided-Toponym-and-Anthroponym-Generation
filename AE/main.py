import random
import editdistance

import matplotlib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AE import AE
from CharVocab import CharVocab
from NameDataset import NameDataset
from config import ALLOWED_CHARS
from utils import load_all, normalise

if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model hyperparameters (there's also dropout, L2 regularisation, Adam vs other optimisers)
    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = 512, 32, 32, 2, 0.001, 30

    print("Batch size: ", batch_size)
    print("Embedding dimension: ", embed_dim)
    print("Hidden dimension: ", hidden_dim)
    print("Number of layers: ", num_layers)
    print("Learning rate: ", lr)
    print("Epochs: ", epochs)
    print("Optimiser: Adam")
    print("No regularisation or dropout")
    print("Bidirectional encoder")
    print("Early stopping")

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms (list of name_romanised)
    # names = load_toponyms()

    # Anthroponyms (list of name_romanised)
    # names = load_anthroponyms()

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all()

    # List of name_romanised after splitting diacritics and lowercasing
    names_normalised = [normalise(n) for n in names]

    # 80/20/20 split
    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    train_dataset = NameDataset(train_names, vocab)
    val_dataset = NameDataset(val_names, vocab)
    test_dataset = NameDataset(test_names, vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Levenshtein
    lev_indices = random.sample(range(len(val_dataset)), 1000)
    lev_subset = Subset(val_dataset, lev_indices)
    lev_dataloader = DataLoader(lev_subset, batch_size=batch_size, shuffle=False)

    # Autoencoder
    model = AE(vocab, embed_dim, hidden_dim, num_layers)

    # Move Autoencoder to device
    model.to(device)

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # Adam (Adaptive Moment Estimation) dynamically adjusts the learning rate for every parameter in the model
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_steps = []

    val_losses = []
    val_steps = []

    global_step = 0

    # For early stopping
    best_loss = float('inf')
    patience, wait = 50, 0
    early_stopping = False

    for epoch in range(epochs):

        if early_stopping:
            break

        epoch_train_losses = []

        for train_batch in train_dataloader:

            sequences, lengths = train_batch

            sequences, lengths = sequences.to(device), lengths.cpu()

            # Zero out the gradients
            optimiser.zero_grad()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict.
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # Returns (batch_size, seq_len, len(vocab))
            logits = model(sequences, lengths)

            # reshape converts logits from (batch, seq_len, len(vocab)) to (batch * seq_len, len(vocab))
            # reshape converts target from (batch, seq_len) to (batch * seq_len,)
            # Automatically does softmax on the logits, converting them to probabilities.
            loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

            # Backprop (compute gradients, update model params via backpropagation)
            loss.backward()
            optimiser.step()

            global_step += 1

            train_losses.append(loss.item())
            train_steps.append(global_step)
            epoch_train_losses.append(loss.item())

            if global_step % 2_000 == 0:

                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        sequences, lengths = val_batch
                        sequences, lengths = sequences.to(device), lengths.cpu()

                        target = sequences[:, 1:]
                        logits = model(sequences, lengths)
                        loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

                        val_loss += loss.item()

                # For early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    torch.save(model.state_dict(), "best_model.pt")

                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping")
                        early_stopping = True
                        break

                val_losses.append(val_loss / len(val_dataloader))
                val_steps.append(global_step)

                model.train()

                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Step {global_step}, "
                    f"Avg validation loss = {val_losses[-1]:.4f}"
                )

                if early_stopping:
                    break

            if global_step % 15_000 == 0:

                model.eval()

                total_lev = 0.00
                count = 0

                with torch.no_grad():
                    for lev_batch in lev_dataloader:
                        sequences, lengths = lev_batch
                        sequences, lengths = sequences.to(device), lengths.cpu()

                        target = sequences[:, 1:]
                        logits = model(sequences, lengths)

                        # (batch_size, seq_len)
                        pred_indices = logits.argmax(dim=-1)

                        for p, t in zip(pred_indices, target):
                            eos_idx = vocab.char2idx['<EOS>']
                            p_list = p.tolist()
                            p_list = p_list[:p_list.index(eos_idx)] if eos_idx in p_list else p_list
                            pred_str = vocab.decode(p_list)
                            target_str = vocab.decode(t.tolist())
                            total_lev += editdistance.eval(pred_str, target_str) / max(len(pred_str), len(target_str))
                            if count < 5:
                                print(
                                    f"Epoch {epoch + 1}/{epochs}, "
                                    f"Step {global_step}, "
                                    f"Target name = {target_str}, "
                                    f"Name reconstructed = {pred_str}"
                                )
                            count += 1

                model.train()

                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Step {global_step}, "
                    f"Avg Levenshtein distance = {(total_lev / count):.4f}"
                )

        print(f"Epoch {epoch + 1}/{epochs}, Avg train loss: {sum(epoch_train_losses) / len(epoch_train_losses):.4f}")

    plt.plot(train_steps, train_losses, label="Training")
    plt.plot(val_steps, val_losses, label="Validation")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()
