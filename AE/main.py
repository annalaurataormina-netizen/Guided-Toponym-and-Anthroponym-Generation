import random

import matplotlib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AE.AE import AE
from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from utils import load_toponyms, normalise

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = 32, 16, 16, 1, 0.001, 1

    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms
    names = load_toponyms()
    random.shuffle(names)
    names = names[:100000]

    # Anthroponyms
    # names = load_anthroponyms()

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

    model = AE(vocab, embed_dim, hidden_dim, num_layers)

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # Adam (Adaptive Moment Estimation) is an optimization algorithm used to train deep learning models.
    # It dynamically adjusts the learning rate for every parameter in the model
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_steps = []

    val_losses = []
    val_steps = []

    global_step = 0

    for epoch in range(epochs):

        epoch_train_losses = []

        for train_batch in train_dataloader:

            sequences, lengths = train_batch

            sequences, lengths = sequences.to(device), lengths.cpu()

            optimiser.zero_grad()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict.
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # Returns (batch_size, seq_len, len(vocab))
            logits = model(sequences, lengths)

            # reshape converts logits from (batch, seq_len, len(vocab)) to (batch * seq_len, len(vocab))
            # reshape converts target from (batch, seq_len) to (batch * seq_len,)
            loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

            # Backprop
            loss.backward()
            optimiser.step()

            global_step += 1

            train_losses.append(loss.item())
            train_steps.append(global_step)
            epoch_train_losses.append(loss.item())

            if global_step % 2000 == 0:

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

                val_losses.append(val_loss / len(val_dataloader))
                val_steps.append(global_step)

                model.train()

                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Step {global_step}, "
                    f"avg validation loss = {val_losses[-1]:.4f}"
                )

        print(f"Epoch {epoch + 1}/{epochs}, avg train loss: {sum(epoch_train_losses) / len(epoch_train_losses):.4f}")

    plt.plot(train_steps, train_losses, label="Training")
    plt.plot(val_steps, val_losses, label="Validation")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()
