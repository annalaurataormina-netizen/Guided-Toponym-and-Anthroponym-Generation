import random

import editdistance
import matplotlib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .VAE import VAE
from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from utils import load_all, normalise, cyclical_beta


def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model hyperparameters (there's also dropout, L2 regularisation, Adam vs other optimisers)
    batch_size, embed_dim, hidden_dim, num_layers, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up, free_bits = 512, 64, 64, 2, 64, 0.0015, 30, 0.005, 5, 0.05
    n_cycles, ratio = 4, 0.5

    # Hyperparameter used for early stopping: if performance doesn't improve for patience times when evaluating
    # the model (done every 2000 batches) on the entire validation set, then early stopping is triggered
    patience = 10

    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of layers: {num_layers}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print("Optimiser: Adam")
    print("No regularisation or dropout")
    print("Bidirectional encoder")
    print(f"Early stopping (with patience {patience})")
    # print(f"Linear ramp-up of beta over the first {n_epochs_ramp_up} epochs from 0 to {beta_max}")
    print(f"Cyclical ramp-up of beta from 0 to {beta_max} over {n_cycles} cycles and with ratio of {ratio}")
    # print(f"Free bits with {free_bits}")
    print("No free bits")
    print(f"Word dropout at 25%")

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all()

    # List of name_romanised after normalising (i.e., splitting diacritics)
    names_normalised = [normalise(n) for n in names]

    # 80/20/20 split of the dataset into train/validation/test
    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    train_dataset = NameDataset(train_names, vocab)
    val_dataset = NameDataset(val_names, vocab)

    # Same seed as the one used to split the dataset into train, validation and test, for consistency
    g = torch.Generator()
    g.manual_seed(1996)

    # Shuffling means that batches are random, which is important when training the model
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Levenshtein (uses the same 1000 random samples from the validation set)
    rng = random.Random(1996)
    lev_indices = rng.sample(range(len(val_dataset)), 1000)
    lev_subset = Subset(val_dataset, lev_indices)
    lev_dataloader = DataLoader(lev_subset, batch_size=batch_size, shuffle=False)

    # Variational Autoencoder (hidden_dim and num_layers are the same for both the encoder and decoder)
    model = VAE(vocab, embed_dim, hidden_dim, num_layers, latent_dim)

    # Move model to device
    model.to(device)

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # Adam (Adaptive Moment Estimation) dynamically adjusts the learning rate for every parameter in the model
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Training losses, one for each batch
    train_losses = []
    train_kl_losses = []
    train_kl_losses_adj = []
    train_reconstruction_losses = []
    train_steps = []

    # Validation losses (for the whole validation set), one every 2000 batches
    val_losses = []
    val_kl_losses = []
    val_kl_losses_adj = []
    val_reconstruction_losses = []
    val_steps = []

    # Tracks the number of batches
    global_step = 0

    best_loss = float('inf')

    # For early stopping
    wait = 0
    early_stopping = False

    for epoch in range(epochs):

        if early_stopping:
            break

        epoch_train_losses = []
        epoch_train_kl_losses = []
        epoch_train_kl_losses_adj = []
        epoch_train_reconstruction_losses = []

        for batch_idx, train_batch in enumerate(train_dataloader):

            warmup_steps = len(train_dataloader) * n_epochs_ramp_up

            # Linear annealing
            '''
            if global_step < warmup_steps:
                beta = beta_max * global_step / warmup_steps
            else:
                beta = beta_max
            '''

            # Cyclical annealing
            total_steps = len(train_dataloader) * epochs
            beta = cyclical_beta(global_step, total_steps, n_cycles, ratio, beta_max)

            sequences, lengths = train_batch
            sequences, lengths = sequences.to(device), lengths.cpu()

            # Zero out the gradients
            optimiser.zero_grad()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # Returns (batch_size, seq_len, len(vocab)), (batch_size, latent_dim), (batch_size, latent_dim)
            logits, mu, logvar = model(sequences, lengths)

            # reshape converts logits from (batch, seq_len, len(vocab)) to (batch * seq_len, len(vocab))
            # reshape converts target from (batch, seq_len) to (batch * seq_len,)
            # CrossEntropyLoss internally applies log_softmax to logits and computes the negative log likelihood loss
            reconstruction_loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

            # KL divergence (w/free bits)
            '''
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
            kl_loss = kl_per_dim.sum(dim=1).mean()
            '''

            # KL divergence (w/o free bits)
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            # TotalLoss = Reconstructionloss + Beta * KLDivergence
            loss = reconstruction_loss + beta * kl_loss

            # Backprop (compute gradients, update model params via backpropagation)
            loss.backward()
            optimiser.step()

            global_step += 1

            train_losses.append(loss.item())
            train_kl_losses.append(kl_loss.item())
            train_kl_losses_adj.append(kl_loss.item() * beta)
            train_reconstruction_losses.append(reconstruction_loss.item())
            train_steps.append(global_step)

            epoch_train_losses.append(loss.item())
            epoch_train_reconstruction_losses.append(reconstruction_loss.item())
            epoch_train_kl_losses.append(kl_loss.item())
            epoch_train_kl_losses_adj.append(kl_loss.item() * beta)

            # Every 2000 batches, model is evaluated on the full evaluation set
            if global_step % 2_000 == 0:

                model.eval()
                val_loss = 0
                val_kl_loss = 0
                val_kl_loss_adj = 0
                val_reconstruction_loss = 0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        sequences, lengths = val_batch
                        sequences, lengths = sequences.to(device), lengths.cpu()

                        target = sequences[:, 1:]
                        logits, mu, logvar = model(sequences, lengths)

                        reconstruction_loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

                        # KL divergence (w/o free bits)
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / sequences.size(0)

                        # KL divergence (w/free bits)
                        '''
                        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
                        kl_loss = kl_per_dim.sum(dim=1).mean()
                        '''

                        loss = reconstruction_loss + beta * kl_loss

                        val_loss += loss.item()
                        val_kl_loss += kl_loss.item()
                        val_kl_loss_adj += kl_loss.item() * beta
                        val_reconstruction_loss += reconstruction_loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                avg_kl_loss = val_kl_loss / len(val_dataloader)
                avg_kl_loss_adj = val_kl_loss_adj / len(val_dataloader)
                avg_reconstruction_loss = val_reconstruction_loss / len(val_dataloader)

                # For early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    wait = 0
                    model_name = f'VAE/models/best_model_bs{batch_size}_ed{embed_dim}_hd{hidden_dim}_nl{num_layers}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'
                    torch.save(model.state_dict(), model_name)

                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping")
                        early_stopping = True
                        break

                val_losses.append(avg_val_loss)
                val_kl_losses.append(avg_kl_loss)
                val_kl_losses_adj.append(avg_kl_loss_adj)
                val_reconstruction_losses.append(avg_reconstruction_loss)
                val_steps.append(global_step)

                model.train()

                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Step {global_step}, "
                    f"Beta = {beta:.5f}, "
                    f"Avg validation loss (full validation set) = {val_losses[-1]:.4f}, "
                    f"Avg validation reconstruction loss (full validation set) = {val_reconstruction_losses[-1]:.4f}, "
                    f"Avg validation KL divergence (full validation set) = {val_kl_losses[-1]:.4f}, "
                    f"Avg validation beta-adjusted KL divergence (full validation set) = {val_kl_losses_adj[-1]:.4f}, "
                    f"Avg training loss (last 2000 batches) = {sum(train_losses[-2000:]) / 2000:.4f}, "
                    f"Avg training reconstruction loss (last 2000 batches) = {sum(train_reconstruction_losses[-2000:]) / 2000:.4f}, "
                    f"Avg training beta-adjusted KL divergence (last 2000 batches) = {sum(train_kl_losses_adj[-2000:]) / 2000:.4f}"
                )

                if early_stopping:
                    break

            # Every 15000 batches, Levenshtein distance is calculated on 1000 (always the same) random
            # samples from the validation set
            if global_step % 15_000 == 0:

                model.eval()

                total_lev = 0
                count = 0

                with torch.no_grad():
                    for lev_batch in lev_dataloader:
                        sequences, lengths = lev_batch
                        sequences, lengths = sequences.to(device), lengths.cpu()

                        target = sequences[:, 1:]
                        logits, mu, logvar = model(sequences, lengths)

                        # (batch_size, seq_len)
                        pred_indices = logits.argmax(dim=-1)

                        for p, t in zip(pred_indices, target):

                            eos_idx = vocab.char2idx['<EOS>']
                            p_list = p.tolist()
                            p_list = p_list[:p_list.index(eos_idx)] if eos_idx in p_list else p_list
                            pred_str = vocab.decode(p_list)
                            target_str = vocab.decode(t.tolist())

                            # Normalised Levenshtein distance between target and predicted strings
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
                    f"Avg normalised Levenshtein distance (portion of validation set) = {(total_lev / count):.4f}"
                )

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Avg train loss per epoch: {sum(epoch_train_losses) / len(epoch_train_losses):.4f}, "
            f"Avg reconstruction loss per epoch: {sum(epoch_train_reconstruction_losses) / len(epoch_train_reconstruction_losses):.4f}, "
            f"Avg KL divergence per epoch: {sum(epoch_train_kl_losses) / len(epoch_train_kl_losses):.4f}, "
            f"Avg beta-adjusted KL divergence per epoch: {sum(epoch_train_kl_losses_adj) / len(epoch_train_kl_losses_adj):.4f}"
        )

    plt.plot(train_steps, train_losses, label="Training")
    plt.plot(val_steps, val_losses, label="Validation")
    plt.plot(val_steps, val_reconstruction_losses, label="Validation Reconstruction")
    plt.plot(val_steps, val_kl_losses, label="Validation KL")
    plt.plot(train_steps, train_reconstruction_losses, label="Training Reconstruction")
    plt.plot(train_steps, train_kl_losses, label="Training KL")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    fig_name = f'VAE/plots/loss_bs{batch_size}_ed{embed_dim}_hd{hidden_dim}_nl{num_layers}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.png'
    plt.savefig(fig_name)
    plt.close()

    model.eval()


if __name__ == "__main__":
    train()
