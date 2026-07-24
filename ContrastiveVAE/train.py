import random

import editdistance
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from ContrastiveVAE.LabelBalancedBatchSampler import LabelBalancedBatchSampler
from ContrastiveVAE.losses import SupConLoss

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ContrastiveVAE.ContrastiveVAE import ContrastiveVAE
from AE.CharVocab import CharVocab
from ContrastiveVAE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from utils import load_all, normalise


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

    # Model hyperparameters (there's also dropout, L2 regularisation, Adam vs other optimisers)
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5
    proj_hidden_dim, proj_output_dim, temperature, lambda_supcon = 128, 64, 0.1, 0.25

    # Hyperparameter used for early stopping: if performance doesn't improve for patience times when evaluating
    # the model (done every 2000 batches) on the entire validation set, then early stopping is triggered
    patience = 10

    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Hidden dimension of the encoder: {hidden_dim_encoder}")
    print(f"Hidden dimension of the decoder: {hidden_dim_decoder}")
    print(f"Number of layers of the encoder: {num_layers_encoder}")
    print(f"Number of layers of the decoder: {num_layers_decoder}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print("Optimiser: Adam")
    print("Bidirectional encoder")
    print(f"Early stopping (with patience {patience})")
    print(f"Linear ramp-up of beta over the first {n_epochs_ramp_up} epochs from 0 to {beta_max}")
    # print(f"Cyclical ramp-up of beta from 0 to {beta_max} over {n_cycles} cycles and with ratio of {ratio}")
    # print(f"Free bits with {free_bits}")
    print("No free bits")
    print(f"Character dropout at 25%")
    print("Contrastive loss: Supervised Contrastive Loss (SupCon)")
    print(f"Projection head hidden dimension: {proj_hidden_dim}")
    print(f"Projection head output dimension: {proj_output_dim}")
    print(f"Temperature: {temperature}")
    print(f"Lambda: {lambda_supcon}")

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
    val_names, _ = train_test_split(temp_names, test_size=0.5, random_state=seed, shuffle=True)

    train_dataset = NameDataset(train_names, vocab)
    val_dataset = NameDataset(val_names, vocab)


    # Same seed as the one used to split the dataset into train, validation and test, for consistency
    g = torch.Generator()
    g.manual_seed(seed)

    # DataLoader with LabelBalancedBatchSampler
    '''
    labels = [label for _, _, label in train_dataset]
    batch_sampler = LabelBalancedBatchSampler(labels=labels, batch_size=batch_size, samples_per_class=4)
    # Shuffling means that batches are random, which is important when training the model
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    '''

    # Plain DataLoader
    # Shuffling means that batches are random, which is important when training the model
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Levenshtein (uses the same 1000 random samples from the validation set)
    rng = random.Random(seed)
    lev_indices = rng.sample(range(len(val_dataset)), 1000)
    lev_subset = Subset(val_dataset, lev_indices)
    lev_dataloader = DataLoader(lev_subset, batch_size=batch_size, shuffle=False)

    # Variational Autoencoder (hidden_dim and num_layers are the same for both the encoder and decoder)
    model = ContrastiveVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, proj_hidden_dim, proj_output_dim)

    # Move model to device
    model.to(device)

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # SupCon criterion
    supcon_criterion = SupConLoss(temperature=temperature)

    # Adam (Adaptive Moment Estimation) dynamically adjusts the learning rate for every parameter in the model
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Training losses, one for each batch
    train_steps = []
    train_losses = []
    train_reconstruction_losses = []
    train_kl_losses = []
    train_kl_losses_adj = []
    train_supcon_losses = []
    train_supcon_losses_adj = []

    # Validation losses (for the whole validation set), one every 2000 batches
    val_steps = []
    val_losses = []
    val_reconstruction_losses = []
    val_kl_losses = []
    val_kl_losses_adj = []
    val_supcon_losses = []
    val_supcon_losses_adj = []

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
        epoch_train_reconstruction_losses = []
        epoch_train_kl_losses = []
        epoch_train_kl_losses_adj = []
        epoch_train_supcon_losses = []
        epoch_train_supcon_losses_adj = []

        for batch_idx, train_batch in enumerate(train_dataloader):

            warmup_steps = len(train_dataloader) * n_epochs_ramp_up

            # Linear annealing
            if global_step < warmup_steps:
                beta = beta_max * global_step / warmup_steps
            else:
                beta = beta_max

            # Cyclical annealing
            '''
            total_steps = len(train_dataloader) * epochs
            beta = cyclical_beta(global_step, total_steps, n_cycles, ratio, beta_max)
            '''

            sequences, lengths, labels = train_batch
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

            # Zero out the gradients
            optimiser.zero_grad()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # Returns (batch_size, seq_len, len(vocab)), (batch_size, latent_dim), (batch_size, latent_dim)
            logits, mu, logvar, projection = model(sequences, lengths)

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

            # SupCon loss
            projection = F.normalize(projection, dim=1)
            projection = projection.unsqueeze(1)
            supcon_loss = supcon_criterion(projection, labels)

            # Total Loss = Reconstruction loss + Beta * KL Divergence + Lambda * SupCon Loss
            loss = reconstruction_loss + beta * kl_loss + lambda_supcon * supcon_loss

            # Backprop (compute gradients, update model params via backpropagation)
            loss.backward()
            optimiser.step()

            global_step += 1

            train_steps.append(global_step)
            train_losses.append(loss.item())
            train_reconstruction_losses.append(reconstruction_loss.item())
            train_kl_losses.append(kl_loss.item())
            train_kl_losses_adj.append(kl_loss.item() * beta)
            train_supcon_losses.append(supcon_loss.item())
            train_supcon_losses_adj.append(lambda_supcon * supcon_loss.item())

            epoch_train_losses.append(loss.item())
            epoch_train_reconstruction_losses.append(reconstruction_loss.item())
            epoch_train_kl_losses.append(kl_loss.item())
            epoch_train_kl_losses_adj.append(kl_loss.item() * beta)
            epoch_train_supcon_losses.append(supcon_loss.item())
            epoch_train_supcon_losses_adj.append(lambda_supcon * supcon_loss.item())

            # Every 2000 batches, model is evaluated on the full evaluation set
            if global_step % 2_000 == 0:

                model.eval()
                val_loss = 0
                val_reconstruction_loss = 0
                val_kl_loss = 0
                val_kl_loss_adj = 0
                val_supcon_loss = 0
                val_supcon_loss_adj = 0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        sequences, lengths, labels = val_batch
                        sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

                        target = sequences[:, 1:]
                        logits, mu, logvar, projection = model(sequences, lengths)

                        reconstruction_loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

                        # KL divergence (w/o free bits)
                        kl_loss = -0.5 * torch.mean(
                            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                        )

                        # KL divergence (w/free bits)
                        '''
                        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
                        kl_loss = kl_per_dim.sum(dim=1).mean()
                        '''

                        # SupCon loss
                        projection = F.normalize(projection, dim=1)
                        projection = projection.unsqueeze(1)
                        supcon_loss = supcon_criterion(projection, labels)

                        loss = reconstruction_loss + beta * kl_loss + lambda_supcon * supcon_loss

                        val_loss += loss.item()
                        val_reconstruction_loss += reconstruction_loss.item()
                        val_kl_loss += kl_loss.item()
                        val_kl_loss_adj += beta * kl_loss.item()
                        val_supcon_loss += supcon_loss.item()
                        val_supcon_loss_adj += lambda_supcon * supcon_loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                avg_reconstruction_loss = val_reconstruction_loss / len(val_dataloader)
                avg_kl_loss = val_kl_loss / len(val_dataloader)
                avg_kl_loss_adj = val_kl_loss_adj / len(val_dataloader)
                avg_supcon_loss = val_supcon_loss / len(val_dataloader)
                avg_supcon_loss_adj = val_supcon_loss_adj / len(val_dataloader)

                # For early stopping (can only kick in after beta has stabilised)
                if epoch >= n_epochs_ramp_up and avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    wait = 0
                    model_name = f'ContrastiveVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_phd{proj_hidden_dim}_pod{proj_output_dim}_t{temperature}_l{lambda_supcon}.pt'
                    torch.save(model.state_dict(), model_name)

                elif epoch >= n_epochs_ramp_up:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping")
                        early_stopping = True
                        break

                val_steps.append(global_step)
                val_losses.append(avg_val_loss)
                val_reconstruction_losses.append(avg_reconstruction_loss)
                val_kl_losses.append(avg_kl_loss)
                val_kl_losses_adj.append(avg_kl_loss_adj)
                val_supcon_losses.append(avg_supcon_loss)
                val_supcon_losses_adj.append(avg_supcon_loss_adj)

                model.train()

                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Step {global_step}, "
                    f"Beta = {beta:.5f}, "
                    f"Avg validation loss (full validation set) = {val_losses[-1]:.4f}, "
                    f"Avg validation reconstruction loss (full validation set) = {val_reconstruction_losses[-1]:.4f}, "
                    f"Avg validation KL divergence (full validation set) = {val_kl_losses[-1]:.4f}, "
                    f"Avg validation beta-adjusted KL divergence (full validation set) = {val_kl_losses_adj[-1]:.4f}, "
                    f"Avg validation SupCon loss (full validation set) = {val_supcon_losses[-1]:.4f}, "
                    f"Avg validation lambda-adjusted SupCon loss (full validation set) = {val_supcon_losses_adj[-1]:.4f}, "
                    f"Avg training loss (last 2000 batches) = {sum(train_losses[-2000:]) / 2000:.4f}, "
                    f"Avg training reconstruction loss (last 2000 batches) = {sum(train_reconstruction_losses[-2000:]) / 2000:.4f}, "
                    f"Avg training beta-adjusted KL divergence (last 2000 batches) = {sum(train_kl_losses_adj[-2000:]) / 2000:.4f}, "
                    f"Avg training SupCon loss (last 2000 batches) = {sum(train_supcon_losses[-2000:]) / 2000:.4f}, "
                    f"Avg training lambda-adjusted SupCon loss (last 2000 batches) = {sum(train_supcon_losses_adj[-2000:]) / 2000:.4f}"
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
                        sequences, lengths, labels = lev_batch
                        sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

                        target = sequences[:, 1:]
                        logits, mu, logvar, projection = model(sequences, lengths)

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
            f"Avg beta-adjusted KL divergence per epoch: {sum(epoch_train_kl_losses_adj) / len(epoch_train_kl_losses_adj):.4f}, "
            f"Avg SupCon loss per epoch: {sum(epoch_train_supcon_losses) / len(epoch_train_supcon_losses):.4f}, "
            f"Avg lambda-adjusted SupCon loss per epoch: {sum(epoch_train_supcon_losses_adj) / len(epoch_train_supcon_losses_adj):.4f}"
        )

    base_fig_name = f'loss_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_phd{proj_hidden_dim}_pod{proj_output_dim}_t{temperature}_l{lambda_supcon}'

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Training")
    plt.plot(val_steps, val_losses, label="Validation")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Total Loss over time")
    plt.legend()
    plt.savefig(f"ContrastiveVAE/plots/total_{base_fig_name}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_reconstruction_losses, label="Training Reconstruction")
    plt.plot(val_steps, val_reconstruction_losses, label="Validation Reconstruction")
    plt.plot(train_steps, train_kl_losses_adj, label="Beta-adjusted Training KL")
    plt.plot(val_steps, val_kl_losses_adj, label="Beta-adjusted Validation KL")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("VAE Loss over time")
    plt.legend()
    plt.savefig(f"ContrastiveVAE/plots/vae_{base_fig_name}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_supcon_losses, label="Training SupCon")
    plt.plot(val_steps, val_supcon_losses, label="Validation SupCon")
    plt.plot(train_steps, train_supcon_losses_adj, label="Lambda-adjusted Training SupCon")
    plt.plot(val_steps, val_supcon_losses_adj, label="Lambda-adjusted Validation SupCon")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("SupCon Loss over time")
    plt.legend()
    plt.savefig(f"ContrastiveVAE/plots/supcon_{base_fig_name}.png", bbox_inches="tight")
    plt.close()

    model.eval()


if __name__ == "__main__":
    train()
