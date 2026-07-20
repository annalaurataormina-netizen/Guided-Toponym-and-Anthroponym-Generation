import editdistance
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from utils import load_all, normalise
from .VAE import VAE

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all()

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up  = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    # free_bits = 0.05
    # n_cycles, ratio = 4, 0.5

    # List of name_romanised after normalising (i.e., splitting diacritics)
    names_normalised = [normalise(n) for n in names]

    # 80/20/20 split of the dataset into train/validation/test (uses same seed as when training the model)
    _, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    _, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    # Test dataset
    test_dataset = NameDataset(test_names, vocab)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Recreate the model architecture first, then load the weights from the saved model
    model = VAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim)
    model_name = f'VAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)

    print(f"Model name: {model_name}")

    # Put the model in evaluation mode if you're doing inference
    model.to(device)
    model.eval()

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # Tracks the number of batches
    global_step = 0

    # Loss tracking
    total_reconstruction_loss = 0
    total_kl_loss = 0

    # Levenshtein distance tracking
    total_lev = 0
    total_count = 0

    with torch.no_grad():
        for test_batch in test_dataloader:

            # Batch Levenshtein distance
            batch_lev = 0
            batch_count = 0

            sequences, lengths = test_batch
            sequences, lengths = sequences.to(device), lengths.cpu()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # logits, mu, logvar are (batch_size, seq_len, len(vocab))
            logits, mu, logvar = model(sequences, lengths)

            # Convert predicted indices to characters
            pred_indices = logits.argmax(dim=-1)

            for p, t in zip(pred_indices, target):
                eos_idx = vocab.char2idx['<EOS>']

                # Remove everything after <EOS>
                p_list = p.tolist()
                p_list = p_list[:p_list.index(eos_idx)] if eos_idx in p_list else p_list
                pred_str = vocab.decode(p_list)
                target_str = vocab.decode(t.tolist())

                # Normalised Levenshtein distance
                distance = editdistance.eval(pred_str, target_str) / max(len(pred_str), len(target_str))

                batch_lev += distance
                total_lev += distance
                batch_count += 1
                total_count += 1

            # reshape converts logits from (batch, seq_len, len(vocab)) to (batch * seq_len, len(vocab))
            # reshape converts target from (batch, seq_len) to (batch * seq_len,)
            # CrossEntropyLoss internally applies log_softmax to logits and computes the negative log likelihood loss
            reconstruction_loss = criterion(
                logits.reshape(-1, len(vocab)),
                target.reshape(-1)
            )

            # KL divergence
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1))

            global_step += 1

            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()

            # Prints the loss for every batch
            print(
                f"Step {global_step}, "
                f"Reconstruction loss = {reconstruction_loss.item():.4f}, "
                f"KL divergence = {kl_loss.item():.4f}, "
                f"Avg normalised Levenshtein distance: {batch_lev / batch_count:.4f}"
            )

    print(f"Avg reconstruction loss: {total_reconstruction_loss / len(test_dataloader):.4f}")
    print(f"Avg KL divergence: {total_kl_loss / len(test_dataloader):.4f}")
    print(f"Avg normalised Levenshtein distance: {total_lev / total_count:.4f}")


if __name__ == "__main__":
    test()
