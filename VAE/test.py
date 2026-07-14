import editdistance
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from AE.utils import load_all, normalise
from .VAE import VAE


def test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all()

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim, num_layers, latent_dim, lr, epochs, beta = 512, 64, 64, 2, 64, 0.001, 30, 0.001

    # List of name_romanised after normalising (i.e., splitting diacritics)
    names_normalised = [normalise(n) for n in names]

    # 80/20/20 split of the dataset into train/validation/test (uses same seed as when training the model)
    _, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    _, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    # Test dataset
    test_dataset = NameDataset(test_names, vocab)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Recreate the model architecture first, then load the weights from the saved model
    model = VAE(vocab, embed_dim, hidden_dim, num_layers, latent_dim)
    state_dict = torch.load("best_model_bs512_ed64_hd64_nl2_ld64_lr0.001_ep30_b0.001.pt", map_location=device)
    model.load_state_dict(state_dict)

    # Put the model in evaluation mode if you're doing inference
    model.to(device)
    model.eval()

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # Tracks the number of batches
    global_step = 0

    total_reconstruction_loss = 0
    total_kl_loss = 0
    total_loss = 0
    total_lev = 0
    total_count = 0

    with torch.no_grad():
        for test_batch in test_dataloader:

            batch_lev = 0
            batch_count = 0

            sequences, lengths = test_batch

            sequences, lengths = sequences.to(device), lengths.cpu()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # Returns (batch_size, seq_len, len(vocab))
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
            kl_loss = -0.5 * torch.mean(
                torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(),
                    dim=1
                )
            )

            loss = reconstruction_loss + beta * kl_loss

            global_step += 1

            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()

            # Prints the loss for every batch
            print(
                f"Step {global_step}, "
                f"Reconstruction loss = {reconstruction_loss.item():.4f}, "
                f"KL loss = {kl_loss.item():.4f}, "
                f"Beta KL = {(beta * kl_loss).item():.4f}, "
                f"Total loss = {loss.item():.4f}, "
                f"Average normalised Levenshtein distance: {batch_lev / batch_count:.4f}"
            )

    print(f"Average reconstruction loss: {total_reconstruction_loss / len(test_dataloader):.4f}")
    print(f"Average KL loss: {total_kl_loss / len(test_dataloader):.4f}")
    print(f"Average total loss: {total_loss / len(test_dataloader):.4f}")
    print(f"Average normalised Levenshtein distance: {total_lev / total_count:.4f}")


if __name__ == "__main__":
    test()
