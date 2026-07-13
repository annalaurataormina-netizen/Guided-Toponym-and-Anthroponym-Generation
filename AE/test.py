import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AE import AE
from CharVocab import CharVocab
from NameDataset import NameDataset
from config import ALLOWED_CHARS
from utils import load_all, normalise


def test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Toponyms and Anthroponyms (list of name_romanised)
    names = load_all()

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = 512, 64, 64, 2, 0.001, 30

    # List of name_romanised after normalising (i.e., splitting diacritics)
    names_normalised = [normalise(n) for n in names]

    # 80/20/20 split of the dataset into train/validation/test (uses same seed as when training the model)
    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    # Test dataset
    test_dataset = NameDataset(test_names, vocab)

    # Same seed as the one used to split the dataset into train, validation and test, for consistency
    g = torch.Generator()
    g.manual_seed(1996)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Recreate the model architecture first, then load the weights from the saved model
    model = AE(vocab, embed_dim, hidden_dim, num_layers)
    state_dict = torch.load("best_model_bs512_ed64_hd64_nl2_lr0.001_ep30.pt", map_location=device)
    model.load_state_dict(state_dict)

    # Put the model in evaluation mode if you're doing inference
    model.to(device)
    model.eval()

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # Tracks the number of batches
    global_step = 0

    with torch.no_grad():
        for test_batch in test_dataloader:
            sequences, lengths = test_batch

            sequences, lengths = sequences.to(device), lengths.cpu()

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # Returns (batch_size, seq_len, len(vocab))
            logits = model(sequences, lengths)

            # reshape converts logits from (batch, seq_len, len(vocab)) to (batch * seq_len, len(vocab))
            # reshape converts target from (batch, seq_len) to (batch * seq_len,)
            # Automatically does softmax on the logits, converting them to probabilities
            loss = criterion(logits.reshape(-1, len(vocab)), target.reshape(-1))

            global_step += 1

            # Prints the loss for every batch
            print(
                f"Step {global_step}, "
                f"Avg test loss (last 2000 batches) = {loss.item():.4f}"
            )


if __name__ == "__main__":
    test()
