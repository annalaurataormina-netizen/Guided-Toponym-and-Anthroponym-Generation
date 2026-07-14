import torch

from .AE import AE
from .CharVocab import CharVocab
from .NameDataset import NameDataset
from .config import ALLOWED_CHARS


def interpolate():
    # Set device
    device = torch.device('cpu')

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = 512, 64, 64, 2, 0.001, 30

    # Recreate the model architecture first, then load the weights from the saved model
    model = AE(vocab, embed_dim, hidden_dim, num_layers)
    state_dict = torch.load("best_model_bs512_ed64_hd64_nl2_lr0.001_ep30.pt", map_location=device)
    model.load_state_dict(state_dict)

    # Evaluation mode
    model.eval()

    names = ['Ludovico', 'Francesco']
    data = NameDataset(names, vocab)

    x1, len1 = data[0]
    x2, len2 = data[1]

    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)

    len1, len2 = torch.tensor([len1]), torch.tensor([len2])

    with torch.no_grad():
        h1, c1 = model.encoder(x1, len1)
        h2, c2 = model.encoder(x2, len2)

        alpha = 0.5
        h = alpha * h1 + (1 - alpha) * h2
        c = alpha * c1 + (1 - alpha) * c2

        generated = model.decoder.generate(h, c)

        print(generated)


if __name__ == "__main__":
    interpolate()
