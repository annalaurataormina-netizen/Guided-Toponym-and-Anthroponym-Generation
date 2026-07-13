import torch

from AE.CharVocab import CharVocab
from AE.NameDataset import NameDataset
from AE.config import ALLOWED_CHARS
from VAE import VAE


def interpolate():
    # Set device
    device = torch.device('cpu')

    # Vocabulary of characters
    vocab = CharVocab(ALLOWED_CHARS)

    # Model hyperparameters
    batch_size, embed_dim, hidden_dim, num_layers, latent_dim, lr, epochs, beta = 512, 64, 64, 2, 64, 0.001, 30, 0.001

    # Recreate the model architecture first, then load the weights from the saved model
    model = VAE(vocab, embed_dim, hidden_dim, num_layers, latent_dim)
    state_dict = torch.load("best_model_bs512_ed64_hd64_nl2_ld64_lr0.001_ep30_b0.001.pt", map_location=device)
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
        _ , mu1, _ = model.encoder(x1, len1)
        _, mu2, _ = model.encoder(x2, len2)

        for alpha in torch.linspace(0, 1, 20):
            z = (1 - alpha) * mu1 + alpha * mu2
            generated = model.decoder.generate(z)
            print(alpha.item(), generated)

if __name__ == "__main__":
    interpolate()
