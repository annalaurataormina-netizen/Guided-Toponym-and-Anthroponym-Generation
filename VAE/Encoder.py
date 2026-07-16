import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from AE.CharVocab import CharVocab


class Encoder(nn.Module):

    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int, latent_dim: int):
        super().__init__()

        # Character vocabulary
        self.vocab = vocab

        # Dimensionality of character embeddings
        self.embed_dim = embed_dim

        # Dimensionality of the hidden state
        self.hidden_dim = hidden_dim

        # Number of layers
        self.num_layers = num_layers

        # Size of the latent representation
        self.latent_dim = latent_dim

        # Embedding layer with size (len(vocab), embed_dim)
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        # Bidirectional LSTM
        # class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        # batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        # batch_first returns (batch, seq_len, hidden_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, bias=True, batch_first=True, bidirectional=True)

        # Projection layer from (batch_size, num_layers * hidden_dim * 2 * 2) to (batch_size, latent_dim)
        self.fc_mu = nn.Linear(num_layers * hidden_dim * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(num_layers * hidden_dim * 2 * 2, latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        # Initial hidden states (*2 because it's bidirectional)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)
        # Initial cell states (*2 because it's bidirectional)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)

        # x is (batch_size, seq_len) because each name is a list of indices
        # embedded is (batch, seq_len, embed_dim) because the embedding convert each index
        # to an embedding of size embed_dim
        embedded = self.embedding(x)

        # Converts the padded, embedded tensor into a PackedSequence object — a special format that internally
        # records each sequence's true length, so the LSTM knows exactly how many real timesteps to process per sample.
        # Tells PyTorch to stop processing the sequence at its true length to avoid polluting the final hidden state
        # with all the <PAD> characters.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # packed_output is a PackedSequence object; after calling pad_packed_sequence becomes (batch, seq_len, hidden_dim).
        # hn, cn are (num_layers * 2, batch_size, hidden_dim)
        packed_output, (hn, cn) = self.rnn(packed, (h0, c0))

        # Reshape to separate directions: (num_layers, 2, batch_size, hidden_dim)
        hn = hn.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cn = cn.view(self.num_layers, 2, batch_size, self.hidden_dim)

        # Concatenate forward and backward directions: (num_layers, batch_size, hidden_dim * 2)
        hn = torch.cat([hn[:, 0], hn[:, 1]], dim=-1)
        cn = torch.cat([cn[:, 0], cn[:, 1]], dim=-1)

        # (batch_size, num_layers, hidden_dim * 2)
        hn, cn = hn.permute(1, 0, 2), cn.permute(1, 0, 2)

        # (batch_size, num_layers * hidden_dim * 2)
        hn, cn = hn.reshape(batch_size, -1), cn.reshape(batch_size, -1)

        # (batch_size, num_layers * hidden_dim * 2 * 2)
        z_input = torch.cat([hn, cn], dim=-1)

        # (batch_size, latent_dim)
        # log-variance is numerically more stable to predict
        mu = self.fc_mu(z_input)
        logvar = self.fc_logvar(z_input)

        # Standard deviation
        std = torch.exp(0.5 * logvar)

        # Random noise
        eps = torch.randn_like(std)

        # Sample latent vector
        z = mu + eps * std

        # z, mu, logvar are (batch_size, latent_dim)
        return z, mu, logvar
