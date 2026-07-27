import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from AE import CharVocab


class CultureClassifier(nn.Module):

    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int, num_cultures: int):
        super().__init__()

        # Character vocabulary
        self.vocab = vocab

        # Dimensionality of character embeddings
        self.embed_dim = embed_dim

        # Dimensionality of the hidden state
        self.hidden_dim = hidden_dim

        # Number of layers
        self.num_layers = num_layers

        # Number of cultures
        self.num_cultures = num_cultures

        # Embedding layer with size (len(vocab), embed_dim)
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        # Bidirectional LSTM
        # class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        # batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        # batch_first returns (batch, seq_len, hidden_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, bias=True, batch_first=True, bidirectional=True)

        # Classifier head
        self.fc = nn.Linear(hidden_dim * 2, num_cultures)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Number of samples
        batch_size = x.size(0)

        # x is (batch_size, seq_len) because each name is a list of indices
        # embedded is (batch, seq_len, embed_dim) because the embedding convert each index
        # to an embedding of size embed_dim
        embedded = self.embedding(x)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # hn, cn are (num_layers * 2, batch_size, hidden_dim)
        _, (hn, _) = self.rnn(packed)

        # Reshape to separate directions: (num_layers, 2, batch_size, hidden_dim)
        hn = hn.view(self.num_layers, 2, batch_size, self.hidden_dim)

        # Concatenate forward and backward directions: (num_layers, batch_size, hidden_dim * 2)
        hn = torch.cat([hn[:, 0], hn[:, 1]], dim=-1)

        # Use final layer hidden state
        hn = hn[-1]

        return self.fc(hn)
