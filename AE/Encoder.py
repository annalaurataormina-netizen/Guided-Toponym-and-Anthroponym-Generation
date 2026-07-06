import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from CharVocab import CharVocab


class Encoder(nn.Module):

    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.vocab = vocab

        # Dimensionality of character embeddings
        self.embed_dim = embed_dim
        # Dimensionality of the hidden state
        self.hidden_dim = hidden_dim
        # Number of layers in the RNN
        self.num_layers = num_layers

        # Embedding layer with size (len(vocab), embed_dim)
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        # Unidirectional LSTM
        # class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        # batch_first=False, dropout=0.0, bidirectional=False, proj_size=0,
        # device=None, dtype=None)
        # batch_first returns (batch, seq_len, hidden_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, bias=True, batch_first=True, bidirectional=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Number of samples
        batch_size = x.size(0)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        # Initial cell state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        # x is (batch_size, seq_len)
        # embedded is (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Converts the padded, embedded tensor into a PackedSequence object — a special format that internally
        # records each sequence's true length, so the LSTM knows exactly how many real timesteps to process per sample.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # packed_output is a PackedSequence object; after calling pad_packed_sequence becomes (batch, seq_len, hidden_dim).
        # hn, cn are (num_layers, batch_size, hidden_dim)
        packed_output, (hn, cn) = self.rnn(packed, (h0, c0))

        return hn, cn
