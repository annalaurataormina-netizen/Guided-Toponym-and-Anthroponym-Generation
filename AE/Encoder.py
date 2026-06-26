import torch
import torch.nn as nn
from AE.CharVocab import CharVocab

class Encoder(nn.Module):

    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        # class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        # batch_first=False, dropout=0.0, bidirectional=False, proj_size=0,
        # device=None, dtype=None)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        return hn.squeeze(0)
