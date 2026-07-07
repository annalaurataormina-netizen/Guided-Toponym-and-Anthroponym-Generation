import torch
import torch.nn as nn

from CharVocab import CharVocab
from Decoder import Decoder
from Encoder import Encoder


class AE(nn.Module):
    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.encoder = Encoder(vocab, embed_dim, hidden_dim, num_layers)
        self.decoder = Decoder(vocab, embed_dim, hidden_dim, num_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hn_encoder, cn_encoder = self.encoder(x, lengths)
        # You don't feed <EOS> since nothing comes after that.
        # Uses teacher forcing.
        decoder_input = x[:, :-1]
        return self.decoder(decoder_input, hn_encoder, cn_encoder)
