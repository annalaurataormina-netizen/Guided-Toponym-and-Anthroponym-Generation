import torch
import torch.nn as nn

from AE.CharVocab import CharVocab
from .Decoder import Decoder
from .Encoder import Encoder


class VAE(nn.Module):
    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim_encoder: int, hidden_dim_decoder: int, num_layers_encoder: int, num_layers_decoder: int, latent_dim: int):
        super().__init__()

        # Encoder
        self.encoder = Encoder(vocab, embed_dim, hidden_dim_encoder, num_layers_encoder, latent_dim)

        # Decoder
        self.decoder = Decoder(vocab, embed_dim, hidden_dim_decoder, num_layers_decoder, latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Encode input into latent distribution and sample z
        z, mu, logvar = self.encoder(x, lengths)

        # You don't feed <EOS> since nothing comes after that. Uses teacher forcing.
        decoder_input = x[:, :-1]

        # The decoder reconstructs the sequence from z using teacher forcing
        return self.decoder(decoder_input, z), mu, logvar
