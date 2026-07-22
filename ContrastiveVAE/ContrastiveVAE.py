from typing import Any

import torch
import torch.nn as nn

from AE.CharVocab import CharVocab
from VAE.Decoder import Decoder
from VAE.Encoder import Encoder


class ContrastiveVAE(nn.Module):
    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim_encoder: int, hidden_dim_decoder: int, num_layers_encoder: int, num_layers_decoder: int, latent_dim: int, proj_hidden_dim: int, proj_output_dim: int):
        super().__init__()

        # Encoder
        self.encoder = Encoder(vocab, embed_dim, hidden_dim_encoder, num_layers_encoder, latent_dim)

        # Decoder
        self.decoder = Decoder(vocab, embed_dim, hidden_dim_decoder, num_layers_decoder, latent_dim)

        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[Any, Any, Any, Any]:
        # Encode input into latent distribution and sample z
        z, mu, logvar = self.encoder(x, lengths)

        # You don't feed <EOS> since nothing comes after that. Uses teacher forcing.
        decoder_input = x[:, :-1]

        # Logits are (batch_size, seq_len, len(vocab))
        logits = self.decoder(decoder_input, z)

        # SupCon projection is (batch_size, proj_output_dim)
        projection = self.projection_head(mu)

        # The decoder reconstructs the sequence from z using teacher forcing
        return logits, mu, logvar, projection