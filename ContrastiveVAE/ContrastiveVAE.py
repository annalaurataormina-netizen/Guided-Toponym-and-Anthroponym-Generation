from typing import Any

import torch
import torch.nn as nn

from AE.CharVocab import CharVocab
from VAE.Decoder import Decoder
from VAE.Encoder import Encoder


class ContrastiveVAE(nn.Module):
    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim_encoder: int, hidden_dim_decoder: int,
                 num_layers_encoder: int, num_layers_decoder: int, latent_dim: int, proj_hidden_dim: int,
                 proj_output_dim: int, culture_stats_path=None):
        super().__init__()

        if culture_stats_path is not None:
            self.culture_stats = torch.load(culture_stats_path, map_location="cpu")
        else:
            self.culture_stats = None

        self.latent_dim = latent_dim

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

    def generate(self, culture, n, max_length=50):
        self.eval()

        generated_names = []

        device = next(self.parameters()).device

        with torch.no_grad():

            culture_mu = self.culture_stats[culture]["mean"].to(device)
            culture_std = self.culture_stats[culture]["std"].to(device)

            z = culture_mu + culture_std * torch.randn(n, self.latent_dim, device=device)

            current = torch.full((n, 1), self.decoder.vocab.sos_idx, dtype=torch.long, device=device)

            finished = torch.zeros(n, dtype=torch.bool, device=device)

            for _ in range(max_length):

                logits = self.decoder(current, z)

                next_tokens = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)

                current = torch.cat([current, next_tokens], dim=1)

                finished |= next_tokens.squeeze(1) == self.decoder.vocab.eos_idx

                if finished.all():
                    break

            for sequence in current:
                generated_names.append(self.decoder.vocab.decode(sequence.tolist()))

        return generated_names
