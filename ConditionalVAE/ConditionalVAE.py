import torch
import torch.nn as nn

from AE.CharVocab import CharVocab
from .Decoder import Decoder
from .Encoder import Encoder


class ConditionalVAE(nn.Module):
    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim_encoder: int, hidden_dim_decoder: int,
                 num_layers_encoder: int, num_layers_decoder: int, latent_dim: int, num_cultures: int,
                 culture_embed_dim: int, culture_stats_path=None):
        super().__init__()

        if culture_stats_path is not None:
            self.culture_stats = torch.load(culture_stats_path, map_location="cpu")
        else:
            self.culture_stats = None

        self.latent_dim = latent_dim

        culture_embedding = nn.Embedding(num_cultures, culture_embed_dim)

        # Encoder
        self.encoder = Encoder(vocab, embed_dim, hidden_dim_encoder, num_layers_encoder, latent_dim, num_cultures,
                               culture_embed_dim, culture_embedding)

        # Decoder
        self.decoder = Decoder(vocab, embed_dim, hidden_dim_decoder, num_layers_decoder, latent_dim, num_cultures,
                               culture_embed_dim, culture_embedding)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode input into latent distribution and sample z
        z, mu, logvar = self.encoder(x, lengths, labels)

        # You don't feed <EOS> since nothing comes after that. Uses teacher forcing.
        decoder_input = x[:, :-1]

        logits, decoder_hidden = self.decoder(decoder_input, z, labels)

        # The decoder reconstructs the sequence from z using teacher forcing
        return logits, decoder_hidden, mu, logvar

    @torch.no_grad()
    def generate(self, culture=None, culture_embedding=None, n=1, max_length=50, temperature=1):
        '''
        self.eval()

        device = next(self.parameters()).device

        z = temperature * torch.randn(n, self.latent_dim, device=device)

        if culture_embedding is not None:
            culture_embedding = culture_embedding.to(device)
            culture_embedding = culture_embedding.expand(n, -1)

        elif culture is not None:
            labels = torch.full((n,), culture, dtype=torch.long, device=device)
            culture_embedding = self.decoder.culture_embedding(labels)

        else:
            raise ValueError("Either culture or culture_embedding must be provided.")

        names = self.decoder.generate(z, culture_embedding=culture_embedding, max_len=max_length)
        return names
        '''

        self.eval()

        device = next(self.parameters()).device

        with torch.no_grad():

            culture_mu = self.culture_stats[culture]["mean"].to(device)
            culture_std = self.culture_stats[culture]["std"].to(device)

            z = culture_mu + culture_std * torch.randn(n, self.latent_dim, device=device)

            if culture_embedding is not None:
                culture_embedding = culture_embedding.to(device)
                culture_embedding = culture_embedding.expand(n, -1)

            elif culture is not None:
                labels = torch.full((n,), culture, dtype=torch.long, device=device)
                culture_embedding = self.decoder.culture_embedding(labels)

            else:
                raise ValueError("Either culture or culture_embedding must be provided.")

            names = self.decoder.generate(z, culture_embedding=culture_embedding, max_len=max_length)
            return names