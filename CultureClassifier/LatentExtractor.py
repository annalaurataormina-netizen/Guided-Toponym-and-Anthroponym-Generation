import torch
from torch.utils.data import DataLoader

from ContrastiveVAE.NameDataset import NameDataset
from VAE.Encoder import Encoder


class LatentExtractor:

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def extract(self, dataset: NameDataset, batch_size: int, device: torch.device) -> (torch.Tensor, torch.Tensor):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        latent_vectors = []
        cultures = []

        self.encoder.eval()

        with torch.no_grad():
            for batch in dataloader:
                sequences, lengths, labels = batch
                sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

                _, mu, _, _ = self.encoder(sequences, lengths)

                latent_vectors.append(mu)
                cultures.append(labels)

        latent_vectors = torch.cat(latent_vectors)
        cultures = torch.cat(cultures)

        return latent_vectors, cultures
