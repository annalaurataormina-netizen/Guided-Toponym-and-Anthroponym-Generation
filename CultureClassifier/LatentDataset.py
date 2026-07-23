import torch


class LatentDataset:

    def __init__(self, latent_vectors: torch.Tensor, labels: torch.Tensor):
        self.latent_vectors = latent_vectors
        self.labels = labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latent_vectors[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.latent_vectors)

    @property
    def num_cultures(self):
        return len(self.labels.unique())
