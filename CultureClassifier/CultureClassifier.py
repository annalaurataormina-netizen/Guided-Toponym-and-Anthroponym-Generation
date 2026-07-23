import torch
from torch import nn


class CultureClassifier(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, cultures: int):
        super().__init__()
        self.input_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = cultures
        self.hidden_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.hidden_layer(x)
        y = self.relu(y)
        y = self.output_layer(y)
        return y
