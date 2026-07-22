import torch

from AE.CharVocab import CharVocab


class NameDataset:

    def __init__(self, data: list[list[str | int]], vocab: CharVocab):
        # Data (list of name_romanised with their cultural label)
        self.data = data

        # Character vocabulary
        self.vocab = vocab

        # Max length among the strings in the dataset (used for padding).
        self.max_len = max(len(self.vocab.encode(name[0])) for name in data)

    # Encodes (and pads up to max_len in the dataset) name in the dataset.
    def encode(self, idx: int) -> list[int]:
        encoded = self.vocab.encode(self.data[idx][0])
        return self.vocab.pad(encoded, self.max_len)

    # Returns encoded and padded string as a list of indices and length of the string.
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        return torch.tensor(self.encode(idx)), len(self.vocab.encode(self.data[idx][0])), torch.tensor(
            self.data[idx][1])

    # Returns the number of names in the dataset.
    def __len__(self) -> int:
        return len(self.data)
