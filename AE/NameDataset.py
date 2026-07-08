import torch

from CharVocab import CharVocab


class NameDataset:

    def __init__(self, data: list[str], vocab: CharVocab):
        # Data (list of strings)
        self.data = data

        # Character vocabulary
        self.vocab = vocab

        # Max length among the strings in the dataset (+2 because of the <SOS> and <EOS> characters).
        self.max_len = max(len(self.vocab.encode(name)) for name in data)

    # Encodes and pads string corresponding to the index.
    def encode(self, idx: int) -> list[int]:
        encoded = self.vocab.encode(self.data[idx])
        return self.vocab.pad(encoded, self.max_len)

    # Returns encoded and padded string as a list of indices and length of the string.
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.encode(idx)), len(self.vocab.encode(self.data[idx]))

    # Returns the number of samples.
    def __len__(self) -> int:
        return len(self.data)
