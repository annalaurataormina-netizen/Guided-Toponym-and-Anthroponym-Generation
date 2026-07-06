import torch

from AE.CharVocab import CharVocab


class NameDataset:

    def __init__(self, data: list[str], vocab: CharVocab):
        self.data = data
        self.vocab = vocab

        # Max length among the strings in the dataset (+2 because of the <SOS> and <EOS> characters).
        self.max_len = max(map(len, data)) + 2

    # Encodes and pads string corresponding to the index.
    def encode(self, idx: int) -> list[int]:
        encoded = self.vocab.encode(self.data[idx])
        return self.vocab.pad(encoded, self.max_len)

    # Returns encoded and padded string (list of indices) and length of the string.
    def __getitem__(self, idx: int):
        return torch.tensor(self.encode(idx)), len(self.data[idx]) + 2

    # Returns the number of samples.
    def __len__(self):
        return len(self.data)
