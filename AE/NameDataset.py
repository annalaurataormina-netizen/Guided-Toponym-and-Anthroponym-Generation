import torch

from AE.CharVocab import CharVocab


class NameDataset:

    def __init__(self, data: list[str], vocab: CharVocab):
        self.data = data
        self.vocab = vocab
        self.max_len = max(map(len, data)) + 2

    def encode(self, idx: int) -> list[int]:
        encoded = self.vocab.encode(self.data[idx])
        return self.vocab.pad(encoded, self.max_len)

    def __getitem__(self, idx: int):
        return torch.tensor(self.encode(idx))

    def __len__(self):
        return len(self.data)
