from config import ALLOWED_CHARS

class CharVocab:

    def __init__(self, allowed_chars: list[chr]):
        self.BASIC_CHARS = ['<SOS>', '<EOS>', '<PAD>']
        self.char2idx = {c: i for i, c in enumerate(self.BASIC_CHARS)}
        self.idx2ch
        ar = {i: c for i, c in enumerate(self.BASIC_CHARS)}
        for i, c in enumerate(allowed_chars):
            self.char2idx[c] = i + len(self.BASIC_CHARS)
            self.idx2char[i + len(self.BASIC_CHARS)] = c

    def encode(self, name: str) -> list[int]:
        sequence = ['<SOS>'] + list(name) + ['<EOS>']
        return [self.char2idx[c] for c in sequence]

    def decode(self, name: list[int]) -> str:
        return ''.join([self.idx2char[i] for i in name if self.idx2char[i] not in self.BASIC_CHARS])

    def pad(self, encoded: list[int], max_length: int) -> list[int]:
        while len(encoded) < max_length:
            encoded.append(self.char2idx['<PAD>'])
        return encoded

    def __len__(self) -> int:
        return len(self.char2idx)

'''
# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")
'''

if __name__ == '__main__':
    vocab = CharVocab(ALLOWED_CHARS)
    print(vocab.encode('hello'))
    print(vocab.decode(vocab.encode('hello')))
    assert vocab.encode('hello') != vocab.decode(vocab.encode('world'))