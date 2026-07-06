from config import ALLOWED_CHARS


class CharVocab:

    def __init__(self, allowed_chars: list[chr]):

        # Start of Sequence, End of Sequence and Padding characters
        self.SPECIAL_CHARS = ['<SOS>', '<EOS>', '<PAD>']

        # Dictionary mapping each character to index
        self.char2idx = {c: i for i, c in enumerate(self.SPECIAL_CHARS)}

        # Dictionary mapping each index to a character
        self.idx2char = {i: c for i, c in enumerate(self.SPECIAL_CHARS)}

        for i, c in enumerate(allowed_chars):
            self.char2idx[c] = i + len(self.SPECIAL_CHARS)
            self.idx2char[i + len(self.SPECIAL_CHARS)] = c

    # Encodes a string, adding <SOS> and <EOS> characters and returning a list of indices, one for each character.
    def encode(self, name: str) -> list[int]:
        sequence = ['<SOS>'] + list(name) + ['<EOS>']
        return [self.char2idx[c] for c in sequence]

    # Decodes a list of indices, stripping special characters and returning a string.
    def decode(self, name: list[int]) -> str:
        return ''.join([self.idx2char[i] for i in name if self.idx2char[i] not in self.SPECIAL_CHARS])

    # Pads until the target length is reached.
    def pad(self, encoded: list[int], target_length: int) -> list[int]:
        while len(encoded) < target_length:
            encoded.append(self.char2idx['<PAD>'])
        return encoded

    # Returns the number of characters in the vocabulary.
    def __len__(self) -> int:
        return len(self.char2idx)


if __name__ == '__main__':
    vocab = CharVocab(ALLOWED_CHARS)
    print(vocab.encode('hello'))
    print(vocab.decode(vocab.encode('hello')))
    assert vocab.encode('hello') != vocab.decode(vocab.encode('world'))
