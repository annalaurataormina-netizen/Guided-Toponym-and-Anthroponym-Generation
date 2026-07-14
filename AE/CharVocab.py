from .config import ALLOWED_CHARS


class CharVocab:

    def __init__(self, allowed_chars: list[str]):

        # Start of Sequence, End of Sequence, Padding and Upper characters
        # The <UPPER> character uppercases the following (lowercase) character
        self.SPECIAL_CHARS = ['<SOS>', '<EOS>', '<PAD>', '<UPPER>']

        # Dictionary mapping each character to an index
        self.char2idx = {c: i for i, c in enumerate(self.SPECIAL_CHARS)}

        # Dictionary mapping each index to a character
        self.idx2char = {i: c for i, c in enumerate(self.SPECIAL_CHARS)}

        # Populate the dictionary based on the allowed characters
        # (already populated with the special characters above)
        for i, c in enumerate(allowed_chars):
            self.char2idx[c] = i + len(self.SPECIAL_CHARS)
            self.idx2char[i + len(self.SPECIAL_CHARS)] = c

    # Encodes a string, adding <SOS> and <EOS> characters, normalising uppercase characters
    # and returning a list of indices, one for each character, using the dictionary.
    def encode(self, name: str) -> list[int]:

        sequence = [self.char2idx['<SOS>']]

        for c in name:
            if c == c.lower():
                sequence.append(self.char2idx[c])
            else:
                # Upper characters are replaced with <UPPER> special character and lowercase character.
                sequence.append(self.char2idx['<UPPER>'])
                sequence.append(self.char2idx[c.lower()])

        sequence.append(self.char2idx['<EOS>'])
        return sequence

    # Decodes a list of indices using the dictionary, stripping special characters,
    # and returning a string.
    def decode(self, name: list[int]) -> str:
        name_str = ''
        upper = False

        for c in name:

            if self.idx2char[c] == '<UPPER>':
                upper = True

            elif self.idx2char[c] in self.SPECIAL_CHARS:
                continue

            # Lowercase characters that come after <UPPER> are uppercased.
            elif upper:
                name_str += self.idx2char[c].upper()
                upper = False

            else:
                name_str += self.idx2char[c]

        return name_str

    # Adds padding to a string (list of indices) until the target length is reached.
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
