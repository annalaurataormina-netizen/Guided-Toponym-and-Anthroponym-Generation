import unicodedata

# Combining grapheme joiner
EXCLUDE_CHARS = {'\u034f'}

# Allowed characters
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
SPACE = ' '
PUNCTUATION = '-\''
OTHER = '·ˌ'
COMBINING_CHARS = ''.join(
    chr(i) for i in range(0x0300, 0x0370) if unicodedata.category(chr(i)) == 'Mn' and chr(i) not in EXCLUDE_CHARS)
RARE_CHARS = 'ɦəßđþøıæœłðɣŧɛǝŋɔʒħ'
ALLOWED_CHARS = [c for c in (LETTERS + SPACE + PUNCTUATION + OTHER + COMBINING_CHARS + RARE_CHARS)]

if __name__ == '__main__':
    print(ALLOWED_CHARS)
