from utils import split_diacritics
import unicodedata

if __name__ == '__main__':
    result = unicodedata.normalize('NFD', 'Kurów')
    print([hex(ord(c)) for c in result])