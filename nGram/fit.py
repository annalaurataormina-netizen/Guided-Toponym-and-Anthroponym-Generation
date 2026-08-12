from nGram.nGram import nGram
from utils import load_all, normalise


def fit():
    # Toponyms and Anthroponyms (name_romanised, label)
    names = load_all(culture=True)

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), lang]
        for name, lang in names
    ]

    for n in range(2, 5):
        model = nGram(n)
        model.fit(names_normalised)

        names = [['Anna', 'Italian'], ['XXX', 'Italian'], ['A B C', 'Italian']]
        for name in names:
            print(model.sequence_probability(name))


if __name__ == '__main__':
    fit()
