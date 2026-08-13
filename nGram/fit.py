from sklearn.model_selection import train_test_split

from nGram.nGram import nGram
from utils import load_all, normalise


def fit():

    seed = 1996

    # Toponyms and Anthroponyms (name_romanised, label)
    names = load_all(culture=True)

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), lang]
        for name, lang in names
    ]

    train_names, _ = train_test_split(names_normalised, test_size=0.2, random_state=seed, shuffle=True)

    for n in range(2, 5):
        model = nGram(n)
        model.fit(train_names)
        model.save()

if __name__ == '__main__':
    fit()
