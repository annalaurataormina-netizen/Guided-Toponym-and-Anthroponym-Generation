import json
from collections import Counter

from utils import normalise, load_all


def temp():

    names = load_all(culture=True)

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    culture_counts = Counter(label for _, label in names_normalised)
    min_samples = 1000
    names_normalised = [
        x for x in names_normalised
        if culture_counts[x[1]] >= min_samples
    ]

    # Re-index remaining cultures
    remaining_cultures = sorted(
        set(label for _, label in names_normalised)
    )

    old_to_new = {
        old: new
        for new, old in enumerate(remaining_cultures)
    }

    names_normalised = [
        [name, old_to_new[label]]
        for name, label in names_normalised
    ]

    language_to_id = {
        language: old_to_new[label]
        for language, label in language_to_id.items()
        if label in old_to_new
    }

    with open("language_to_id_filtered.json", "w") as f:
        f.write(language_to_id)