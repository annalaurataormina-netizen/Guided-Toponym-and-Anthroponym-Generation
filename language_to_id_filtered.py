import json
from collections import Counter

from utils import load_all

def save():
    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    dataset = load_all(culture=True)

    culture_counts = Counter(language for _, language in dataset)

    min_culture_names = 10000

    language_to_id_filtered = {
        l: i for l, i in language_to_id.items()
        if culture_counts[l] >= min_culture_names
    }

    with open("language_to_id_filtered.json", "w") as f:
        json.dump(language_to_id_filtered, f, indent=2)

    print(f"Saved {len(language_to_id_filtered)} filtered languages to language_to_id_filtered.json")

if __name__ == "__main__":
    save()