from sklearn.model_selection import train_test_split

from utils import load_all, normalise


from collections import Counter

def split():
    names = load_all(culture=True)

    names_normalised = [[normalise(n[1]), n[1]] for n in names]

    train_names, temp_names = train_test_split(
        names_normalised,
        test_size=0.2,
        random_state=1996,
        shuffle=True
    )

    val_names, test_names = train_test_split(
        temp_names,
        test_size=0.5,
        random_state=1996,
        shuffle=True
    )

    print(f"Training: {len(train_names)}")
    print(f"Validation: {len(val_names)}")
    print(f"Test: {len(test_names)}")

    train_cultures = set(c for n, c in train_names)
    val_cultures = set(c for n, c in val_names)
    test_cultures = set(c for n, c in test_names)

    print(f"Training: {len(train_cultures)}")
    print(f"Validation: {len(val_cultures)}")
    print(f"Test: {len(test_cultures)}")

    # Count examples per language in the training set
    train_counts = Counter(c for n, c in train_names)

    # Languages with more than 1,000 training examples
    languages_over_1000 = {
        language: count
        for language, count in train_counts.items()
        if count > 1000
    }

    print(f"Languages with >1,000 training examples: {len(languages_over_1000)}")

    # Optional: print the languages and their counts
    for language, count in sorted(languages_over_1000.items(), key=lambda x: x[1], reverse=True):
        print(f"{language}: {count}")



if __name__ == "__main__":
    split()
