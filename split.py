from sklearn.model_selection import train_test_split

from utils import load_all, normalise


def split():
    names = load_all(culture=True)

    names_normalised = [normalise(n) for n in names]

    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    print(f"Training: {len(train_names)}")
    print(f"Validation: {len(val_names)}")
    print(f"Test: {len(test_names)}")

    train_cultures = set([c for n, c in train_names])
    val_cultures = set([c for n, c in val_names])
    test_cultures = set([c for n, c in test_names])

    print(f"Training: {len(train_cultures)}")
    print(f"Validation: {len(val_cultures)}")
    print(f"Test: {len(test_cultures)}")


if __name__ == "__main__":
    split()