import os
import unicodedata

import psycopg2.extras
from dotenv import load_dotenv

from .NameDataset import NameDataset


# Returns the string after splitting the diacritics from the underlying character.
def split_diacritics(name: str) -> str:
    return unicodedata.normalize('NFD', name)


# Returns the string after splitting the diacritics from the underlying character.
def normalise(name: str) -> str:
    return split_diacritics(name)


# Returns a list of name_romanised for all anthroponyms or toponyms in the database.
def load_from_database(target: str) -> list[str]:
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        user=os.getenv("PGUSER"),
        dbname=os.getenv("PGDATABASE"),
        password=os.getenv("PGPASSWORD"),
        options='-c client_encoding=UTF8'
    )

    cur = conn.cursor()

    entries = target + '.entries'

    cur.execute(f"SELECT name_romanised FROM {entries}")
    names = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return names


# Returns a list of strings (name_romanised) for all anthroponyms in the dataset.
def load_anthroponyms() -> list[str]:
    return load_from_database('anthroponyms')


# Returns a list of strings (name_romanised) for all toponyms in the dataset.
def load_toponyms() -> list[str]:
    return load_from_database('toponyms')


# Returns a list of strings (name_romanised) for all anthroponyms and toponyms in the dataset.
def load_all() -> list[str]:
    return load_anthroponyms() + load_toponyms()


# Returns the percentage of generated names that doesn't appear in the training dataset.
def compute_novelty(generated: list[str], train_dataset: NameDataset) -> float:
    # Extract training names
    train_names = set()

    for i in range(len(train_dataset)):
        name, _ = train_dataset[i]
        train_names.add(name)

    # Count generated names not seen in training
    novel_count = sum(name not in train_names for name in generated)

    novelty = novel_count / len(generated)

    return novelty


# Computes the proportion of character n-grams in generated names that also occur in the training set.
def compute_ngram_coverage(generated, train_dataset, n=3):

    # Build the training n-gram vocabulary
    train_ngrams = set()

    for i in range(len(train_dataset)):
        name, _ = train_dataset[i]

        for j in range(len(name) - n + 1):
            train_ngrams.add(name[j:j+n].lower())

    # Count generated n-grams
    total = 0
    matched = 0

    for name in generated:
        name = name.lower()

        for j in range(len(name) - n + 1):
            total += 1
            if name[j:j+n] in train_ngrams:
                matched += 1

    return matched / total if total > 0 else 0.0