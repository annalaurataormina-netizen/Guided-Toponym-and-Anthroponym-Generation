import os
import unicodedata

import psycopg2.extras
from dotenv import load_dotenv


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
