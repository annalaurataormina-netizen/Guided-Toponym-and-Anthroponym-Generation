import gzip
import json
import os
import unicodedata
from typing import Dict, List

import psycopg2.extras
import requests
from dotenv import load_dotenv
from icu import Transliterator

from AE.NameDataset import NameDataset

TRANSLITERATOR = Transliterator.createInstance('Any-Latin; NFC')

# Norther Cyprus, Somaliland, Kingdom of Denmark and Kingdom of the Netherlands.
EXCLUDED_COUNTRIES = {'Q23681', 'Q34754', 'Q29999', 'Q756617'}


# Returns a dictionary used to filter anthroponyms or toponyms from the dump based on their P31.
# The key is a qid, while the value is a list of types, e.g., ['human settlement'].
def get_qids_and_types(category: str) -> Dict[str, str]:
    url = 'https://query.wikidata.org/sparql'

    # SPARQL query to get all the subclasses of Q486972 ('human settlement'), Q56061 ('administrative territorial entity'),
    # Q6256 ('country'), Q271669 ('landform'), Q15324 ('body of water'), Q23442 ('island').
    # P279* allows to get subclasses recursively.
    # Order matters due to the way get_gids() processes them.
    if category == 'toponyms':
        query = '''
        SELECT DISTINCT ?item ?type WHERE {
          {?item wdt:P279* wd:Q486972 . BIND("settlement" AS ?type)}
          UNION {?item wdt:P279* wd:Q56061 . BIND("region" AS ?type)}
          UNION {?item wdt:P279* wd:Q271669 . BIND("landform" AS ?type)}
          UNION {?item wdt:P279* wd:Q15324 . BIND("body of water" AS ?type)}
          UNION {?item wdt:P279* wd:Q23442 . BIND("island" AS ?type)}
          UNION {?item wdt:P279* wd:Q6256 . BIND("country" AS ?type)}
        }
        '''

    # SPARQL query to get all the subclasses of Q12308941 ('male given name'),
    # Q11879590 ('female given name'), Q3409032 ('unisex given name') and Q101352 ('family name').
    # P279* allows to get subclasses recursively.
    if category == 'anthroponyms':
        query = '''
        SELECT DISTINCT ?item ?type WHERE {
          {?item wdt:P279* wd:Q12308941 . BIND("male given name" AS ?type)}
          UNION {?item wdt:P279* wd:Q11879590 . BIND("female given name" AS ?type)}
          UNION {?item wdt:P279* wd:Q3409032 . BIND("unisex given name" AS ?type)}
          UNION {?item wdt:P279* wd:Q101352 . BIND("family name" AS ?type)}
        }
        '''

    headers = {'User-Agent': 'Guided-Toponym-and-Anthroponym-Generation/1.0 (anna.taormina25@imperial.ac.uk)'}
    r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers)
    data = r.json()['results']['bindings']

    qid_to_types = {}

    for item in data:
        qid = item['item']['value'].split('/')[-1]
        type_ = item['type']['value']
        if qid in qid_to_types.keys():
            qid_to_types[qid].append(type_)
        else:
            qid_to_types[qid] = [type_]

    return qid_to_types


# Returns a list of values (qids) for a given property among entity's claims.
def get_claims(claims: Dict, pid: str) -> list[str]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
        return result if result else []
    except (KeyError, IndexError, TypeError):
        return []


# Returns a dictionary for a given property among entity's monolingual claims.
# Can be used to extract native labels when called with claims = entity['info']['claims'] and pid = 'P1705'.
def get_monolingual_claims(claims: Dict, pid: str) -> Dict[str, str]:
    try:
        result = {claim['mainsnak']['datavalue']['value']['language']: claim['mainsnak']['datavalue']['value']['text']
                  for claim in claims[pid]}
        return result if result else {}
    except (KeyError, IndexError, TypeError):
        return {}


# Returns a list of values (qids) for a given property among entity's time claims.
# Can be used on humans to extract their date of birth.
def get_time_claims(claims: Dict, pid: str) -> list[str]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['time'] for claim in claims[pid]]
        return result if result else []
    except (KeyError, IndexError, TypeError):
        return []


# Returns a dictionary where the key is a country (qid) and the value is a list of languages (ISO codes).
def country_to_languages() -> Dict[str, List[str]]:
    url = 'https://query.wikidata.org/sparql'

    '''
    SPARQL query to get countries and languages.
    '''
    query = '''
    SELECT ?country ?lang WHERE {
      ?country wdt:P31 wd:Q3624078 .
      ?country wdt:P37 ?language .
      ?language wdt:P218 ?lang .
    }
    '''
    headers = {'User-Agent': 'Guided-Toponym-and-Anthroponym-Generation/1.0 (anna.taormina25@imperial.ac.uk)'}
    r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers)
    data = r.json()

    country_to_languages = {}

    for item in data['results']['bindings']:

        country = item['country']['value'].split('/')[-1]
        lang = item['lang']['value']

        # Norther Cyprus, Somaliland, Kingdom of Denmark and Kingdom of Netherlands.
        if country in EXCLUDED_COUNTRIES:
            continue

        if country in country_to_languages.keys():
            country_to_languages[country].append(lang)

        else:
            country_to_languages[country] = [lang]

    # China
    if 'Q148' not in country_to_languages.keys():
        country_to_languages['Q148'] = ['zh']
    elif 'zh' not in country_to_languages['Q148']:
        country_to_languages['Q148'].append('zh')

    # US
    country_to_languages['Q30'] = ['en']

    # Vatican
    country_to_languages['Q237'] = ['it']

    # Georgia
    if country_to_languages.get('Q230', []):
        country_to_languages['Q230'].append('kat')
        country_to_languages['Q230'] = list(set(country_to_languages['Q230']))
    else:
        country_to_languages['Q230'] = ['kat']

    # Australia
    country_to_languages['Q408'] = ['en']

    # India
    country_to_languages['Q668'] = ['hi', 'as', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'ta', 'te',
                                    'ur']

    # Congo
    if country_to_languages.get('Q974', []):
        country_to_languages['Q974'].extend(['sw', 'ln', 'kg', 'lua'])
        country_to_languages['Q974'] = list(set(country_to_languages['Q974']))
    else:
        country_to_languages['Q974'] = ['fr', 'sw', 'ln', 'kg', 'lua']

    # Burkina Faso
    country_to_languages['Q965'] = ['fr', 'mos', 'dyu', 'ful']

    # Algeria
    country_to_languages['Q262'] = ["ar", "fr", "kab", "ber"]

    # Spain
    country_to_languages['Q29'] = ["es", "ca", "gl", "eu"]

    # Norway
    country_to_languages['Q20'] = ['no']

    return country_to_languages


# Based on a mapping of countries to languages as done by get_country_languages(),
# returns a list of unique languages for a list of countries.
def countries_to_languages(mapping: dict, countries: list) -> List[str]:
    if not countries:
        return []
    languages = []
    for country in countries:
        languages.extend(mapping.get(country, []))
    return list(set(languages)) if languages else []


# Returns a dictionary mapping each place (qid) to the country (qid) where it is located.
def place_to_country() -> Dict[str, str]:
    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:
        mapping = {}

        for line in input:
            entity = json.loads(line)

            id = entity.get('id', None)
            country = entity.get('country', None)

            mapping[id] = country[0] if country else None

    return mapping


# Returns the romanised version of a string or an empty string, if the original contains characters that are now allowed.
def romanise(name):
    name_romanised = unicodedata.normalize('NFKC', TRANSLITERATOR.transliterate(name))

    valid = ''

    for char in name_romanised:

        cat = unicodedata.category(char)
        block = unicodedata.name(char, '')

        # Return '' if original string contains phonetic extension characters.
        if char in ['ɂ', 'Ɂ', 'ʋ', 'ʕ', 'Ɲ', 'ȝ', 'ɬ']:
            return ''

        if char in '︠︡\u034f':
            continue

        # Allow Latin letters
        if cat in ('Ll', 'Lu', 'Lt', 'Lm') and 'LATIN' in block:
            valid += char
            continue

        # Allow combining diacritics (Latin only)
        if cat == 'Mn' and 'ARABIC' not in block and 'MYANMAR' not in block and 'PRESENTATION FORM' not in block:
            valid += char
            continue

        # Standardise spaces
        if cat == 'Zs' or char == ' ':
            valid += ' '
            continue

        # Standardise hyphens
        if char in '-–—‑‐\u2011\u2013\u2014':
            valid += '-'
            continue

        # Standardise apostrophes
        if char in "'''''\u2019":
            valid += "'"
            continue

        # Keep these
        if char in '·ˌ':
            valid += char
            continue

        # Drop symbols, punctuation, digits
        if cat in ('Po', 'Ps', 'Pe', 'Pi', 'Pf', 'Nd', 'No', 'Nl', 'So', 'Sm', 'Sk', 'Cf', 'Co'):
            continue

        # Drop strings containing non-Latin letters
        if cat in ('Ll', 'Lu', 'Lt', 'Lm', 'Lo', 'Mc') and 'LATIN' not in block:
            return ''

    return valid.strip() or ''


# Returns a dictionary where, for each country, the key is its qid and the value is its name.
def qid_to_country_name(ids: list) -> dict[str, str]:
    countries = {}

    url = 'https://query.wikidata.org/sparql'

    batch_size = 200

    batches = len(ids) // batch_size

    for batch in range(batches):
        ids_ = ids[batch * batch_size:(batch + 1) * batch_size]
        values = ' '.join(f'wd:{id}' for id in ids_)
        query = f'''
        SELECT ?country ?countryLabel WHERE {{
          VALUES ?country {{ {values} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        '''
        headers = {'User-Agent': 'Guided-Toponym-and-Anthroponym-Generation/1.0 (anna.taormina25@imperial.ac.uk)'}
        r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers)

        data = r.json()['results']['bindings']
        for country in data:
            countries[country['country']['value'].split('/')[-1]] = country['countryLabel']['value']

    ids_ = ids[batches * batch_size:]

    if not ids_:
        return countries

    values = ' '.join(f'wd:{id}' for id in ids_)

    query = f'''
    SELECT ?country ?countryLabel WHERE {{
      VALUES ?country {{ {values} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    '''
    headers = {'User-Agent': 'Guided-Toponym-and-Anthroponym-Generation/1.0 (anna.taormina25@imperial.ac.uk)'}
    r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers)
    data = r.json()['results']['bindings']
    for country in data:
        countries[country['country']['value'].split('/')[-1]] = country['countryLabel']['value']

    return countries


# Returns a dictionary where, for each language, the key is its qid and the value is its ISO code.
def qid_to_iso() -> dict[str, str]:
    url = "https://query.wikidata.org/sparql"

    query = """
    SELECT ?language ?isoCode WHERE {
      ?language wdt:P218 ?isoCode.
    }
    """

    headers = {
        "User-Agent": "Guided-Toponym-and-Anthroponym-Generation/1.0 (anna.taormina25@imperial.ac.uk)"
    }

    r = requests.get(url, params={"format": "json", "query": query}, headers=headers)
    data = r.json()

    qid_to_iso = {}

    for item in data["results"]["bindings"]:
        qid = item["language"]["value"].split("/")[-1]
        iso = item["isoCode"]["value"]

        qid_to_iso[qid] = iso

    return qid_to_iso


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
def compute_ngram_coverage(generated: list[str], train_names: list[str], n=3) -> float:
    train_ngrams = set()

    for name in train_names:
        name = name.lower()
        for j in range(len(name) - n + 1):
            train_ngrams.add(name[j:j + n])

    total = 0
    matched = 0

    for name in generated:
        name = name.lower()
        for j in range(len(name) - n + 1):
            total += 1
            if name[j:j + n] in train_ngrams:
                matched += 1

    return matched / total if total > 0 else 0.0


def cyclical_beta(step: int, total_steps: int, n_cycles: int, ratio: float, beta_max: float) -> float:
    cycle_length = total_steps / n_cycles
    cycle_position = (step % cycle_length) / cycle_length

    if cycle_position < ratio:
        beta = beta_max * (cycle_position / ratio)
    else:
        beta = beta_max

    return beta
