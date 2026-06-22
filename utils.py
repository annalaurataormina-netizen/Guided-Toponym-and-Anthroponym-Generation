import gzip
import json
import unicodedata
from typing import Dict, List

import requests
from icu import Transliterator

TRANSLITERATOR = Transliterator.createInstance('Any-Latin; NFC')

# Returns the list of qids to be used to filter anthroponyms or toponyms
# from the dump based on the "instance of" field.
def get_qids(target: str) -> Dict[str, str]:
    url = 'https://query.wikidata.org/sparql'

    '''
    SPARQL query to get all the subclasses of Q486972 ('human settlement'), Q56061 ('administrative territorial entity'),
    Q6256 ('country'), Q271669 ('landform'), Q15324 ('body of water'), Q23442 ('island').
    P279* is to get subclasses recursively.
    Order matters due to the way get_gids() processes them.
    '''
    if target == 'toponyms':
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

    '''
    SPARQL query to get all the subclasses of Q1243157 ('given name') and Q101352 ('family name').
    P279* is to get subclasses recursively.
    '''
    if target == 'anthroponyms':
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

    qids = {}
    for item in data:
        item_id = item['item']['value'].split('/')[-1]
        item_type = item['type']['value']
        if item_id in qids.keys():
            qids[item_id].append(item_type)
        else:
            qids[item_id] = [item_type]
    return qids


# Returns the list of values for a given property among entity's claims.
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


# Returns the list of values for a given property among entity's time claims.
# Can be used on humans to extract their date of birth.
def get_time_claims(claims: Dict, pid: str) -> list[str]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['time'] for claim in claims[pid]]
        return result if result else []
    except (KeyError, IndexError, TypeError):
        return []


# Returns a dictionary where the key is a country (ID) and the value is a list of languages (ISO codes).
def get_country_languages() -> Dict[str, List[str]]:
    url = 'https://query.wikidata.org/sparql'

    '''
    SPARQL query to get countries and languages.
    '''
    query = '''
    SELECT ?country ?lang WHERE {
      ?country wdt:P31 wd:Q6256 .
      ?country wdt:P37 ?language .
      ?language wdt:P218 ?lang .
    }
    '''
    headers = {'User-Agent': 'Guided-Toponym-and-Anthroponym-Generation/1.0 (anna.taormina25@imperial.ac.uk)'}
    r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers)
    data = r.json()

    country_languages = {}

    for item in data['results']['bindings']:

        country = item['country']['value'].split('/')[-1]
        lang = item['lang']['value']

        if country in country_languages.keys():
            country_languages[country].append(lang)

        else:
            country_languages[country] = [lang]

    # China
    if 'Q148' not in country_languages.keys():
        country_languages['Q148'] = ['zh']
    elif 'zh' not in country_languages['Q148']:
        country_languages['Q148'].append('zh')

    # US
    country_languages['Q30'] = ['en']

    # Vatican
    country_languages['Q237'] = ['it']

    # Georgia
    if country_languages.get('Q230', []):
        country_languages['Q230'].append('kat')
        country_languages['Q230'] = list(set(country_languages['Q230']))
    else:
        country_languages['Q230'] = ['kat']

    # Australia
    country_languages['Q408'] = ['en']

    # India
    country_languages['Q668'] = ['hi', 'as', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'ta', 'te',
                                 'ur']

    # Congo
    if country_languages.get('Q974', []):
        country_languages['Q974'].append('sw')
        country_languages['Q974'].append('ln')
        country_languages['Q974'].append('kg')
        country_languages['Q974'].append('lua')
        country_languages['Q974'] = list(set(country_languages['Q974']))
    else:
        country_languages['Q974'] = ['fr', 'sw', 'ln', 'kg', 'lua']

    # Burkina Faso
    country_languages['Q965'] = ['fr', 'mos', 'dyu', 'ful']

    # Algeria
    country_languages['Q262'] = ["ar", "fr", "kab", "ber"]

    # Spain
    country_languages['Q29'] = ["es", "ca", "gl", "eu"]

    return country_languages


# Based on a mapping, returns a list of unique languages for a list of countries.
def get_languages(mapping: dict, countries: list) -> List[str]:
    if not countries:
        return []
    languages = []
    for country in countries:
        languages.extend(mapping.get(country, []))
    return list(set(languages)) if languages else []


# Returns a dictionary mapping each place (ID) to a country (ID).
def get_place_country() -> Dict[str, str]:
    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:
        mapping = {}

        for line in input:
            entity = json.loads(line)

            id = entity.get('id', None)
            country = entity.get('country', None)

            mapping[id] = country[0] if country else None

    return mapping


# Returns romanised version of a string or an empty string, if the original contains characters that are now allowed.
def get_romanised(name):

    name_romanised = unicodedata.normalize('NFKC', TRANSLITERATOR.transliterate(name))

    valid = ''

    for char in name_romanised:

        cat = unicodedata.category(char)
        block = unicodedata.name(char, '')

        # Allow Latin letters
        if cat in ('Ll', 'Lu', 'Lt', 'Lm') and 'LATIN' in block:
            valid += char
            continue

        # Allow combining diacritics (Latin only)
        if cat == 'Mn' and 'ARABIC' not in block and 'MYANMAR' not in block:
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

        # Drop non-Latin letters
        if cat in ('Ll', 'Lu', 'Lt', 'Lm', 'Lo', 'Mc') and 'LATIN' not in block:
            return ''

    return valid.strip() or ''


def split_diacritics(name: str) -> str:
    return ''.join(unicodedata.normalize('NFD', char) for char in name)


def get_countries_names(ids: list) -> dict[str, str]:
    country_id_name = {}

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
            country_id_name[country['country']['value'].split('/')[-1]] = country['countryLabel']['value']

    ids_ = ids[batches * batch_size:]

    if not ids_:
        return country_id_name

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
        country_id_name[country['country']['value'].split('/')[-1]] = country['countryLabel']['value']

    return country_id_name
