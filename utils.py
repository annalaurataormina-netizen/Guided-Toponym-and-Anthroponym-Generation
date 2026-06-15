import gzip
import json
from typing import Dict, Optional, List

import requests


# Returns the list of qids to be used to obtain anthroponyms or toponyms
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
          UNION {?item wdt:P279* wd:Q101352 . BIND("family name" AS ?type)}
        }
        '''

    r = requests.get(url, params={'format': 'json', 'query': query})
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
def get_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None


# Returns the list of values for a given property among entity's monolingual claims.
def get_monolingual_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = {claim['mainsnak']['datavalue']['value']['language']: claim['mainsnak']['datavalue']['value']['text']
                  for claim in claims[pid]}
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None


# Returns a dictionary where the key is a country (ID) and the value is a list of official languages (ISO codes).
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
    headers = {'User-Agent': 'YourProjectName/1.0 (anna.taormina25@imperial.ac.uk)'}
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

    if 'Q148' not in country_languages.keys():
        country_languages['Q148'] = ['zh']

    return country_languages


# Based on a mapping, returns a list of unique languages for a list of countries.
def get_languages(mapping: dict, countries: list) -> Optional[List[str]]:
    if not countries:
        return None
    languages = []
    for country in countries:
        languages.extend(mapping.get(country, []))
    return list(set(languages)) if languages else None


# Returns a dictionary mapping each place (ID) to a country (ID)
def get_place_country() -> Dict[str, str]:
    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:
        mapping = {}

        for line in input:
            entity = json.loads(line)

            id = entity.get('id', None)
            country = entity.get('country', None)

            mapping[id] = country

    return mapping
