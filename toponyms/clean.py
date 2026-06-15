import gzip
import json
import os
import sys
from typing import Optional, Dict, List

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_monolingual_claims, get_claims

import requests


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
    headers = {'User-Agent': 'YourProjectName/1.0 (your-email@imperial.ac.uk)'}
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

        if 'Q128' not in country_languages.keys():
            country_languages['Q128'] = 'zh'

    return country_languages


COUNTRY_LANGUAGES = get_country_languages()


def get_languages(countries: list) -> Optional[List[str]]:
    if not countries:
        return None
    languages = []
    for country in countries:
        languages.extend(COUNTRY_LANGUAGES.get(country, []))
    return list(set(languages)) if languages else None


def main():
    counter = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_country_counter = 0

    with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in input:

                counter += 1

                entity = json.loads(line)

                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

                # The value is a list of IDs for the countries.
                entity['country'] = get_claims(entity['info']['claims'], 'P17')
                # The value is a dictionary where each entry is of this type: "Italian": {'name': 'Anna Laura', 'code': 'it'}.
                entity['name'] = ({Language.get(lang).language_name(): {'name': name, 'code': lang} for lang, name in
                                   native_labels.items()} if native_labels else None)

                # P1705 is missing for virtually all toponyms. If missing, resort to labels.
                if entity['name'] is None:
                    languages = get_languages(entity['country'])
                    labels = entity['info'].get('labels', None)
                    entity['name'] = (
                        {Language.get(lang).language_name(): {'name': labels.get(lang, {}).get('value'), 'code': lang}
                         for lang in
                         languages if labels and labels.get(lang, None) is not None} if languages else None)

                if entity['id'] is None:
                    missing_id_counter += 1

                if entity['name'] is None:
                    missing_name_counter += 1

                if entity['country'] is None:
                    missing_country_counter += 1

                # print(entity.keys())
                # print(entity['info'].keys())
                # print(entity['info']['claims'].keys())

                # print(entity['id'])
                # print(entity['type'])
                # print(entity['name'])
                # print(entity['country'])

                output.write(json.dumps(entity) + '\n')

    print('# of items: ', counter)
    print('# of items missing ID: ', missing_id_counter)
    print('# of items missing name: ', missing_name_counter)
    print('# of items missing country: ', missing_country_counter)


if __name__ == '__main__':
    main()
