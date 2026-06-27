import gzip
import json
import os
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_monolingual_claims, get_claims, country_to_languages, countries_to_languages

# Dictionary mapping each country to a list of languages.
COUNTRY_TO_LANGUAGES = country_to_languages()


def main():
    counter = 0

    excluded_dissolved_counter = 0
    excluded_historical = 0
    excluded_russian_empire = 0
    excluded_missing_labels = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_country_counter = 0

    with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # 'P576' signals a dissolved, abolished or demolished state.
                if 'P576' in entity['info']['claims']:
                    excluded_dissolved_counter += 1
                    continue

                # entity['country'] contains a list of qids for the countries where the place is located.
                entity['country'] = get_claims(entity['info']['claims'], 'P17')

                # 'Q34266' is the Russian Empire.
                if 'Q34266' in entity['country']:
                    excluded_russian_empire += 1
                    continue

                # Exclude historical entities.
                claims = get_claims(entity['info']['claims'], 'P31')
                if any(q in claims for q in ('Q1620908', 'Q3024240', 'Q50068795', 'Q28171280', 'Q2072238', 'Q1250464')):
                    excluded_historical += 1
                    continue

                # As a first step, use native labels (P1705).
                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')
                entity['name'] = {Language.get(iso).language_name(): {'name': name, 'code': iso,
                                                                      'language': Language.get(
                                                                          iso).language_name()}
                                  for iso, name in
                                  native_labels.items()}

                # Get labels (dictionary mapping ISO code to a string).
                labels = entity['info'].get('labels')

                # Get list of languages spoken in the countries where the entity is located.
                languages = countries_to_languages(COUNTRY_TO_LANGUAGES, entity['country'])

                # Nothing to be done here (entity has neither native labels nor labels).
                if not entity['name'] and not labels:
                    excluded_missing_labels += 1
                    continue

                # If no native labels, use labels linked to the languages of the country.
                if not entity['name']:
                    entity['name'] = {
                        Language.get(iso).language_name(): {
                            'name': labels[iso]['value'],
                            'code': iso,
                            'language': Language.get(iso).language_name()
                        }
                        for iso in languages
                        if labels.get(iso, {}).get('value')
                    }

                # If no native labels or labels in the languages of the country, use the labels associated to
                # the languages below as a fallback and assign that to all languages of the country.
                if not entity['name']:
                    name = ''
                    for iso in ['mul', 'sv', 'vi', 'nl', 'ceb', 'en']:
                        if labels.get(iso):
                            name = labels.get(iso).get('value')
                            break
                    if name:
                        entity['name'] = {
                            Language.get(iso).language_name(): {'name': name,
                                                                'code': iso,
                                                                'language': Language.get(iso).language_name()}
                            for iso in languages}

                if entity['id'] is None:
                    missing_id_counter += 1

                if not entity['name']:
                    missing_name_counter += 1

                if not entity['country']:
                    missing_country_counter += 1

                # print(entity.keys())
                # print(entity['info'].keys())
                # print(entity['info']['claims'].keys())

                # print(entity['id'])
                # print(entity['type'])
                # print(entity['name'])
                # print(entity['country'])

                counter += 1

                output.write(json.dumps(entity) + '\n')

    print('# of items (ex. excluded): ', counter, '\n')
    print('# of items missing ID: ', missing_id_counter, '\n')
    print('# of items missing name: ', missing_name_counter, '\n')
    print('# of items missing country: ', missing_country_counter, '\n')
    print('# of items excluded because dissolved: ', excluded_dissolved_counter, '\n')
    print('# of items excluded because historical: ', excluded_historical, '\n')
    print('# of items excluded because belonging to Russian Empire: ', excluded_russian_empire, '\n')
    print('# of items excluded because missing native labels and labels: ', excluded_missing_labels, '\n')


if __name__ == '__main__':
    main()
