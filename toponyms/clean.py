import gzip
import json
import os
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_monolingual_claims, get_claims, get_country_languages, get_languages

# Dictionary mapping country to a list of languages.
COUNTRY_LANGUAGES = get_country_languages()


def main():
    counter = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_country_counter = 0

    with (gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input):
        with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # 'P576' signals a dissolved, abolished or demolished state.
                if 'P576' in entity['info']['claims']:
                    continue

                # The value is a list of IDs for the countries.
                entity['country'] = get_claims(entity['info']['claims'], 'P17')

                # 'Q34266' is the Russian Empire.
                if 'Q34266' in entity['country']:
                    continue

                # Exclude historical entities.
                claims = get_claims(entity['info']['claims'], 'P31')
                if 'Q1620908' in claims or 'Q3024240' in claims or 'Q50068795' in claims or 'Q28171280' in claims or 'Q2072238' in claims or 'Q1250464' in claims:
                    continue

                # As a first step, use native labels.
                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')
                entity['name'] = {Language.get(language).language_name(): {'name': name, 'code': language,
                                                                           'language': Language.get(
                                                                               language).language_name()}
                                  for language, name in
                                  native_labels.items()}

                # Get labels (dictionary).
                labels = entity['info'].get('labels', None)

                # Get list of languages spoken in the country where the entity is located.
                languages = get_languages(COUNTRY_LANGUAGES, entity['country'])

                # Nothing to be done here (entity has neither native labels nor labels).
                if not entity['name'] and not labels:
                    continue

                # If no native labels, use labels linked to the languages of the country.
                if not entity['name']:
                    entity['name'] = {
                        Language.get(language).language_name(): {'name': labels.get(language, {}).get('value'),
                                                                 'code': language,
                                                                 'language': Language.get(language).language_name()}
                        for language in languages if labels.get(language, {})}

                # If no native labels or labels in the languages of the country, use the
                # labels associated to the languages below as a fallback and assign that to all languages
                # of the country.
                if not entity['name']:
                    name = ''
                    if labels.get('mul', None):
                        name = labels.get('mul').get('value')
                    elif labels.get('sv', None):
                        name = labels.get('sv').get('value')
                    elif labels.get('vi', None):
                        name = labels.get('vi').get('value')
                    elif labels.get('nl', None):
                        name = labels.get('nl').get('value')
                    elif labels.get('ceb', None):
                        name = labels.get('ceb').get('value')
                    elif labels.get('en', None):
                        name = labels.get('en').get('value')
                    if name:
                        entity['name'] = {
                            Language.get(language).language_name(): {'name': name,
                                                                     'code': language,
                                                                     'language': Language.get(language).language_name()}
                            for language in languages}

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

    print('# of items: ', counter)
    print('# of items missing ID: ', missing_id_counter)
    print('# of items missing name: ', missing_name_counter)
    print('# of items missing country: ', missing_country_counter)


if __name__ == '__main__':
    main()
