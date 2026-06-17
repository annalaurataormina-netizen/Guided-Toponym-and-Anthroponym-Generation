import gzip
import json
import os
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_monolingual_claims, get_claims, get_country_languages, get_languages

COUNTRY_LANGUAGES = get_country_languages()


def main():
    counter = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_country_counter = 0

    with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                if 'P576' in entity['info']['claims']:
                    continue

                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

                # The value is a list of IDs for the countries.
                entity['country'] = get_claims(entity['info']['claims'], 'P17')

                if 'Q34266' in entity['country']:
                    continue

                # The value is a dictionary where each entry is of this type: "Italian": {'name': 'Anna Laura', 'code': 'it'}.
                entity['name'] = {Language.get(language).language_name(): {'name': name, 'code': language, 'language': language}
                                  for language, name in
                                  native_labels.items()}

                labels = entity['info'].get('labels', None)
                languages = get_languages(COUNTRY_LANGUAGES, entity['country'])

                if not labels or not languages:
                    continue

                # P1705 is missing for virtually all toponyms. If missing, resort to labels.
                if not entity['name']:
                    entity['name'] = {
                        Language.get(language).language_name(): {'name': labels.get(language, {}).get('value'),
                                                                 'code': language,
                                                                 'language': language}
                        for language in languages if labels.get(language, {})}

                # Use, in order, ceb and en as fallbacks.
                if not entity['name']:
                    name = ''
                    if labels.get('ceb', None):
                        name = labels.get('ceb').get('value')
                    elif labels.get('en', None):
                        name = labels.get('en').get('value')
                    if name:
                        entity['name'] = {
                            Language.get(language).language_name(): {'name': name,
                                                                     'code': language,
                                                                     'language': language}
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
    # print(COUNTRY_LANGUAGES)
