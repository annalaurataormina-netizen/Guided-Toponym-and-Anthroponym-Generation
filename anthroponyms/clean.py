import gzip
import json
import os
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_claims, get_monolingual_claims, get_place_country

PLACE_COUNTRY = get_place_country()


def main():
    counter_anthroponyms = 0
    counter_humans = 0

    missing_id_counter = 0
    missing_name_counter = 0

    with gzip.open('/vol/bitbucket/at2225/humans.jsonl.gz', 'rt') as humans:

        given_names_occurrences = {}
        family_names_occurrences = {}

        for line in humans:

            entity = json.loads(line)

            counter_humans += 1

            # List of given names
            given_names = get_claims(entity['info']['claims'], 'P735')

            # List of family names
            family_names = get_claims(entity['info']['claims'], 'P734')

            # Gender
            gender = get_claims(entity['info']['claims'], 'P21')[0]

            # Date of birth
            dates_of_birth = get_claims(entity['info']['claims'], 'P569')
            if dates_of_birth:
                year_of_birth = int(dates_of_birth[0][1:5])
            else:
                year_of_birth = None

            # Place of birth
            places_of_birth = get_claims(entity['info']['claims'], 'P19')
            if places_of_birth:
                place_of_birth = places_of_birth[0]
            else:
                place_of_birth = None
            country_of_birth = PLACE_COUNTRY.get(place_of_birth, None)

            for given_name in given_names:

                if given_name not in given_names_occurrences:
                    given_names_occurrences[given_name] = {
                        'occurrences': 0,
                        'country_of_birth': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = given_names_occurrences[given_name]

                entry['occurrences'] += 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1

            for family_name in family_names:
                if family_name not in family_names_occurrences:
                    family_names_occurrences[family_name] = {
                        'occurrences': 0,
                        'country_of_birth': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = family_names_occurrences[family_name]

                entry['occurrences'] += 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1

    print('# of humans: ', counter_humans)

    with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'rt') as anthroponyms:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in anthroponyms:

                entity = json.loads(line)

                counter_anthroponyms += 1

                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

                entity['name'] = ({Language.get(lang).language_name(): {'name': name, 'code': lang} for lang, name in
                                   native_labels.items()} if native_labels else None)

                if entity['id'] is None:
                    missing_id_counter += 1

                if entity['name'] is None:
                    missing_name_counter += 1

                # print(entity.keys())
                # print(entity['info'].keys())
                # print(entity['info']['claims'].keys())
                # print(entity['info']['claims']['P407'])

                # print(entity['id'])
                # print(entity['type'])
                # print(entity['name'])

                output.write(json.dumps(entity) + '\n')

    print('# of anthroponyms: ', counter_anthroponyms)
    print('# of anthroponyms missing ID: ', missing_id_counter)
    print('# of anthroponyms missing name: ', missing_name_counter)


if __name__ == '__main__':
    main()
