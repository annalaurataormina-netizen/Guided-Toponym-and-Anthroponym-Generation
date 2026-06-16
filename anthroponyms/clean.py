import gzip
import json
import os
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_claims, get_monolingual_claims, get_time_claims, get_place_country

PLACE_COUNTRY = get_place_country()


def main():
    counter_anthroponyms = 0
    counter_humans = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_occurrences_counter = 0

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
            genders = get_claims(entity['info']['claims'], 'P21')
            if genders:
                gender = genders[0]
            else:
                gender = None

            # Date of birth
            dates_of_birth = get_time_claims(entity['info']['claims'], 'P569')
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
                        'count': 0,
                        'country_of_birth': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = given_names_occurrences[given_name]

                entry['id'] = given_name
                entry['count'] += 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1

            for family_name in family_names:
                if family_name not in family_names_occurrences:
                    family_names_occurrences[family_name] = {
                        'count': 0,
                        'country_of_birth': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = family_names_occurrences[family_name]

                entry['id'] = family_name
                entry['count'] += 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1

    print('# of humans: ', counter_humans)

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_humans_cleaned.jsonl.gz', 'wt') as output:

        for entry in given_names_occurrences.values():
            output.write(json.dumps(entry) + '\n')

        for entry in family_names_occurrences.values():
            output.write(json.dumps(entry) + '\n')

    print('''# of humans' given names: ''', len(given_names_occurrences))
    print('''# of humans' family names: ''', len(family_names_occurrences))

    with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'rt') as anthroponyms:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in anthroponyms:

                entity = json.loads(line)

                counter_anthroponyms += 1

                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

                if 'family name' in entity['type']:
                    entity['occurrences'] = family_names_occurrences.get(entity['id'], {})

                elif 'male given name' in entity['type'] or 'female given name' in entity['type']:
                    entity['occurrences'] = given_names_occurrences.get(entity['id'], {})

                entity['name'] = {Language.get(lang).language_name(): {'name': name, 'code': lang} for lang, name in
                                   native_labels.items()}

                if not entity['id']:
                    missing_id_counter += 1

                if not entity['name']:
                    missing_name_counter += 1

                if entity['occurrences']['count'] == 0 or not entity['occurrences']:
                    missing_occurrences_counter += 1

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
    print('# of anthroponyms missing occurrences: ', missing_occurrences_counter)


if __name__ == '__main__':
    main()
