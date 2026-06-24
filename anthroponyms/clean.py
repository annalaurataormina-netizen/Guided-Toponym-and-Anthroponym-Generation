import gzip
import json
import os
import random
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_claims, get_monolingual_claims, get_time_claims, get_place_country, get_languages, \
    get_country_languages, map_lang_ids_to_iso_codes

# Dictionary mapping each place (ID) to a country (ID).
PLACE_COUNTRY = get_place_country()

# Dictionary mapping country to a list of languages.
COUNTRY_LANGUAGES = get_country_languages()

# Threshold for occurrences in country of birth over which a name is considered to be of that country and its languages.
OCCURRENCE_THRESHOLD = 1

LANGUAGE_IDS_TO_ISO = map_lang_ids_to_iso_codes()

def main():
    counter_anthroponyms = 0
    counter_humans = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_occurrences_counter = 0
    missing_labels_counter = 0

    given_names_occurrences = {}
    family_names_occurrences = {}

    with gzip.open('/vol/bitbucket/at2225/humans.jsonl.gz', 'rt') as humans:

        for line in humans:

            entity = json.loads(line)

            counter_humans += 1

            # List of given names
            given_names = get_claims(entity['info']['claims'], 'P735')

            # List of family names
            family_names = get_claims(entity['info']['claims'], 'P734')

            # Gender
            genders = get_claims(entity['info']['claims'], 'P21')
            gender = genders[0] if genders else None

            # Year of birth
            dates_of_birth = get_time_claims(entity['info']['claims'], 'P569')
            year_of_birth = int(dates_of_birth[0][1:5]) if dates_of_birth else None

            # Country of birth
            places_of_birth = get_claims(entity['info']['claims'], 'P19')
            place_of_birth = places_of_birth[0] if places_of_birth else None
            country_of_birth = PLACE_COUNTRY.get(place_of_birth, None)

            for given_name in given_names:
                if given_name not in given_names_occurrences:
                    given_names_occurrences[given_name] = {
                        'id': given_name,
                        'count': 0,
                        'country_of_birth': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = given_names_occurrences[given_name]

                entry['count'] += 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1

            for family_name in family_names:
                if family_name not in family_names_occurrences:
                    family_names_occurrences[family_name] = {
                        'id': family_name,
                        'count': 0,
                        'country_of_birth': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = family_names_occurrences[family_name]

                entry['count'] += 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1

    print('# of humans: ', counter_humans, '\n')
    print('# of given name occurrences: ', len(given_names_occurrences), '\n')
    print('# of family name occurrences: ', len(family_names_occurrences), '\n')
    print('100 given name occurrences (shuffled): ',
          random.sample(list(given_names_occurrences.values()), min(100, len(given_names_occurrences))), '\n')
    print('100 family name occurrences (shuffled): ',
          random.sample(list(family_names_occurrences.values()), min(100, len(family_names_occurrences))), '\n')

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_humans_cleaned.jsonl.gz', 'wt') as output:

        for entry in given_names_occurrences.values():
            output.write(json.dumps(entry) + '\n')

        for entry in family_names_occurrences.values():
            output.write(json.dumps(entry) + '\n')

    with (gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'rt') as anthroponyms):
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in anthroponyms:

                entity = json.loads(line)

                counter_anthroponyms += 1

                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

                # Family name.
                if 'family name' in entity['type']:
                    entity['occurrences'] = family_names_occurrences.get(entity['id'], {})

                # Given name.
                else:
                    entity['occurrences'] = given_names_occurrences.get(entity['id'], {})

                entity['name'] = {Language.get(language).language_name(): {'name': name, 'code': language,
                                                                           'language': Language.get(
                                                                               language).language_name()} for
                                  language, name in native_labels.items()}

                labels = entity['info'].get('labels', None)

                # Get P407 claims
                languages_of_work_or_name = get_claims(entity['info']['claims'], 'P407')

                if languages_of_work_or_name:
                    for language_id in languages_of_work_or_name:

                        iso = LANGUAGE_IDS_TO_ISO.get(language_id)
                        if not iso:
                            continue

                        lang_obj = Language.get(iso)
                        lang_name = lang_obj.language_name()

                        if lang_name in entity["name"]:
                            continue

                        name = native_labels.get(iso, "")

                        if not name and labels:
                            name = labels.get(iso, {}).get("value", "")

                        if not name and native_labels:
                            name = next(iter(native_labels.values()), "")

                        if not name:
                            continue

                        entity["name"][lang_name] = {
                            "name": name,
                            "code": iso,
                            "language": lang_name
                        }

                # Nothing to be done here (entity has neither native labels nor labels).
                if not entity['name'] and not labels:
                    missing_labels_counter += 1
                    continue

                # Skip (no occurrences)
                if not entity['name'] and (not entity['occurrences'] or entity['occurrences']['count'] == 0):
                    missing_occurrences_counter += 1
                    continue

                # Get the list of countries where the names occurs (using place of birth) more than X times.
                countries_of_occurrence = []
                if entity['occurrences']:
                    countries_of_occurrence = [country for country in entity['occurrences']['country_of_birth'].keys()
                                               if
                                               entity['occurrences']['country_of_birth'][
                                                   country] >= OCCURRENCE_THRESHOLD]

                # Get list of languages spoken in the countries where the name occurs.
                # Expand the list of names with languages from the countries where the name occurs more than the threshold.
                languages = get_languages(COUNTRY_LANGUAGES, countries_of_occurrence)
                if languages:
                    for language_id in languages:

                        iso = LANGUAGE_IDS_TO_ISO.get(language_id)

                        if not iso:
                            continue

                        name = native_labels.get(iso, "")

                        if not name and labels:
                            name = labels.get(iso, {}).get('value', "")
                        if not name and native_labels:
                            name = list(native_labels.values())[0]
                        if not name:
                            continue
                        if Language.get(iso).language_name() not in entity['name']:
                            entity['name'][Language.get(iso).language_name()] = {'name': name, 'code': iso,
                                                                                      'language': Language.get(
                                                                                          iso).language_name()}

                if not entity['id']:
                    missing_id_counter += 1
                    continue

                if not entity['name']:
                    missing_name_counter += 1
                    continue

                # print(entity.keys())
                # print(entity['info'].keys())
                # print(entity['info']['claims'].keys())
                # print(entity['info']['claims']['P407'])

                # print(entity['id'])
                # print(entity['type'])
                # print(entity['name'])

                output.write(json.dumps(entity) + '\n')

    print('# of anthroponyms: ', counter_anthroponyms, '\n')
    print('# of anthroponyms missing ID: ', missing_id_counter, '\n')
    print('# of anthroponyms missing name: ', missing_name_counter, '\n')
    print('# of anthroponyms missing labels: ', missing_labels_counter, '\n')
    print('# of anthroponyms missing occurrences: ', missing_occurrences_counter, '\n')


if __name__ == '__main__':
    main()
