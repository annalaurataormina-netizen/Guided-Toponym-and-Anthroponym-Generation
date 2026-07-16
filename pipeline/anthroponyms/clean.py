import gzip
import json
import random

from langcodes import Language

from utils import get_claims, get_monolingual_claims, get_time_claims, place_to_country, countries_to_languages, \
    country_to_languages, qid_to_iso

# Dictionary mapping each place (qid) to a country (qid).
PLACE_COUNTRY = place_to_country()

# Dictionary  mapping each country to a list of languages.
COUNTRY_TO_LANGUAGES = country_to_languages()

# Dictionary mapping each language (qid) to its ISO code.
QID_TO_ISO = qid_to_iso()

# Threshold for occurrences in country of birth for a name to be considered as linked to that country and its languages.
OCCURRENCE_THRESHOLD = 5

# Threshold for occurrences in country of citizenship for a name to be considered as linked to that country and its languages.
OCCURRENCE_THRESHOLD_CITIZENSHIP = 10


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

            if not country_of_birth:
                country_of_citizenship = get_claims(entity['info']['claims'], 'P27')

            country_of_citizenship = country_of_citizenship[0] if country_of_citizenship else None

            # Keep track of given names' occurrences
            for given_name in given_names:
                if given_name not in given_names_occurrences:
                    given_names_occurrences[given_name] = {
                        'id': given_name,
                        'count': 0,
                        'country_of_birth': {},
                        'country_of_citizenship': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = given_names_occurrences[given_name]

                entry['count'] += 1
                entry['gender'][gender] = entry['gender'].get(gender, 0) + 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                if country_of_citizenship:
                    entry['country_of_citizenship'][country_of_citizenship] = entry['country_of_citizenship'].get(
                        country_of_citizenship, 0) + 1
                entry['year_of_birth'][year_of_birth] = entry['year_of_birth'].get(year_of_birth, 0) + 1

            # Keep track of family names' occurrences
            for family_name in family_names:
                if family_name not in family_names_occurrences:
                    family_names_occurrences[family_name] = {
                        'id': family_name,
                        'count': 0,
                        'country_of_birth': {},
                        'country_of_citizenship': {},
                        'year_of_birth': {},
                        'gender': {}
                    }

                entry = family_names_occurrences[family_name]

                entry['count'] += 1
                entry['country_of_birth'][country_of_birth] = entry['country_of_birth'].get(country_of_birth, 0) + 1
                if country_of_citizenship:
                    entry['country_of_citizenship'][country_of_citizenship] = entry['country_of_citizenship'].get(
                        country_of_citizenship, 0) + 1
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

    with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'rt') as anthroponyms:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in anthroponyms:

                entity = json.loads(line)

                counter_anthroponyms += 1

                # As a first step, use native labels (P1705)
                native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

                # Family name
                if 'family name' in entity['type']:
                    entity['occurrences'] = family_names_occurrences.get(entity['id'], {})

                # Given name
                else:
                    entity['occurrences'] = given_names_occurrences.get(entity['id'], {})

                entity['name'] = {Language.get(iso).language_name(): {'name': name, 'code': iso,
                                                                      'language': Language.get(
                                                                          iso).language_name()} for
                                  iso, name in native_labels.items()}

                # Retrieve all entity's labels
                labels = entity['info'].get('labels', None)

                # Get P407 claims
                languages_of_work_or_name = get_claims(entity['info']['claims'], 'P407')

                # Iterate over all languages of work or name. For each, get the name associated to that languages
                # from the entity's labels or, alternatively, get the first native label.
                if languages_of_work_or_name:
                    for qid in languages_of_work_or_name:

                        iso = QID_TO_ISO.get(qid)
                        if not iso:
                            continue

                        language = Language.get(iso).language_name()

                        if language in entity["name"]:
                            continue

                        name = native_labels.get(iso, "")

                        if not name and labels:
                            name = labels.get(iso, {}).get("value", "")

                        if not name and native_labels:
                            name = next(iter(native_labels.values()), "")

                        if not name:
                            continue

                        entity["name"][language] = {
                            "name": name,
                            "code": iso,
                            "language": language
                        }

                # Skip (entity has neither native labels nor labels).
                if not entity['name'] and not labels:
                    missing_labels_counter += 1
                    continue

                # Skip (no occurrences)
                if not entity['name'] and (not entity['occurrences'] or entity['occurrences']['count'] == 0):
                    missing_occurrences_counter += 1
                    continue

                # Get the list of countries where the names occurrs (using place of birth) more times than a threshold.
                countries_of_occurrence = []
                if entity['occurrences']:
                    countries_of_occurrence = [country for country in entity['occurrences']['country_of_birth'].keys()
                                               if
                                               entity['occurrences']['country_of_birth'][
                                                   country] >= OCCURRENCE_THRESHOLD]

                # Get the list of countries where the names occurrs (using place of citizenship) more times than a threshold.
                for country in entity['occurrences'].get('country_of_citizenship', {}).keys():
                    if country is not None and country not in countries_of_occurrence and \
                            entity['occurrences']['country_of_citizenship'][
                                country] >= OCCURRENCE_THRESHOLD_CITIZENSHIP:
                        countries_of_occurrence.append(country)

                # Get list of languages spoken in the countries where the name occurs, as retrieved above.
                # Iterate over all languages spoken in the countries of occurrence. For each, get the name associated
                # to that languages from the entity's labels or, alternatively, get the first native label.
                languages = countries_to_languages(COUNTRY_TO_LANGUAGES, countries_of_occurrence)
                if languages:
                    for qid in languages:

                        iso = QID_TO_ISO.get(qid)

                        if not iso:
                            continue

                        name = native_labels.get(iso, "")

                        if not name and labels:
                            name = labels.get(iso, {}).get('value', "")

                        if not name and native_labels:
                            name = next(iter(native_labels.values()), "")

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
