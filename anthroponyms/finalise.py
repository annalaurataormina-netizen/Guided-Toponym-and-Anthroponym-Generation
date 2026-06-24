import gzip
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_romanised, get_countries_names, get_country_languages

from clean import OCCURRENCE_THRESHOLD

MIN_LENGTH_THRESHOLD = 2
MAX_LENGTH_THRESHOLD = 30

COUNTRY_LANGUAGES = get_country_languages()


def main():
    counter = 0

    character_counter = Counter()

    breakdown_by_language = Counter()
    breakdown_by_country = Counter()
    breakdown_by_length = Counter()

    excluded_characters = 0
    excluded_length = 0

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_final.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # Skip entities without names.
                if not entity['name']:
                    continue

                for language, name in entity['name'].items():

                    # Get the romanised version of the name.
                    name_romanised = get_romanised(name['name'])

                    if not name_romanised:
                        excluded_characters += 1
                        continue

                    if len(name_romanised) < MIN_LENGTH_THRESHOLD or len(name_romanised) > MAX_LENGTH_THRESHOLD:
                        excluded_length += 1
                        continue

                    length = len(name_romanised)

                    anthroponym = {
                        'name_romanised': name_romanised,
                        'name': name['name'],
                        'language': language,
                        'language_code': name['code'],
                        'id': entity['id'],
                        'type': entity['type'],
                        'country': [country for country in entity['occurrences']['country_of_birth'].keys() if
                                    entity['occurrences']['country_of_birth'][
                                        country] >= OCCURRENCE_THRESHOLD],
                    }

                    output.write(json.dumps(anthroponym) + '\n')

                    counter += 1

                    character_counter.update(name_romanised)
                    breakdown_by_language.update([language])
                    breakdown_by_length[length] += 1

                    for country in entity['country']:
                        breakdown_by_country.update([country])

    countries_id_names = get_countries_names(list(breakdown_by_country.keys()))

    print('# of anthroponyms: ', counter, '\n')

    print('Character occurrences (romanised): ', character_counter, '\n')

    print('Breakdown by language: ', breakdown_by_language, '\n')
    print('Breakdown by country: ', {countries_id_names.get(k, k): v for k, v in breakdown_by_country.items()}, '\n')
    print('Breakdown by length (romanised): ', breakdown_by_length, '\n')

    print('# of anthroponyms excluded due to characters: ', excluded_characters, '\n')
    print('# of anthroponyms excluded due to length: ', excluded_length, '\n')


if __name__ == '__main__':
    main()
