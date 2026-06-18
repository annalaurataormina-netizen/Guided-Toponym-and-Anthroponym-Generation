import gzip
import json
import os
import sys
from collections import Counter, defaultdict

from icu import Transliterator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_romanised, get_countries_names, split_diacritics

MIN_LENGTH_THRESHOLD = 2
MAX_LENGTH_THRESHOLD = 25


def main():
    counter = 0

    character_counter = Counter()

    breakdown_by_language = Counter()
    breakdown_by_country = Counter()
    breakdown_by_length = Counter()

    length_by_country = defaultdict(Counter)
    length_by_language = defaultdict(Counter)

    excluded_characters = 0
    excluded_length = 0
    excluded_language = 0

    with (gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input):
        with gzip.open('/vol/bitbucket/at2225/toponyms_final.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # Skip entities without names.
                if not entity['name']:
                    continue

                for language, name in entity['name'].items():

                    # Get the romanised version of the name.
                    name_romanised = get_romanised(name['name'])

                    if language in ('Unknown language', 'Unknown language [eml]', 'Uncoded languages',
                                    'Multiple languages', 'Australian languages',
                                    'Ancient Egyptian', 'Mycenaean Greek', 'Sumerian', 'Akkadian',
                                    'Elamite', 'Phoenician', 'Ancient Greek', 'Old Norse', 'Old English', 'Old French',
                                    'Old Turkish', 'Church Slavic', 'Ancient Hebrew', 'Pali', 'Latin', 'Aramaic'):
                        excluded_language += 1
                        continue

                    if name_romanised == '':
                        excluded_characters += 1
                        continue

                    if len(name_romanised) < MIN_LENGTH_THRESHOLD or len(name_romanised) > MAX_LENGTH_THRESHOLD:
                        excluded_length += 1
                        continue

                    name_romanised = split_diacritics(name_romanised)

                    toponym = {
                        'name_romanised': name_romanised,
                        'name': name['name'],
                        'language': language,
                        'language_code': name['code'],
                        'id': entity['id'],
                        'type': entity['type'],
                        'country': entity['country'],
                    }

                    output.write(json.dumps(toponym) + '\n')

                    counter += 1

                    character_counter.update(name_romanised)

                    breakdown_by_language.update([language])

                    breakdown_by_length[len(name_romanised)] += 1

                    for country in entity['country']:
                        breakdown_by_country.update([country])

                        length_by_country[country][len(name_romanised)] += 1

                    length_by_language[language][len(name_romanised)] += 1

    countries_id_names = get_countries_names(list(breakdown_by_country.keys()))

    print('# of toponyms: ', counter, '\n')

    print('Character occurrences (romanised): ', character_counter, '\n')

    print('Breakdown by language: ', breakdown_by_language, '\n')
    print('Breakdown by country: ', {countries_id_names.get(k, k): v for k, v in breakdown_by_country.items()}, '\n')
    print('Breakdown by length (romanised): ', breakdown_by_length, '\n')

    print('Length by country (romanised): ', {countries_id_names.get(k, k): v for k, v in length_by_country.items()},
          '\n')
    print('Length by language (romanised): ', length_by_language, '\n')

    print('# of toponyms excluded due to characters: ', excluded_characters, '\n')
    print('# of toponyms excluded due to length: ', excluded_length, '\n')
    print('# of toponyms excluded due to language: ', excluded_language, '\n')


if __name__ == '__main__':
    main()
