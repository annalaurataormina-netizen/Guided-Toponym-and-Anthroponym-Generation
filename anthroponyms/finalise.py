import gzip
import json
from collections import Counter, defaultdict

from icu import Transliterator


def main():
    transliterator = Transliterator.createInstance("Any-Latin")
    counter = 0

    character_counter = Counter()
    character_counter_romanised = Counter()

    breakdown_by_language = Counter()
    breakdown_by_country = Counter()
    breakdown_by_length = Counter()
    breakdown_by_length_romanised = Counter()

    length_by_country = defaultdict(Counter)
    length_by_language = defaultdict(Counter)

    length_by_country_romanised = defaultdict(Counter)
    length_by_language_romanised = defaultdict(Counter)

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_final.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # Skip entities without names.
                if not entity['name']:
                    continue

                for language, name in entity['name'].items():

                    # Get the romanised version of the name.
                    name_romanised = transliterator.transliterate(name['name'])

                    # CONTINUE FROM HERE
                    anthroponym = {
                        'name_romanised': name_romanised,
                        'name': name['name'],
                        'language': language,
                        'language_code': name['code'],
                        'id': entity['id'],
                        'type': entity['type'],
                        'country': [country for country in entity['occurrences']['country'].ke,
                    }

                    output.write(json.dumps(anthroponym) + '\n')

                    counter += 1

                    character_counter.update(name['name'])
                    character_counter_romanised.update(name_romanised)

                    breakdown_by_language.update([language])

                    breakdown_by_length[len(name['name'])] += 1
                    breakdown_by_length_romanised[len(name_romanised)] += 1

                    for country in entity['country']:
                        breakdown_by_country.update([country])

                        length_by_country[country][len(name['name'])] += 1
                        length_by_country_romanised[country][len(name_romanised)] += 1

                    length_by_language[language][len(name['name'])] += 1
                    length_by_language_romanised[language][len(name_romanised)] += 1

    print('# of anthroponym: ', counter)
    print('Character occurrences: ', character_counter)
    print('Character occurrences in romanised names: ', character_counter_romanised)
    print('Breakdown by language: ', breakdown_by_language)
    print('Breakdown by country: ', breakdown_by_country)
    print('Breakdown by length: ', breakdown_by_length)
    print('Breakdown by length (romanised): ', breakdown_by_length_romanised)
    print('Length by country: ', length_by_country)
    print('Length by language: ', length_by_language)
    print('Length by country (romanised): ', length_by_country_romanised)
    print('Length by language (romanised): ', length_by_language_romanised)


if __name__ == '__main__':
    main()
