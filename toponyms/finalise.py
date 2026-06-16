import gzip
import json
from collections import Counter

from icu import Transliterator, defaultdict


def main():
    transliterator = Transliterator.createInstance("Any-Latin")
    counter = 0

    character_counter = Counter()

    breakdown_by_language = Counter()
    breakdown_by_country = Counter()
    breakdown_by_length = Counter()

    length_by_country = defaultdict(Counter)
    length_by_language = defaultdict(Counter)

    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/toponyms_finalised.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                if not entity['name']:
                    continue

                for language, name in entity['name']:
                    name_romanised = transliterator.transliterate(name['name'])

                    toponym = {
                        'name': name_romanised,
                        'name_original': name['name'],
                        'language': language,
                        'language_code': name['code'],
                        'id': entity['id'],
                        'type': entity['type'],
                        'country': entity['country'],
                    }

                    output.write(json.dumps(toponym) + '\n')

                    counter += 1

                    character_counter.update(name_romanised)

                    breakdown_by_language[language] += 1

                    for country in entity['country']:
                        breakdown_by_country[country] += 1

                    breakdown_by_length[len(name_romanised)] += 1

                    length_by_country[country][len(name_romanised)] += 1
                    length_by_language[language][len(name_romanised)] += 1

    print('# of toponyms: ', counter)
    print('Character occurrences: ', character_counter)
    print('Breakdown by language: ', breakdown_by_language)
    print('Breakdown by country: ', breakdown_by_country)
    print('Breakdown by length: ', breakdown_by_length)
    print('Length by country: ', length_by_country)
    print('Length by language: ', length_by_language)

if __name__ == '__main__':
    main()
