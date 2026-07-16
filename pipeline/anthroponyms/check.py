import gzip
import json
import random
from collections import Counter


def main():
    given_names = 0
    family_names = 0
    entities_missing_name = []
    entities_missing_id = 0
    entities_missing_occurrences = []
    types_missing_name = Counter()
    types_missing_id = Counter()
    types_missing_occurrences = Counter()

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'rt') as input:

        for line in input:

            entity = json.loads(line)

            if not entity.get('name'):
                entities_missing_name.append(entity.get('id'))
                types_missing_name.update(entity.get('type', []))

            if not entity.get('id'):
                entities_missing_id += 1
                types_missing_id.update(entity.get('type', []))

            if 'family name' in entity['type']:
                family_names += 1
            else:
                given_names += 1

            if not entity.get('occurrences'):
                entities_missing_occurrences.append(entity.get('id'))
                types_missing_occurrences.update(entity.get('type', []))


    print('FOR ENTITIES MISSING NAMES')
    print('#: ', len(entities_missing_name))
    print('100 IDs (shuffled): ', random.sample(entities_missing_name, min(100, len(entities_missing_name))))
    print('breakdown by type: ', types_missing_name)

    print('FOR ENTITIES MISSING IDS')
    print('#: ', str(entities_missing_id))
    print('breakdown by type: ', types_missing_id)

    print('FOR ENTITIES MISSING OCCURRENCES')
    print('#: ', len(entities_missing_occurrences))
    print('100 IDs (shuffled): ',
          random.sample(entities_missing_occurrences, min(100, len(entities_missing_occurrences))))
    print('breakdown by type: ', types_missing_occurrences)

    print('OVERALL')
    print('# given names: ', str(given_names))
    print('# family names: ', str(family_names))


if __name__ == '__main__':
    main()
