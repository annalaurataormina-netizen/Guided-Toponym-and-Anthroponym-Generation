import gzip
import json
import random
from collections import Counter


def main():
    entities_missing_name = []
    types_missing_name = Counter()

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'rt') as input:

        for line in input:

            entity = json.loads(line)

            if not entity.get('name'):
                entities_missing_name.append(entity.get('id'))
                types_missing_name.update(entity.get('type', []))

    print('FOR ENTITIES MISSING NAMES')
    print('#: ', len(entities_missing_name))
    print('100 IDs (shuffled): ', random.sample(entities_missing_name, min(100, len(entities_missing_name))))
    print('breakdown by type: ', types_missing_name)


if __name__ == '__main__':
    main()
