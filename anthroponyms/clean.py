import gzip
import json
import os
import sys

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_monolingual_claims


def main():
    counter = 0

    missing_id_counter = 0
    missing_name_counter = 0
    missing_language_counter = 0

    with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'wt') as output:

            for line in input:

                counter += 1

                entity = json.loads(line)

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
                # print(entity['country'])

                output.write(json.dumps(entity) + '\n')

    print('# of items: ', counter)
    print('# of items missing ID: ', missing_id_counter)
    print('# of items missing name: ', missing_name_counter)
    print('# of items missing language: ', missing_language_counter)


if __name__ == '__main__':
    main()
