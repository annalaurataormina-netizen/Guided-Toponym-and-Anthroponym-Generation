import gzip
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'wt') as output:
            for line in input:
                entity = json.loads(line)

                # entity['name'] = get_monolingual_claims(entity['claims'], 'P17')
                # entity['language'] = get_monolingual_claims(entity['info']['claims'], 'P1705')

                # print(entity.keys())
                # print(entity['info'].keys())
                # print(entity['info']['claims'].keys())
                # print(entity['info']['claims']['P407'])

                # print(entity['id'])
                # print(entity['type'])
                # print(entity['name'])
                # print(entity['country'])

                # break

                output.write(json.dumps(entity) + '\n')


if __name__ == '__main__':
    main()
