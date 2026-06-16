import gzip
import json
import random


def main():
    ids = []
    types = {}
    countries = {}

    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:

        for line in input:

            entity = json.loads(line)

            if not entity.get('name'):
                ids.append(entity.get('id'))
                for type in entity.get('type'):
                    if type in types.keys():
                        types[type] += 1
                    else:
                        types[type] = 1
                if entity.get('country'):
                    for country in entity.get('country'):
                        if country not in countries.keys():
                            countries[country] = 1
                        else:
                            countries[country] += 1

    print('# missing name: ', len(ids))
    print('First 100 IDs (shuffled list): ', random.sample(ids, 100))
    print('breakdown by type: ', types)


if __name__ == '__main__':
    main()
