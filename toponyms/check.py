import gzip, json


def main():

    ids = []
    types = {}

    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:

        for line in input:

            entity = json.loads(line)

            if entity.get('name') is None:
                ids.append(entity.get('id'))
                for type in entity.get('type'):
                    if type in types.keys():
                        types[type] += 1
                    else:
                        types[type] = 1

    print('# missing name:', len(ids))
    print('breakdown by type:', types)

if __name__ == '__main__':
    main()
