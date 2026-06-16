import gzip
import json


def main():
    counter = 0

    with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/toponyms_finalised.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                if not entity['name']:
                    continue

                for language in entity['name'].keys():
                    toponym = {
                        'name': entity['name'][language]['name'],
                        'language': language,
                        'language_code': entity['name'][language]['code'],
                        'id': entity['id'],
                        'type': entity['type'],
                        'country': entity['country'],
                    }

                    output.write(json.dumps(toponym) + '\n')

                    counter += 1

    print('# of toponyms: ', counter)


if __name__ == '__main__':
    main()
