import gzip
import json

from utils import get_qids


def main():
    # Dictionary where the key is the qid and the value is the type, as set in the SPARQL query.
    qids_toponyms = get_qids('toponyms')
    qids_anthroponyms = get_qids('anthroponyms')

    with gzip.open('/vol/bitbucket/at2225/latest-all.json.gz', 'rt') as dump:

        #with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'wt') as toponyms:
            with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'wt') as anthroponyms:
                #with gzip.open('/vol/bitbucket/at2225/humans.jsonl.gz', 'wt') as humans:

                    for line in dump:

                        line = line.strip().rstrip(',')
                        if line in ('[', ']', ''):
                            continue

                        line = json.loads(line)

                        # Get the list of claims for the entity.
                        claims = line.get('claims', {})
                        # Get the P31 ('instance of') claims.
                        p31 = claims.get('P31', [])

                        toponym = False
                        anthroponym = False
                        human = False

                        type_ = []

                        # Go through 'instance of' claims.
                        for claim in p31:

                            try:
                                val = claim['mainsnak']['datavalue']['value']['id']

                                # Entity is an instance of 'toponym'.
                                if val in qids_toponyms and anthroponym is False and human is False:
                                    toponym = True
                                    type_.extend(qids_toponyms[val])

                                # Entity is an instance of 'anthroponym'.
                                if val in qids_anthroponyms and toponym is False and human is False:
                                    anthroponym = True
                                    type_.extend(qids_anthroponyms[val])

                                # Entity is an instance of 'human'.
                                if val == 'Q5' and toponym is False and anthroponym is False:
                                    human = True
                                    type_ = ['human']
                                    break

                            except (KeyError, TypeError):
                                pass

                        # Make type_ a list of unique items.
                        type_ = list(set(type_))

                        # Each entity is a dictionary.
                        entity = {
                            # ID, like 'Q38'.
                            'id': line.get('id', None),
                            # List of unique types, like ['human'].
                            'type': type_,
                            # All the information, as pulled from WikiData, for that entity.
                            'info': line,
                        }

                        # Entity is a toponym.
                        # if toponym:
                        #     toponyms.write(json.dumps(entity) + '\n')

                        # Entity is an anthroponym.
                        if anthroponym:
                            anthroponyms.write(json.dumps(entity) + '\n')

                        # Entity is a human.
                        # if human:
                        #     humans.write(json.dumps(entity) + '\n')


if __name__ == '__main__':
    main()
