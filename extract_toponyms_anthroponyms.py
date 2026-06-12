import gzip, json
from typing import Dict, Optional

'''
To get the latest Wikidata dump:
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz -P /vol/bitbucket/at2225/
'''

'''
SPARQL query to get all the subclasses of Q486972 ('human settlement'), Q56061 ('administrative territorial entity'),
Q6256 ('country'), Q271669 ('landform'), Q15324 ('body of water'), Q23442 ('island').
P279* is to get subclasses recursively.
Order matters due to the way get_gids() processes them.
The result of this query is saved in Toponyms/query.json.

SELECT DISTINCT ?item ?type WHERE {
  {?item wdt:P279* wd:Q486972 . BIND("settlement" AS ?type)}
  UNION {?item wdt:P279* wd:Q56061 . BIND("region" AS ?type)}
  UNION {?item wdt:P279* wd:Q271669 . BIND("landform" AS ?type)}
  UNION {?item wdt:P279* wd:Q15324 . BIND("body of water" AS ?type)}
  UNION {?item wdt:P279* wd:Q23442 . BIND("island" AS ?type)}
  UNION {?item wdt:P279* wd:Q6256 . BIND("country" AS ?type)}
}
'''

'''
SPARQL query to get all the subclasses of Q1243157 ('given name') and Q101352 ('family name').
P279* is to get subclasses recursively.
The result of this query is saved in Anthroponyms/query.json.

SELECT DISTINCT ?item ?type WHERE {
  {?item wdt:P279* wd:Q12308941 . BIND("male given name" AS ?type)}
  UNION {?item wdt:P279* wd:Q11879590 . BIND("female given name" AS ?type)}
  UNION {?item wdt:P279* wd:Q101352 . BIND("family name" AS ?type)}
}
'''

# Returns the list of qids to be used to obtain toponyms from the dump based on the "instance of" field.
def get_qids(path: str) -> Dict[str,str]:
    with open(path) as file:
        items = json.load(file)
        qids = {}
        for item in items:
            item_id = item['item'].split('/')[-1]
            item_type = item['type']
            if item_id in qids.keys():
                qids[item_id].append(item_type)
            else:
                qids[item_id] = [item_type]
        return qids

# Returns the list of values for a given property on an entity's claims.
def get_clean_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None

def main():

    # Dictionary where the key is the qid and the value is the type, as set in the SPARQL query.
    qids_toponyms = get_qids("Toponyms/query.json")
    qids_anthroponyms = get_qids("Anthroponyms/query.json")

    with gzip.open('/vol/bitbucket/at2225/latest-all.json.gz', 'rt') as dump:

        with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'wt') as toponyms:
            with gzip.open('/vol/bitbucket/at2225/anthroponyms.jsonl.gz', 'wt') as anthroponyms:

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

                    type_ = []

                    # Go through 'instance of' claims.
                    for claim in p31:

                        try:
                            # Check if entity is instance of any of the gids selected as toponyms.
                            val = claim['mainsnak']['datavalue']['value']['id']

                            if val in qids_toponyms and anthroponym is False:
                                toponym = True
                                type_.extend(qids_toponyms[val])
                                type_ = list(set(type_))

                            if val in qids_anthroponyms and toponym is False:
                                anthroponym = True
                                type_.extend(qids_anthroponyms[val])
                                type_ = list(set(type_))

                        except (KeyError, TypeError):
                            pass

                    # Entity is a toponym.
                    if toponym:

                        entity = {
                            'id': line.get('id', None),
                            'type': type_,
                            'info': line,
                        }

                        toponyms.write(json.dumps(entity) + '\n')

                    # Entity is an anthroponym.
                    if anthroponym:

                        entity = {
                            'id': line.get('id', None),
                            'type': type_,
                            'info': line,
                        }
                        
                        anthroponyms.write(json.dumps(entity) + '\n')

if __name__ == '__main__':
    main()
