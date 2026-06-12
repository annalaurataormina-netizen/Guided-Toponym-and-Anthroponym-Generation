import gzip, json
from typing import Dict, Optional

'''
SPARQL query to get countries and languages.
The result of this query is not saved yet.

SELECT ?country ?lang WHERE {
  ?country wdt:P31 wd:Q6256 .
  ?country wdt:P37 ?language .
  ?language wdt:P218 ?lang .
}
'''

# Returns the list of values for a given property on an entity's claims.
def get_clean_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None

def main():

    with gzip.open('toponyms.jsonl.gz', 'rt') as input:
        with open('toponyms.json', 'w') as output:

            for line in input:

                line = json.loads(line)

                # Get the list of claims for the entity.
                claims = line.get('claims', {})
                # Get the P31 ('instance of') claims.
                p31 = claims.get('P31', [])

                match = False

                # Go through 'instance of' claims.
                for claim in p31:

                    try:
                        # Check if entity is instance of any of the gids selected as toponyms.
                        val = claim['mainsnak']['datavalue']['value']['id']
                        if val in qids:
                            match = True
                            type_ = qids[val]
                            break

                    except (KeyError, TypeError):
                        pass

                # Entity is a toponym.
                if match:
                    entity = {
                        'id': line.get('id', None),
                        'name': line[''],
                        'country': get_clean_claims(claims, "P17"),
                        'region': get_clean_claims(claims, "P131"),
                        'language': get_clean_claims(claims, "P37"),
                        'type': type_
                    }

                    output.write(json.dumps(entity) + '\n')

if __name__ == '__main__':
    main()

'''
Notes:
- How to infer name? Ideally, should use the language of the place itself but seems messy.
'''