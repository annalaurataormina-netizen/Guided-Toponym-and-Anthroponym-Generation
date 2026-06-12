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

    with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

            for entity in input:

                '''
                entity['name'] = ''
                entity['country'] = ''
                entity['region'] = ''
                entity['language'] = ''
                '''

                print(entity['info'].keys())

                '''
                    'country': get_clean_claims(claims, "P17"),
                    'region': get_clean_claims(claims, "P131"),
                    'language': get_clean_claims(claims, "P37"),
                '''

                output.write(json.dumps(entity) + '\n')

                break

if __name__ == '__main__':
    main()