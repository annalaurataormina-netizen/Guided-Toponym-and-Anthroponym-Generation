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
def get_claims(claims: Dict, pid: str) -> Optional[list[str]]:
	try:
		result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
		return result if result else None
	except (KeyError, IndexError, TypeError):
		return None

# Returns the list of values for a given property on an entity's claims.
def get_monolingual_claims(claims: Dict, pid: str) -> Optional[list[str]]:
	try:
		result = {claim['mainsnak']['datavalue']['value']['language']: claim['mainsnak']['datavalue']['value']['text'] for claim in claims[pid]}
		return result if result else None
	except (KeyError, IndexError, TypeError):
		return None

def main():

	with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input:
		with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

			for line in input:

				entity = json.loads(line)

				entity['name'] = get_monolingual_claims(entity['info']['claims'], 'P1705')
				entity['country'] = get_claims(entity['info']['claims'], 'P17')

				#print(entity.keys())
				#print(entity['info'].keys())
				#print(entity['info']['claims'].keys())

				print(entity['id'])
				print(entity['type'])
				print(entity['name'])
				print(entity['country'])

				output.write(json.dumps(entity) + '\n')

if __name__ == '__main__':
    main()
