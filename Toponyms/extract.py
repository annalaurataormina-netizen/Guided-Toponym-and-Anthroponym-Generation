import gzip, json
from typing import Dict, Optional

'''
SPARQL query to get all the subclasses of Q486972 ('human settlement'), Q56061 ('administrative territorial entity'),
Q6256 ('country'), Q271669 ('landform'), Q15324 ('body of water'), Q23442 ('island').
P279* is to get subclasses recursively.
Order matters due to the way get_gids() processes them.
The result of this query is saved in query.json.

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
SPARQL query to get countries and languages.

SELECT ?country ?lang WHERE {
  ?country wdt:P31 wd:Q6256 .
  ?country wdt:P37 ?language .
  ?language wdt:P218 ?lang .
}
'''

# Returns the list of qids to be used to obtain toponyms from the dump based on the "instance of" field.
def get_qids() -> Dict[str,str]:
    with open("query.json") as file:
        items = json.load(file)
        qids = {item['item'].split('/')[-1]: item['type'] for item in items}
    return qids

# Returns the list of values for a given property on an entity's claims.
def get_clean_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None

def main():
	
	count = 0

    # Dictionary where the key is the qid and the value is the type, as set in the SPARQL query.
	qids = get_qids()

	with gzip.open('/vol/bitbucket/at2225/latest-all.json.gz', 'rt') as input:
		with open('/vol/bitbucket/at2225/toponyms.json', 'w') as output:

			for line in input:

				if count >= 1000:
					return

				count += 1

				line = line.strip().rstrip(',')
				if line in ('[', ']', ''):
					continue

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
                        'name': '',
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
- How to deal with the huge dump? Is the dump even the right approach?
- How to infer name? Ideally, should use the language of the place itself but seems messy.
'''
