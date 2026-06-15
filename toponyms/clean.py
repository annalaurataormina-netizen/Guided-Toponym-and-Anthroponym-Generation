import gzip, json
from langcodes import Language
from utils import get_monolingual_claims, get_claims

'''
SPARQL query to get countries and languages.
The result of this query is not saved yet.

SELECT ?country ?lang WHERE {
  ?country wdt:P31 wd:Q6256 .
  ?country wdt:P37 ?language .
  ?language wdt:P218 ?lang .
}
'''

def main():

	counter = 0

	with gzip.open('/vol/bitbucket/at2225/toponyms.jsonl.gz', 'rt') as input:
		with gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'wt') as output:

			for line in input:

				counter += 1

				if counter > 1000:
					return

				entity = json.loads(line)

				native_labels = get_monolingual_claims(entity['info']['claims'], 'P1705')

				entity['country'] = get_claims(entity['info']['claims'], 'P17')
				entity['name'] = {Language.get(lang).language_name(): {'name': name, 'code': lang} for lang, name in
								  native_labels.items()}

				#print(entity.keys())
				#print(entity['info'].keys())
				#print(entity['info']['claims'].keys())

				#print(entity['id'])
				#print(entity['type'])
				#print(entity['name'])
				#print(entity['country'])

				output.write(json.dumps(entity) + '\n')

if __name__ == '__main__':
    main()
