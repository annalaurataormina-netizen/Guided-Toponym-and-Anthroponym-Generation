import gzip, json

'''
SPARQL query
P279* to get subclasses recursively.
Below to get all the subclasses of Q486972 ('human settlement'), Q56061 ('administrative territorial entity'),
Q6256 ('country'), Q271669 ('landform'), Q15324 ('body of water'), Q23442 ('island').

SELECT ?item WHERE {
  {?item wdt:P279* wd:Q486972 .}
  UNION {?item wdt:P279* wd:Q56061 .}
  UNION {?item wdt:P279* wd:Q6256 .}
  UNION {?item wdt:P279* wd:Q271669 .}
  UNION {?item wdt:P279* wd:Q15324 .}
  UNION {?item wdt:P279* wd:Q23442 .}
}
'''

# Returns the list of items that we can consider as toponyms based on query.json which is the
# result of the SPARQL query above
def get_qids():
    with open("query.json") as file:
        items = json.load(file)
        qids = [item['item'].split('/')[-1] for item in items]
    return qids

# Gets the first claim of a category of claims
def get_first(claims, pid):
    try:
        return claims[pid][0]['mainsnak']['datavalue']['value']
    except (KeyError, IndexError, TypeError):
        return None

def main():

    qids = set(get_qids())

    with gzip.open('latest-all.json.gz', 'rt') as input:
        with open('toponyms.json', 'w') as output:

            for line in input:

                line = line.strip().rstrip(',')
                if line in ('[', ']', ''):
                    continue

                line = json.loads(line)

                # Claims link entities
                claims = line.get('claims', {})
                # P31 are 'instance of' claims
                p31 = claims.get('P31', [])

                match = False

                for claim in p31:
                    try:
                        val = claim['mainsnak']['datavalue']['value']['id']
                        if val in qids:
                            match = True
                            break
                    except (KeyError, TypeError):
                        pass

                if match:
                    entity = {}
                    entity['id'] = line.get('id', None)
                    entity['name'] = line['']
                    entity['country'] = get_first(claims, "P17")
                    entity['region'] = get_first(claims, "P131")
                    entity['language'] = get_first(claims, "P37")
                    entity['type'] = get_first(claims, "P31")

                    output.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    main()

'''
Notes:
- How to deal with the huge dump? Is the dump even the right approach?
- Should try and keep the name of the place since the beginning? From the SPARQL query.
- Sensible to pick the first country, region, language?
- How to infer name? Ideally, should use the language of the place itself but seems messy.
'''