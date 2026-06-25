import gzip
import json

if __name__ == '__main__':

    target_id = 'Q113098'

    with gzip.open('/vol/bitbucket/at2225/humans.jsonl.gz', 'rt') as f:
        for line in f:
            entity = json.loads(line)
            if entity['id'] == target_id:
                print(json.dumps(entity['occurrences'], indent=2))
                break