import gzip
import json
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    user=os.getenv("PGUSER"),
    dbname=os.getenv("PGDATABASE"),
    password=os.getenv("PGPASSWORD"),
)

cur = conn.cursor()

with gzip.open('/vol/bitbucket/at2225/toponyms_final.jsonl.gz', 'rt') as f:
    batch = []
    for line in f:
        t = json.loads(line)
        batch.append((
            t['id'],
            t['language_code'],
            t['name'],
            t['name_romanised'],
            t['language'],
            t['country'],
            t['type'],
        ))
        if len(batch) == 1000:
            cur.executemany("""
                INSERT INTO toponyms.entries 
                (id, language_code, name, name_romanised, language, countries, types)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id, language_code) DO NOTHING
            """, batch)
            conn.commit()
            batch = []

    if batch:
        cur.executemany("""
            INSERT INTO toponyms.entries 
            (id, language_code, name, name_romanised, language, countries, types)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id, language_code) DO NOTHING
        """, batch)
        conn.commit()

cur.close()
conn.close()
print("Done")