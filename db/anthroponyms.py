import gzip
import json
from dotenv import load_dotenv
import os
import psycopg2.extras

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    user=os.getenv("PGUSER"),
    dbname=os.getenv("PGDATABASE"),
    password=os.getenv("PGPASSWORD"),
    options='-c client_encoding=UTF8'
)

psycopg2.extras.register_default_jsonb(conn)

cur = conn.cursor()

with gzip.open('/vol/bitbucket/at2225/anthroponyms_final.jsonl.gz', 'rt', encoding='utf-8') as f:
    batch = []
    for line in f:
        a = json.loads(line)
        batch.append((
            a['id'],
            a['language_code'],
            a['name'],
            a['name_romanised'],
            a['language'],
            a['country'],
            a['type'],
        ))

        if len(batch) == 1000:
            cur.executemany("""
                INSERT INTO anthroponyms.entries 
                (id, language_code, name, name_romanised, language, countries, types)
                VALUES (%s, %s, %s, %s, %s, %s::text[], %s::text[])
                ON CONFLICT (id, language_code) DO NOTHING
            """, batch)
            conn.commit()
            batch = []

    if batch:
        cur.executemany("""
            INSERT INTO anthroponyms.entries 
            (id, language_code, name, name_romanised, language, countries, types)
            VALUES (%s, %s, %s, %s, %s, %s::text[], %s::text[])
            ON CONFLICT (id, language_code) DO NOTHING
        """, batch)
        conn.commit()

cur.close()
conn.close()
print("Done")