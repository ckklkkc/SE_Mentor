"""Validate locally by default; insert missing IDs only when --apply is supplied."""
import argparse
import json
import os
from pathlib import Path
from collections import Counter
from validate_bank import load_and_validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',default=str(Path(__file__).with_name('diagnosis_quiz_all.json')))
    p.add_argument('--uri',default=os.environ.get('MONGODB_URI','mongodb://localhost:27017/'))
    p.add_argument('--database',default='TABotAI_quiz')
    p.add_argument('--collection',default='diagnosis_quiz')
    p.add_argument('--status',choices=['approved','pending'],default='approved')
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    docs=load_and_validate(args.file)
    print(json.dumps({'questions':len(docs),'database':args.database,'collection':args.collection,'status_for_new_documents':args.status,'chapter_counts':dict(Counter(q['chapter'] for q in docs))},ensure_ascii=False,indent=2))
    if not args.apply:
        print('DRY RUN: local validation only; no MongoDB connection or changes. Add --apply to insert.')
        return
    try:
        from pymongo import MongoClient, UpdateOne
        from bson import json_util
    except ImportError as exc:
        raise SystemExit('Install pymongo in this interpreter: python -m pip install pymongo') from exc
    # json_util turns Extended JSON $oid/$date into BSON ObjectId/datetime values.
    bson_docs=json_util.loads(json.dumps(docs,ensure_ascii=False))
    for doc in bson_docs:
        doc['review_status']=args.status
    with MongoClient(args.uri,serverSelectionTimeoutMS=10000) as client:
        client.admin.command('ping')
        collection=client[args.database][args.collection]
        operations=[]
        for doc in bson_docs:
            oid=doc['_id']
            content={k:v for k,v in doc.items() if k!='_id'}
            operations.append(UpdateOne({'_id':oid},{'$setOnInsert':content},upsert=True))
        # Repeated imports preserve existing documents, including later teacher edits.
        result=collection.bulk_write(operations,ordered=True)
        print(json.dumps({'inserted':result.upserted_count,'existing_preserved':result.matched_count,'modified':result.modified_count},indent=2))
        print('Existing approved questions were preserved and can still be sampled by title. Use this package course_unit_id to select this bank only.')


if __name__=='__main__':
    main()
