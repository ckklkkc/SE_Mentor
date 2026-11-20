"""Offline integrity checks. Python standard library only."""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


def load_and_validate(path):
    path = Path(path)
    raw = path.read_text(encoding='utf-8-sig')
    docs = json.loads(raw) if path.suffix != '.jsonl' else [json.loads(s) for s in raw.splitlines() if s.strip()]
    if not isinstance(docs, list) or not docs:
        raise ValueError('Expected a nonempty JSON array or JSONL documents')
    ids, stems, targets = set(), set(), set()
    chapters = defaultdict(list)
    for q in docs:
        code = q.get('question_code', '?')
        assert isinstance(q, dict), code
        oid = q['_id']['$oid']
        assert re.fullmatch(r'[0-9a-f]{24}', oid) and oid not in ids, code
        ids.add(oid)
        assert isinstance(q['question'], str) and q['question'].strip(), code
        assert q['question'] not in stems, code
        stems.add(q['question'])
        opts = q['options']
        assert len(opts) == 4 and len(set(opts)) == 4, code
        assert all(isinstance(v,str) and v.strip() for v in opts), code
        assert type(q['answer']) is int and 0 <= q['answer'] <= 3, code
        assert q['review_status'] in ('approved', 'pending', 'rejected'), code
        assert q['assessment_target_id'] not in targets, code
        targets.add(q['assessment_target_id'])
        assert q['concept'] and q['assessment_target'] and q['analysis'], code
        assert q['course_unit_title'] == q['chapter'], code
        assert q['course_unit_id'].startswith(q['bank_batch']+':unit:'), code
        assert len(q['knowledge_point_ids']) == 1, code
        diagnoses=q['option_diagnoses']
        assert sorted(d['option_index'] for d in diagnoses) == list(range(4)), code
        for d in diagnoses:
            assert d['knowledge_point_id'] in q['knowledge_point_ids'], code
            if d['option_index'] == q['answer']:
                assert d['misconception'] is None, code
            else:
                assert d['misconception'], code
        pages=q['source_evidence']['pdf_pages']
        assert pages and all(type(p) is int and p>0 for p in pages), code
        assert len(q['source_chunk_ids']) == len(pages), code
        assert all(x.startswith('manual_pdf:') for x in q['source_chunk_ids']), code
        # These are package-local IDs, not false links into a pre-existing analysis.
        assert q['source_analysis_id'] is None and q['section_ids']==[], code
        assert len(q['question'])+sum(map(len,opts))+160 < 2000, code
        chapters[q['chapter']].append(q)
    for chapter,qs in chapters.items():
        assert 15<=len(qs)<=20, chapter
        counts=Counter(q['answer'] for q in qs)
        assert max(counts.values())-min(counts.values())<=1 and len(counts)==4, chapter
        assert len({q['assessment_target_id'] for q in qs}) >= 10, chapter
    return docs


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file',nargs='?',default=str(Path(__file__).with_name('diagnosis_quiz_all.json')))
    args=parser.parse_args()
    docs=load_and_validate(args.file)
    print(json.dumps({'valid':True,'questions':len(docs),'chapters':dict(Counter(q['chapter'] for q in docs))},ensure_ascii=False,indent=2))
