#!/usr/bin/env python3
"""Session evidence retrieval on a deterministic stratified LongMemEval-S subset.

This is NOT the official answer-generation score. Labels never enter retrieval.
A dev sample and disjoint test sample are selected by SHA256 within each type.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


def select_cases(data, split, per_type, test_offset=1):
    groups = defaultdict(list)
    for item in data:
        groups[item['question_type']].append(item)
    selected = []
    for kind in sorted(groups):
        group = sorted(groups[kind], key=lambda x: hashlib.sha256(
            ('jevosyne-v1:' + x['question_id']).encode()).hexdigest())
        selected.extend(group[:1] if split == 'dev' else group[test_offset:test_offset + per_type])
    return selected


def corpus(case):
    """Fixed character chunks, identical for both arms; no answer metadata."""
    rows, mapping = [], {}
    for index, (sid, date, turns) in enumerate(zip(case['haystack_session_ids'],
                                                case['haystack_dates'], case['haystack_sessions'])):
        text = '\n'.join(f"{t['role']}: {t['content']}" for t in turns)
        for part, offset in enumerate(range(0, len(text), 6000)):
            mid = f's{index:04d}c{part:04d}'
            rows.append(dict(id=mid, content=f'Session date: {date}\n' + text[offset:offset+6000],
                             source='benchmark', memory_type='fact', veracity='stated', importance=.5))
            mapping[mid] = sid
    return rows, mapping


def run(args):
    root = Path(args.source_root).resolve()
    sys.path.insert(0, str(root))
    if args.api_key_file:
        key = args.api_key_file.read_text().strip()
        os.environ['OPENROUTER_API_KEY'] = key
        os.environ['MNEMOSYNE_EMBEDDING_API_KEY'] = key
    os.environ['MNEMOSYNE_DECISION_BACKEND'] = 'baseline'
    os.environ['MNEMOSYNE_WRITE_CLASSIFIER'] = 'off'
    os.environ['MNEMOSYNE_CROSS_SESSION'] = '0'
    os.environ['MNEMOSYNE_WM_MAX_ITEMS'] = '10000'
    os.environ['MNEMOSYNE_POLYPHONIC_RECALL'] = '0'
    if args.backend == 'jev' or not args.dense:
        os.environ['MNEMOSYNE_NO_EMBEDDINGS'] = '1'
    else:
        os.environ.pop('MNEMOSYNE_NO_EMBEDDINGS', None)
        os.environ['MNEMOSYNE_EMBEDDING_MODEL'] = 'openai/text-embedding-3-small'
        os.environ['MNEMOSYNE_EMBEDDING_API_URL'] = 'https://openrouter.ai/api/v1'
    data_bytes = args.dataset.read_bytes()
    cases = select_cases(json.loads(data_bytes), args.split, args.per_type, args.test_offset)
    if args.case_ids:
        wanted = set(args.case_ids.split(','))
        cases = [case for case in cases if case['question_id'] in wanted]
        if {case['question_id'] for case in cases} != wanted:
            raise ValueError('Requested cases must belong to the selected split')
    if args.limit: cases = cases[:args.limit]
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core import embeddings
    embed_stats = dict(requests=0, input_tokens=0, cost_usd=0., upstream_cost_usd=0.)
    if args.dense:
        if not embeddings.available():
            raise RuntimeError('Dense baseline unavailable')
        # Record real response usage without recording input text or credentials.
        original_open = embeddings.urllib.request.OpenerDirector.open
        def tracked_open(opener, *a, **kw):
            response = original_open(opener, *a, **kw)
            class RecordedResponse:
                def __enter__(self): return self
                def __exit__(self, *exc): response.close()
                def __getattr__(self, name): return getattr(response, name)
                def read(self, *read_args):
                    raw = response.read(*read_args)
                    body = json.loads(raw)
                    usage = body.get('usage', {})
                    embed_stats['requests'] += 1
                    embed_stats['input_tokens'] += usage.get('prompt_tokens', usage.get('input_tokens', 0))
                    embed_stats['cost_usd'] += usage.get('cost', 0.)
                    embed_stats['upstream_cost_usd'] += (usage.get('cost_details') or {}).get('upstream_inference_cost', 0.)
                    if embed_stats['requests'] > 500 or max(embed_stats['cost_usd'], embed_stats['upstream_cost_usd']) > args.max_cost:
                        raise RuntimeError('Embedding evaluation budget exceeded')
                    return raw
            return RecordedResponse()
        embeddings.urllib.request.OpenerDirector.open = tracked_open
    client = None
    if args.backend == 'jev':
        from mnemosyne.core import jev
        client = jev.client()
        post = client._transport
        def bounded(payload, timeout):
            stats = client.snapshot()
            if stats['requests'] > 15000 or stats['cost_usd'] >= args.max_cost:
                raise jev.JevError('Evaluation budget exceeded')
            return post(payload, timeout)
        client._transport = bounded
    criteria_map = json.loads(args.evidence_questions.read_text()) if args.evidence_questions else {}
    if args.evidence_questions and any(c['question_id'] not in criteria_map for c in cases):
        raise ValueError('Missing caller-authored evidence questions')
    output = dict(status='running', protocol='jevosyne-longmemeval-session-retrieval-v4',
        dataset_sha256=hashlib.sha256(data_bytes).hexdigest(), split=args.split,
        per_type=args.per_type, test_offset=args.test_offset, backend=args.backend, dense=args.dense,
        selected_ids=[x['question_id'] for x in cases], chunk_chars=6000,
        retrieved_chunk_budget=20, evaluated_session_k=5,
        saves_exact_returned_records=args.save_context,
        query_mode='caller_criteria' if args.evidence_questions else 'original',
        criteria_sha256=hashlib.sha256(args.evidence_questions.read_bytes()).hexdigest() if args.evidence_questions else None,
        source_sha256=hashlib.sha256(b''.join(str(p.relative_to(root)).encode()+p.read_bytes() for p in sorted((root/'mnemosyne').rglob('*.py')))).hexdigest(),
        ranking=os.environ.get('MNEMOSYNE_JEV_RANKING', 'evidence') if args.backend == 'jev' else 'upstream_hybrid',
        results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        rows, mapping = corpus(case)
        with tempfile.TemporaryDirectory(prefix='jevosyne-lme-') as directory:
            os.environ['MNEMOSYNE_DATA_DIR'] = directory
            os.environ['MNEMOSYNE_DECISION_BACKEND'] = 'baseline'
            beam = BeamMemory(session_id='evaluation', db_path=Path(directory)/'memory.db')
            start = time.perf_counter()
            ids = beam.remember_batch(rows)
            mapping = {mid: mapping[row["id"]] for mid, row in zip(ids, rows)}
            if args.dense and beam.conn.execute("SELECT count(*) FROM memory_embeddings").fetchone()[0] != len(rows):
                raise RuntimeError("Dense baseline produced no embeddings")
            ingest_seconds = time.perf_counter()-start
            # Check no age/capacity trim silently reduced the corpus.
            count = beam.conn.execute('SELECT count(*) FROM working_memory').fetchone()[0]
            if count != len(rows): raise RuntimeError('Ingestion coverage mismatch')
            os.environ['MNEMOSYNE_DECISION_BACKEND'] = args.backend
            start = time.perf_counter()
            query = case['question'] + '\nQuestion date: ' + case['question_date']
            criteria = criteria_map.get(case['question_id'])
            kwargs = {'evidence_questions': criteria} if criteria and args.backend == 'jev' else {}
            if criteria and args.backend == 'baseline':
                query += '\nEvidence sought:\n' + '\n'.join(criteria)
            hits = beam.recall(query, top_k=20, **kwargs)
            seconds = time.perf_counter()-start
            retrieved = list(dict.fromkeys(mapping[h['id']] for h in hits))[:5]
            expected = set(case['answer_session_ids']) & set(case['haystack_session_ids'])
            correct = len(expected & set(retrieved))
            row = dict(unanswerable=case['question_id'].endswith('_abs'), id=case['question_id'], kind=case['question_type'], corpus_chunks=count,
                retrieved_sessions=retrieved, expected_sessions=sorted(expected),
                criteria=criteria,
                reciprocal_rank=next((1/(i+1) for i,sid in enumerate(retrieved) if sid in expected), 0.) if expected else None,
                hit_at_1=float(bool(retrieved) and retrieved[0] in expected) if expected else None,
                ndcg_at_5=(sum(1/math.log2(i+2) for i,sid in enumerate(retrieved) if sid in expected) / sum(1/math.log2(i+2) for i in range(min(5,len(expected))))) if expected else None,
                recall_at_5=correct/len(expected) if expected else None,
                all_evidence=expected.issubset(retrieved) if expected else None,
                abstention_correct=not retrieved if not expected else None,
                precision_at_5=correct/5, seconds=seconds, ingest_seconds=ingest_seconds)
            if args.save_context:
                # Only the records actually returned by Beam.recall, in order.
                # Never expand to whole sessions or add labelled evidence.
                row['returned_records'] = [dict(id=h['id'], session_id=mapping[h['id']],
                    content=h['content']) for h in hits]
            output['results'].append(row)
            beam.conn.close()
        output['usage'] = client.snapshot() if client else embed_stats
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps({'completed':len(output['results']), 'total':len(cases),
                          'id':row['id'], 'recall_at_5':row['recall_at_5']}), flush=True)
    output['status'] = 'completed'
    output['summary'] = {key: statistics.mean([r[key] for r in output['results'] if r[key] is not None])
        if any(r[key] is not None for r in output['results']) else None
        for key in ('recall_at_5','all_evidence','abstention_correct','precision_at_5','ndcg_at_5','reciprocal_rank','hit_at_1')}
    answerable = [r for r in output['results'] if not r['unanswerable']]
    unanswerable = [r for r in output['results'] if r['unanswerable']]
    output['answerable_count'] = len(answerable)
    output['answerable_summary'] = {key: statistics.mean(r[key] for r in answerable)
        for key in ('recall_at_5','all_evidence','precision_at_5','ndcg_at_5','reciprocal_rank','hit_at_1')} if answerable else {}
    output['unanswerable_count'] = len(unanswerable)
    output['unanswerable_empty_rate'] = statistics.mean(not r['retrieved_sessions'] for r in unanswerable) if unanswerable else None
    times = sorted(r['seconds'] for r in output['results'])
    output['summary']['p50_seconds'] = statistics.median(times)
    output['summary']['max_seconds'] = max(times)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--backend', choices=['baseline','jev'], required=True)
    parser.add_argument('--evidence-questions', type=Path)
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--per-type', type=int, default=2)
    parser.add_argument('--test-offset', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--case-ids', help='Comma-separated IDs within the selected split')
    parser.add_argument('--save-context', action='store_true',
                        help='Save exact returned records for downstream answer evaluation')
    parser.add_argument('--max-cost', type=float, default=.25)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        failure = json.loads(args.output.read_text()) if args.output.exists() else {}
        failure.update(status='failed', error_type=type(exc).__name__)
        args.output.write_text(json.dumps(failure, indent=2)+'\n')
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}), flush=True)
        raise SystemExit(1)
