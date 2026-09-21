#!/usr/bin/env python3
"""Benchmark the actual caller -> memory tool -> answer loop.

Luna is the external caller for both systems. No generative query component is
added to Jevosyne. Gold labels never enter the caller or memory tool messages.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from evaluate_answers import ANSWER_SYSTEM, ChatClient, digest, judge_messages
from evaluate_longmemeval import corpus, select_cases


SEARCH_PARAMETERS = {'query', 'limit', 'evidence_questions', 'temporal_weight',
                     'query_time', 'temporal_halflife'}


def search_schema(root):
    """Read the real description without importing the alternative runtime."""
    module = ast.parse((root/'mnemosyne/tool_schemas.py').read_text())
    schema = next(ast.literal_eval(node.value) for node in module.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'RECALL_SCHEMA' for t in node.targets))
    schema['parameters']['properties'] = {k: v for k, v in schema['parameters']['properties'].items()
                                           if k in SEARCH_PARAMETERS}
    schema['parameters']['additionalProperties'] = False
    return schema


def initial_messages(case, searches, records):
    return [{'role': 'system', 'content': ANSWER_SYSTEM +
        f'\nUse the memory search tool before answering. You may make at most {searches} '
        f'search calls and read at most {records} returned records in total. Follow the '
        'tool description when forming queries. Search again if a needed relationship '
        'or detail is missing and budget remains. Do not invent missing information.'},
        {'role': 'user', 'content': json.dumps(dict(question=case['question'],
            question_date=case['question_date']), ensure_ascii=False)}]


def exposed_records(hits, remaining_chars):
    records = []
    for hit in hits:
        if remaining_chars <= 0:
            break
        text = hit['content'][:remaining_chars]
        records.append(dict(id=hit['id'], content=text, truncated=len(text) < len(hit['content'])))
        remaining_chars -= len(text)
    return records, remaining_chars


def agent_answer(case, beam, schema, client, args):
    messages = initial_messages(case, args.searches, args.records)
    searches, remaining_records, remaining_chars = [], args.records, args.context_chars
    name = schema['name']
    while True:
        available = len(searches) < args.searches and remaining_records > 0 and remaining_chars > 0
        choice = client.request(args.caller_model, messages, args.answer_max_tokens, 'caller',
            args.reasoning_effort, tools=[dict(type='function', function=schema)],
            tool_choice=({'type': 'function', 'function': {'name': name}} if not searches
                         else 'auto' if available else 'none'))
        message = choice['message']
        tool_calls = message.get('tool_calls') or []
        if not tool_calls:
            answer = message.get('content')
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError('Caller returned neither a tool call nor an answer')
            return answer.strip(), searches
        # Keep provider reasoning state only in this in-memory conversation.
        # It is never written to benchmark outputs or shown as an explanation.
        messages.append({k: v for k, v in message.items()
                         if k in ('role', 'content', 'tool_calls', 'reasoning_details')})
        for call in tool_calls:
            if not available or len(searches) >= args.searches or remaining_records <= 0 or remaining_chars <= 0:
                raise ValueError('Caller exceeded the announced search budget')
            if call.get('type') != 'function' or call['function']['name'] != name:
                raise ValueError('Caller selected an unknown tool')
            arguments = json.loads(call['function']['arguments'])
            if not isinstance(arguments, dict) or set(arguments) - set(schema['parameters']['properties']):
                raise ValueError('Unsupported search arguments')
            query = arguments.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('Search query must be nonempty')
            limit = arguments.get('limit', 5)
            if type(limit) is not int or limit <= 0:
                raise ValueError('Search result limit must be a positive integer')
            limit = min(limit, remaining_records)
            kwargs = {k: v for k, v in arguments.items() if k not in ('query', 'limit')}
            start = time.perf_counter()
            hits = beam.recall(query, top_k=limit, **kwargs)
            seconds = time.perf_counter()-start
            records, remaining_chars = exposed_records(hits, remaining_chars)
            remaining_records -= len(records)
            searches.append(dict(arguments=arguments, effective_limit=limit,
                                 returned_records=records, seconds=seconds))
            result = dict(records=records, remaining_searches=args.searches-len(searches),
                          remaining_records=remaining_records)
            messages.append(dict(role='tool', tool_call_id=call['id'],
                                 content=json.dumps(result, ensure_ascii=False)))
            print(json.dumps(dict(id=case['question_id'], event='search_completed',
                                  search=len(searches), records=len(records))), flush=True)


def run(args):
    if any(value <= 0 for value in (args.searches, args.records, args.context_chars,
            args.answer_max_tokens, args.caller_budget, args.embedding_budget,
            args.jev_budget, args.max_jev_requests)):
        raise ValueError('All budgets must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    key = args.api_key_file.read_text().strip()
    os.environ.update(OPENROUTER_API_KEY=key, MNEMOSYNE_EMBEDDING_API_KEY=key,
        MNEMOSYNE_DECISION_BACKEND='baseline', MNEMOSYNE_WRITE_CLASSIFIER='off',
        MNEMOSYNE_CROSS_SESSION='0', MNEMOSYNE_WM_MAX_ITEMS='10000',
        MNEMOSYNE_POLYPHONIC_RECALL='0')
    if args.backend == 'jev':
        os.environ['MNEMOSYNE_NO_EMBEDDINGS'] = '1'
    else:
        os.environ.pop('MNEMOSYNE_NO_EMBEDDINGS', None)
        os.environ.update(MNEMOSYNE_EMBEDDING_MODEL='openai/text-embedding-3-small',
                         MNEMOSYNE_EMBEDDING_API_URL='https://openrouter.ai/api/v1')
    raw = args.dataset.read_bytes()
    cases = select_cases(json.loads(raw), args.split, args.per_type, args.test_offset)
    cases = cases[args.start_case:]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError('No selected cases')
    schema = search_schema(root)
    runtime_hash = digest(b''.join(str(p.relative_to(root)).encode()+p.read_bytes()
                         for p in sorted((root/'mnemosyne').rglob('*.py'))))
    plan = dict(protocol='jevosyne-agentic-memory-v1', scope=args.scope,
        backend=args.backend, dataset_sha256=digest(raw), selected_ids=[c['question_id'] for c in cases],
        source_sha256=runtime_hash, schema_sha256=digest(json.dumps(schema, sort_keys=True).encode()),
        harness_sha256=digest(Path(__file__).read_bytes()),
        client_sha256=digest((Path(__file__).parent/'evaluate_answers.py').read_bytes()),
        caller_model=args.caller_model, reasoning_effort=args.reasoning_effort,
        judge_model=args.judge_model, answer_max_tokens=args.answer_max_tokens,
        searches=args.searches, records=args.records, context_chars=args.context_chars,
        chunk_chars=6000, surface='Actual tool description; common query/limit/temporal subset plus Jev criteria')
    report = dict(status='running', plan=plan, results=[], calls=[])
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report['plan'] != plan:
            raise ValueError('Output belongs to a different frozen run')
        if report['status'] == 'completed':
            return
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core import embeddings
    embed_stats = report.get('embedding_usage', dict(requests=0, input_tokens=0,
        cost_usd=0., upstream_cost_usd=0.))
    if args.backend == 'baseline':
        if not embeddings.available():
            raise RuntimeError('Dense baseline unavailable')
        original_open = embeddings.urllib.request.OpenerDirector.open
        def tracked_open(opener, request, *a, **kw):
            url = request.get_full_url() if hasattr(request, 'get_full_url') else request
            response = original_open(opener, request, *a, **kw)
            if not str(url).rstrip('/').endswith('/embeddings'):
                return response
            class RecordedResponse:
                def __enter__(self): return self
                def __exit__(self, *exc): response.close()
                def __getattr__(self, name): return getattr(response, name)
                def read(self, *read_args):
                    body = response.read(*read_args)
                    usage = json.loads(body).get('usage') or {}
                    embed_stats['requests'] += 1
                    embed_stats['input_tokens'] += usage.get('prompt_tokens', 0)
                    embed_stats['cost_usd'] += usage.get('cost', 0) or 0
                    embed_stats['upstream_cost_usd'] += (usage.get('cost_details') or {}).get('upstream_inference_cost', 0) or 0
                    if max(embed_stats['cost_usd'], embed_stats['upstream_cost_usd']) >= args.embedding_budget:
                        raise RuntimeError('Embedding budget exceeded')
                    return body
            return RecordedResponse()
        embeddings.urllib.request.OpenerDirector.open = tracked_open
    decision_client = None
    prior_jev = report.get('jev_usage', {})
    if args.backend == 'jev':
        from mnemosyne.core import jev
        decision_client = jev.client()
        transport = decision_client._transport
        def bounded(payload, timeout):
            usage = decision_client.snapshot()
            if (usage['requests'] + prior_jev.get('requests', 0) >= args.max_jev_requests
                    or usage['cost_usd'] + prior_jev.get('cost_usd', 0) >= args.jev_budget):
                raise jev.JevError('Agent evaluation Jev budget exceeded')
            return transport(payload, timeout)
        decision_client._transport = bounded
    client = ChatClient(key, args.caller_budget)
    client.calls = report['calls']
    def save():
        report['calls'] = client.calls
        report['embedding_usage'] = embed_stats
        if decision_client:
            report['jev_usage'] = decision_client.snapshot()
            for field in ('requests', 'input_tokens', 'output_tokens', 'cache_hits',
                          'failures', 'seconds', 'cost_usd', 'priced_responses'):
                report['jev_usage'][field] += prior_jev.get(field, 0)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    completed = {r['id'] for r in report['results']}
    try:
        for case in cases:
            if case['question_id'] in completed:
                continue
            with tempfile.TemporaryDirectory(prefix='jevosyne-agent-') as directory:
                os.environ['MNEMOSYNE_DATA_DIR'] = directory
                os.environ['MNEMOSYNE_DECISION_BACKEND'] = 'baseline'
                beam = BeamMemory(session_id='evaluation', db_path=Path(directory)/'memory.db')
                try:
                    rows, _ = corpus(case)
                    ids = beam.remember_batch(rows)
                    if len(ids) != len(rows) or beam.conn.execute('SELECT count(*) FROM working_memory').fetchone()[0] != len(rows):
                        raise RuntimeError('Incomplete corpus ingestion')
                    if args.backend == 'baseline' and beam.conn.execute('SELECT count(*) FROM memory_embeddings').fetchone()[0] != len(rows):
                        raise RuntimeError('Incomplete embedding coverage')
                    os.environ['MNEMOSYNE_DECISION_BACKEND'] = args.backend
                    answer, searches = agent_answer(case, beam, schema, client, args)
                    label = client.call(args.judge_model, judge_messages(case, answer), 5, 'judge').lower()
                    if label not in ('yes', 'no'):
                        raise ValueError('Invalid judge label')
                    report['results'].append(dict(id=case['question_id'], kind=case['question_type'],
                        unanswerable=case['question_id'].endswith('_abs'), answer=answer,
                        correct=label == 'yes', searches=searches))
                    save()
                    print(json.dumps(dict(completed=len(report['results']), total=len(cases),
                        id=case['question_id'], correct=label == 'yes')), flush=True)
                finally:
                    beam.conn.close()
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        save()
        raise
    report.update(status='completed', summary=dict(cases=len(report['results']),
        errors=sum(not r['correct'] for r in report['results'])))
    report.pop('error_type', None)
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['baseline', 'jev'], required=True)
    parser.add_argument('--scope', choices=['diagnostic', 'frozen-independent'], default='diagnostic')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--per-type', type=int, default=2)
    parser.add_argument('--test-offset', type=int, default=1)
    parser.add_argument('--start-case', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--caller-model', default='openai/gpt-5.6-luna')
    parser.add_argument('--reasoning-effort', default='high', choices=['none','low','medium','high','xhigh','max'])
    parser.add_argument('--judge-model', default='openai/gpt-4.1')
    parser.add_argument('--answer-max-tokens', type=int, default=16384)
    parser.add_argument('--searches', type=int, default=3)
    parser.add_argument('--records', type=int, default=20)
    parser.add_argument('--context-chars', type=int, default=120000)
    parser.add_argument('--caller-budget', type=float, default=1.)
    parser.add_argument('--embedding-budget', type=float, default=.1)
    parser.add_argument('--jev-budget', type=float, default=.5)
    parser.add_argument('--max-jev-requests', type=int, default=30000)
    try:
        run(parser.parse_args())
    except Exception as exc:
        print(json.dumps(dict(status='failed', error_type=type(exc).__name__)), flush=True)
        raise SystemExit(1)
