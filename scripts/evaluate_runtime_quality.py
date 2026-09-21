"""Paired development check of Jev request formats with an actual calling agent.

Uses the frozen public caller/judge protocol but loads fixtures directly into
SQLite: ingestion decisions and removed embedding modules are not part of this
retrieval comparison. Output may contain public-dataset answers and traces;
keep raw reports outside source control and publish only reduced summaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-mode', choices=['single', 'question'], required=True)
    parser.add_argument('--per-type', type=int, default=2)
    parser.add_argument('--test-offset', type=int, default=10)
    parser.add_argument('--caller-model', default='openai/gpt-5.6-luna')
    parser.add_argument('--reasoning-effort', default='high')
    parser.add_argument('--judge-model', default='openai/gpt-4.1')
    parser.add_argument('--caller-budget', type=float, default=1.)
    parser.add_argument('--jev-budget', type=float, default=3.)
    parser.add_argument('--max-jev-requests', type=int, default=50000)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not args.live or not os.environ.get('OPENROUTER_API_KEY'):
        parser.error('--live and OPENROUTER_API_KEY are required for paid model calls')
    if args.output.exists():
        parser.error('Use a new output path; retain earlier attempts')
    if not 1 <= args.per_type <= 20 or min(args.caller_budget, args.jev_budget, args.max_jev_requests) <= 0:
        parser.error('Use 1..20 cases per type and positive budgets')
    root = Path(__file__).resolve().parents[1]
    frozen = root / 'benchmarks/historical/confirmation'
    sys.path[:0] = [str(root), str(frozen)]
    from evaluate_agentic_memory import agent_answer, search_schema
    from evaluate_answers import ChatClient, judge_messages
    from evaluate_longmemeval import corpus, select_cases
    raw = args.dataset.read_bytes()
    cases = select_cases(json.loads(raw), 'test', args.per_type, args.test_offset)
    source = b''.join(str(path.relative_to(root)).encode() + path.read_bytes()
                      for package in ['mnemosyne', 'hermes_memory_provider', 'perfectrecall']
                      for path in sorted((root / package).rglob('*.py')))
    args.searches, args.records, args.context_chars, args.answer_max_tokens = 3, 20, 120000, 16384
    schema = search_schema(root)
    report = dict(status='running', plan=dict(scope='development',
        batch_mode=args.batch_mode, selected_ids=[case['question_id'] for case in cases],
        dataset_sha256=hashlib.sha256(raw).hexdigest(), runtime_sha256=hashlib.sha256(source).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caller_model=args.caller_model, reasoning_effort=args.reasoning_effort,
        judge_model=args.judge_model, searches=args.searches, records=args.records,
        context_chars=args.context_chars, chunk_chars=6000), results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='perfectrecall-quality-') as directory:
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_', 'PERFECTRECALL_', 'JEVOSYNE_')):
                del os.environ[key]
        os.environ.update(HERMES_HOME=directory, PERFECTRECALL_LLM_ENABLED='0',
                          PERFECTRECALL_HOST_LLM_ENABLED='0', PERFECTRECALL_JEV_BATCH_MODE=args.batch_mode,
                          PERFECTRECALL_JEV_HTTP_TRANSPORT='pooled', PERFECTRECALL_CROSS_SESSION='0')
        import perfectrecall
        perfectrecall.configure()
        from mnemosyne.core.beam import BeamMemory
        from mnemosyne.core import jev
        decision = jev.client()
        transport = decision._transport
        def bounded(payload, timeout):
            usage = decision.snapshot()
            if usage['requests'] > args.max_jev_requests or usage['cost_usd'] >= args.jev_budget:
                raise jev.JevError('Quality experiment budget reached')
            return transport(payload, timeout)
        decision._transport = bounded
        caller = ChatClient(os.environ['OPENROUTER_API_KEY'], args.caller_budget)

        def save():
            report['jev_usage'] = decision.snapshot()
            report['calls'] = caller.calls
            args.output.write_text(json.dumps(report, indent=2)+'\n')

        save()  # Freeze all case IDs and settings before seeing any answers.
        try:
            for index, case in enumerate(cases):
                with tempfile.TemporaryDirectory(dir=directory) as case_dir:
                    beam = BeamMemory(session_id='evaluation', db_path=Path(case_dir) / 'memory.db')
                    try:
                        rows, _ = corpus(case)
                        beam.conn.executemany(
                            'INSERT INTO working_memory(id,content,session_id,source,memory_type,veracity,importance,scope,timestamp) '
                            'VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                            [(row['id'], row['content'], 'evaluation', row['source'], row['memory_type'],
                              row['veracity'], row['importance'], 'session') for row in rows])
                        beam.conn.commit()
                        before = decision.snapshot()
                        started = time.monotonic()
                        answer, searches = agent_answer(case, beam, schema, caller, args)
                        label = caller.call(args.judge_model, judge_messages(case, answer), 5, 'judge').lower()
                        if label not in {'yes', 'no'}:
                            raise ValueError('Invalid judge label')
                        after = decision.snapshot()
                        report['results'].append(dict(id=case['question_id'], kind=case['question_type'],
                            answer=answer, correct=label == 'yes', searches=searches,
                            seconds=time.monotonic()-started, usage={key: after[key]-before[key] for key in before if isinstance(before[key], (int, float))}))
                        save()
                        print(json.dumps(dict(completed=index+1, total=len(cases), id=case['question_id'], correct=label == 'yes')), flush=True)
                    finally:
                        beam.conn.close()
        except Exception as exc:
            report.update(status='failed', error_type=type(exc).__name__)
            save()
            raise
        finally:
            decision.close()
        report.update(status='completed', summary=dict(cases=len(cases), errors=sum(not row['correct'] for row in report['results'])))
        save()


if __name__ == '__main__':
    main()
