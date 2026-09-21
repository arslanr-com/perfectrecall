"""Measure sequential Hermes conversation costs on a synthetic, changing bank.

Default is offline contract testing, not quality evidence. --live uses paid Jev.
Each mode starts with its own profile and empty client caches. No user data is read.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from measure_recall_performance import corpus


TURNS = [
    ('Which production database does Project Cedar use?', 'PostgreSQL', False),
    ('Put that same database fact in one bullet point.', 'PostgreSQL', True),
    ('Rewrite that same database fact in simpler words.', 'PostgreSQL', True),
    ('In which city does Project Cedar deploy?', 'Frankfurt', False),
    ('Turn that same deployment city fact into a short sentence.', 'Frankfurt', True),
    ('Which database does Project Maple-00016 use?', 'SQLite', False),
    ('When does Project Cedar back up its database?', 'evening', False),
    ('Keep only that backup schedule fact as one bullet.', 'evening', True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-root', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, default=256)
    parser.add_argument('--turns', type=int, default=len(TURNS))
    parser.add_argument('--mode', choices=['both', 'strict', 'economy'], default='both')
    parser.add_argument('--max-cost', type=float, default=.5)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not 32 <= args.records <= 10000 or not 1 <= args.turns <= len(TURNS):
        parser.error('Use 32..10000 records and 1..8 turns')
    if not 0 < args.max_cost <= 2:
        parser.error('Use a cost stop threshold above zero and at most $2')
    modes = ['strict', 'economy'] if args.mode == 'both' else [args.mode]
    fixtures = corpus(args.records, 'mixed')
    report = dict(synthetic=True, live_api=args.live, records=args.records,
                  wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(json.dumps(fixtures).encode()).hexdigest(),
                  hermes_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.hermes_root, text=True).strip(),
                  quality_measure='Expected evidence present in injected context; no answer model or independent quality benchmark',
                  cost_limit_note='Stop threshold checked before requests; concurrent in-flight charges may exceed it',
                  modes={})
    spent = 0.
    with tempfile.TemporaryDirectory(prefix='perfectrecall-conversation-') as directory:
        root = Path(directory)
        site = root / 'site'
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(site)
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_', 'PERFECTRECALL_', 'JEVOSYNE_')):
                del os.environ[key]
        os.environ.update(HERMES_HOME=str(root/'bootstrap'), PERFECTRECALL_HOST_LLM_ENABLED='0',
                          PERFECTRECALL_LLM_ENABLED='0')
        sys.path[:0] = [str(site), str(args.hermes_root.resolve())]
        from perfectrecall.install import configure_hermes
        from plugins.memory import load_memory_provider
        from agent.memory_manager import MemoryManager
        from mnemosyne.core import jev
        live_client = jev.client
        for mode in modes:
            home = root / mode
            os.environ['HERMES_HOME'] = str(home)
            os.environ['MNEMOSYNE_DATA_DIR'] = str(home / 'mnemosyne' / 'data')
            configure_hermes(home)
            if args.live:
                jev._client.cache_clear()
                client = live_client()
                underlying = client._transport
                def bounded(payload, timeout):
                    usage = client.snapshot()
                    if spent + usage['cost_usd'] >= args.max_cost or usage['requests'] > 3000:
                        raise jev.JevError('Conversation experiment budget reached')
                    return underlying(payload, timeout)
                client._transport = bounded
            else:
                def transport(payload, timeout):
                    answers = {}
                    for key, q in payload['questions'].items():
                        if q['type'] == 'choice':
                            label = next(iter(q['criteria']))
                            answers[key] = dict(type='choice', choice=label, confidence=1.,
                                probabilities={x: float(x == label) for x in q['criteria']})
                        else:
                            if key in {'reuse', 'search_needed'}:
                                score = .99 if payload['state']['new_message'] in [t[0] for t in TURNS if t[2]] else .01
                                if key == 'search_needed': score = 1 - score
                            elif isinstance(q['instructions'], dict):
                                instruction = q['instructions']
                                text = instruction.get('memory', instruction.get('candidate', ''))
                                question = instruction['question']
                                expected = next((t[1] for t in TURNS if t[0] in question), 'PostgreSQL')
                                score = .99 if expected in text else .01
                                if 'same substantive claims' in question:
                                    score = .01
                            else:
                                score = .99
                            answers[key] = dict(type='noul', noul=score)
                    return dict(model='scripted-test-double', answers=answers,
                                usage=dict(input_tokens=0, output_tokens=0))
                client = jev.JevClient('test-double', transport=transport)
            jev.client = lambda: client
            provider = load_memory_provider('perfectrecall', register_skills=False)
            provider.initialize('conversation', hermes_home=str(home), auto_sleep=False, prefetch_mode=mode)
            assert provider._beam is not None, provider._init_error
            assert provider._beam.db_path.is_relative_to(home.resolve())
            assert provider._beam.conn.execute('SELECT COUNT(*) FROM working_memory').fetchone()[0] == 0
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            provider._beam.conn.executemany(
                'INSERT INTO working_memory(id,content,session_id,source,importance,scope,timestamp) VALUES (?,?,?,?,?,?,?)',
                [(key, text, provider._beam.session_id, 'fact', .5, 'session',
                  (now - timedelta(seconds=60 if i < 17 else 120)).isoformat())
                 for i, (key, text) in enumerate(fixtures)])
            provider._beam.conn.commit()
            manager = MemoryManager()
            manager.add_provider(provider)
            runs = []
            result = report['modes'][mode] = dict(turns=runs)
            for index, (query, expected, can_reuse) in enumerate(TURNS[:args.turns]):
                with client.usage_scope() as usage:
                    started = time.monotonic()
                    context = manager.prefetch_all(query, session_id='conversation')
                    elapsed = time.monotonic() - started
                trace = dict(provider._last_prefetch)
                reused = trace.get('conversation', {}).get('action') == 'reuse_question'
                runs.append(dict(turn=index + 1, seconds=elapsed, expected_evidence_present=expected in context,
                                 expected_reuse=can_reuse, reused=reused, false_reuse=reused and not can_reuse,
                                 usage=dict(usage), prefetch=trace,
                                 scanned=provider._beam._last_jev_recall.get('scanned')))
                if index == 0:
                    # Exercise actual post-turn capture while the active query
                    # cache is populated. A useful new fact must be evaluated.
                    manager.sync_all('Project Cedar uses PostgreSQL for its new billing service.',
                                     'Understood.', session_id='conversation')
                    assert manager.flush_pending(timeout=30)
                    assert provider._sync_turn_diagnostics()['completed'] == 1
            result['evidence_hits'] = sum(t['expected_evidence_present'] for t in runs)
            result['false_reuses'] = sum(t['false_reuse'] for t in runs)
            result['usage'] = client.snapshot()
            result['automatic_capture'] = True
            result['final_records'] = provider._beam.conn.execute('SELECT COUNT(*) FROM working_memory').fetchone()[0]
            spent += result['usage']['cost_usd']
            manager.shutdown_all()
            client.close() if hasattr(client, 'close') else None
        report['total_cost_usd'] = spent
        report['status'] = 'passed' if all(
            not t['false_reuse'] and t['seconds'] < 8
            and t['prefetch']['status'] == 'completed'
            for mode in report['modes'].values() for t in mode['turns']) else 'failed'
        report['economy_evidence_check_passed'] = (
            all(t['expected_evidence_present'] for t in report['modes']['economy']['turns'])
            if 'economy' in report['modes'] else None)
        if args.mode == 'both':
            strict = report['modes']['strict']['usage']['cost_usd']
            economy = report['modes']['economy']['usage']['cost_usd']
            report['cost_reduction_percent'] = 100 * (strict - economy) / strict if strict else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if report['status'] != 'passed' or report['economy_evidence_check_passed'] is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
