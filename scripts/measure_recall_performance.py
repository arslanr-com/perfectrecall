"""Measure complete Hermes prefetch on reproducible synthetic memory corpora.

Run with Hermes' Python and a built wheel. --live explicitly enables paid calls.
Reports contain aggregate metrics and synthetic IDs, never memory text or keys.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile


def corpus(records, shape):
    facts = [
        'uses PostgreSQL as its production database', 'deploys in Frankfurt',
        'is maintained by the Orion team', 'uses Python for its backend',
        'serves daily financial reports', 'backs up its database every evening',
        'retains application logs for thirty days', 'uses port 8443 for HTTPS',
        'runs its integration tests on every commit', 'uses Redis for ephemeral caching',
        'stores documents in an S3 bucket', 'sends alerts to the on-call engineer',
        'deploys a staging environment on Tuesdays', 'keeps audit records for one year',
        'uses Terraform to manage infrastructure', 'uses OAuth for sign-in',
    ]
    rows = []
    for index in range(records):
        if index < len(facts):
            content = f'Project Cedar {facts[index]}.'
        else:
            project = f'Maple-{index:05d}'
            content = f'Project {project} uses SQLite for telemetry in Oslo.'
            if shape == 'mixed' and index % 19 == 0:
                content = '\n'.join(
                    f'{"user" if turn % 2 == 0 else "assistant"}: '
                    + f'Project {project} checkpoint {turn}: sensor data stays in Oslo. ' * 8
                    for turn in range(8))
            elif shape == 'mixed' and index % 5 == 0:
                content = ' '.join(f'Project {project} technical detail {part}: local telemetry uses SQLite and is reviewed by team Maple.'
                                   for part in range(15))
            elif index == records - 1:
                content = f'user: {content}\nassistant: Project {project} has no relation to Cedar.'
        rows.append((f'fixture-{index:05d}', content))
    return rows


def serial_dedup(rows, top_k):
    """Original 0.1.0a1 selection loop, retained only for A/B reproduction."""
    from mnemosyne.core import jev
    kept = []
    for row in rows:
        if len(kept) >= top_k:
            break
        if any(row['content'] == old['content'] for old in kept):
            continue
        comparable = [r for r in kept if len(jev._json(r['content'])) <= 2000]
        if comparable and len(jev._json(row['content'])) <= 2000:
            scores = jev.duplicate_scores(row['content'], [r['content'] for r in comparable])
            if max(scores) >= jev.threshold('DEDUP_THRESHOLD', .97):
                continue
        kept.append(row)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-root', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, default=241)
    parser.add_argument('--shape', choices=['short', 'mixed'], default='short')
    parser.add_argument('--batch-mode', choices=['single', 'question'], default='question')
    parser.add_argument('--http-transport', choices=['urllib', 'pooled'], default='pooled')
    parser.add_argument('--request-bytes', type=int, choices=[24000, 48000])
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--serial-dedup', action='store_true')
    parser.add_argument('--legacy-prefetch', action='store_true',
                        help='Reproduce the old unbudgeted prefetch in this temporary profile only; at most 1024 records')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--max-requests', type=int, default=40000)
    parser.add_argument('--max-cost', type=float, default=2.)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; this script makes paid API requests')
    if not 16 <= args.records <= 10000 or not 1 <= args.workers <= 256 or not 1 <= args.iterations <= 20:
        parser.error('Use 16..10000 records, 1..256 workers and 1..20 iterations')
    if args.max_requests < 1 or not 0 < args.max_cost <= 20:
        parser.error('Use a positive request limit and a cost limit of at most $20')
    if args.legacy_prefetch and args.records > 1024:
        parser.error('--legacy-prefetch is restricted to at most 1024 synthetic records')
    rows = corpus(args.records, args.shape)
    query = 'What are the recorded facts about Project Cedar?'
    report = dict(synthetic=True, records=args.records, shape=args.shape,
                  batch_mode=args.batch_mode, http_transport=args.http_transport, workers=args.workers,
                  serial_dedup=args.serial_dedup, legacy_prefetch=args.legacy_prefetch,
                  fixture_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                  corpus_characters=sum(len(text) for _, text in rows),
                  wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(), runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='perfectrecall-performance-') as directory:
        root = Path(directory)
        site, home = root / 'site', root / 'profile'
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(site)
        for name in tuple(os.environ):
            if name.startswith(('PERFECTRECALL_', 'MNEMOSYNE_', 'JEVOSYNE_')):
                del os.environ[name]
        os.environ.update(HERMES_HOME=str(home), PERFECTRECALL_HOST_LLM_ENABLED='0',
                          PERFECTRECALL_LLM_ENABLED='0', PERFECTRECALL_JEV_WORKERS=str(args.workers),
                          PERFECTRECALL_JEV_BATCH_MODE=args.batch_mode,
                          PERFECTRECALL_JEV_HTTP_TRANSPORT=args.http_transport)
        sys.path[:0] = [str(site), str(args.hermes_root.resolve())]
        from perfectrecall.install import configure_hermes
        from mnemosyne.core import jev, jev_evidence, jev_recall
        from plugins.memory import load_memory_provider
        from agent.memory_manager import MemoryManager
        report['spans'] = sum(sum(1 for _ in jev_evidence.evidence_spans(text)) for _, text in rows)
        report['hermes_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.hermes_root, text=True).strip()
        if args.serial_dedup:
            jev_recall._diverse_top = serial_dedup
        configure_hermes(home)
        provider = load_memory_provider('perfectrecall', register_skills=False)
        provider.initialize('performance', hermes_home=str(home), auto_sleep=False)
        assert provider._beam is not None, provider._init_error
        if args.legacy_prefetch:
            def legacy_prefetch(query, *, session_id=''):
                with provider._ensure_beam_access_lock():
                    return provider._prefetch_locked(query, session_id=session_id)
            provider.prefetch = legacy_prefetch
        provider._beam.conn.executemany(
            'INSERT INTO working_memory(id,content,session_id,source,importance,scope,timestamp) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
            [(key, text, provider._beam.session_id, 'fact', .5, 'session') for key, text in rows])
        provider._beam.conn.commit()
        manager = MemoryManager()
        manager.add_provider(provider)
        client = jev.client()
        if args.request_bytes is not None:
            client.request_bytes = args.request_bytes
        report['request_bytes'] = client.request_bytes
        transport = client._transport
        durations, latency_lock = [], threading.Lock()

        def bounded(payload, timeout):
            usage = client.snapshot()
            if usage['requests'] > args.max_requests or usage['cost_usd'] >= args.max_cost:
                raise jev.JevError('Performance experiment request or cost limit reached')
            start = time.monotonic()
            try:
                return transport(payload, timeout)
            finally:
                with latency_lock:
                    durations.append(time.monotonic() - start)

        client._transport = bounded
        try:
            for iteration in range(args.iterations):
                for temperature in ['cold', 'warm']:
                    if temperature == 'cold':
                        with client._lock:
                            client._cache.clear()
                            client._decision_cache.clear()
                    before = client.snapshot()
                    start = time.monotonic()
                    context = manager.prefetch_all(query, session_id='performance')
                    elapsed = time.monotonic() - start
                    worker = manager._external_prefetch_threads.get('perfectrecall')
                    alive = bool(worker and worker.is_alive())
                    entry = dict(iteration=iteration, temperature=temperature, host_elapsed_seconds=elapsed,
                                 context_injected=bool(context), cedar_context='Project Cedar' in context,
                                 unrelated_context='Maple-' in context, worker_alive_at_return=alive)
                    if alive:
                        followup_start = time.monotonic()
                        followup = manager.prefetch_all(query, session_id='performance')
                        entry.update(immediate_followup_seconds=time.monotonic()-followup_start,
                                     immediate_followup_empty=not bool(followup))
                        while worker.is_alive():
                            worker.join(25)
                            if worker.is_alive():
                                print('Host window elapsed; observing underlying worker cleanup.', flush=True)
                    after = client.snapshot()
                    entry['usage'] = {key: after[key] - before[key] for key in before
                                      if isinstance(before[key], (int, float))}
                    trace = getattr(provider._beam, '_last_jev_recall', {})
                    entry['recall'] = {key: trace[key] for key in ('status', 'scanned', 'elapsed_seconds', 'stages', 'error_type') if key in trace}
                    entry['prefetch'] = getattr(provider, '_last_prefetch', {})
                    report['runs'].append(entry)
                    report['usage'] = after
                    args.output.write_text(json.dumps(report, indent=2)+'\n')
                    print(json.dumps(entry), flush=True)
            report['api_latency_seconds'] = dict(min=min(durations), median=statistics.median(durations), max=max(durations)) if durations else {}
            report['all_injected'] = all(run['cedar_context'] and not run['worker_alive_at_return'] for run in report['runs'])
            args.output.write_text(json.dumps(report, indent=2)+'\n')
        finally:
            manager.shutdown_all()


if __name__ == '__main__':
    main()
