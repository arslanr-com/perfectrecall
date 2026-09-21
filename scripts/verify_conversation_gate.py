"""Small synthetic gate probes separate from the cost-development conversation.

No bank, answer model or user data. --live costs Jev calls; default is scripted.
False reuse is a risk to evidence completeness. Rejected safe reuse only costs more.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


CASES = [
    ('Show that information as a table.', True),
    ('Translate the database fact into French.', True),
    ('Give me a shorter version of the same fact.', True),
    ('Thanks for the information about the Cedar database.', True),
    ('Use that fact in a sentence for the README.', True),
    ('Which PostgreSQL version does Cedar use?', False),
    ('Who decided to use that database?', False),
    ('Was Cedar previously on MySQL?', False),
    ('Which database does Project Birch use?', False),
    ('Compare Cedar and Birch database choices.', False),
    ('Check our memory again to verify this.', False),
    ('That is outdated. Cedar moved to MariaDB.', False),
    ('What else do you remember about Cedar?', False),
    ('When did we migrate to that database?', False),
    ('Where are its backups stored?', False),
    ('What did I say about database licensing costs?', False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='perfectrecall-gate-') as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(root/'site')
        for key in tuple(os.environ):
            if key.startswith(('PERFECTRECALL_', 'MNEMOSYNE_', 'JEVOSYNE_')):
                del os.environ[key]
        os.environ['HERMES_HOME'] = str(root/'home')
        sys.path.insert(0, str(root/'site'))
        import perfectrecall  # noqa: F401
        from mnemosyne.core import jev
        from hermes_memory_provider._conversation_prefetch import ConversationPrefetch
        if not args.live:
            labels = dict(CASES)
            def transport(payload, timeout):
                reuse = labels[payload['state']['new_message']]
                return dict(model='test-double', answers={
                    'reuse': dict(type='noul', noul=.99 if reuse else .01),
                    'search_needed': dict(type='noul', noul=.01 if reuse else .99)})
            client = jev.JevClient('test-double', transport=transport)
            jev.client = lambda: client
        else:
            client = jev.client()
            original = client._transport
            def bounded(payload, timeout):
                if client.snapshot()['requests'] > 40 or client.snapshot()['cost_usd'] >= .01:
                    raise jev.JevError('Gate probe budget reached')
                return original(payload, timeout)
            client._transport = bounded
        row = dict(id='cedar', tier='working', content='Project Cedar uses PostgreSQL as its production database.')
        class Beam:
            conn = type('Connection', (), {'cursor': lambda _: None})()
            def _fetch_polyphonic_row(self, *args): return dict(row)
            def _polyphonic_row_passes_filters(self, row, **kwargs): return True
        anchor = 'Which production database does Project Cedar use?'
        cases = []
        for query, expected in CASES:
            cache = ConversationPrefetch()
            cache.remember(('probe',), anchor, anchor, [row], {})
            trace = {}
            selected = cache.select(Beam(), ('probe',), query, None, trace)
            reused = selected == anchor
            cases.append(dict(query=query, expected_reuse=expected, reused=reused, trace=trace))
        report = dict(synthetic=True, live_api=args.live,
                      wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                      probe_sha256=hashlib.sha256(json.dumps(CASES).encode()).hexdigest(),
                      cases=cases, false_reuses=sum(c['reused'] and not c['expected_reuse'] for c in cases),
                      missed_savings=sum(not c['reused'] and c['expected_reuse'] for c in cases),
                      usage=client.snapshot())
        report['status'] = 'passed' if report['false_reuses'] == 0 and not any(
            c['trace']['reason'] == 'gate_failed' for c in cases) else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if report['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
