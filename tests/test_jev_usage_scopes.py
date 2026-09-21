"""Concurrent callers must not inherit each other's API request counts."""
from concurrent.futures import ThreadPoolExecutor
import threading

from mnemosyne.core import jev


def test_nested_scopes_follow_workers_but_exclude_other_callers():
    barrier = threading.Barrier(2)
    def transport(payload, timeout):
        barrier.wait(2)
        return dict(model='test', answers={key: dict(type='noul', noul=.9)
                    for key in payload['questions']}, usage=dict(input_tokens=7, output_tokens=1, cost=.001))
    client = jev.JevClient('test', transport=transport)
    def call(name):
        with client.usage_scope() as outer:
            with client.usage_scope() as inner, ThreadPoolExecutor(max_workers=1) as worker:
                jev.submit(worker, client.evaluate, name, {'a': jev.noul('Relevant?')}).result()
            client.evaluate(name, {'a': jev.noul('Relevant?')})
        return outer, inner
    with ThreadPoolExecutor(max_workers=2) as callers:
        first, second = [future.result() for future in [callers.submit(call, 'one'), callers.submit(call, 'two')]]
    for outer, inner in (first, second):
        assert outer['requests'] == inner['requests'] == 1
        assert outer['input_tokens'] == inner['input_tokens'] == 7
        assert outer['cost_usd'] == inner['cost_usd'] == .001
        assert outer['cache_hits'] == 1 and inner['cache_hits'] == 0
    assert client.snapshot()['requests'] == 2
    assert client.snapshot()['cache_hits'] == 2


def test_provider_performance_diagnostics_omit_query_and_criteria(monkeypatch):
    import json
    from hermes_memory_provider import MnemosyneMemoryProvider
    from mnemosyne import diagnose
    provider = MnemosyneMemoryProvider()
    provider._beam = type('Beam', (), {'_last_jev_recall': {
        'status': 'completed', 'scanned': 10000, 'elapsed_seconds': 3.2,
        'evidence_questions': ['private criterion'], 'query': 'private query',
        'stages': {'ranking': {'status': 'completed', 'seconds': 2.1}},
        'usage': {'requests': 100},
    }})()
    provider._last_prefetch = {'status': 'completed', 'elapsed_seconds': 3.3}
    monkeypatch.setattr(diagnose, 'run_diagnostics', lambda **kwargs: {})
    result = json.loads(provider._handle_diagnose({}))
    assert result['jev_recall']['scanned'] == 10000
    assert result['prefetch']['status'] == 'completed'
    assert 'private' not in json.dumps(result)
