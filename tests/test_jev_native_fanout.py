"""Native batches overlap without changing question content or dropping work."""
from collections import Counter
import threading

import pytest

from mnemosyne.core import jev


def test_bounded_fanout_matches_serial_packing_and_preserves_keys(monkeypatch):
    questions = {str(i): jev.noul('Relevant?', f'Unique candidate {i} ' * 30) for i in range(16)}
    seen, barrier = [], None
    def transport(payload, timeout):
        seen.append(payload)
        if barrier:
            barrier.wait(2)
        return dict(model='test', answers={key: dict(type='noul', noul=.9) for key in payload['questions']})
    client = jev.JevClient('test', transport=transport, cache_size=0)
    client.request_bytes = 1300
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '1')
    serial = client.fanout({'query': 'Cedar'}, questions)
    original = Counter(jev._json(body) for body in seen)
    assert len(seen) == 16
    seen.clear()
    barrier = threading.Barrier(4)
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '4')
    assert client.fanout({'query': 'Cedar'}, questions) == serial
    assert Counter(jev._json(body) for body in seen) == original
    assert list(serial) == list(questions)


def test_failed_batch_cannot_return_partial_facts(monkeypatch):
    from mnemosyne.core.jev_recall import _rank
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '2')
    def transport(payload, timeout):
        if any('broken' in q['instructions']['candidate'] for q in payload['questions'].values()):
            raise jev.JevError('injected failure')
        return dict(model='test', answers={key: dict(type='noul', noul=.9) for key in payload['questions']})
    client = jev.JevClient('test', transport=transport)
    monkeypatch.setattr(jev, 'client', lambda: client)
    rows = [dict(id=str(i), content=f'Fact {i}') for i in range(10000)]
    rows[-1]['content'] = 'broken'
    with pytest.raises(jev.JevError, match='injected failure'):
        _rank('query', rows)
