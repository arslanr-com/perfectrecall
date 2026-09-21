"""A prefetch budget must reach workers and not leave subsequent calls blocked."""
import threading
import time

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core import jev, jev_evidence


@pytest.mark.parametrize('mode', ['single', 'question'])
def test_budget_reaches_workers_and_stops_scheduling(monkeypatch, mode):
    timeouts = []
    def transport(payload, timeout):
        timeouts.append(timeout)
        time.sleep(timeout + .005)
        raise TimeoutError()
    client = jev.JevClient('test', transport=transport)
    monkeypatch.setattr(jev, 'client', lambda: client)
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', mode)
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '2')
    rows = [dict(id=str(i), content=f'Fact {i}') for i in range(1000)]
    started = time.monotonic()
    with pytest.raises(jev.JevDeadlineExceeded), jev.decision_budget(.1):
        jev_evidence.rank('query', rows)
    assert time.monotonic() - started < 1
    assert 1 <= len(timeouts) <= 2
    assert all(0 < timeout <= .1 for timeout in timeouts)
    assert client.snapshot()['failures'] == 0
    assert client.snapshot()['deadline_exceeded'] == len(timeouts)
    # The request budget is scoped, not leaked into a subsequent explicit call.
    jev.check_deadline()


def test_prefetch_timeout_releases_worker_for_following_turn(monkeypatch):
    monkeypatch.setenv('MNEMOSYNE_PREFETCH_BUDGET_SECONDS', '.1')
    provider = MnemosyneMemoryProvider()
    provider._beam = type('Beam', (), {})()

    def first(*args, **kwargs):
        while True:
            jev.check_deadline()
            time.sleep(.005)

    monkeypatch.setattr(provider, '_prefetch_locked', first)
    output = []
    worker = threading.Thread(target=lambda: output.append(provider.prefetch('first')))
    worker.start()
    worker.join(1)
    assert not worker.is_alive() and output == ['']
    assert provider._last_prefetch['status'] == 'timed_out'
    monkeypatch.setattr(provider, '_prefetch_locked', lambda *args, **kwargs: 'next context')
    assert provider.prefetch('next') == 'next context'
    assert provider._last_prefetch['status'] == 'completed'


def test_prefetch_lock_contention_is_bounded_without_changing_scope(monkeypatch):
    monkeypatch.setenv('MNEMOSYNE_PREFETCH_BUDGET_SECONDS', '.05')
    provider = MnemosyneMemoryProvider()
    lock = provider._ensure_beam_access_lock()
    entered, release = threading.Event(), threading.Event()
    def holder():
        with lock:
            entered.set()
            release.wait(2)
    worker = threading.Thread(target=holder)
    worker.start()
    assert entered.wait(1)
    try:
        start = time.monotonic()
        assert provider.prefetch('query') == ''
        assert time.monotonic() - start < 1
        assert provider._last_prefetch['status'] == 'lock_timeout'
    finally:
        release.set()
        worker.join(1)


def test_nested_budget_cannot_extend_outer_deadline():
    with jev.decision_budget(.01):
        with jev.decision_budget(30):
            time.sleep(.02)
            with pytest.raises(jev.JevDeadlineExceeded):
                jev.check_deadline()
    jev.check_deadline()


def test_late_valid_response_is_accounted_but_not_cached():
    def transport(payload, timeout):
        time.sleep(.025)
        return dict(model='test', answers={'a': dict(type='noul', noul=.9)}, usage={'cost': .001})
    client = jev.JevClient('test', transport=transport)
    with jev.decision_budget(.01), pytest.raises(jev.JevDeadlineExceeded):
        client.evaluate('evidence', {'a': jev.noul('Relevant?')})
    assert not client._cache
    assert client.snapshot()['deadline_exceeded'] == 1
    assert client.snapshot()['failures'] == 0
    assert client.snapshot()['cost_usd'] == .001
