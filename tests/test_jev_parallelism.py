"""Scheduling must preserve full-corpus decisions and fail as a whole."""
import threading

import pytest

from mnemosyne.core import jev, jev_evidence


@pytest.fixture(autouse=True)
def single_span_format(monkeypatch):
    # These tests pin the retained one-span/request scheduling contract.
    # Native batches and their complete coverage are tested separately.
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', 'single')


@pytest.mark.parametrize('raw', ['0', '-1', '257', 'many', '2.5', ''])
def test_invalid_concurrency_fails_before_evaluating(monkeypatch, raw):
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', raw)
    monkeypatch.setattr(jev, 'client', lambda: pytest.fail('API must not be called'))
    with pytest.raises(ValueError, match='1 to 256'):
        jev_evidence.rank('q', [dict(id='a', content='evidence')])


def test_default_and_explicit_concurrency(monkeypatch):
    monkeypatch.delenv('MNEMOSYNE_JEV_WORKERS', raising=False)
    assert jev_evidence.worker_count() == 128
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '16')
    assert jev_evidence.worker_count() == 16


def test_parallelism_preserves_payloads_and_sorted_results(monkeypatch):
    rows = [dict(id=str(n), content=('user: '+str(n)+' evidence.\nassistant: noted.'))
            for n in range(12)]
    calls = []
    lock = threading.Lock()
    barrier = None
    class Client:
        def evaluate(self, state, questions):
            with lock:
                calls.append((state, questions))
            if barrier is not None:
                barrier.wait(timeout=5)
            value = (.8 if int(state.split()[1]) % 2 == 0 else .4) if state.startswith('user:') else .1
            return {key: dict(noul=value) for key in questions}
    monkeypatch.setattr(jev, 'client', lambda: Client())
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '1')
    serial = jev_evidence.rank('question', rows)
    expected_calls = list(calls)
    calls.clear()
    barrier = threading.Barrier(4)
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '4')
    parallel = jev_evidence.rank('question', rows)
    assert parallel == serial
    assert [row['id'] for row in parallel[0]] == ['0', '10', '2', '4', '6', '8']
    assert sorted(calls, key=lambda p: p[0]) == sorted(expected_calls, key=lambda p: p[0])
    assert len(calls) == 24  # Every user and assistant span was evaluated.


def test_empty_corpus_does_not_create_workers(monkeypatch):
    monkeypatch.setattr(jev_evidence, 'ThreadPoolExecutor', lambda **kw: pytest.fail('Empty corpus'))
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '64')
    assert jev_evidence.rank('question', []) == ([], 0)


def test_spans_of_one_long_memory_are_evaluated_concurrently(monkeypatch):
    barrier = threading.Barrier(4)
    class Client:
        def evaluate(self, state, questions):
            barrier.wait(timeout=5)
            return {key: dict(noul=.9) for key in questions}
    monkeypatch.setattr(jev, 'client', lambda: Client())
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '4')
    rows, scanned = jev_evidence.rank('q', [dict(id='long', content='user: first\nassistant: second\nuser: third\nassistant: fourth')])
    assert scanned == 1 and rows[0]['score'] == .9


def test_one_failed_decision_never_returns_partial_recall(monkeypatch):
    class Client:
        def evaluate(self, state, questions):
            if state == 'bad':
                raise jev.JevError('failed decision')
            return {key: dict(noul=.9) for key in questions}
    monkeypatch.setattr(jev, 'client', lambda: Client())
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '8')
    with pytest.raises(jev.JevError, match='failed decision'):
        jev_evidence.rank('q', [dict(id=str(n), content=c) for n,c in enumerate(['good', 'bad', 'good'])])
