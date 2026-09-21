"""Parallel duplicate checks must preserve the ordered greedy decisions."""
import random
import threading

import pytest

from mnemosyne.core import jev
from mnemosyne.core.jev_recall import _diverse_top


def serial_reference(rows, top_k, score):
    kept = []
    for row in rows:
        if len(kept) >= top_k:
            break
        if any(row['content'] == previous['content'] for previous in kept):
            continue
        comparable = [r['content'] for r in kept if len(jev._json(r['content'])) <= 2000]
        if comparable and len(jev._json(row['content'])) <= 2000:
            if max(score(row['content'], comparable)) >= .97:
                continue
        kept.append(row)
    return kept


def test_sixteen_distinct_hits_overlap_fifteen_comparisons(monkeypatch):
    barrier = threading.Barrier(15)
    calls = []
    lock = threading.Lock()

    def score(text, others):
        with lock:
            calls.append((text, tuple(others)))
        barrier.wait(timeout=5)
        return [.01] * len(others)

    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '128')
    monkeypatch.setattr(jev, 'duplicate_scores', score)
    rows = [dict(id=str(i), content=f'Unique fact {i}') for i in range(241)]
    assert _diverse_top(rows, 16) == rows[:16]
    assert len(calls) == 15
    assert sorted(calls) == sorted((rows[i]['content'], tuple(r['content'] for r in rows[:i]))
                                   for i in range(1, 16))


def test_rejected_intermediary_cannot_suppress_later_hit(monkeypatch):
    calls = []

    def score(text, others):
        calls.append((text, tuple(others)))
        return [float((text, old) in {('b', 'a'), ('c', 'b')}) for old in others]

    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '128')
    monkeypatch.setattr(jev, 'duplicate_scores', score)
    rows = [dict(id=text, content=text) for text in 'abcd']
    assert [r['id'] for r in _diverse_top(rows, 3)] == ['a', 'c', 'd']
    assert ('c', ('a', 'b')) in calls  # Optimistic comparison.
    assert ('c', ('a',)) in calls  # Corrected comparison, actually used.


@pytest.mark.parametrize('workers', [1, 4, 128])
def test_matches_serial_even_when_scores_depend_on_entire_question_group(monkeypatch, workers):
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', str(workers))
    rng = random.Random(101)
    for _ in range(30):
        texts = ['fact ' + str(rng.randrange(18)) for _ in range(30)]
        texts[7] = 'long ' * 600
        texts[12] = texts[7]
        rows = [dict(id=str(i), content=text) for i, text in enumerate(texts)]
        salt = rng.randrange(100)

        def score(text, others):
            # Deliberately context-sensitive and non-transitive. Filtering a
            # speculative answer would be wrong: changed groups must be redone.
            checksum = sum(map(ord, text + '|'.join(others))) + salt
            return [1. if (checksum + i) % 7 == 0 else .01 for i in range(len(others))]

        monkeypatch.setattr(jev, 'duplicate_scores', score)
        limit = rng.randrange(1, 20)
        assert _diverse_top(rows, limit) == serial_reference(rows, limit, score)


def test_failed_required_comparison_is_not_treated_as_distinct(monkeypatch):
    def fail(*args):
        raise jev.JevError('unavailable')
    monkeypatch.setattr(jev, 'duplicate_scores', fail)
    with pytest.raises(jev.JevError):
        _diverse_top([dict(content='a'), dict(content='b')], 2)


def test_irrelevant_speculation_error_does_not_change_serial_outcome(monkeypatch):
    def score(text, others):
        if text == 'c' and others == ['a', 'b']:
            raise jev.JevError('unused speculative group failed')
        return [float(text == 'b')] * len(others)
    monkeypatch.setattr(jev, 'duplicate_scores', score)
    rows = [dict(content=text) for text in 'abc']
    assert _diverse_top(rows, 3) == [rows[0], rows[2]]


def test_exact_duplicates_oversized_rows_and_zero_limit_need_no_api(monkeypatch):
    monkeypatch.setattr(jev, 'duplicate_scores', lambda *args: pytest.fail('Unexpected API call'))
    rows = [dict(content='x' * 2001), dict(content='x' * 2001), dict(content='short')]
    assert _diverse_top(rows, 0) == []
    assert _diverse_top(rows, 10) == [rows[0], rows[2]]
