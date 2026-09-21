"""Every span/criterion must survive byte-bounded request packing and caching."""
import json
import threading

import pytest

from mnemosyne.core import jev, jev_evidence


def client_for(monkeypatch, limit=24000, decision_cache_size=32768):
    sent = []
    lock = threading.Lock()
    def transport(payload, timeout):
        assert len(jev._json(payload)) <= limit
        assert payload['state'] == {}
        with lock:
            sent.append(payload)
        return dict(model='test-double', answers={key: dict(type='noul', noul=.9)
                    for key in payload['questions']}, usage={})
    client = jev.JevClient('test', transport=transport, decision_cache_size=decision_cache_size)
    client.request_bytes = limit
    monkeypatch.setattr(jev, 'client', lambda: client)
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', 'question')
    monkeypatch.setenv('MNEMOSYNE_JEV_WORKERS', '4')
    return client, sent


def test_ten_thousand_candidates_all_evaluated_with_fewer_http_requests(monkeypatch):
    client, sent = client_for(monkeypatch)
    rows = [dict(id=str(i), content=f'Memory {i}: project details.') for i in range(10000)]
    ranked, count = jev_evidence.rank('query', rows, ['Is this relevant?', 'Does it link the requested entity?'])
    assert count == len(ranked) == 10000
    assert len(sent) < 500
    questions = [question for body in sent for question in body['questions'].values()]
    assert len(questions) == 20000
    assert {q['instructions']['memory'] for q in questions} == {r['content'] for r in rows}
    before = client.snapshot()['requests']
    again, _ = jev_evidence.rank('query', rows, ['Is this relevant?', 'Does it link the requested entity?'])
    assert again == ranked and client.snapshot()['requests'] == before
    assert client.snapshot()['decision_cache_hits'] == 20000


def test_unicode_long_turns_and_multiple_criteria_have_complete_coverage(monkeypatch):
    _, sent = client_for(monkeypatch, limit=3000)
    rows = [dict(id='a', content='user: ' + 'caf\u00e9 ' * 1000 + '\nassistant: A separate reply.')]
    criteria = ['Does this give the answer?', 'Does this resolve the entity?', 'Does it state a date?']
    expected = [(span, criterion) for span in jev_evidence.evidence_spans(rows[0]['content']) for criterion in criteria]
    ranked, count = jev_evidence.rank('query', rows, criteria)
    actual = [(q['instructions']['memory'], q['instructions']['question'].removeprefix(jev.DATA_RULE))
              for body in sent for q in body['questions'].values()]
    assert sorted(actual) == sorted(expected)
    assert count == len(ranked) == 1


def test_mutation_and_different_packing_reuse_only_unchanged_decisions(monkeypatch):
    client, sent = client_for(monkeypatch)
    rows = [dict(id=str(i), content=f'Unique fact {i}') for i in range(100)]
    criteria = ['Does this match the requested entity?']
    jev_evidence.rank('query', rows, criteria)
    before = client.snapshot()['requests']
    sent.clear()
    rows.insert(0, dict(id='new', content='New fact'))
    rows[5] = dict(id=rows[5]['id'], content='Changed fact')
    jev_evidence.rank('query', rows, criteria)
    assert client.snapshot()['requests'] > before
    transmitted = [q['instructions']['memory'] for body in sent for q in body['questions'].values()]
    assert sorted(transmitted) == ['Changed fact', 'New fact']
    assert client.snapshot()['decision_cache_hits'] == 99
    # Criteria are part of the cache key, even when evidence is identical.
    sent.clear()
    jev_evidence.rank('query', rows, ['Is this an entirely different condition?'])
    assert sum(len(body['questions']) for body in sent) == len(rows)


def test_invalid_batch_never_populates_per_decision_cache(monkeypatch):
    client, _ = client_for(monkeypatch)
    client._transport = lambda *_: dict(model='test', answers={})
    with pytest.raises(jev.JevError):
        jev_evidence.rank('query', [dict(id='a', content='evidence')])
    assert len(client._decision_cache) == 0


def test_decision_cache_is_bounded_and_has_no_plaintext_keys(monkeypatch):
    client, _ = client_for(monkeypatch, decision_cache_size=2)
    jev_evidence.rank('query', [dict(id=str(i), content=f'Unique fact {i}') for i in range(6)])
    assert len(client._decision_cache) == 2
    assert all(isinstance(key, bytes) and len(key) == 32 for key in client._decision_cache)
    assert all(set(json.loads(value)) == {'type', 'noul'} for value in client._decision_cache.values())
