"""Stateful conversation contracts with scripted Jev; not model quality evidence."""
import json
import sqlite3

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core import jev
from tests.test_jev import beam, response  # noqa: F401


@pytest.fixture
def conversation(beam, monkeypatch):
    monkeypatch.setenv('MNEMOSYNE_JEV_RANKING', 'evidence')
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', 'question')
    beam.conn.execute("INSERT INTO working_memory(id,content,source,importance,session_id,scope,timestamp) "
                      "VALUES ('cedar','Project Cedar uses PostgreSQL.','fact',.8,'test','session',CURRENT_TIMESTAMP)")
    beam.conn.commit()
    payloads = []
    gate = {'score': .99, 'fail': False}
    def transport(payload, timeout):
        payloads.append(payload)
        if 'reuse' in payload['questions']:
            assert timeout <= .9
            if gate['fail']:
                raise jev.JevError('synthetic gate failure')
            result = response(payload['questions'], gate['score'])
            result['answers']['search_needed']['noul'] = gate.get('new', 1 - gate['score'])
            return result
        result = response(payload['questions'])
        for key, q in payload['questions'].items():
            instruction = q['instructions']
            text = instruction.get('memory', instruction.get('candidate', ''))
            result['answers'][key]['noul'] = (.99 if 'Cedar' in text else .01)
            if 'same substantive claims' in instruction['question']:
                result['answers'][key]['noul'] = .01
        return result
    client = jev.JevClient('test', transport=transport)
    monkeypatch.setattr(jev, 'client', lambda: client)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider._prefetch_mode = 'economy'
    return provider, client, payloads, gate


def search(provider, query='Which database does Project Cedar use?', session='one'):
    return provider.prefetch(query, session_id=session)


def test_followups_scan_fresh_storage_but_only_pay_for_gate_and_additions(conversation):
    provider, client, payloads, gate = conversation
    assert 'PostgreSQL' in search(provider)
    before = client.snapshot()
    assert 'PostgreSQL' in search(provider, 'Put that fact in a bullet point.')
    assert client.snapshot()['requests'] - before['requests'] == 1
    assert client.snapshot()['decision_cache_hits'] - before['decision_cache_hits'] == 1
    assert provider._last_prefetch['conversation']['action'] == 'reuse_question'
    # A separate writer adds relevant evidence and a record in another session.
    with sqlite3.connect(provider._beam.db_path) as writer:
        writer.executemany("INSERT INTO working_memory(id,content,source,importance,session_id,scope,timestamp) "
                           "VALUES (?,?, 'fact',.8,?,'session',CURRENT_TIMESTAMP)",
                           [('new', 'Project Cedar database backups run at midnight.', 'test'),
                            ('hidden', 'PRIVATE HIDDEN FACT', 'other')])
    before = client.snapshot()
    payloads.clear()
    assert 'midnight' in search(provider, 'Keep the same information, but make it shorter.')
    assert provider._beam._last_jev_recall['scanned'] == 2
    assert client.snapshot()['decision_cache_hits'] > before['decision_cache_hits']
    assert 'midnight' in json.dumps(payloads) and 'PRIVATE HIDDEN FACT' not in json.dumps(payloads)


@pytest.mark.parametrize('sql', [
    "DELETE FROM working_memory WHERE id='cedar'",
    "UPDATE working_memory SET content='REPLACED PRIVATE TEXT' WHERE id='cedar'",
    "UPDATE working_memory SET session_id='other' WHERE id='cedar'",
    "UPDATE working_memory SET valid_until='2000-01-01' WHERE id='cedar'",
    "UPDATE working_memory SET superseded_by='replacement' WHERE id='cedar'",
    "UPDATE working_memory SET importance=.1 WHERE id='cedar'",
])
def test_mutated_evidence_never_reaches_gate(conversation, sql):
    provider, client, payloads, gate = conversation
    search(provider)
    with sqlite3.connect(provider._beam.db_path) as writer:
        writer.execute(sql)
    payloads.clear()
    search(provider, 'Summarize that.')
    assert not any('reuse' in p['questions'] for p in payloads)
    assert provider._last_prefetch['conversation']['reason'] == 'evidence_changed'


def test_topic_change_and_uncertainty_use_new_question(conversation):
    provider, client, payloads, gate = conversation
    search(provider)
    gate['score'] = .79
    search(provider, 'What is the launch date for Project Birch?')
    trace = provider._last_prefetch['conversation']
    assert trace['action'] == 'full_search' and trace['reason'] == 'new_evidence_needed'
    assert 'launch date' in payloads[-1]['questions']['0']['instructions']['question']


def test_gate_failure_falls_back_then_recovers(conversation):
    provider, client, payloads, gate = conversation
    search(provider)
    gate['fail'] = True
    assert 'PostgreSQL' in search(provider, 'Explain that information.')
    assert provider._last_prefetch['conversation']['reason'] == 'gate_failed'
    gate['fail'] = False
    assert 'PostgreSQL' in search(provider, 'Now shorten that.')
    assert provider._last_prefetch['conversation']['action'] == 'reuse_question'


@pytest.mark.parametrize('change', ['session', 'ttl', 'turns', 'reset', 'size'])
def test_refresh_boundaries_do_not_call_gate(conversation, change):
    provider, client, payloads, gate = conversation
    search(provider)
    cache = provider._conversation_prefetch
    session = 'one'
    query = 'Summarize that.'
    if change == 'session': session = 'two'
    if change == 'ttl': cache.anchor.created -= 301
    if change == 'turns': cache.anchor.reuses = 6
    if change == 'reset':
        provider._session_id = 'test'
        provider.on_session_switch('one', reset=True)
    if change == 'size': query = 'x' * 9000
    payloads.clear()
    search(provider, query, session)
    assert not any('reuse' in p['questions'] for p in payloads)
    assert provider._last_prefetch['conversation']['action'] == 'full_search'


def test_strict_default_and_explicit_recall_are_not_gated(conversation):
    provider, client, payloads, gate = conversation
    assert MnemosyneMemoryProvider()._prefetch_mode == 'strict'
    search(provider)
    payloads.clear()
    result = provider._handle_recall({'query': 'What is the Cedar database?', 'limit': 5})
    assert 'PostgreSQL' in result
    assert not any('reuse' in p['questions'] for p in payloads)
    assert provider._conversation_prefetch.anchor is None
    provider._prefetch_mode = 'strict'
    search(provider, 'A different query')
    assert provider._last_prefetch['conversation']['reason'] == 'strict_mode'


def test_configuration_and_private_diagnostics(conversation):
    provider, client, payloads, gate = conversation
    provider._apply_provider_config({'prefetch_mode': 'economy'})
    search(provider)
    search(provider, 'Put that in a bullet point.')
    diagnostic = json.dumps(provider._last_prefetch)
    assert 'PostgreSQL' not in diagnostic and 'bullet point' not in diagnostic
    assert provider._last_prefetch['conversation']['gate_usage']['requests'] == 1
    with pytest.raises(ValueError, match='prefetch_mode'):
        provider._apply_provider_config({'prefetch_mode': 'broken'})


def test_new_evidence_guard_overrides_positive_continuation(conversation):
    provider, client, payloads, gate = conversation
    search(provider)
    gate.update(score=.99, new=.70)
    search(provider, 'And when was it upgraded?')
    assert provider._last_prefetch['conversation']['action'] == 'full_search'


def test_late_gate_response_preserves_time_for_full_search(conversation):
    import time
    provider, client, payloads, gate = conversation
    search(provider)
    original = client._transport
    def slow(payload, timeout):
        if 'reuse' in payload['questions']:
            time.sleep(.95)
        return original(payload, timeout)
    client._transport = slow
    assert 'PostgreSQL' in search(provider, 'Explain that information.')
    trace = provider._last_prefetch['conversation']
    assert trace['reason'] == 'gate_failed'
    assert trace['gate_error_type'] == 'JevDeadlineExceeded'
    assert provider._last_prefetch['status'] == 'completed'


def test_cross_session_setting_matches_recall_and_revocation(conversation, monkeypatch):
    provider, client, payloads, gate = conversation
    provider._beam.conn.execute("UPDATE working_memory SET session_id='another-session'")
    provider._beam.conn.commit()
    from mnemosyne.core import beam as beam_module
    monkeypatch.setattr(beam_module, '_cross_session_enabled', lambda: True)
    assert 'PostgreSQL' in search(provider)
    assert 'PostgreSQL' in search(provider, 'Shorten that fact.')
    assert provider._last_prefetch['conversation']['action'] == 'reuse_question'
    monkeypatch.setattr(beam_module, '_cross_session_enabled', lambda: False)
    payloads.clear()
    assert 'PostgreSQL' not in search(provider, 'Write the fact again.')
    assert provider._last_prefetch['conversation']['reason'] == 'scope_changed'
    assert not any('reuse' in p['questions'] for p in payloads)
    assert 'PostgreSQL' not in json.dumps(payloads)
