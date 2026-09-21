"""Offline contract/integration tests. Scripted answers are NOT quality evidence."""
import json
from urllib.error import HTTPError

import pytest

from mnemosyne.core import jev


def response(questions, probability=.9):
    answers = {}
    for key, q in questions.items():
        if q["type"] == "choice":
            selected = next(iter(q["criteria"]))
            answers[key] = dict(type="choice", choice=selected, confidence=1.,
                                probabilities={x: float(x == selected) for x in q["criteria"]})
        else:
            answers[key] = dict(type="noul", noul=probability)
    return dict(model="jev-1.13.0", answers=answers, usage=dict(input_tokens=10, output_tokens=2))


def test_http_contract_cache_and_usage():
    seen = []
    def transport(payload, timeout):
        seen.append(payload)
        assert 0 < timeout <= 30
        return response(payload["questions"])
    client = jev.JevClient("test-key", transport=transport)
    question = {"a": jev.noul("Is the assertion relevant?")}
    assert client.evaluate("evidence", question)["a"]["noul"] == .9
    assert client.evaluate("evidence", question)["a"]["noul"] == .9
    assert len(seen) == 1 and seen[0]["state"] == "evidence"
    assert seen[0]["model"] == "typesafe/jev-1.13"
    assert client.snapshot()["input_tokens"] == 10
    assert client.snapshot()["cache_hits"] == 1


@pytest.mark.parametrize("answer", [None, {}, {"a": {"type": "choice", "choice": "yes"}},
    {"a": {"type": "noul", "noul": float("nan")}}, {"a": {"type": "noul", "noul": True}},
    {"a": {"type": "noul", "noul": 1.1}}, {"extra": {"type": "noul", "noul": .9}}])
def test_invalid_response_never_becomes_irrelevant(answer):
    client = jev.JevClient("test", transport=lambda *_: dict(model="jev", answers=answer))
    with pytest.raises(jev.JevError):
        client.evaluate("data", {"a": jev.noul("yes?")})
    assert client.snapshot()["failures"] == 1


@pytest.mark.parametrize("url", ["http://api.typesafe.ai/v1", "https://user:pass@example.com",
                                "https://example.com?token=x", "file:///tmp/test"])
def test_no_credentials_to_unsafe_endpoint(url):
    with pytest.raises(ValueError):
        jev.JevClient("secret", base_url=url)


def test_missing_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MNEMOSYNE_JEV_PROVIDER", "openrouter")
    with pytest.raises(jev.JevError, match="OPENROUTER_API_KEY"):
        jev.client()


def test_retry_after_and_permanent_error(monkeypatch):
    sleeps, attempts = [], []
    monkeypatch.setattr(jev.time, "sleep", sleeps.append)
    def transient(payload, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise HTTPError("https://api.typesafe.ai", 429, "limited", {"Retry-After": "3"}, None)
        return response(payload["questions"])
    client = jev.JevClient("test", transport=transient)
    client.evaluate("data", {"a": jev.noul("yes?")})
    assert sleeps == [3] and len(attempts) == 2
    def unauthorized(*_):
        raise HTTPError("https://api.typesafe.ai", 401, "secret server body", {}, None)
    client = jev.JevClient("test", transport=unauthorized)
    with pytest.raises(jev.JevError, match="401") as error:
        client.evaluate("data", {"a": jev.noul("yes?")})
    assert "secret" not in str(error.value)
    assert client.snapshot()["requests"] == 1


def test_batching_long_unicode_and_complete_coverage(monkeypatch):
    seen = []
    def transport(payload, timeout):
        seen.append(payload)
        assert len(jev._json(payload)) <= 48000
        return response(payload["questions"])
    client = jev.JevClient("test", transport=transport)
    monkeypatch.setattr(jev, "client", lambda: client)
    texts = ["字" * 19000] + [f"document {i} " * 100 for i in range(80)]
    scores = jev.relevance("query", texts)
    assert scores == [.9] * 81 and len(seen) > 1
    sent = [question['instructions']['candidate'] for _, question in sorted(
        ((int(key), q) for body in seen for key, q in body['questions'].items()))]
    assert "".join(sent) == "".join(texts)


def test_oversized_shared_state_and_deadline_fail_before_transport():
    client = jev.JevClient("test", transport=lambda *_: pytest.fail("network called"))
    with pytest.raises(jev.JevError, match="budget"):
        client.evaluate("x" * 30000, {"a": jev.noul("yes?")})
    with pytest.raises(jev.JevError, match="deadline"):
        client.evaluate("state", {"a": jev.noul("yes?")}, deadline=0)


@pytest.fixture
def beam(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DECISION_BACKEND", "baseline")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
    monkeypatch.setenv("MNEMOSYNE_JEV_RANKING", "independent")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    from mnemosyne.core.beam import BeamMemory
    obj = BeamMemory(session_id="test", db_path=tmp_path / "test.db")
    yield obj
    obj.conn.close()


def install_fake(monkeypatch, rule=lambda candidate: .95):
    sent = []
    def transport(payload, timeout):
        sent.append(payload)
        data = response(payload["questions"])
        for key, question in payload["questions"].items():
            if question["type"] == "noul":
                instruction = question["instructions"]
                if isinstance(instruction, str):
                    data["answers"][key]["noul"] = rule(payload["state"])
                    continue
                score = rule(instruction.get("candidate", instruction.get("memory", "")))
                if "same substantive claims" in instruction["question"]:
                    score = .01
                data["answers"][key]["noul"] = score
        return data
    client = jev.JevClient("test", transport=transport)
    monkeypatch.setattr(jev, "client", lambda: client)
    monkeypatch.setenv("MNEMOSYNE_DECISION_BACKEND", "jev")
    return sent, client


@pytest.mark.parametrize('batch_mode', ['single', 'question'])
def test_full_corpus_no_shortlist_and_scope_before_disclosure(beam, monkeypatch, batch_mode):
    monkeypatch.setenv('MNEMOSYNE_JEV_RANKING', 'evidence')
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', batch_mode)
    ids = [beam.remember(f"ordinary memo {i}", memory_type="fact") for i in range(260)]
    target = beam.remember("A semantically useful needle with no literal query token.", memory_type="fact")
    hidden = beam.remember("hidden needle", memory_type="fact")
    expired = beam.remember("expired needle", valid_until="2000-01-01", memory_type="fact")
    beam.conn.execute("UPDATE working_memory SET session_id='other' WHERE id=?", (hidden,))
    beam.conn.commit()
    sent, _ = install_fake(monkeypatch, lambda text: .95 if "needle" in text else .01)
    result = beam.recall("What editor should I use?", top_k=1, explain=True)
    assert [r["id"] for r in result["results"]] == [target]
    assert result["explain"]["scanned"] == 261
    assert result["explain"]["candidate_mode"] == "full_corpus"
    encoded = json.dumps(sent)
    assert "hidden needle" not in encoded and "expired needle" not in encoded
    assert beam.conn.execute("SELECT recall_count FROM working_memory WHERE id=?", (target,)).fetchone()[0] == 1
    for memory_id in [ids[0], hidden, expired]:
        assert beam.conn.execute("SELECT recall_count FROM working_memory WHERE id=?", (memory_id,)).fetchone()[0] == 0


def test_recall_failure_no_partial_reinforcement(beam, monkeypatch):
    memory_id = beam.remember("evidence", memory_type="fact")
    _, client = install_fake(monkeypatch)
    client._transport = lambda *_: {}
    with pytest.raises(jev.JevError):
        beam.recall("query")
    assert beam.conn.execute("SELECT recall_count FROM working_memory WHERE id=?", (memory_id,)).fetchone()[0] == 0
    assert beam._last_jev_recall['status'] == 'failed'
    assert beam._last_jev_recall['stages']['ranking']['status'] == 'failed'
    assert beam._last_jev_recall['stages']['ranking']['usage']['failures'] == 1


def test_recall_separates_ranking_and_duplicate_costs(beam, monkeypatch):
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', 'single')
    for text in ['Cedar uses PostgreSQL.', 'Cedar deploys to Paris.', 'Cedar serves reports.']:
        beam.remember(text, memory_type='fact')
    _, client = install_fake(monkeypatch)
    monkeypatch.setenv('MNEMOSYNE_JEV_RANKING', 'evidence')
    result = beam.recall('What do we know about Cedar?', top_k=3, explain=True)
    trace = result['explain']
    assert trace['status'] == 'completed' and trace['scanned'] == 3
    assert list(trace['stages']) == ['ranking', 'weighting', 'deduplication', 'finalization']
    assert trace['stages']['ranking']['usage']['requests'] == 3
    assert trace['stages']['deduplication']['usage']['requests'] == 2
    assert trace['usage']['requests'] == client.snapshot()['requests'] == 5
    assert trace['elapsed_seconds'] >= sum(stage['seconds'] for stage in trace['stages'].values())


def test_filter_no_result_zero_limit_and_superseded(beam, monkeypatch):
    old = beam.remember("old value", source="user", memory_type="fact")
    new = beam.remember("new value", source="user", memory_type="fact")
    beam.invalidate(old, replacement_id=new)
    sent, _ = install_fake(monkeypatch)
    assert beam.recall("q", top_k=0) == [] and sent == []
    assert beam.recall("q", source="different") == [] and sent == []
    assert [x["id"] for x in beam.recall("q", source="user")] == [new]
    assert "old value" not in json.dumps(sent)


def test_removed_embedding_module_cannot_be_imported(monkeypatch):
    import importlib.util
    monkeypatch.setenv("MNEMOSYNE_DECISION_BACKEND", "baseline")
    assert jev.enabled()
    assert importlib.util.find_spec("mnemosyne.core.embeddings") is None


def test_extractive_ingestion_and_direct_batch_gate(beam, monkeypatch):
    from mnemosyne.core.extraction import extract_facts
    from mnemosyne.core.filters import classify_memory_write
    install_fake(monkeypatch, lambda text: .01 if "Thanks" in text else .99)
    monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "strict")
    assert classify_memory_write("Thanks for the update").action == "reject"
    assert extract_facts("I prefer Vim.\nThanks for the update.") == ["I prefer Vim."]
    assert beam.remember("Thanks for the update") is None
    ids = beam.remember_batch([dict(content="Thanks for the update"), dict(content="I prefer Vim.")])
    assert len(ids) == 1
    assert beam.conn.execute("SELECT content FROM working_memory WHERE id=?", (ids[0],)).fetchone()[0] == "I prefer Vim."


@pytest.mark.parametrize("changed,lost,expected", [
    (.99, .02, True),
    (.8, .01, False),  # Uncertain change must not invalidate a stored fact.
    (.99, .6, False),  # A clear change cannot discard an unrelated fact.
    (.02, .01, False),  # Preservation alone does not make a duplicate a change.
])
def test_conflict_uses_typed_probability_and_preserves_new_text(
        monkeypatch, changed, lost, expected):
    from mnemosyne.core.llm_conflict_detector import validate_conflict_pair
    def transport(payload, timeout):
        data = response(payload["questions"])
        data["answers"]["changed"]["noul"] = changed
        data["answers"]["lost"]["noul"] = lost
        return data
    client = jev.JevClient("test", transport=transport)
    monkeypatch.setattr(jev, "client", lambda: client)
    monkeypatch.setenv("MNEMOSYNE_DECISION_BACKEND", "jev")
    actual = validate_conflict_pair("I use Vim", "I now use Emacs", "test")
    assert actual[0] is expected
    assert actual[1] == pytest.approx(min(changed, 1-lost))
    assert actual[2] == "I now use Emacs"


def test_conflict_candidates_do_not_require_embeddings(beam, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DECISION_BACKEND", "jev")
    rows = [dict(id="a", content="old", timestamp="2026-01-01"),
            dict(id="b", content="new", timestamp="2026-02-01"),
            dict(id="c", content="stale", timestamp="2025-01-01", superseded_by="b")]
    assert beam._detect_conflicts(rows) == [("a", "b")]


def test_fact_recall_filters_parent_before_api(beam, monkeypatch):
    from mnemosyne.core.beam import _store_facts_in_table
    good = beam.remember("user likes Vim", memory_type="fact")
    bad = beam.remember("private original", memory_type="fact")
    _store_facts_in_table(beam, good, "", "user", ["safe fact"])
    _store_facts_in_table(beam, bad, "", "user", ["private fact"])
    beam.conn.execute("UPDATE working_memory SET session_id='foreign' WHERE id=?", (bad,))
    beam.conn.commit()
    sent, _ = install_fake(monkeypatch)
    assert any("safe fact" in r["content"] for r in beam.fact_recall("q"))
    assert "private fact" not in json.dumps(sent)


def test_config_namespace_and_new_entry_point(monkeypatch, tmp_path):
    import perfectrecall
    monkeypatch.setenv("PERFECTRECALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PERFECTRECALL_DECISION_BACKEND", "baseline")
    for name in ("DATA_DIR", "DECISION_BACKEND", "PERSONA_FILE", "WRITE_CLASSIFIER"):
        # delenv alone does not track keys that were initially absent.
        monkeypatch.setenv("MNEMOSYNE_" + name, "")
        monkeypatch.delenv("MNEMOSYNE_" + name, raising=False)
    perfectrecall.configure()
    import os
    assert os.environ["MNEMOSYNE_DATA_DIR"] == str(tmp_path)
    assert jev.enabled()
    assert perfectrecall.PerfectRecall.__name__ == "Mnemosyne"


def test_provider_contract_and_key_isolation(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-test')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'ts-test')
    monkeypatch.setenv('MNEMOSYNE_JEV_PROVIDER', 'openrouter')
    client = jev.client()
    assert client.url == 'https://openrouter.ai/api/alpha/decisions'
    assert client.model == 'typesafe/jev-1.13' and client.api_key == 'or-test'
    monkeypatch.setenv('MNEMOSYNE_JEV_PROVIDER', 'typesafe')
    client = jev.client()
    assert client.url == 'https://api.typesafe.ai/v1/systemone'
    assert client.model == 'jev-1.13.0' and client.api_key == 'ts-test'
    monkeypatch.delenv('TYPESAFE_API_KEY')
    with pytest.raises(jev.JevError, match='TYPESAFE_API_KEY'):
        jev.client()


def test_actual_cost_and_invalid_cost():
    def priced(payload, timeout):
        data = response(payload['questions'])
        data['usage']['cost'] = .001
        return data
    client = jev.JevClient('test', transport=priced)
    client.evaluate('data', {'x': jev.noul('question')})
    assert client.snapshot()['cost_usd'] == .001
    assert client.snapshot()['priced_responses'] == 1
    data = response({'x': jev.noul('question')})
    data['usage']['cost'] = -1
    with pytest.raises(jev.JevError, match='cost'):
        jev.JevClient._validate(data, {'x': jev.noul('question')})


def test_tournament_covers_corpus_and_abstains(monkeypatch):
    from mnemosyne.core.jev_tournament import rank
    seen = []
    def transport(payload, timeout):
        data = response(payload['questions'])
        for key, question in payload['questions'].items():
            options = payload['state']['memories']
            seen.extend(options.values())
            selected = next((k for k, v in options.items() if 'needle' in v), 'none')
            data['answers'][key] = dict(type='choice', choice=selected, confidence=1.,
                probabilities={k: float(k == selected) for k in question['criteria']})
        return data
    client = jev.JevClient('test', transport=transport)
    monkeypatch.setattr(jev, 'client', lambda: client)
    rows = [dict(id=str(i), content=f'irrelevant {i}') for i in range(9)]
    rows.append(dict(id='target', content='needle'))
    assert [r['id'] for r in rank('q', rows, 3)] == ['target']
    assert set(r['content'] for r in rows).issubset(seen)
    assert rank('q', rows[:-1], 3) == []


def test_shmr_filters_foreign_facts_before_decisions(beam, monkeypatch):
    from mnemosyne.core.beam import _store_facts_in_table
    from mnemosyne.core.jev_shmr import echo_candidates
    good = beam.remember('safe evidence', memory_type='fact')
    bad = beam.remember('private evidence', memory_type='fact')
    _store_facts_in_table(beam, good, '', 'user', ['safe fact'])
    _store_facts_in_table(beam, bad, '', 'user', ['private fact'])
    beam.conn.execute("UPDATE working_memory SET session_id='foreign' WHERE id=?", (bad,))
    beam.conn.commit()
    rows = echo_candidates(beam, 100)
    assert any('safe fact' in r['object'] for r in rows)
    assert not any('private' in r['object'] for r in rows)


def test_jev_write_replaces_builtin_noise_heuristic(monkeypatch):
    from mnemosyne.core.filters import classify_memory_write
    install_fake(monkeypatch, lambda _: .99)
    assert classify_memory_write('heartbeat monitoring is required for production').action == 'allow'
    assert classify_memory_write('heartbeat monitoring is required', ignore_patterns=['heartbeat']).action == 'reject'


def test_compression_preserves_indivisible_evidence(monkeypatch):
    install_fake(monkeypatch)
    text = 'word ' * 100
    assert jev.compress_extractively(text, 20) == text


@pytest.mark.parametrize('batch_mode', ['single', 'question'])
def test_caller_criteria_are_applied_to_every_memory(monkeypatch, batch_mode):
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', batch_mode)
    from mnemosyne.core import jev_evidence
    seen = []
    def transport(payload, timeout):
        data = response(payload['questions'])
        for key, question in payload['questions'].items():
            memory = payload['state'] if batch_mode == 'single' else question['instructions']['memory']
            seen.append(memory)
            data['answers'][key]['noul'] = .95 if 'answer' in memory else .05
        return data
    monkeypatch.setattr(jev, 'client', lambda: jev.JevClient('test', transport=transport))
    rows = [dict(id=str(i), content=f'noise {i}') for i in range(10)] + [dict(id='good', content='answer')]
    ranked, count = jev_evidence.rank('question', iter(rows), ['Does it mention the exact answer?'])
    assert count == 11 and len(seen) == 11
    assert [r['id'] for r in ranked] == ['good']


@pytest.mark.parametrize('questions', [[], [''], [' '], ['x'] * 4, 'question', [None], ['x' * 601]])
def test_evidence_questions_reject_invalid_input_before_network(questions):
    from mnemosyne.core.jev_evidence import rank
    with pytest.raises(ValueError, match='evidence_questions'):
        rank('query', [], questions)


@pytest.mark.parametrize('batch_mode', ['single', 'question'])
def test_evidence_pipeline_scans_visible_spans_and_preserves_rows(beam, monkeypatch, batch_mode):
    monkeypatch.setenv('MNEMOSYNE_JEV_BATCH_MODE', batch_mode)
    wanted = beam.remember('prefix ' * 600 + 'needle evidence', memory_type='fact')
    hidden = beam.remember('secret foreign evidence', memory_type='fact')
    beam.conn.execute("UPDATE working_memory SET session_id='other' WHERE id=?", (hidden,))
    beam.conn.commit()
    seen = []
    def transport(payload, timeout):
        data = response(payload['questions'])
        for key, question in payload['questions'].items():
            memory = payload['state'] if batch_mode == 'single' else question['instructions']['memory']
            criterion = question['instructions'] if batch_mode == 'single' else question['instructions']['question']
            seen.append(memory)
            assert 'secret foreign' not in memory
            assert 'Does it contain a needle?' in criterion
            data['answers'][key]['noul'] = .95 if 'needle' in memory else .01
        return data
    client = jev.JevClient('test', transport=transport)
    monkeypatch.setattr(jev, 'client', lambda: client)
    monkeypatch.setenv('MNEMOSYNE_DECISION_BACKEND', 'jev')
    result = beam.recall_enhanced('query', evidence_questions=['Does it contain a needle?'], explain=True)
    assert [r['id'] for r in result['results']] == [wanted]
    assert result['results'][0]['content'].endswith('needle evidence')
    assert result['explain']['ranking'] == 'evidence'
    assert result['explain']['scanned'] == 1
    # Request cache can reuse repeated spans, but no source text is truncated.
    assert any('needle evidence' in text for text in seen)


def test_tool_contract_exposes_caller_questions_in_both_schemas():
    from mnemosyne.tool_schemas import RECALL_SCHEMA, SHARED_RECALL_SCHEMA
    from hermes_memory_provider import RECALL_SCHEMA as bundled
    for schema in (RECALL_SCHEMA, SHARED_RECALL_SCHEMA, bundled):
        assert 'evidence_questions' in schema['parameters']['properties']
        assert 'do not invent facts' in schema['description']
        assert schema['parameters']['properties']['evidence_questions']['maxItems'] == 3


def test_evidence_spans_preserve_turns_and_speakers():
    from mnemosyne.core.jev_evidence import evidence_spans
    content = 'Session date: 2024-01-01\nuser: I sold 40 dozen eggs.\nassistant: You could try other crops.\nuser: ' + 'extended statement ' * 250
    spans = list(evidence_spans(content))
    assert all(s.startswith('Session date: 2024-01-01\n') for s in spans)
    assert any('user: I sold 40 dozen eggs.' in s for s in spans)
    assert not any('I sold 40' in s and 'You could try' in s for s in spans)
    assert all('user:' in s for s in spans if 'extended' in s)
    assert ''.join(s.split('\n', 1)[1].removeprefix('user: ') for s in spans[2:]) == 'extended statement ' * 250
    plain = 'ordinary text ' * 1000
    assert ''.join(evidence_spans(plain)) == plain
