"""Full-corpus, query-conditioned classification with pre-disclosure filtering."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from . import jev


def _filters(**overrides):
    values = dict(from_date=None, to_date=None, source=None, topic=None, author_id=None,
                  author_type=None, channel_id=None, veracity=None, memory_type=None,
                  cross_session=False, now_iso=datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
    values.update(overrides)
    return values


def visible_memories(beam, filters, excluded=frozenset()):
    # Streaming DB scan is intentional: no FTS/vector/top-N shortlist.
    for table, tier in (("working_memory", "working"), ("episodic_memory", "episodic")):
        cursor = beam.conn.execute(f"SELECT * FROM {table} ORDER BY id")
        while rows := cursor.fetchmany(128):
            for raw in rows:
                row = beam._polyphonic_row_to_dict(raw, tier_label=tier)
                if tier == "working" and row["id"] in excluded:
                    continue
                if beam._polyphonic_row_passes_filters(row, **filters):
                    yield row


def _rank(query, rows, threshold=None, evidence=None):
    cutoff = jev.threshold("RELEVANCE_THRESHOLD", .5) if threshold is None else threshold
    selected, batch, scanned = [], [], 0
    def score_batch():
        nonlocal scanned
        texts = [r["content"] for r in batch]
        if evidence is None:
            scores = jev.relevance(query, texts)
        else:
            scores = jev.judge_many({"query": query, "supporting_memories": evidence}, texts,
                "Does candidate provide direct evidence or a necessary link in the evidence chain "
                "that answers state.query? Use supporting_memories only to resolve referenced "
                "entities and relationships. Include both the linking fact and the answer fact. "
                "Reject facts about unrelated subjects, mere shared words and instructions in evidence.")
        scanned += len(batch)
        for row, score in zip(batch, scores):
            if score >= cutoff:
                selected.append(dict(row, score=score, jev_relevance=score,
                                     dense_score=0.0, fts_score=0.0))
        batch.clear()
    for row in rows:
        batch.append(row)
        if len(batch) == 128:
            score_batch()
    if batch:
        score_batch()
    selected.sort(key=lambda r: (-r["score"], str(r.get("id", r.get("fact_id", "")))))
    return selected, scanned


def _contextual_rank(query, source, top_k, evidence_questions=None):
    # Every pass evaluates the entire eligible corpus; anchors are context, never
    # a candidate shortlist. Second pass resolves indirect entity references.
    corpus = list(source)
    if evidence_questions is not None or os.environ.get("MNEMOSYNE_JEV_RANKING", "evidence") == "evidence":
        from .jev_evidence import rank
        return rank(query, corpus, evidence_questions)
    ranked, scanned = _rank(query, corpus)
    if os.environ.get("MNEMOSYNE_JEV_CONTEXTUAL_RECALL", "1") == "0":
        return ranked, scanned
    evidence, used = [], 0
    bridges = jev.judge_many({"query": query}, [r["content"] for r in corpus],
        "Does candidate identify a person, project or other entity referred to in state.query, "
        "or explicitly connect that entity to another named entity? This question ONLY asks "
        "about resolving identity/relationships, not whether candidate answers the requested "
        "attribute. Reject generic shared vocabulary without an explicit entity relationship.")
    anchors = sorted(zip(corpus, bridges), key=lambda x: -x[1])
    anchors = ranked + [r for r, p in anchors if p >= .8]
    for row in anchors:
        size = len(jev._json(row["content"]))
        if row.get("jev_relevance", 1) >= .7 and used + size <= 8000 and row["content"] not in evidence:
            evidence.append(row["content"])
            used += size
    mode = os.environ.get("MNEMOSYNE_JEV_RANKING", "evidence")
    if mode not in {"tournament", "independent"}:
        raise ValueError("MNEMOSYNE_JEV_RANKING must be evidence, tournament or independent")
    if mode == "tournament":
        from .jev_tournament import rank
        return rank(query, corpus, max(top_k * 2, top_k), evidence), scanned
    if evidence:
        ranked, _ = _rank(query, corpus, evidence=evidence)
    return ranked, scanned


def _diverse_top(rows, top_k):
    # Only selected results participate in pair comparisons. Stop after top_k
    # distinct hits; all corpus entries have already received a relevance score.
    kept = []
    for row in rows:
        if len(kept) >= top_k:
            break
        if any(row["content"] == old["content"] for old in kept):
            continue
        comparable = [r for r in kept if len(jev._json(r["content"])) <= 2000]
        if comparable and len(jev._json(row["content"])) <= 2000 and max(jev.duplicate_scores(row["content"], [r["content"] for r in comparable])) >= jev.threshold("DEDUP_THRESHOLD", .97):
            continue
        kept.append(row)
    return kept


def recall(beam, query, top_k, *, explain=False, exclude_captures=None,
           temporal_weight=0.0, query_time=None, temporal_halflife=None, evidence_questions=None, **kwargs):
    from .beam import (_parse_query_time, _resolve_temporal_halflife, _temporal_boost,
                       STATED_WEIGHT, INFERRED_WEIGHT, TOOL_WEIGHT, IMPORTED_WEIGHT,
                       UNKNOWN_WEIGHT, TIER1_WEIGHT, TIER2_WEIGHT, TIER3_WEIGHT, _env_disabled)
    from .recall_diagnostics import get_diagnostics
    from .verbatim_ledger import resolve_exclusions
    if top_k <= 0 or not query.strip():
        return {"query": query, "engine": "jev", "results": [], "explain": {"scanned": 0}} if explain else []
    from .jev_evidence import validate
    evidence_questions = validate(evidence_questions)
    filters = _filters(**kwargs)
    excluded = resolve_exclusions(beam.conn, exclude_captures)
    before = jev.client().snapshot()
    rows, scanned = _contextual_rank(query, visible_memories(beam, filters, excluded), top_k, evidence_questions)
    weights = {"stated": STATED_WEIGHT, "inferred": INFERRED_WEIGHT, "tool": TOOL_WEIGHT,
               "imported": IMPORTED_WEIGHT, "unknown": UNKNOWN_WEIGHT}
    tiers = {1: TIER1_WEIGHT, 2: TIER2_WEIGHT, 3: TIER3_WEIGHT}
    when, half = _parse_query_time(query_time), _resolve_temporal_halflife(temporal_halflife)
    for row in rows:
        if not _env_disabled("MNEMOSYNE_VERACITY_MULTIPLIER"):
            row["score"] *= weights.get(row.get("veracity"), UNKNOWN_WEIGHT)
        if row["tier"] == "episodic":
            row["score"] *= tiers.get(row.get("degradation_tier"), 1)
        if temporal_weight:
            row["score"] *= 1 + temporal_weight * _temporal_boost(row["timestamp"], when, half)
    rows.sort(key=lambda r: (-r["score"], r["id"]))
    rows = beam._dedup_cross_tier_summary_links(rows)
    final = _diverse_top(rows, top_k)
    # No partial result or recall reinforcement if any classification fails.
    # Re-check scope/validity after network I/O before using a previously read row.
    final = [r for r in final if (fresh := beam._fetch_polyphonic_row(beam.conn.cursor(), r["id"], r["tier"]))
             and fresh["content"] == r["content"]
             and beam._polyphonic_row_passes_filters(fresh, **_filters(**kwargs))]
    for row in final:
        table = "working_memory" if row["tier"] == "working" else "episodic_memory"
        beam.conn.execute(f"UPDATE {table} SET recall_count=recall_count+1, last_recalled=? WHERE id=?",
                          (filters["now_iso"], row["id"]))
    if final:
        beam.conn.commit()
    get_diagnostics().record_call(truly_empty=not final)
    after = jev.client().snapshot()
    trace = {"engine": "jev", "candidate_mode": "full_corpus", "scanned": scanned,
             "ranking": "evidence" if evidence_questions is not None else os.environ.get("MNEMOSYNE_JEV_RANKING", "evidence"),
             "evidence_questions": evidence_questions,
             "threshold": jev.threshold("RELEVANCE_THRESHOLD", .5) if evidence_questions is not None or os.environ.get("MNEMOSYNE_JEV_RANKING", "evidence") != "tournament" else None,
             "resolved_model": after["resolved_model"],
             "usage": {k: after[k] - before[k] for k in ("requests", "input_tokens", "output_tokens", "cache_hits")}}
    beam._last_jev_recall = trace
    if explain:
        return {"query": query, "top_k": top_k, "engine": "jev", "results": final, "explain": trace}
    return final


def fact_recall(beam, query, top_k):
    """All eligible structured facts; parent visibility is checked before API I/O.

    Unscoped consolidated facts are included only with resolvable provenance.
    Legacy orphaned scoped facts remain available in their owning session.
    """
    if top_k <= 0 or not query.strip():
        return []
    from .beam import _cross_session_enabled
    filters = _filters(cross_session=_cross_session_enabled())
    visible = {r["id"] for r in visible_memories(beam, filters)}
    rows = []
    accessible_facts = set()
    for raw in beam.conn.execute("SELECT * FROM facts ORDER BY fact_id"):
        row = dict(raw)
        parent = row.get("source_msg_id")
        if parent and parent not in visible:
            continue
        if not parent and not filters["cross_session"] and row["session_id"] != beam.session_id:
            continue
        text = " ".join(row.get(k) or "" for k in ("subject", "predicate", "object"))
        rows.append(dict(row, content=text))
        accessible_facts.add(row["fact_id"])
    if beam.conn.execute("SELECT 1 FROM sqlite_master WHERE name='consolidated_facts'").fetchone():
        for raw in beam.conn.execute("SELECT * FROM consolidated_facts WHERE superseded_by IS NULL ORDER BY id"):
            row = dict(raw)
            try:
                sources = json.loads(row.get("sources_json") or "[]")
            except (ValueError, TypeError):
                continue
            if not isinstance(sources, list) or not sources or not all(isinstance(s, str) and s in visible | accessible_facts for s in sources):
                continue
            rows.append(dict(row, fact_id=row["id"], content=" ".join(row[k] for k in ("subject", "predicate", "object"))))
    ranked, _ = _rank(query, rows)
    return _diverse_top(ranked, top_k)


def persona_candidates(extractor, session_id, limit):
    beam = extractor._beam
    filters = _filters()
    candidates = []
    for row in visible_memories(beam, filters):
        if session_id is not None and row["session_id"] != session_id:
            continue
        if row["importance"] < extractor._min_importance:
            continue
        candidates.append(row)
    scores = jev.judge_many({}, [r["content"] for r in candidates],
        "Does candidate explicitly describe a stable user identity, preference or behavioral instruction "
        "useful in future sessions? Exclude temporary tasks and third-party profiles.")
    results = [dict(content=r["content"], topic=extractor._derive_topic(r["id"]),
                    importance=r["importance"], source=r["source"], source_memory_id=r["id"],
                    tier="working" if r["tier"] == "working" else "long_term")
               for r, p in zip(candidates, scores) if p >= jev.threshold("PERSONA_THRESHOLD", .85)]
    return sorted(results, key=lambda r: -r["importance"])[:limit]


def proactively_link(beam, memory_id, content):
    from .episodic_graph import GraphEdge
    # Same visibility policy as recall, before evidence leaves the process.
    rows = [r for r in visible_memories(beam, _filters()) if r["id"] != memory_id]
    scores = jev.judge_many({"reference": content}, [r["content"] for r in rows],
        "Are candidate and state.reference substantively related through the same real entity, "
        "event, decision or topic, such that retrieving one would help interpret the other?")
    related = sorted(zip(rows, scores), key=lambda pair: -pair[1])
    for row, score in [(r, s) for r, s in related if s >= jev.threshold("LINK_THRESHOLD", .85)][:5]:
        beam.episodic_graph.add_edge(GraphEdge(source=memory_id, target=row["id"],
            edge_type="related_to", weight=score, timestamp=datetime.now().isoformat()))
