"""Jev's bounded decisions within SHMR; novel belief prose stays generative."""
import json

from . import jev
from .jev_recall import _diverse_top, _filters, _rank, visible_memories


def _text(item):
    return " ".join(str(item.get(k) or "") for k in ("subject", "predicate", "object"))


def echo_candidates(beam, batch_size):
    """Bounded maintenance batch, scoped before classification or generation."""
    from .beam import _cross_session_enabled
    filters = _filters(cross_session=_cross_session_enabled())
    visible = {r["id"]: r for r in visible_memories(beam, filters)}
    result = []
    for raw in beam.conn.execute("SELECT * FROM facts ORDER BY created_at DESC"):
        row = dict(raw)
        parent = row.get("source_msg_id")
        if parent and parent not in visible:
            continue
        if not parent and not filters["cross_session"] and row["session_id"] != beam.session_id:
            continue
        result.append(dict(row, source="fact", embedding=None))
        if len(result) >= batch_size:
            break
    episodes = sorted((r for r in visible.values() if r["tier"] == "episodic"),
                      key=lambda r: r.get("timestamp", ""), reverse=True)
    for row in episodes[:max(0, batch_size // 2)]:
        result.append(dict(fact_id="ep_" + row["id"], subject="memory", predicate="contains",
            object=row["content"], confidence=row["importance"], timestamp=row["timestamp"],
            source="episodic", embedding=None))
    return result


def cluster(items, threshold):
    adjacency = {i: set() for i in range(len(items))}
    for i, item in enumerate(items):
        scores = jev.judge_many({"reference": _text(item)}, [_text(x) for x in items[i + 1:]],
            "Do candidate and state.reference concern the same real subject and closely related "
            "property or event? Mere generic vocabulary in common is insufficient.")
        for j, score in enumerate(scores, i + 1):
            if score >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen, result = set(), []
    for start in range(len(items)):
        if start in seen:
            continue
        queue, group = [start], []
        while queue:
            index = queue.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(items[index])
            queue.extend(adjacency[index] - seen)
        result.append(group)
    return result


def harmony(beliefs, sources):
    if not beliefs or not sources:
        return 0.0
    support = jev.judge_many({"evidence": [_text(x) for x in sources]}, [_text(b) for b in beliefs],
        "Is candidate a supported summary or inference from state.evidence, without introducing "
        "unsupported entities, relationships or values?")
    # Every accepted belief must be grounded, not just the average belief.
    return min(support)


def classify_actions(beliefs, sources):
    accepted = []
    ids = {s.get("fact_id") for s in sources}
    for belief in beliefs:
        target = belief.get("target_fact_id")
        if target and target not in ids:
            continue
        action, probability = jev.choose({"evidence": [_text(s) for s in sources],
            "candidate": _text(belief), "target": next((_text(s) for s in sources if s.get("fact_id") == target), None)},
            "Choose an action for candidate based solely on evidence. Never infer chronological order from dates.",
            {"create": "Supported useful new synthesis; no existing fact needs changing",
             "update": "Evidence explicitly corrects target and candidate preserves all unaffected claims",
             "dampen": "Evidence explicitly contradicts target; its confidence should decrease",
             "ignore": "Unsupported, redundant, uncertain or irrelevant"})
        if action == "ignore" or probability < .9 or (action in {"update", "dampen"} and not target):
            continue
        accepted.append(dict(belief, action=action, confidence=probability))
    return accepted


def recall_beliefs(beam, query, top_k):
    if top_k <= 0 or not query.strip():
        return []
    from .beam import _cross_session_enabled
    visible = {r["id"] for r in visible_memories(beam, _filters(cross_session=_cross_session_enabled()))}
    allowed = visible | {"ep_" + value for value in visible}
    for row in beam.conn.execute("SELECT fact_id, source_msg_id, session_id FROM facts"):
        if row["source_msg_id"] in visible or (not row["source_msg_id"] and row["session_id"] == beam.session_id):
            allowed.add(row["fact_id"])
    rows = []
    for raw in beam.conn.execute("SELECT * FROM harmonic_beliefs ORDER BY belief_id"):
        row = dict(raw)
        try:
            provenance = json.loads(row.get("provenance") or "[]")
        except (ValueError, TypeError):
            continue
        if not isinstance(provenance, list) or not provenance or not all(isinstance(x, str) and x in allowed for x in provenance):
            continue
        rows.append(dict(row, content=_text(row), source="harmonic_belief"))
    ranked, _ = _rank(query, rows)
    return _diverse_top(ranked, top_k)
