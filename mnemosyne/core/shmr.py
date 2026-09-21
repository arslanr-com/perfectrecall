"""
Self-Harmonizing Memory Reasoning (SHMR)
========================================
Built on ECHO-OR research (AxDSan/ECHO-OR) but fully rearchitected for
continuous local memory orchestration inside Mnemosyne's BEAM architecture.

Core idea: related memories "echo" each other in the background, negotiating
contradictions, surfacing hidden patterns, and converging into stable beliefs.

This is Mnemosyne's signature reasoning layer -- no Honcho dreams, no Hindsight
reflections, no Mem0 static graphs. Memories actively resonate and self-correct.
"""

from __future__ import annotations
import os
import time
import logging
import json
from typing import TYPE_CHECKING, List, Dict, Optional

if TYPE_CHECKING:
    pass
else:
    np = None
logger = logging.getLogger("mnemosyne.shmr")
SHMR_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_SHMR_BATCH_SIZE", "50"))
SHMR_MAX_ITERATIONS = int(os.environ.get("MNEMOSYNE_SHMR_MAX_ITERATIONS", "3"))
SHMR_SIMILARITY_THRESHOLD = float(
    os.environ.get("MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD", "0.70")
)
SHMR_HARMONY_THRESHOLD = float(
    os.environ.get("MNEMOSYNE_SHMR_HARMONY_THRESHOLD", "0.60")
)
SHMR_MODEL = os.environ.get("MNEMOSYNE_SHMR_MODEL", "")
SHMR_MIN_CLUSTER_SIZE = int(os.environ.get("MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE", "2"))
SHMR_TEMPERATURE = float(os.environ.get("MNEMOSYNE_SHMR_TEMPERATURE", "0.2"))
EMBEDDING_DIM = 0
FACTS_SCHEMA_SQL = "\nCREATE TABLE IF NOT EXISTS harmonic_beliefs (\n    belief_id TEXT PRIMARY KEY,\n    subject TEXT,\n    predicate TEXT,\n    object TEXT NOT NULL,\n    confidence REAL DEFAULT 0.5,\n    provenance TEXT,   -- JSON array of source fact_ids or memory_ids\n    cluster_id TEXT,\n    iteration INTEGER DEFAULT 0,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);\n\nCREATE TABLE IF NOT EXISTS memory_resonance_log (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n    session_id TEXT,\n    cluster_count INTEGER,\n    beliefs_generated INTEGER,\n    contradictions_resolved INTEGER,\n    harmony_score_avg REAL,\n    duration_ms INTEGER,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);\n\nCREATE INDEX IF NOT EXISTS idx_beliefs_subject ON harmonic_beliefs(subject);\nCREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON harmonic_beliefs(predicate);\nCREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON harmonic_beliefs(confidence);\n"


def _init_schema(conn):
    """Ensure SHMR tables exist."""
    conn.executescript(FACTS_SCHEMA_SQL)
    conn.commit()


def _cluster_by_similarity(items: List[Dict], threshold: float) -> List[List[Dict]]:
    "Cluster related items using Jev decisions."
    if not items:
        return []
    from . import jev
    from .jev_shmr import cluster

    return cluster(items, threshold)


def _format_cluster_for_llm(cluster: List[Dict]) -> str:
    """Format a memory cluster as a prompt for the LLM harmonizer."""
    lines = ["=== MEMORY CLUSTER ==="]
    for i, item in enumerate(cluster):
        subject = item.get("subject", "unknown")
        predicate = item.get("predicate", "stated")
        obj = item.get("object", item.get("content", ""))
        confidence = item.get("confidence", 0.5)
        source = item.get("source", "fact")
        lines.append(
            f"[{i}] ({source}, conf={confidence:.2f}) {subject} | {predicate} | {obj}"
        )
    return "\n".join(lines)


HARMONY_PROMPT = 'You are the Self-Harmonizing Memory Reasoner for Mnemosyne.\nThese memories belong to the same semantic cluster -- they all relate to the\nsame entities, topics, or events. Your job is to harmonize them:\n\n1. **Resolve contradictions**: If two memories conflict, determine which is more\n   likely true based on recency, specificity, and internal consistency. Flag the\n   weaker one as dampened, not deleted.\n2. **Extract higher-order beliefs**: Find patterns that span multiple memories.\n   What does this cluster as a whole tell us? What\'s the stable truth?\n3. **Dampen noise, amplify signal**: Low-confidence or stale memories get lower\n   weight. Corroborated facts get reinforced.\n4. **Output only stable beliefs**: Return NEW or UPDATED facts with confidence\n   scores. Don\'t regurgitate every input fact -- synthesize.\n\nOutput as JSON array of belief objects:\n[{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0,\n  "action": "create"|"update"|"dampen", "target_fact_id": null|"fact_id",\n  "rationale": "one sentence explaining why"}]\n\nRULES:\n- Confidence 0.9+ = highly corroborated (multiple sources agree)\n- Confidence 0.5-0.8 = reasonable inference from the cluster\n- Confidence <0.4 = speculative, mark as such\n- Use "dampen" to reduce confidence of contradicted facts (never delete)\n- Use "update" to modify an existing fact with new information\n- Output 1-5 beliefs per cluster (don\'t over-generate)'


def _call_llm(prompt: str, system: str = "") -> str:
    """Call the configured LLM for harmonization.

    Uses the same LLM chain as mnemosyne_sleep's summarization:
    local_llm first, fallback to cloud extraction client.
    """
    try:
        from mnemosyne.core.local_llm import _call_local_llm

        result = _call_local_llm((system + "\n\n" if system else "") + prompt)
        if result and len(result.strip()) > 10:
            return result
    except Exception:
        pass
    try:
        from mnemosyne.extraction import ExtractionClient

        client = ExtractionClient(model=SHMR_MODEL or None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = client.chat(messages, temperature=SHMR_TEMPERATURE)
        if result:
            return result
    except Exception:
        logger.debug("SHMR cloud fallback failed", exc_info=True)
    return ""


def _compute_harmony_score(beliefs: List[Dict], cluster: List[Dict]) -> float:
    "Evaluate belief coverage with Jev decisions."
    from . import jev
    from .jev_shmr import harmony

    return harmony(beliefs, cluster)


def _extract_json_from_llm_output(text: str) -> List[Dict]:
    """Robust JSON extraction from LLM output (handles markdown wrappers)."""
    import re

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "beliefs" in parsed:
            return parsed["beliefs"]
    except (json.JSONDecodeError, TypeError):
        pass
    json_match = re.search("```(?:json)?\\s*(\\[.*?\\])\\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    array_match = re.search("\\[\\s*\\{.*?\\}\\s*\\]", text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    objects = re.findall("\\{[^{}]*\\}", text)
    results = []
    for obj_str in objects:
        try:
            results.append(json.loads(obj_str))
        except (json.JSONDecodeError, TypeError):
            continue
    return results


def _apply_beliefs(conn, beliefs: List[Dict], cluster: List[Dict], cluster_id: str):
    """Write harmonized beliefs to the database and update source facts."""
    import hashlib

    cursor = conn.cursor()
    now = __import__("datetime").datetime.now().isoformat()
    for belief in beliefs:
        action = belief.get("action", "create")
        subject = belief.get("subject", "entity")
        predicate = belief.get("predicate", "related_to")
        obj = belief.get("object", "")
        confidence = max(0.1, min(1.0, belief.get("confidence", 0.5)))
        belief_id = hashlib.sha256(
            f"{cluster_id}:{subject}:{predicate}:{obj[:50]}".encode()
        ).hexdigest()[:24]
        if action == "dampen":
            target_id = belief.get("target_fact_id")
            if target_id:
                cursor.execute(
                    "UPDATE facts SET confidence = MAX(0.1, confidence - 0.15) WHERE fact_id = ?",
                    (target_id,),
                )
        elif action == "update":
            target_id = belief.get("target_fact_id")
            if target_id:
                cursor.execute(
                    "UPDATE facts SET object = ?, confidence = ? WHERE fact_id = ?",
                    (obj, confidence, target_id),
                )
        try:
            cursor.execute(
                "\n                INSERT OR REPLACE INTO harmonic_beliefs\n                (belief_id, subject, predicate, object, confidence,\n                 provenance, cluster_id, iteration, updated_at)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    belief_id,
                    subject,
                    predicate,
                    obj,
                    confidence,
                    json.dumps(
                        [c.get("fact_id", "") for c in cluster if c.get("fact_id")]
                    ),
                    cluster_id,
                    0,
                    now,
                ),
            )
        except Exception:
            continue
    conn.commit()


def harmonize(
    beam,
    batch_size: int = None,
    max_iterations: int = None,
    similarity_threshold: float = None,
) -> Dict:
    "Run a bounded reasoning cycle. Jev groups and scores memories; the configured LLM may synthesize beliefs."
    if batch_size is None:
        batch_size = SHMR_BATCH_SIZE
    if max_iterations is None:
        max_iterations = SHMR_MAX_ITERATIONS
    if similarity_threshold is None:
        similarity_threshold = SHMR_SIMILARITY_THRESHOLD
    from . import jev

    t0 = time.perf_counter()
    _init_schema(beam.conn)
    cursor = beam.conn.cursor()
    candidates = []
    from .jev_shmr import echo_candidates

    candidates = echo_candidates(beam, batch_size)
    if len(candidates) < SHMR_MIN_CLUSTER_SIZE:
        return {
            "clusters_found": 0,
            "beliefs_generated": 0,
            "contradictions_resolved": 0,
            "harmony_score_avg": 0.0,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "status": "insufficient_candidates",
        }
    clusters = _cluster_by_similarity(candidates, similarity_threshold)
    clusters = [c for c in clusters if len(c) >= SHMR_MIN_CLUSTER_SIZE]
    total_beliefs = 0
    total_contradictions = 0
    harmony_scores = []
    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = f"shmr_{int(time.time())}_{cluster_idx}"
        for iteration in range(max_iterations):
            context = _format_cluster_for_llm(cluster)
            full_prompt = context + "\n\n" + HARMONY_PROMPT
            try:
                llm_output = _call_llm(full_prompt)
                if not llm_output:
                    continue
                beliefs = _extract_json_from_llm_output(llm_output)
                if not beliefs:
                    continue
                from .jev_shmr import classify_actions

                beliefs = classify_actions(beliefs, cluster)
                if not beliefs:
                    continue
                score = _compute_harmony_score(beliefs, cluster)
                harmony_scores.append(score)
                if score >= SHMR_HARMONY_THRESHOLD:
                    _apply_beliefs(beam.conn, beliefs, cluster, cluster_id)
                    total_beliefs += len(
                        [b for b in beliefs if b.get("action") in ("create", "update")]
                    )
                    total_contradictions += len(
                        [b for b in beliefs if b.get("action") == "dampen"]
                    )
                    break
            except Exception as e:
                logger.warning(f"SHMR cluster {cluster_id} iteration {iteration}: {e}")
                continue
    duration_ms = int((time.perf_counter() - t0) * 1000)
    avg_score = sum(harmony_scores) / len(harmony_scores) if harmony_scores else 0.0
    try:
        cursor.execute(
            "\n            INSERT INTO memory_resonance_log\n            (session_id, cluster_count, beliefs_generated,\n             contradictions_resolved, harmony_score_avg, duration_ms)\n            VALUES (?, ?, ?, ?, ?, ?)\n        ",
            (
                beam.session_id,
                len(clusters),
                total_beliefs,
                total_contradictions,
                round(avg_score, 4),
                duration_ms,
            ),
        )
        beam.conn.commit()
    except Exception:
        pass
    return {
        "clusters_found": len(clusters),
        "beliefs_generated": total_beliefs,
        "contradictions_resolved": total_contradictions,
        "harmony_score_avg": round(avg_score, 4),
        "duration_ms": duration_ms,
        "status": "harmonized" if total_beliefs > 0 else "no_convergence",
    }


def recall_beliefs(beam, query: str, top_k: int = 10) -> List[Dict]:
    "Recall visible stored beliefs using Jev decisions."
    from . import jev
    from .jev_shmr import recall_beliefs as jev_recall_beliefs

    _init_schema(beam.conn)
    return jev_recall_beliefs(beam, query, top_k)


REFLECTION_PROMPT = "You are a memory reasoning assistant. You have retrieved facts\nfrom a conversation database and need to synthesize a coherent answer.\n\nQUESTION: {question}\n\nRETRIEVED FACTS:\n{fact_context}\n\nBased on these facts, provide a concise synthesis (2-4 sentences) that:\n1. Answers the question directly if the facts are sufficient\n2. Identifies any contradictions or gaps in the facts\n3. Notes temporal context (dates, order of events) if present\n4. If facts are insufficient, states what's missing clearly\n\nSYNTHESIS:"


def reflect(
    beam, question: str, facts: List[Dict] = None, top_k: int = 10
) -> Optional[str]:
    """Single-pass reflective synthesis over retrieved facts.

    Takes a question and a list of fact dicts (from fact_recall()), sends them
    to an LLM, and returns a coherent synthesis paragraph. This synthesis is
    then injected as additional context for the final answering LLM.

    This is Phase 3A: lightweight, works with any LLM, no iteration needed.
    Phase 3B (SHMR harmonize()) replaces this with multi-iteration harmony loop.

    Args:
        beam: BeamMemory instance (for fact_recall if facts not provided)
        question: The question to synthesize for
        facts: Pre-retrieved facts (if None, calls fact_recall automatically)
        top_k: Max facts to include in the reflection

    Returns:
        Synthesis string, or None if no facts available.
    """
    if facts is None and beam is not None:
        try:
            facts = beam.fact_recall(question, top_k=top_k)
        except Exception:
            return None
    if not facts:
        return None
    sorted_facts = sorted(facts, key=lambda f: f.get("score", 0), reverse=True)[:top_k]
    fact_lines = []
    for i, f in enumerate(sorted_facts):
        content = f.get("content", "")
        score = f.get("score", 0.5)
        source = f.get("source", "fact")
        fact_lines.append(f"[{i}] ({source}, conf={score:.2f}) {content}")
    fact_context = "\n".join(fact_lines)
    prompt = REFLECTION_PROMPT.format(question=question, fact_context=fact_context)
    synthesis = _call_llm(prompt)
    if synthesis and len(synthesis.strip()) > 10:
        return synthesis.strip()
    return None


def get_resonance_log(beam, limit: int = 10) -> List[Dict]:
    """Get recent harmonization run logs."""
    cursor = beam.conn.cursor()
    _init_schema(beam.conn)
    try:
        rows = cursor.execute(
            "\n            SELECT * FROM memory_resonance_log\n            ORDER BY created_at DESC LIMIT ?\n        ",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
