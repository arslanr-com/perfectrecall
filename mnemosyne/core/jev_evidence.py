"""Caller-authored evidence questions, evaluated by Jev over every visible span.

No generative model, embeddings, candidate shortlist or query rewriting.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
import re
from . import jev


def worker_count():
    """Bound concurrent Jev requests without changing evidence or coverage."""
    raw = os.environ.get('MNEMOSYNE_JEV_WORKERS', '128')
    try:
        workers = int(raw)
    except (ValueError, TypeError):
        raise ValueError('MNEMOSYNE_JEV_WORKERS must be an integer from 1 to 256') from None
    if not 1 <= workers <= 256:
        raise ValueError('MNEMOSYNE_JEV_WORKERS must be an integer from 1 to 256')
    return workers


def validate(questions):
    if questions is None:
        return None
    if (not isinstance(questions, (list, tuple)) or not 1 <= len(questions) <= 3
            or not all(isinstance(q, str) and q.strip() and len(q) <= 600 for q in questions)):
        raise ValueError("evidence_questions must contain 1-3 nonempty strings of at most 600 characters")
    return [q.strip() for q in questions]


def evidence_spans(content):
    """Keep speaker turns separate so surrounding advice does not negate a fact.

    This is structural splitting, not relevance filtering: every turn and every
    character remains represented. Long turns retain their speaker on each span.
    """
    metadata = content.split('\n', 1)[0] if content.startswith('Session date: ') else ''
    turns = re.split(r'(?m)(?=^(?:user|assistant|system|tool):)', content)
    for turn in turns:
        if not turn or (len(turns) > 1 and metadata and turn.strip() == metadata):
            continue
        match = re.match(r'^(user|assistant|system|tool):', turn)
        speaker = match.group(0) if match else ''
        for span in jev.chunks(turn, size=2000):
            if speaker and not span.startswith(speaker):
                span = speaker + ' ' + span
            if metadata and not span.startswith(metadata):
                span = metadata + '\n' + span
            yield span


def rank(query, source, evidence_questions=None):
    workers = worker_count()
    criteria = validate(evidence_questions) or [
        "Does this memory contain any partial evidence relevant to this question, even if it does not state the complete answer: " + query]
    rows = list(source)
    if not rows:
        return [], 0
    questions = {str(i): {'type':'noul', 'instructions': jev.DATA_RULE + criterion}
                 for i, criterion in enumerate(criteria)}
    def score(span):
        answers = jev.client().evaluate(span, questions)
        return max(a['noul'] for a in answers.values())
    spans = iter((owner, span) for owner, row in enumerate(rows)
                 for span in evidence_spans(row['content']))
    scores = [0.] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        def fill():
            # Keep at most one outstanding request per worker. Do not eagerly
            # materialize every span/future of a large corpus.
            while len(pending) < workers:
                try:
                    owner, span = next(spans)
                except StopIteration:
                    break
                pending[pool.submit(score, span)] = owner
        try:
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    owner = pending.pop(future)
                    scores[owner] = max(scores[owner], future.result())
                fill()
        except Exception:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    ranked = [dict(row, score=score, jev_relevance=score, dense_score=0., fts_score=0.)
              for row, score in zip(rows, scores)]
    cutoff = jev.threshold('RELEVANCE_THRESHOLD', .5)
    ranked = [row for row in ranked if row['score'] >= cutoff]
    ranked.sort(key=lambda row: (-row['score'], row['id']))
    return ranked, len(rows)
