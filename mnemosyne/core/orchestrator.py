"""
Expert recall orchestrator.

Provides a small compatibility entry point for callers that want a single
"best available" recall function without depending directly on BeamMemory.
All retrieval/scoring logic remains in beam.py and related recall modules.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def orchestrate_recall(
    query: str,
    conn: Optional[Any] = None,
    top_k: int = 20,
    *,
    session_id: str = "default",
    beam: Optional[Any] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run recall through the best available BeamMemory path.

    Args:
        query: Natural language search query.
        conn: Optional SQLite connection. Used only when ``beam`` is not
            provided; lets legacy callers pass a raw connection.
        top_k: Maximum results to return.
        session_id: Session id for a temporary BeamMemory wrapper.
        beam: Optional BeamMemory instance. Preferred because it preserves the
            caller's existing DB path, session/channel scope, and cached helpers.
        **kwargs: Passed through to ``BeamMemory.recall``.

    Returns:
        Recall result dictionaries. Returns an empty list only if recall raises.
    """
    try:
        if beam is not None:
            return beam.recall(query, top_k=top_k, **kwargs)

        from mnemosyne.core.beam import BeamMemory

        temp_beam = BeamMemory(session_id=session_id)
        if conn is not None:
            # Keep legacy raw-connection callers on their provided DB.
            temp_beam.conn = conn
        return temp_beam.recall(query, top_k=top_k, **kwargs)
    except Exception:
        logger.warning(
            "orchestrate_recall: BeamMemory recall failed; returning empty list",
            exc_info=True,
        )
        return []
