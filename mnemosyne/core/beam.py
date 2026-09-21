"""SQLite storage inherited from Mnemosyne; all semantic recall uses Jev."""

from __future__ import annotations
import contextlib
import logging
import sqlite3
import json
import hashlib
import threading
import math
from dataclasses import dataclass
from mnemosyne.core._connection_gc import collect_connection_cycles
from mnemosyne.core.config import resolve_beam_runtime
from mnemosyne.core.journal import journal_mode

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Set, Union, Tuple, Callable, Sequence
from pathlib import Path


class MemoryTransactionStateError(RuntimeError):
    """consolidate_to_episodic() was asked to emit MEMORY_CONSOLIDATED while
    a caller-owned transaction is open: the event cannot be ordered after
    the outer commit, so the call is rejected before any write."""


def _event_date_valid(value: str) -> bool:
    """True when value is a real calendar date in strict ASCII YYYY-MM-DD.

    Shape alone accepts 2026-02-31, and permissive parsers accept
    non-padded or non-ASCII forms that sort wrong under the lexicographic
    date filters — so validity is shape AND calendar. Single definition
    shared by the sanitizer (import), the public validator
    (consolidate_to_episodic) and the degrade path (sleep aggregation).
    """
    import re as _re
    import datetime as _dt

    if not _re.fullmatch("[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _sanitize_import_event_date(value, precision):
    """Sanitize an imported event_date/precision pair. Invalid dates are
    sanitized to (None, 'unknown') + warning — the row survives, only the
    derived field is reset (consistent with the consolidation-side policy:
    the row's chronology is preserved even when the derived date is
    garbage)."""
    if isinstance(value, str):
        value = value.strip()
    else:
        if value is not None:
            logger.warning(
                "import_from_dict: event_date %r is not a string; sanitized to undated",
                value,
            )
        return (None, "unknown")
    if not _event_date_valid(value):
        logger.warning(
            "import_from_dict: event_date %r is not a real calendar date (strict YYYY-MM-DD); sanitized to undated",
            value,
        )
        return (None, "unknown")
    if not isinstance(precision, str) or precision not in _EVENT_DATE_PRECISIONS:
        logger.warning(
            "import_from_dict: event_date_precision %r invalid; sanitized to 'unknown'",
            precision,
        )
        precision = "unknown"
    return (value, precision or "unknown")


def _import_timestamp_ok(value) -> bool:
    """An imported row timestamp must place the row in time. None/blank/
    unparseable values would otherwise epoch-degrade into immediate trim
    candidates (round-4 probe: 1 valid + 1 None row -> 1 row after trim)."""
    if value is None:
        return False
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        _parse_iso_datetime_utc(value.strip())
        return True
    except (ValueError, TypeError, OverflowError):
        return False


try:
    from mnemosyne.core.typed_memory import classify_memory, MemoryType
except ImportError:
    classify_memory = None
    MemoryType = None
_mib = None
_hamming = None
EMBEDDING_DIM = None
try:
    from mnemosyne.core.episodic_graph import EpisodicGraph, GraphEdge
except ImportError:
    EpisodicGraph = None
    GraphEdge = None
try:
    from mnemosyne.core.veracity_consolidation import (
        VeracityConsolidator,
        VERACITY_WEIGHTS,
        clamp_veracity,
        aggregate_veracity,
    )

    _VW_DEFAULTS = VERACITY_WEIGHTS
except ImportError:
    VeracityConsolidator = None
    VERACITY_WEIGHTS = {}
    _VW_DEFAULTS = {
        "stated": 1.0,
        "inferred": 0.7,
        "tool": 0.5,
        "imported": 0.6,
        "unknown": 0.8,
    }
    logger.warning(
        "mnemosyne.core.veracity_consolidation unavailable; using fallback clamp_veracity. Non-canonical veracity labels will be clamped silently (no per-call WARNING). Operators should resolve the import to restore full audit logging."
    )

    def aggregate_veracity(source_veracities) -> str:
        """Fallback aggregator when veracity_consolidation is unavailable.
        Returns 'unknown' unconditionally so consolidation doesn't crash."""
        return "unknown"

    def clamp_veracity(raw, *, context: str = "veracity") -> str:
        """Fallback when veracity_consolidation is unavailable.
        Mirrors the canonical helper's API and clamps non-canonical
        labels to 'unknown'. Does NOT log per-call warnings -- the
        import-time warning above is the audit signal. Operators
        should fix the import to restore full observability.
        """
        if raw is None:
            return "unknown"
        norm = str(raw).strip().lower()
        if not norm:
            return "unknown"
        if norm in {"stated", "inferred", "tool", "imported", "unknown"}:
            return norm
        return "unknown"


try:
    from mnemosyne.core.weibull import weibull_boost
except ImportError:
    weibull_boost = None
try:
    from mnemosyne.core.query_intent import classify_intent, adjust_weights
except ImportError:
    classify_intent = None
    adjust_weights = None
try:
    from mnemosyne.core.mmr import mmr_rerank
except ImportError:
    mmr_rerank = None
try:
    from mnemosyne.core.synonyms import expand_query, normalize_query
except ImportError:
    expand_query = None
    normalize_query = None
QueryCache = None
try:
    from mnemosyne.core.temporal_parser import extract_temporal, parse_nl_date
except ImportError:
    extract_temporal = None
    parse_nl_date = None
TRUST_TIER_MAP = {
    "conversation": "STATED",
    "user": "STATED",
    "cli": "STATED",
    "mcp": "EXTERNAL_WRITE",
    "import": "IMPORTED",
    "mem0": "IMPORTED",
    "honcho_import": "IMPORTED",
    "honcho_summary": "IMPORTED",
    "consolidation": "DERIVED",
    "sleep_consolidation": "DERIVED",
    "regex": "DERIVED",
    "extraction": "DERIVED",
    "unknown": "STATED",
}


def _source_to_trust_tier(source: str) -> str:
    """Map ingestion source to trust_tier for prompt-injection defense.

    Plugin-first design: callers describe WHAT they are (via `source`),
    Mnemosyne decides HOW to trust it (via trust_tier mapping). New
    ingestion paths only need to set `source` honestly — the mapping
    centralizes the trust policy.
    """
    if not source:
        return "STATED"
    if source in TRUST_TIER_MAP:
        return TRUST_TIER_MAP[source]
    if "import" in source.lower():
        return "IMPORTED"
    if "mcp" in source.lower():
        return "EXTERNAL_WRITE"
    return "STATED"


def _clamp_memory_type(value: Optional[str]) -> Optional[str]:
    """Validate an explicit memory_type label, or return None.

    None passes through, which is the signal to fall back to the content
    classifier. An unrecognized label also returns None, with a WARNING: a
    typo should degrade to classification, not strip the type off the row.

    Mirrors the posture of clamp_veracity -- validate at the lowest public
    ingest path rather than trusting callers, since the value reaches a
    column that recall filters on.
    """
    if value is None:
        return None
    if MemoryType is None:
        return None
    candidate = str(value).strip().lower()
    if candidate in {m.value for m in MemoryType}:
        return candidate
    logger.warning(
        "remember: unknown memory_type %r; falling back to the classifier", value
    )
    return None


np = None
from mnemosyne.core import plugins as _plugins

sqlite_vec = None
import os
import re

_VERSION_STRING_RE = re.compile(
    "([A-Z][a-zA-Z]+(?:\\s+[A-Z][a-zA-Z]+)*)\\s+v?(\\d+\\.\\d+(?:\\.\\d+)?)"
)
_DEFAULT_ROOT = Path(
    os.environ.get("HERMES_HOME")
    or (
        Path(os.environ["HOME"]) / ".hermes"
        if os.environ.get("HOME")
        else Path.home() / ".hermes"
    )
)
DEFAULT_DATA_DIR = _DEFAULT_ROOT / "mnemosyne" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"
_thread_local = threading.local()


def _env_truthy(name: str) -> bool:
    """Parse an env var as truthy. Accepts `1`/`true`/`yes`/`on`
    (case-insensitive, whitespace-stripped). Everything else
    (including unset, empty, garbage) is False.

    Complement of `_env_disabled` (defined below) -- they exist for
    different default-state use cases. Use `_env_truthy` when the
    feature is default-OFF and an env var opts it on; use
    `_env_disabled` when the feature is default-ON and an env var
    opts it off.

    Mirrors the helper of the same name in `_benchmarks/evaluate_beam_end_to_end.py`
    for env-parsing consistency across the codebase.
    """
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


_BEAM_MODE = _env_truthy("MNEMOSYNE_BEAM_OPTIMIZATIONS")
if os.environ.get("MNEMOSYNE_DATA_DIR"):
    DEFAULT_DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR"))
    DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"


def _default_data_dir() -> Path:
    """Return the current default data directory, honoring runtime env changes."""
    if os.environ.get("MNEMOSYNE_DATA_DIR"):
        return Path(os.environ["MNEMOSYNE_DATA_DIR"])
    return DEFAULT_DATA_DIR


def _default_db_path() -> Path:
    """Return the current default DB path, honoring runtime env changes."""
    return _default_data_dir() / "mnemosyne.db"


EMBEDDING_DIM = 0
WORKING_MEMORY_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_WM_MAX_ITEMS", "10000"))
WORKING_MEMORY_TTL_HOURS = int(os.environ.get("MNEMOSYNE_WM_TTL_HOURS", "168"))
WM_BUMP_CAP_HOURS = int(os.environ.get("MNEMOSYNE_WM_BUMP_CAP_HOURS", "24"))
WM_PINNED_IDS = set(
    (
        pid.strip()
        for pid in os.environ.get("MNEMOSYNE_WM_PINNED_IDS", "").split(",")
        if pid.strip()
    )
)
EPISODIC_RECALL_LIMIT = int(os.environ.get("MNEMOSYNE_EP_LIMIT", "50000"))
SLEEP_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_SLEEP_BATCH", "5000"))
SCRATCHPAD_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_SP_MAX", "1000"))
RECENCY_HALFLIFE_HOURS = float(os.environ.get("MNEMOSYNE_RECENCY_HALFLIFE", "168"))
TIER2_DAYS = int(os.environ.get("MNEMOSYNE_TIER2_DAYS", "30"))
TIER3_DAYS = int(os.environ.get("MNEMOSYNE_TIER3_DAYS", "180"))
TIER1_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER1_WEIGHT", "1.0"))
TIER2_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER2_WEIGHT", "0.5"))
TIER3_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER3_WEIGHT", "0.25"))
DEGRADE_BATCH_SIZE = 100
SMART_COMPRESS = os.environ.get("MNEMOSYNE_SMART_COMPRESS", "1") not in (
    "0",
    "false",
    "no",
)
TIER3_MAX_CHARS = int(os.environ.get("MNEMOSYNE_TIER3_MAX_CHARS", "300"))


def _env_disabled(name: str) -> bool:
    """A/B toggle helper: return True iff the env var is explicitly
    set to a falsy value (`0`/`false`/`no`/`off`, case-insensitive,
    whitespace-stripped).

    Used by experiment ablation toggles where the feature is ON by
    default (production behavior) and operators can disable it
    explicitly via env var. Distinct from `_env_truthy` from the
    benchmark harness -- that one defaults to OFF, this one defaults
    to ON. See `docs/benchmarking.md` for the full toggle reference.

    Unset / empty / non-falsy → False (feature enabled).
    """
    val = os.environ.get(name, "").strip().lower()
    return val in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    """Parse an env var as float; fall back to `default` on empty or
    invalid values rather than crashing at module load.

    Pre-fix `float(os.environ.get("MNEMOSYNE_STATED_WEIGHT", "1.0"))`
    raised ValueError when the env var was set to empty (`export
    MNEMOSYNE_STATED_WEIGHT=`) because `os.environ.get` returns `""`
    (the value), not the default -- `float("")` then crashed import
    BEFORE the C32 override-WARN could fire. Restored from PR #91
    after the merge stripped it.
    """
    raw = os.environ.get(name, "")
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a valid float; falling back to default %s",
            name,
            raw[:80],
            default,
        )
        return default


def _env_int(name: str, default: int) -> int:
    """Parse a positive integer knob, falling back on empty/invalid values."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning("%s is not a positive integer; using default %s", name, default)
    return default


_CONFLICT_PAIR_BUDGET = _env_int("MNEMOSYNE_CONFLICT_PAIR_BUDGET", 20)
_CONFLICT_TIME_BUDGET_S = _env_float("MNEMOSYNE_CONFLICT_TIME_BUDGET_S", 300.0)
if not math.isfinite(_CONFLICT_TIME_BUDGET_S) or _CONFLICT_TIME_BUDGET_S <= 0:
    logger.warning(
        "MNEMOSYNE_CONFLICT_TIME_BUDGET_S must be finite and positive; using 300s"
    )
    _CONFLICT_TIME_BUDGET_S = 300.0
STATED_WEIGHT = _env_float("MNEMOSYNE_STATED_WEIGHT", _VW_DEFAULTS["stated"])
INFERRED_WEIGHT = _env_float("MNEMOSYNE_INFERRED_WEIGHT", _VW_DEFAULTS["inferred"])
TOOL_WEIGHT = _env_float("MNEMOSYNE_TOOL_WEIGHT", _VW_DEFAULTS["tool"])
IMPORTED_WEIGHT = _env_float("MNEMOSYNE_IMPORTED_WEIGHT", _VW_DEFAULTS["imported"])
UNKNOWN_WEIGHT = _env_float("MNEMOSYNE_UNKNOWN_WEIGHT", _VW_DEFAULTS["unknown"])
_INT8_SATURATION_RMS = 50.0
_VEC_NORM_BIT = 536870912
_SIGN_BOOST_CAP = 0.15
_POPCOUNT_TABLE_256 = None


def _classify_vec_store_regime(conn, table: str = "vec_episodes") -> str:
    return "unused"


_legacy_warning_emitted = False
_unknown_marker_warning_emitted = False
_unknown_marker_warning_lock = threading.Lock()
EM_VEC_ADMIT = 0


def _detect_veracity_weight_overrides() -> List[str]:
    """C32: return a list of `MNEMOSYNE_*_WEIGHT` env vars set to a
    non-empty value. Filters out empty-string values (`export
    MNEMOSYNE_STATED_WEIGHT=`) since `_env_float` falls back to default
    on empties -- counting them would confuse the WARN message.
    """
    return [
        name
        for name in (
            "MNEMOSYNE_STATED_WEIGHT",
            "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT",
            "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        )
        if os.environ.get(name, "").strip()
    ]


_VERACITY_WARN_EMITTED = False


def _warn_about_veracity_weight_overrides(force: bool = False) -> bool:
    """Log a WARNING if any `MNEMOSYNE_*_WEIGHT` env var is overridden.

    Idempotent per-process: subsequent calls return False without
    re-emitting unless `force=True` (tests use this to verify the WARN
    fires per call). Multi-worker setups (uvicorn `--workers`,
    pytest-xdist) get one WARN per process instead of N per startup.
    """
    global _VERACITY_WARN_EMITTED
    if _VERACITY_WARN_EMITTED and (not force):
        return False
    overrides = _detect_veracity_weight_overrides()
    if not overrides:
        return False
    logger.warning(
        "Veracity weight env overrides detected: %s. Recall scoring will honor the override, but consolidation Bayesian compounding (veracity_consolidation.VERACITY_WEIGHTS) does NOT -- the two will drift. Set matching values in veracity_consolidation.py OR accept that 'consolidated-as-N also ranks at N' invariant is broken until the consolidator is taught the same overrides.",
        ", ".join(overrides),
    )
    _VERACITY_WARN_EMITTED = True
    return True


_warn_about_veracity_weight_overrides()


def _cross_session_enabled() -> bool:
    """Return whether session scoping should be disabled for recall."""
    return resolve_beam_runtime().cross_session


def _session_scope_filter(
    extra_col: str = "", *, cross_session: Optional[bool] = None
) -> str:
    """Return a WHERE clause for session scoping from one runtime snapshot."""
    if cross_session is None:
        cross_session = _cross_session_enabled()
    if cross_session:
        return "(1=1)"
    if extra_col:
        return f"(session_id = ? OR scope = 'global' OR {extra_col} = ?)"
    return "(session_id = ? OR scope = 'global')"


def _session_scope_params(
    session_id: str, extra_value=None, *, cross_session: Optional[bool] = None
) -> list:
    """Return bind params matching a scope filter from one runtime snapshot."""
    if cross_session is None:
        cross_session = _cross_session_enabled()
    if cross_session:
        return []
    if extra_value is not None:
        return [session_id, extra_value]
    return [session_id]


def _episodic_recall_where(
    *,
    session_id: str,
    now_iso: str,
    cross_session: bool,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    author_id: Optional[str] = None,
    author_type: Optional[str] = None,
    channel_id: Optional[str] = None,
    veracity: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """Build the shared episodic eligibility predicate used before limits."""
    clauses = [
        "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
        "superseded_by IS NULL",
    ]
    params: List[Any] = [now_iso]
    if channel_id:
        clauses.append(_session_scope_filter("channel_id", cross_session=cross_session))
        params.extend(
            _session_scope_params(session_id, channel_id, cross_session=cross_session)
        )
    elif author_id or author_type:
        clauses.append("(1=1)")
    else:
        clauses.append(_session_scope_filter(cross_session=cross_session))
        params.extend(_session_scope_params(session_id, cross_session=cross_session))
    if from_date:
        clauses.append("timestamp >= ?")
        params.append(f"{from_date}T00:00:00")
    if to_date:
        clauses.append("timestamp <= ?")
        params.append(f"{to_date}T23:59:59")
    if source:
        clauses.append("source = ?")
        params.append(source)
    if topic:
        clauses.append("source = ?")
        params.append(topic)
    if veracity:
        clauses.append("veracity = ?")
        params.append(veracity)
    if memory_type:
        clauses.append("memory_type = ?")
        params.append(memory_type)
    if author_id:
        clauses.append("author_id = ?")
        params.append(author_id)
    if author_type:
        clauses.append("author_type = ?")
        params.append(author_type)
    if channel_id:
        clauses.append("channel_id = ?")
        params.append(channel_id)
    return (" AND ".join(clauses), params)


VEC_TYPE = os.environ.get("MNEMOSYNE_VEC_TYPE", "int8").lower()
if VEC_TYPE not in ("float32", "int8", "bit"):
    VEC_TYPE = "float32"


def _get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Get thread-local database connection with extensions loaded.

    Returns a `_BeamConnection` (sqlite3.Connection subclass) so
    `remember_batch`'s enrichment loop can defer commits via
    `_deferred_commits`. Connection is otherwise identical to a
    plain sqlite3.Connection.
    """
    path = (Path(db_path) if db_path else _default_db_path()).expanduser().resolve()
    needs_reconnect = (
        not hasattr(_thread_local, "conn")
        or _thread_local.conn is None
        or getattr(_thread_local, "db_path", None) != str(path)
    )
    if not needs_reconnect:
        try:
            _thread_local.conn.execute("SELECT 1")
        except Exception:
            needs_reconnect = True
    if needs_reconnect:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(path), check_same_thread=False, factory=_BeamConnection
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA journal_mode={journal_mode()}")
        try:
            _busy_ms = int(os.environ.get("MNEMOSYNE_BUSY_TIMEOUT_MS", "5000"))
        except ValueError:
            _busy_ms = 5000
        conn.execute(f"PRAGMA busy_timeout={_busy_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn._mnemosyne_vec_loaded = False
        _thread_local.conn = conn
        _thread_local.db_path = str(path)
        collect_connection_cycles()
    return _thread_local.conn


_VEC_TABLE_NAMES = ("vec_episodes", "vec_working", "vec_facts")


@dataclass(frozen=True)
class BeamInitResult:
    """Outcome of BEAM schema initialization.

    This is additive status only: callers that ignore ``init_beam()``'s return
    value retain the historical initialization behavior. ``stored_dims`` is a
    fixed-order immutable ``(table, dimension)`` tuple; ``existing_dim`` is set
    only when those stored tables have one uniform dimension. A mismatch remains
    recoverable and does not prevent the non-vector schema from being initialized.
    """

    vec_dim_mismatch: bool
    existing_dim: Optional[int]
    configured_dim: int
    stored_dims: Tuple[Tuple[str, int], ...] = ()


_schema_locks = {}
_schema_locks_guard = threading.Lock()


@contextlib.contextmanager
def _schema_init_lock(path):
    """Serialize schema initialization across threads and local processes."""
    path = Path(path).expanduser().resolve()
    with _schema_locks_guard:
        lock = _schema_locks.setdefault(str(path), threading.RLock())
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".init.lock", "a+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"\x00")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield path
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def init_beam(db_path: Path = None) -> BeamInitResult:
    """Initialize BEAM under one canonical-path thread/process boundary."""
    with _schema_init_lock(
        db_path if db_path is not None else _default_db_path()
    ) as path:
        return _init_beam_locked(path)


def _init_beam_locked(db_path: Path) -> BeamInitResult:
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS working_memory (\n            id TEXT PRIMARY KEY,\n            content TEXT NOT NULL,\n            source TEXT,\n            timestamp TEXT,\n            session_id TEXT DEFAULT 'default',\n            importance REAL DEFAULT 0.5,\n            metadata_json TEXT,\n            veracity TEXT DEFAULT 'unknown',\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_timestamp ON working_memory(timestamp)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_source ON working_memory(source)")
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS episodic_memory (\n            rowid INTEGER PRIMARY KEY AUTOINCREMENT,\n            id TEXT UNIQUE NOT NULL,\n            content TEXT NOT NULL,\n            source TEXT,\n            timestamp TEXT,\n            session_id TEXT DEFAULT 'default',\n            importance REAL DEFAULT 0.5,\n            metadata_json TEXT,\n            summary_of TEXT DEFAULT '',\n            veracity TEXT DEFAULT 'unknown',\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_session ON episodic_memory(session_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_timestamp ON episodic_memory(timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_source ON episodic_memory(source)"
    )
    _add_column_if_missing(conn, "episodic_memory", "tier", "INTEGER DEFAULT 1")
    _add_column_if_missing(conn, "episodic_memory", "degraded_at", "TEXT")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_tier ON episodic_memory(tier)")
    _add_column_if_missing(conn, "working_memory", "veracity", "TEXT DEFAULT 'unknown'")
    _add_column_if_missing(
        conn, "episodic_memory", "veracity", "TEXT DEFAULT 'unknown'"
    )
    _add_column_if_missing(
        conn, "working_memory", "memory_type", "TEXT DEFAULT 'unknown'"
    )
    _add_column_if_missing(
        conn, "episodic_memory", "memory_type", "TEXT DEFAULT 'unknown'"
    )
    _add_column_if_missing(conn, "episodic_memory", "binary_vector", "BLOB")
    _e3_column_added = _add_column_if_missing(
        conn, "working_memory", "consolidated_at", "TEXT"
    )
    if _e3_column_added:
        cursor.execute(
            "UPDATE working_memory SET consolidated_at = ? WHERE consolidated_at IS NULL",
            (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),),
        )
    _add_column_if_missing(conn, "working_memory", "consolidation_claimed_at", "TEXT")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_unconsolidated ON working_memory(session_id, timestamp) WHERE consolidated_at IS NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_consolidation_claims ON working_memory(consolidation_claimed_at) WHERE consolidation_claimed_at IS NOT NULL"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS scratchpad (\n            id TEXT PRIMARY KEY,\n            content TEXT NOT NULL,\n            session_id TEXT DEFAULT 'default',\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sp_session ON scratchpad(session_id)"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memory_events (\n            event_id TEXT PRIMARY KEY,\n            memory_id TEXT NOT NULL,\n            operation TEXT NOT NULL CHECK(operation IN ('CREATE','UPDATE','DELETE','CONSOLIDATE')),\n            timestamp TEXT NOT NULL,\n            device_id TEXT NOT NULL,\n            payload TEXT,\n            parent_event_ids TEXT DEFAULT '[]',\n            importance REAL DEFAULT 0.5,\n            expiry TEXT,\n            event_hash TEXT,\n            synced_at TEXT\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_timestamp ON memory_events(timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_memory_id ON memory_events(memory_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_device_id ON memory_events(device_id)"
    )
    for col, ddl in {
        "event_hash": "event_hash TEXT",
        "synced_at": "synced_at TEXT",
        "parent_event_ids": "parent_event_ids TEXT DEFAULT '[]'",
        "expiry": "expiry TEXT",
    }.items():
        _add_column_if_missing(conn, "memory_events", col, ddl.split(" ", 1)[1])
    vec_dim_mismatch = False
    existing_dim = None
    stored_dims = ()
    cursor.execute(
        "\n        CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes USING fts5(\n            content,\n            content='episodic_memory',\n            content_rowid='rowid'\n        )\n    "
    )
    cursor.execute(
        "\n        CREATE VIRTUAL TABLE IF NOT EXISTS fts_working USING fts5(\n            id UNINDEXED,\n            content\n        )\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS em_ai AFTER INSERT ON episodic_memory BEGIN\n            INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS em_ad AFTER DELETE ON episodic_memory BEGIN\n            INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS em_au AFTER UPDATE ON episodic_memory BEGIN\n            INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);\n            INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS wm_ai AFTER INSERT ON working_memory BEGIN\n            INSERT INTO fts_working(id, content) VALUES (new.id, new.content);\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS wm_ad AFTER DELETE ON working_memory BEGIN\n            DELETE FROM fts_working WHERE id = old.id;\n        END\n    "
    )
    cursor.execute("DROP TRIGGER IF EXISTS wm_au")
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS wm_au AFTER UPDATE OF content ON working_memory BEGIN\n            DELETE FROM fts_working WHERE id = old.id;\n            INSERT INTO fts_working(id, content) VALUES (new.id, new.content);\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memoria_facts (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT DEFAULT 'default',\n            message_idx INTEGER,\n            fact_type TEXT,\n            key TEXT,\n            value TEXT,\n            context_snippet TEXT,\n            importance REAL DEFAULT 0.5,\n            timestamp TEXT,\n            version_id INTEGER DEFAULT 0,\n            previous_value TEXT,\n            updated_msg_idx INTEGER,\n            valid_from_msg_idx INTEGER,\n            valid_to_msg_idx INTEGER,\n            source_memory_id TEXT\n        )\n    "
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON memoria_facts(key)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_type ON memoria_facts(fact_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_session ON memoria_facts(session_id)"
    )
    for col in [
        "version_id",
        "previous_value",
        "updated_msg_idx",
        "valid_from_msg_idx",
        "valid_to_msg_idx",
        "source_memory_id",
    ]:
        _add_column_if_missing(
            conn,
            "memoria_facts",
            col,
            {
                "version_id": "INTEGER DEFAULT 0",
                "previous_value": "TEXT",
                "updated_msg_idx": "INTEGER",
                "valid_from_msg_idx": "INTEGER",
                "valid_to_msg_idx": "INTEGER",
                "source_memory_id": "TEXT",
            }.get(col, "TEXT"),
        )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memoria_timelines (\n            event_id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT DEFAULT 'default',\n            date TEXT,\n            message_idx INTEGER,\n            description TEXT,\n            source TEXT,\n            source_memory_id TEXT\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timelines_date ON memoria_timelines(date)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timelines_session ON memoria_timelines(session_id)"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memoria_instructions (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT DEFAULT 'default',\n            message_idx INTEGER,\n            instruction TEXT,\n            active INTEGER DEFAULT 1,\n            topic TEXT,\n            context_snippet TEXT,\n            source_memory_id TEXT\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_instr_session ON memoria_instructions(session_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_instr_active ON memoria_instructions(active)"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memoria_preferences (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT DEFAULT 'default',\n            message_idx INTEGER,\n            preference TEXT,\n            topic TEXT,\n            evolution TEXT,\n            context_snippet TEXT,\n            source_memory_id TEXT\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_pref_session ON memoria_preferences(session_id)"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memoria_kg (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT DEFAULT 'default',\n            subject TEXT,\n            predicate TEXT,\n            object TEXT,\n            message_idx INTEGER,\n            confidence REAL DEFAULT 0.7,\n            source_memory_id TEXT\n        )\n    "
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_subject ON memoria_kg(subject)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_kg_predicate ON memoria_kg(predicate)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_kg_session ON memoria_kg(session_id)"
    )
    for table in (
        "memoria_timelines",
        "memoria_instructions",
        "memoria_preferences",
        "memoria_kg",
    ):
        _add_column_if_missing(conn, table, "source_memory_id", "TEXT")
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memoria_persona (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT DEFAULT 'default',\n            tier TEXT NOT NULL CHECK(tier IN ('permanent','long_term','working')),\n            topic TEXT NOT NULL,\n            content TEXT NOT NULL,\n            confidence REAL DEFAULT 0.7,\n            source_memory_id TEXT,\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n            last_reinforced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n            reinforcement_count INTEGER DEFAULT 0,\n            promotion_reason TEXT\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_persona_session_tier ON memoria_persona(session_id, tier)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_persona_tier_topic ON memoria_persona(tier, topic)"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS consolidation_log (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            session_id TEXT,\n            items_consolidated INTEGER,\n            summary_preview TEXT,\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n    "
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memory_embeddings (\n            memory_id TEXT PRIMARY KEY,\n            embedding_json TEXT NOT NULL,\n            model TEXT,\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n    "
    )
    conn.commit()
    _add_column_if_missing(conn, "working_memory", "recall_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(
        conn, "working_memory", "last_recalled", "TIMESTAMP DEFAULT NULL"
    )
    _add_column_if_missing(conn, "episodic_memory", "recall_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(
        conn, "episodic_memory", "last_recalled", "TIMESTAMP DEFAULT NULL"
    )
    _add_column_if_missing(conn, "working_memory", "pinned", "INTEGER DEFAULT 0")
    _add_column_if_missing(
        conn, "working_memory", "valid_until", "TIMESTAMP DEFAULT NULL"
    )
    _add_column_if_missing(conn, "working_memory", "superseded_by", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "scope", "TEXT DEFAULT 'global'")
    _add_column_if_missing(
        conn, "episodic_memory", "valid_until", "TIMESTAMP DEFAULT NULL"
    )
    _add_column_if_missing(
        conn, "episodic_memory", "superseded_by", "TEXT DEFAULT NULL"
    )
    _add_column_if_missing(conn, "episodic_memory", "scope", "TEXT DEFAULT 'global'")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_scope_imp\n        ON episodic_memory(scope, importance) WHERE superseded_by IS NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_session_recall\n        ON working_memory(session_id, last_recalled) WHERE valid_until IS NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_context_session\n        ON working_memory(session_id, importance DESC, timestamp DESC)\n        WHERE superseded_by IS NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_context_global\n        ON working_memory(scope, importance DESC, timestamp DESC)\n        WHERE superseded_by IS NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_mem_emb_type\n        ON memory_embeddings(memory_id, model)"
    )
    _add_column_if_missing(conn, "working_memory", "author_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "author_type", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "channel_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "author_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "author_type", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "channel_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(
        conn, "working_memory", "trust_tier", "TEXT DEFAULT 'STATED'"
    )
    _add_column_if_missing(
        conn, "episodic_memory", "trust_tier", "TEXT DEFAULT 'STATED'"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_author ON working_memory(author_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_channel ON working_memory(channel_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_author ON episodic_memory(author_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_channel ON episodic_memory(channel_id)"
    )
    _add_column_if_missing(conn, "working_memory", "validator", "TEXT DEFAULT NULL")
    _add_column_if_missing(
        conn, "working_memory", "validated_at", "TIMESTAMP DEFAULT NULL"
    )
    _add_column_if_missing(
        conn, "working_memory", "validation_count", "INTEGER DEFAULT 0"
    )
    _add_column_if_missing(conn, "episodic_memory", "validator", "TEXT DEFAULT NULL")
    _add_column_if_missing(
        conn, "episodic_memory", "validated_at", "TIMESTAMP DEFAULT NULL"
    )
    _add_column_if_missing(
        conn, "episodic_memory", "validation_count", "INTEGER DEFAULT 0"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_validator ON working_memory(validator)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_validated_at ON working_memory(validated_at)"
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS memory_validations (\n            validation_id INTEGER PRIMARY KEY AUTOINCREMENT,\n            memory_id TEXT NOT NULL,\n            validator TEXT NOT NULL,\n            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n            action TEXT NOT NULL,\n            new_content TEXT,\n            note TEXT\n        )\n    "
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_validations_memory ON memory_validations(memory_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_validations_validator ON memory_validations(validator)"
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS trim_validations_to_3\n        AFTER INSERT ON memory_validations\n        BEGIN\n            DELETE FROM memory_validations\n            WHERE memory_id = NEW.memory_id\n              AND validation_id NOT IN (\n                  SELECT validation_id FROM memory_validations\n                  WHERE memory_id = NEW.memory_id\n                  ORDER BY validation_id DESC\n                  LIMIT 3\n              );\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TABLE IF NOT EXISTS facts (\n            fact_id TEXT PRIMARY KEY,\n            session_id TEXT NOT NULL,\n            subject TEXT NOT NULL,\n            predicate TEXT NOT NULL,\n            object TEXT NOT NULL,\n            timestamp TEXT,\n            source_msg_id TEXT,\n            confidence REAL DEFAULT 1.0,\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n    "
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_msg_id)"
    )
    cursor.execute(
        "\n        CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(\n            subject, predicate, object, content='facts'\n        )\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN\n            INSERT INTO fts_facts(rowid, subject, predicate, object)\n            VALUES (new.rowid, new.subject, new.predicate, new.object);\n        END\n    "
    )
    cursor.execute(
        "\n        CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN\n            INSERT INTO fts_facts(fts_facts, rowid, subject, predicate, object)\n            VALUES ('delete', old.rowid, old.subject, old.predicate, old.object);\n        END\n    "
    )
    _add_column_if_missing(conn, "working_memory", "event_date", "TEXT DEFAULT NULL")
    _add_column_if_missing(
        conn, "working_memory", "event_date_precision", "TEXT DEFAULT 'unknown'"
    )
    _add_column_if_missing(conn, "working_memory", "temporal_tags", "TEXT DEFAULT '[]'")
    _add_column_if_missing(
        conn, "working_memory", "corrected_by", "INTEGER DEFAULT NULL"
    )
    _add_column_if_missing(conn, "episodic_memory", "event_date", "TEXT DEFAULT NULL")
    _add_column_if_missing(
        conn, "episodic_memory", "event_date_precision", "TEXT DEFAULT 'unknown'"
    )
    _add_column_if_missing(
        conn, "episodic_memory", "temporal_tags", "TEXT DEFAULT '[]'"
    )
    _add_column_if_missing(
        conn, "episodic_memory", "corrected_by", "INTEGER DEFAULT NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_event_date ON working_memory(event_date)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_em_event_date ON episodic_memory(event_date)"
    )
    return BeamInitResult(
        vec_dim_mismatch=vec_dim_mismatch,
        existing_dim=existing_dim,
        configured_dim=0,
        stored_dims=stored_dims,
    )


class _BeamConnection(sqlite3.Connection):
    """sqlite3.Connection subclass that supports deferring commits.

    Used by BeamMemory so `remember_batch`'s enrichment loop can wrap
    many sub-helper commits in a single transaction. The substores
    (AnnotationStore, EpisodicGraph, VeracityConsolidator) each call
    `self.conn.commit()` after their per-row writes; pre-E2-hardening
    that produced 10-15 commits per batch row × 250K rows = millions
    of fsync round-trips. /review army (4-source CRITICAL on commit 1)
    estimated 3-10 hours wall clock for the BEAM-recovery benchmark.

    When `_defer_commit` is True, `commit()` becomes a no-op. The
    `_deferred_commits` context manager flips the flag, runs the
    block, then calls `_real_commit()` once at the end (or rolls back
    on exception).

    Subclassing is required because `sqlite3.Connection.commit` is a
    read-only C-level method -- monkey-patching it raises
    `AttributeError`. The factory= parameter on `sqlite3.connect` is
    the supported integration point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._defer_commit = False
        self._savepoint_counter = 0
        self._vec_working_count_cache: Optional[Tuple[int, int, int]] = None

    def _next_savepoint_name(self, purpose: str) -> str:
        """Return a connection-local, SQLite-safe savepoint identifier."""
        self._savepoint_counter += 1
        return f"mnemosyne_{purpose}_{self._savepoint_counter}"

    def commit(self) -> None:
        if self._defer_commit:
            return
        super().commit()

    def _real_commit(self) -> None:
        """Force a real commit regardless of the defer flag.
        Used by `_deferred_commits` on successful exit."""
        super().commit()


@contextlib.contextmanager
def _guarded_transaction(conn: sqlite3.Connection):
    """Run the enclosed statements as one transaction, rolling back on ANY
    failure before re-raising.

    Centralizes the guarded commit/rollback pattern shared by
    ``get_context()``'s recall touch and ``forget_working()``'s cascade
    delete. Left unguarded, a failure mid-transaction (for example
    "database is locked" while a consolidation pass is writing) abandons
    the long-lived thread-local connection inside an open, stale
    transaction, and every later write on that thread fails
    "database is locked" instantly (stale-snapshot upgrade; the busy
    handler is not consulted) until something resets it. The rollback is
    itself guarded so a dead connection cannot mask the original
    exception.

    For an already-open ``_BeamConnection`` transaction, guarded operations
    take a local savepoint instead: a connection-wide commit or rollback would
    otherwise steal or erase caller-owned writes.
    """
    savepoint = None
    if isinstance(conn, _BeamConnection) and conn.in_transaction:
        savepoint = conn._next_savepoint_name("guarded_transaction")
        conn.execute(f"SAVEPOINT {savepoint}")
    try:
        yield
        if savepoint is None:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        try:
            if savepoint is None:
                conn.rollback()
            else:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except sqlite3.Error:
            pass
        raise


@contextlib.contextmanager
def _deferred_commits(conn: sqlite3.Connection):
    """Defer nested commits without stealing a caller-owned transaction.

    A BEAM-owned batch starts and commits its own transaction.  When a caller
    already has a transaction open, the batch is isolated in a savepoint: it
    releases on success and rolls back only its own writes on failure.  In
    particular, neither path may commit or roll back the caller's marker rows.
    """
    if not isinstance(conn, _BeamConnection):
        yield
        return
    owns_transaction = not conn.in_transaction
    savepoint = conn._next_savepoint_name("deferred_commits")
    previously_deferred = conn._defer_commit
    if owns_transaction:
        conn.execute("BEGIN")
    else:
        conn.execute(f"SAVEPOINT {savepoint}")
    conn._defer_commit = True
    try:
        yield
    except Exception:
        conn._defer_commit = previously_deferred
        try:
            if owns_transaction:
                conn.rollback()
            else:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except sqlite3.Error:
            pass
        raise
    else:
        conn._defer_commit = previously_deferred
        try:
            if owns_transaction:
                conn._real_commit()
            else:
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except sqlite3.Error as exc:
            logger.error("_deferred_commits: finalization failed: %s", exc)
            try:
                if owns_transaction:
                    conn.rollback()
                else:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            except sqlite3.Error:
                pass
            raise
    finally:
        conn._defer_commit = previously_deferred


def _generate_id(content: str) -> str:
    return hashlib.sha256(
        f"{content}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]


def _sanitize_utf8(text: str) -> str:
    """Ensure text is valid UTF-8, stripping or replacing invalid bytes.

    SQLite TEXT columns enforce UTF-8 encoding. Corrupt bytes (e.g. 0xFE,
    0xFF, or truncated multi-byte sequences from LLM output / AAAK
    compression / buffer errors) cause OperationalError on subsequent reads.
    This function sanitizes content at the write boundary so a single bad
    write doesn't poison episodic_memory and crash future maintenance passes
    (sleep_all_sessions, degrade_episodic).
    """
    if not isinstance(text, str):
        return ""
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, col_type: str
):
    """Add a column, then verify type/default when it already exists.

    Tolerates a concurrent winner: if ALTER raises 'duplicate column name'
    for the exact requested column because another initializer created it
    between our read and write, reread the schema and verify it matches the
    expected declaration exactly before suppressing.
    """
    cursor = conn.cursor()

    def _expected_pieces():
        expected_type, _, expected_default = col_type.partition(" DEFAULT ")
        expected_notnull = " NOT NULL" in expected_type.upper()
        expected_type = expected_type.replace(" NOT NULL", "").strip()
        return (expected_type, expected_notnull, expected_default.strip())

    def _column_matches(rows, strict_default=False):
        matches = [r for r in rows if len(r) >= 5 and r[1] == column]
        if len(matches) != 1:
            return False
        row = matches[0]
        expected_type, expected_notnull, expected_default = _expected_pieces()
        actual_type = row[2].strip().upper()
        expected_type = expected_type.strip().upper()
        if actual_type != expected_type and {actual_type, expected_type} != {
            "TEXT",
            "TIMESTAMP",
        }:
            return False
        if bool(row[3]) != expected_notnull:
            return False
        actual_default = row[4].strip() if row[4] is not None else None
        if expected_default:
            return not strict_default or actual_default == expected_default
        return actual_default is None

    cursor.execute(f"PRAGMA table_info({table})")
    rows = cursor.fetchall()
    cols = {r[1] for r in rows}
    if column not in cols:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()
            return True
        except sqlite3.OperationalError as e:
            msg = str(e)
            if column not in msg or "duplicate column" not in msg.lower():
                raise
            cursor.execute(f"PRAGMA table_info({table})")
            if not _column_matches(cursor.fetchall(), strict_default=True):
                raise
            return False
    cursor.execute(f"PRAGMA table_info({table})")
    rows = cursor.fetchall()
    if not _column_matches(rows):
        actual = next((r for r in rows if len(r) >= 2 and r[1] == column), None)
        raise sqlite3.OperationalError(
            f"schema mismatch for {table}.{column}: expected {col_type}, got {(actual[2] if actual and len(actual) >= 3 else '?')} DEFAULT {(actual[4] if actual and len(actual) >= 5 else '?')}"
        )
    return False


@dataclass(frozen=True)
class _RecallWeightSnapshot:
    """The normalized scoring weights fixed for one recall request."""

    vec: float
    fts: float
    importance: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.vec, self.fts, self.importance)


_DEFAULT_RECALL_WEIGHTS = (0.5, 0.3, 0.2)


def _normalize_recall_weight_values(
    vec_weight: Any, fts_weight: Any, importance_weight: Any
) -> _RecallWeightSnapshot:
    """Normalize one finite request snapshot, or return the safe defaults.

    Compatibility policy: ordinary negative values remain clamped and an
    all-zero triplet still falls back to the historical defaults.  If any
    resolved component is NaN or infinite (or normalization overflows), use
    ``(0.5, 0.3, 0.2)`` for the whole request rather than letting a partial
    value poison scoring or enhanced-cache material.
    """
    try:
        values = (float(vec_weight), float(fts_weight), float(importance_weight))
    except (TypeError, ValueError, OverflowError):
        return _RecallWeightSnapshot(*_DEFAULT_RECALL_WEIGHTS)
    if not all((math.isfinite(value) for value in values)):
        return _RecallWeightSnapshot(*_DEFAULT_RECALL_WEIGHTS)
    clamped = tuple((max(0.0, value) for value in values))
    total = sum(clamped)
    if total == 0.0 or not math.isfinite(total):
        return _RecallWeightSnapshot(*_DEFAULT_RECALL_WEIGHTS)
    normalized = tuple((value / total for value in clamped))
    if not all((math.isfinite(value) for value in normalized)):
        return _RecallWeightSnapshot(*_DEFAULT_RECALL_WEIGHTS)
    return _RecallWeightSnapshot(*normalized)


def _resolve_recall_weights(
    vec_weight: Optional[float],
    fts_weight: Optional[float],
    importance_weight: Optional[float],
) -> _RecallWeightSnapshot:
    """Resolve one finite atomic snapshot with config.yaml > env > defaults."""
    from mnemosyne.core.config import get_config

    config = get_config()
    configured = config.get_many(
        {
            "vec_weight": _DEFAULT_RECALL_WEIGHTS[0],
            "fts_weight": _DEFAULT_RECALL_WEIGHTS[1],
            "importance_weight": _DEFAULT_RECALL_WEIGHTS[2],
        }
    )
    return _normalize_recall_weight_values(
        vec_weight if vec_weight is not None else configured["vec_weight"],
        fts_weight if fts_weight is not None else configured["fts_weight"],
        importance_weight
        if importance_weight is not None
        else configured["importance_weight"],
    )


def _normalize_weights(
    vec_weight: Optional[float],
    fts_weight: Optional[float],
    importance_weight: Optional[float],
) -> tuple[float, float, float]:
    """
    Normalize hybrid scoring weights to sum to 1.0.

    Falls back to env vars, then defaults:
        vec_weight      -> MNEMOSYNE_VEC_WEIGHT      -> 0.5
        fts_weight      -> MNEMOSYNE_FTS_WEIGHT      -> 0.3
        importance_weight -> MNEMOSYNE_IMPORTANCE_WEIGHT -> 0.2

    After normalization: vw + fw + iw == 1.0
    """
    vw = (
        vec_weight
        if vec_weight is not None
        else os.environ.get("MNEMOSYNE_VEC_WEIGHT", "0.5")
    )
    fw = (
        fts_weight
        if fts_weight is not None
        else os.environ.get("MNEMOSYNE_FTS_WEIGHT", "0.3")
    )
    iw = (
        importance_weight
        if importance_weight is not None
        else os.environ.get("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2")
    )
    return _normalize_recall_weight_values(vw, fw, iw).as_tuple()


def _normalize_datetime_utc(dt: datetime) -> datetime:
    """Return dt as a timezone-aware UTC datetime.

    Naive datetimes are treated as UTC to preserve existing naive timestamp
    behavior while avoiding naive/aware comparison crashes.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_datetime_utc(value: str) -> datetime:
    """Parse an ISO datetime string and normalize it to UTC."""
    return _normalize_datetime_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


_EVENT_DATE_PRECISIONS = {"day", "month", "week", "year", "unknown"}


def _utc_cutoff_sql(cutoff: str) -> str:
    """Normalize a cutoff timestamp for the chronological SQL predicates.

    The SQL side compares ``COALESCE(datetime(timestamp), epoch)`` against
    this value as a plain string, so it must arrive in exactly the shape
    ``datetime()`` emits: naive UTC ``YYYY-MM-DD HH:MM:SS``. Normalizing in
    Python (never via SQL ``datetime(?)``) also sidesteps a SQLite version
    dependence: boundary values like ``9999-12-31T23:59:59.999999`` (the
    force-consolidation sentinel) overflow the julian-day range under
    SQL-side parsing on some SQLite builds and come back NULL, which would
    silently make every row ineligible. Unparseable cutoffs fall back to the
    raw string (lexicographic, the pre-chronology behavior).
    """
    try:
        dt = _parse_iso_datetime_utc(cutoff)
        return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OverflowError):
        return cutoff


_SQL_PLACEABLE_TS = "(lower(trim(timestamp)) IN ('now', 'subsec') OR (datetime(trim(timestamp)) IS NOT NULL AND date(substr(trim(timestamp), 1, 10)) = substr(trim(timestamp), 1, 10)))"
_SQL_CHRONO_TS = "CASE WHEN lower(trim(timestamp)) IN ('now', 'subsec') OR datetime(trim(timestamp)) IS NULL THEN '1970-01-01 00:00:00' ELSE datetime(trim(timestamp)) END"


def _latest_iso_string(values, *, normalized: bool = False) -> "Optional[str]":
    """Return the chronologically latest ISO timestamp string from an iterable.

    Each value is parsed with _parse_iso_datetime_utc (naive values treated
    as UTC; mixed offsets normalized), so selection is by instant, not by
    lexicographic order of the serialized forms. Invalid/empty values are
    skipped with a warning (they cannot influence selection, but silently
    dropping a newer row would stamp the summary with an older instant).

    With ``normalized=False`` the ORIGINAL string of the winner is returned.
    With ``normalized=True`` the winner is returned as a NAIVE UTC ISO string
    (offset stripped after conversion) — REQUIRED for storage in
    ``episodic_memory.timestamp``, because recall's date filters compare
    that column lexicographically: an offset-bearing string sorts wrong on
    both boundaries of its day, and even a "+00:00" suffix breaks parity
    with the naive forms every other producer writes.

    Producer contract (post round-4): in-repo producers stamp
    datetime.now(timezone.utc).replace(tzinfo=None) — naive-UTC — so new
    rows compare exactly against these cutoffs on any host. Rows written
    by pre-fix producers are naive-LOCAL wall time: they skew by the
    writing host's offset (up to ±14h, DST makes it offset±1), bounded
    but not corrected — no data migration. Operators running non-UTC
    hosts should expect trim/consolidation windows on legacy rows to be
    off by the host offset in the direction of the offset's sign.
    """
    latest_dt = None
    latest_raw = None
    for raw in values:
        if not raw:
            continue
        try:
            dt = _parse_iso_datetime_utc(str(raw))
        except (OverflowError, TypeError, ValueError):
            logger.warning(
                "sleep: skipping unparseable source timestamp %r during latest-instant selection",
                raw,
            )
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_raw = dt.replace(tzinfo=None).isoformat() if normalized else raw
    return latest_raw


_DATE_ONLY_RE = re.compile("^\\d{4}-\\d{2}-\\d{2}$")


def _normalize_valid_until(value: Optional[str]) -> Optional[str]:
    """Canonicalize a caller-supplied ``valid_until`` to aware UTC ISO.

    Offset-bearing ISO timestamps are converted to UTC so comparisons
    against aware-UTC now stay chronologically correct (e.g.
    ``2026-08-15T11:00:00-02:00`` is 13:00Z but sorts before
    ``12:30:00+00:00`` lexically). Only an exact date-only value
    (``YYYY-MM-DD``) keeps its documented pass-through API semantics;
    everything else is parsed chronologically, so a lowercase ``t``
    separator or ``Z`` suffix is normalized too. Unparseable values pass
    through unchanged.
    """
    if value is None:
        return value
    if not isinstance(value, str):
        return None
    if not value:
        return value
    if _DATE_ONLY_RE.match(value):
        return value
    try:
        return _parse_iso_datetime_utc(value).isoformat()
    except (OverflowError, ValueError, TypeError):
        return value


def _valid_until_active(valid_until: str, now_iso: str) -> bool:
    """True when ``valid_until`` is strictly in the future of ``now_iso``.

    Both operands are parsed chronologically so offset-bearing stored
    values (e.g. ``2026-08-15T11:00:00-02:00`` = 13:00Z) are not
    misjudged by lexical ordering against aware-UTC now. Naive values
    are treated as UTC, matching ``_normalize_datetime_utc``. An
    unparseable value is treated as expired (False), matching the
    julianday-based SQL predicate where it evaluates to NULL and is
    excluded.
    """
    try:
        return _parse_iso_datetime_utc(valid_until) > _parse_iso_datetime_utc(now_iso)
    except (OverflowError, ValueError, TypeError):
        return False


def _recency_decay(
    timestamp_str: str, halflife_hours: float = RECENCY_HALFLIFE_HOURS
) -> float:
    """Calculate recency decay factor. 1.0 = brand new, ~0.5 = one halflife old.

    Exponential decay based on age. Returns 0.5 for unknown/invalid timestamps.
    """
    if not timestamp_str:
        return 0.5
    try:
        ts = _parse_iso_datetime_utc(timestamp_str)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return math.exp(-age_hours / halflife_hours)
    except Exception:
        return 0.5


def _parse_query_time(query_time: Optional[Union[str, datetime]]) -> datetime:
    """Parse query_time parameter into a timezone-aware UTC datetime object.

    - None (or a blank/whitespace-only string) -> current UTC time
    - str  -> parsed from ISO format and normalized to UTC
    - datetime -> normalized to UTC
    Naive values are treated as UTC for backward compatibility.

    A blank string is treated as "unset" for compatibility with callers that
    send ``""`` to mean "omitted" — notably MCP harnesses built against the
    older ``mnemosyne_recall`` schema, which declared ``query_time`` with
    ``"default": ""`` (see #555). Non-string falsey values are still rejected.
    """
    if query_time is None:
        return datetime.now(timezone.utc)
    if isinstance(query_time, str) and (not query_time.strip()):
        return datetime.now(timezone.utc)
    if isinstance(query_time, datetime):
        return _normalize_datetime_utc(query_time)
    if isinstance(query_time, str):
        try:
            return _parse_iso_datetime_utc(query_time)
        except ValueError:
            try:
                return _parse_iso_datetime_utc(f"{query_time}T00:00:00")
            except ValueError:
                raise ValueError(
                    f"Invalid query_time format: {query_time!r}. Expected ISO datetime string."
                )
    raise TypeError(
        f"query_time must be str, datetime, or None; got {type(query_time).__name__}"
    )


_TS_CACHE: Dict[str, datetime] = {}
_TS_CACHE_MAX = 2000


def _parse_ts_fast(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp with LRU-style cache for performance."""
    if not ts:
        return None
    cached = _TS_CACHE.get(ts)
    if cached is not None:
        return cached
    try:
        dt = _parse_iso_datetime_utc(ts)
    except (OverflowError, ValueError, TypeError):
        return None
    if len(_TS_CACHE) >= _TS_CACHE_MAX:
        _TS_CACHE.clear()
    _TS_CACHE[ts] = dt
    return dt


def _temporal_boost(
    memory_timestamp_str: str, query_time: datetime, halflife_hours: float = 24.0
) -> float:
    """Temporal boost factor based on proximity to query_time.

    Formula: exp(-hours_delta / halflife)
    - memory at query_time -> boost = 1.0
    - memory 1 halflife away -> boost = exp(-1) ≈ 0.368
    - memory 3 halflives away -> boost = exp(-3) ≈ 0.050

    Returns 0.0 for invalid timestamps or future timestamps (clamped to now).
    """
    ts = _parse_ts_fast(memory_timestamp_str)
    if ts is None:
        return 0.0
    query_time = _normalize_datetime_utc(query_time)
    if ts > query_time:
        ts = query_time
    hours_delta = (query_time - ts).total_seconds() / 3600.0
    return math.exp(-hours_delta / halflife_hours)


def _resolve_temporal_halflife(value: Any) -> float:
    """Return the finite, positive temporal half-life used by recall."""
    raw_value = (
        os.environ.get("MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS", "24")
        if value is None
        else value
    )
    try:
        resolved = float(raw_value)
    except (TypeError, ValueError, OverflowError):
        return 24.0
    return resolved if math.isfinite(resolved) and resolved > 0 else 24.0


def _vec_available(conn: sqlite3.Connection) -> bool:
    return False


def _wm_vec_available(conn: sqlite3.Connection) -> bool:
    return False


def _extract_and_store_entities(beam: "BeamMemory", memory_id: str, content: str):
    """
    Extract entities from content and store as annotations (post-E6).
    Called internally by remember() when extract_entities=True.

    Pre-E6 wrote to TripleStore with predicate="mentions", which silently
    invalidated prior mentions on the same memory via auto-invalidation
    on (subject, predicate). Post-E6, writes go to AnnotationStore where
    multiple mentions per memory coexist.
    """
    try:
        from mnemosyne.core.entities import extract_entities_regex

        entities = extract_entities_regex(content)
        if not entities:
            return
        beam.annotations.add_many(
            memory_id=memory_id,
            kind="mentions",
            values=entities,
            source="regex",
            confidence=0.8,
        )
    except Exception:
        pass


def _extract_and_store_facts(
    beam: "BeamMemory", memory_id: str, content: str, source: str = ""
):
    """
    Extract structured facts from content using LLM and store as annotations
    + facts table. Called internally by remember() when extract=True.

    Stores in TWO places:
    1. AnnotationStore with kind="fact" (post-E6; was TripleStore pre-E6)
    2. facts table (structured SPO facts for fact_recall())

    Post-E6 note: writes formerly used TripleStore.add_facts() which
    silently invalidated each prior fact via (subject, predicate) auto-
    invalidation. AnnotationStore.add_many is append-only so all facts
    coexist.
    """
    try:
        from mnemosyne.core.extraction import extract_facts_safe
        from mnemosyne.core.annotations import filter_facts

        facts = extract_facts_safe(content)
        if not facts:
            return
        kept = filter_facts(facts)
        if kept:
            beam.annotations.add_many(
                memory_id=memory_id,
                kind="fact",
                values=kept,
                source=source,
                confidence=0.7,
            )
        _store_facts_in_table(beam, memory_id, content, source, facts)
    except Exception:
        pass


def _store_facts_in_table(
    beam: "BeamMemory", memory_id: str, content: str, source: str, facts: list
):
    """Store extracted free-text facts as simple SPO entries in the facts table."""
    import hashlib

    cursor = beam.conn.cursor()
    timestamp = __import__("datetime").datetime.now().isoformat()
    for i, fact_text in enumerate(facts):
        subject = source or "user"
        fact_id = hashlib.sha256(
            f"{memory_id}:fact:{i}:{fact_text[:50]}".encode()
        ).hexdigest()[:24]
        try:
            cursor.execute(
                "\n                INSERT OR IGNORE INTO facts\n                (fact_id, session_id, subject, predicate, object,\n                 timestamp, source_msg_id, confidence)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    fact_id,
                    beam.session_id,
                    subject,
                    "stated",
                    fact_text,
                    timestamp,
                    memory_id,
                    0.7,
                ),
            )
        except Exception:
            continue
    beam.conn.commit()


def _find_memories_by_entity(
    beam: "BeamMemory", entity_name: str, threshold: float = 0.8
) -> List[str]:
    """
    Find memory IDs that mention an entity (or similar entity via fuzzy match).
    Returns list of memory_id strings.

    Post-E6: reads from AnnotationStore. Memories with multiple mentions
    now all surface (silent-destruction bug fixed) -- the pre-E6 path
    against TripleStore returned only the last-written mention per memory
    because of auto-invalidation on (subject, predicate).
    """
    try:
        from mnemosyne.core.entities import find_similar_entities

        known_entities = beam.annotations.get_distinct_values("mentions")
        if not known_entities:
            return []
        matches = find_similar_entities(
            entity_name, known_entities, threshold=threshold
        )
        memory_ids: Set[str] = set()
        for matched_entity, _ in matches:
            results = beam.annotations.query_by_kind("mentions", value=matched_entity)
            for row in results:
                memory_ids.add(row["memory_id"])
        return list(memory_ids)
    except Exception:
        return []


_FACT_MATCH_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "related",
    "should",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "totally",
    "unrelated",
    "use",
    "uses",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "again",
    "into",
    "not",
    "please",
    "somewhere",
    "supposed",
    "them",
    "then",
    "they",
    "whatever",
    "ask",
    "asking",
    "query",
    "question",
    "questions",
    "search",
    "retrieve",
    "recall",
}
_extra_recall_stopwords = {
    token.strip().lower()
    for token in re.split(
        "[,\\s]+", os.environ.get("MNEMOSYNE_RECALL_EXTRA_STOPWORDS", "")
    )
    if len(token.strip()) >= 3
}
_FACT_MATCH_STOPWORDS.update(_extra_recall_stopwords)
_RECALL_TOKEN_RE = re.compile("(?u)[^\\W_][\\w]*(?:[_.:/+-]+[^\\W_][\\w]*)*")
_RECALL_SYNONYMS: Dict[str, tuple[str, ...]] = {
    "branding": ("brand", "positioning", "identity", "wording"),
    "preference": (
        "prefer",
        "prefers",
        "want",
        "wants",
        "reject",
        "rejects",
        "avoid",
        "grounded",
    ),
    "professional": ("software", "builder"),
    "url": ("link", "profile"),
    "current": ("now", "live", "latest"),
    "feeling": ("feel", "feels"),
    "imposter": ("self-doubt", "doubt", "insecure"),
}


def _is_meaningful_recall_token(token: str) -> bool:
    """Return whether a token is eligible for lexical recall matching.

    Non-Hangul tokens need >= 3 characters: a shorter English token is almost
    always a function word. Hangul is the opposite — a two-syllable 어절 is a
    complete, high-information noun (백업, 캐시, 포트), so the English rule
    discards exactly the terms a Korean user searches by.
    """
    minimum = 2 if _has_hangul(token) else 3
    return (
        len(token) >= minimum
        and token not in _FACT_MATCH_STOPWORDS
        and (not token.isdigit())
    )


from mnemosyne.core.verbatim_ledger import (
    ExclusionSnapshot,
    exclusion_sql,
    resolve_exclusions,
)


def _recall_tokens(text: str) -> List[str]:
    """Meaningful lexical tokens for precision gates and fallback scoring.

    Hangul tokens are particle-stripped here, not at the call sites, so that
    candidate generation and final admission share one normalization contract.
    unicode61 indexes Hangul at whitespace granularity, so a particle stays
    glued to its stem; normalizing only the query side lets ``_fts_search()``
    return a row that ``recall()`` then scores at zero.
    """
    tokens: List[str] = []
    for token in _RECALL_TOKEN_RE.findall(text.lower()):
        if not _is_meaningful_recall_token(token):
            continue
        tokens.append(_strip_ko_josa(token) if _has_hangul(token) else token)
    return tokens


def _hyphen_components(token: str) -> List[str]:
    """Return unique hyphen components eligible for lexical recall matching."""
    if "-" not in token:
        return []
    return list(
        dict.fromkeys(
            (part for part in token.split("-") if _is_meaningful_recall_token(part))
        )
    )


_HYPHEN_FRAGMENT_RE = re.compile(
    "(?u)(?<![-\\w])-+[^\\W_][\\w]*(?:-[^\\W_][\\w]*)*(?![-\\w])"
)


def _hyphen_fragment_tokens(text: str) -> List[str]:
    """Return unique components of leading-hyphen fragments in ``text``.

    ``rm -rf`` yields ``['rf']``, ``--force`` yields ``['force']`` and
    ``python -v`` yields ``['v']``. One-character components are kept when
    they are not stopwords or digits so single-letter flags stay recallable.
    Fragments embedded in a word (``git-rebase``) are already covered by
    ``_recall_tokens()`` and are intentionally not matched here.
    """
    tokens: List[str] = []
    for fragment in _HYPHEN_FRAGMENT_RE.findall(text.lower()):
        for part in fragment.split("-"):
            if (
                len(part) >= 1
                and part not in _FACT_MATCH_STOPWORDS
                and (not part.isdigit())
            ):
                tokens.append(part)
    return list(dict.fromkeys(tokens))


def _leading_hyphen_fragments(text: str) -> List[str]:
    """Return unique literal leading-hyphen fragment forms in ``text``.

    ``rm -rf`` yields ``['-rf']``, ``--force`` yields ``['--force']`` and
    ``python -v`` yields ``['-v']``. Unlike ``_hyphen_fragment_tokens()``
    which strips the hyphens into bare components, the literal form is kept
    intact so a query for the CLI flag ``--force`` can be distinguished from
    an ordinary occurrence of the word ``force``. Fragments embedded in a
    word (``git-rebase``, ``git--rebase``) are intentionally not matched.
    """
    return list(dict.fromkeys(_HYPHEN_FRAGMENT_RE.findall(text.lower())))


def _literal_flag_bonus(query_lower: str, content: str) -> float:
    """Precision premium for a literal leading-hyphen flag inside ``content``.

    ``_lexical_relevance()`` already scores an exact ``--force`` match above a
    bare ``force`` occurrence, but both live in ``[0, 1]`` and the final rank
    blends keyword relevance with ``importance``. Without a dedicated signal a
    high-importance row that merely uses the word "force" can otherwise rank
    too closely to a low-importance row that literally contains the flag. This
    additive premium is applied where the other recall bonuses live
    (recency/current-state, graph/fact/binary); the final linear-recall
    selection separately rejects bare-component collisions so the precision
    contract does not depend on this fixed amount. The premium requires an
    exact extracted token match on the content side too: ``--force`` never
    matches ``--forceful`` or ``foo--force``. Fragments embedded in a word
    (``git-rebase``) never match, exactly like ``_leading_hyphen_fragments()``.
    """
    if not query_lower or not content:
        return 0.0
    literals = _leading_hyphen_fragments(query_lower)
    if not literals:
        return 0.0
    content_literals = set(_leading_hyphen_fragments(content.lower()))
    return 0.3 if set(literals) & content_literals else 0.0


_SYMBOLIC_CODE_RE = re.compile(
    "(?u)(?<![A-Za-z0-9_])[A-Za-z]+(?:\\+\\+|#+)[0-9]*(?![A-Za-z0-9_])"
)


def _symbolic_code_tokens(text: str) -> List[str]:
    """Return unique symbolic code tokens in ``text``.

    ``C++`` yields ``['c++']``, ``c#`` yields ``['c#']`` and ``g++`` yields
    ``['g++']``. ``a+b`` (arithmetic) is rejected because a single ``+`` is
    not a symbolic code name; ``git-rebase`` and ``node_modules`` contain
    neither ``++`` nor ``#`` and are intentionally not matched here.
    """
    return list(dict.fromkeys(_SYMBOLIC_CODE_RE.findall(text.lower())))


def _component_unit_weight(components: List[str]) -> int:
    """Return the lexical-unit weight for a token's hyphen components."""
    return len(components) if len(components) >= 2 else 1


def _expand_hyphenated_tokens(tokens: List[str]) -> List[str]:
    """Keep hyphenated tokens and add their meaningful components.

    The full compound remains available for precise matches. Components make
    differently-hyphenated forms such as ``orion-telemetrie`` and
    ``orion-gateway ... telemetrie`` comparable at the lexical gate.
    Other structured separators (paths, versions, identifiers) stay intact.
    """
    expanded: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            expanded.append(token)
        for candidate in _hyphen_components(token):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _is_bare_literal_flag_collision(query: str, content: str) -> bool:
    """Return whether ``content`` only has a bare component of a query flag.

    A literal query such as ``--force`` must not degrade into an ordinary
    search for the word ``force`` when configurable weights favor importance.
    Exact literal matches remain eligible. Prefixes and embedded forms such as
    ``--forceful`` and ``foo--force`` are not exact literals; only the latter
    is a bare-component collision because it still exposes the word ``force``.
    """
    if not query or not content:
        return False
    query_literals = set(_leading_hyphen_fragments(query.lower()))
    if not query_literals:
        return False
    content_lower = content.lower()
    missing_literals = query_literals - set(_leading_hyphen_fragments(content_lower))
    if not missing_literals:
        return False
    bare_components = {
        component
        for literal in missing_literals
        for component in _hyphen_fragment_tokens(literal)
    }
    content_components = set(re.findall("(?u)[^\\W_]+", content_lower))
    return bool(bare_components & content_components)


def _expanded_query_tokens(tokens: List[str]) -> List[str]:
    """Return query tokens plus a bounded synonym expansion.

    Expansion is query-side only and de-duplicated in order. It broadens FTS
    candidate generation without lowering the lexical abstention gate.
    """
    expanded: List[str] = []
    seen: Set[str] = set()
    for token in _expand_hyphenated_tokens(tokens):
        for candidate in (token, *_RECALL_SYNONYMS.get(token, ())):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _minimum_recall_relevance(query_tokens: List[str]) -> float:
    """Raise the lexical gate for broad natural-language queries.

    One matching real word is enough for short lookup-style queries, but not
    for broad nonsense strings like "purple bicycle quantum oatmeal".

    ``MNEMOSYNE_LEXICAL_GATE_MIN`` (float 0.0–1.0) overrides the gate entirely,
    defaulting to the historical thresholds when unset. Setting it to 0.0 admits
    purely-vector candidates (recall-first); the default keeps today's behaviour
    so existing users are unaffected unless they opt in. The env is read on every
    call so operators can tune it without restarting.
    """
    env = os.environ.get("MNEMOSYNE_LEXICAL_GATE_MIN")
    if env is not None:
        try:
            value = float(env)
        except ValueError:
            pass
        else:
            if math.isfinite(value):
                return min(max(value, 0.0), 1.0)
    if len(query_tokens) >= 4:
        return 0.3
    if len(query_tokens) == 3:
        return 0.5
    return 0.15


_CURRENT_STATE_QUERY_TOKENS = {"current", "currently", "latest", "now", "newer"}
_CURRENT_STATE_CONTENT_TOKENS = {"current", "currently", "latest", "newer", "updated"}


def _current_state_recency_bonus(query_tokens: List[str], content: str) -> float:
    """Small tie-breaker for "what is true now/currently" queries.

    Hybrid lexical scoring can leave stale and corrected facts nearly tied when
    both share the same subject/action terms. Only add a bounded bonus when the
    query asks for current state and the row itself carries an explicit
    current-state marker; this keeps ordinary recall ranking unchanged.
    """
    if not _CURRENT_STATE_QUERY_TOKENS & set(query_tokens):
        return 0.0
    content_tokens = set(_recall_tokens(content))
    if _CURRENT_STATE_CONTENT_TOKENS & content_tokens:
        return 0.04
    return 0.0


def _fact_match_tokens(text: str) -> Set[str]:
    """Return meaningful tokens for strict fact matching."""
    return set(_recall_tokens(text))


def _contains_spaceless_cjk(text: str) -> bool:
    return any(
        (
            "一" <= ch <= "鿿" or "\u3040" <= ch <= "ヿ" or "가" <= ch <= "\ud7af"
            for ch in text
        )
    )


def _cjk_fts_terms(text: str) -> List[str]:
    """Generate FTS-safe terms for CJK text.

    The default unicode61 tokenizer indexes each CJK character as an
    individual token. Unquoted character terms match directly. Bigrams
    are quoted as phrases for multi-character matching.
    """
    cjk_chars = [
        ch
        for ch in text
        if "一" <= ch <= "鿿" or "\u3040" <= ch <= "ヿ" or "가" <= ch <= "\ud7af"
    ]
    if not cjk_chars:
        return []
    terms: List[str] = []
    seen: Set[str] = set()
    for ch in cjk_chars:
        if ch not in seen:
            seen.add(ch)
            terms.append(ch)
    for i in range(len(cjk_chars) - 1):
        bigram = cjk_chars[i] + cjk_chars[i + 1]
        if bigram not in seen:
            seen.add(bigram)
            terms.append(f'"{bigram}"')
    return terms


def _lexical_relevance(
    query_tokens: List[str], content: str, query_lower: str = ""
) -> float:
    """Conservative lexical score in [0, 1]. Returns 0 for no real token overlap.

    This replaces the old character-overlap fallback for normal spaced text.
    Character overlap is only useful for CJK/spaceless text; in English it made
    nonsense queries retrieve unrelated high-importance memories.
    """
    content_lower = content.lower()
    query_cjk = {
        ch
        for ch in query_lower
        if "一" <= ch <= "鿿" or "\u3040" <= ch <= "ヿ" or "가" <= ch <= "\ud7af"
    }
    if query_lower:
        query_tokens = [*query_tokens, *_hyphen_fragment_tokens(query_lower)]
        query_tokens = [*query_tokens, *_symbolic_code_tokens(query_lower)]
        query_tokens = [*query_tokens, *_leading_hyphen_fragments(query_lower)]
        query_tokens = list(dict.fromkeys(query_tokens))
    if query_tokens:
        token_chars = {ch for token in query_tokens for ch in token}
        query_cjk &= token_chars
    if not query_tokens and (not query_cjk):
        return 0.0
    component_groups = [_hyphen_components(token) for token in query_tokens]
    lexical_unit_count = sum(
        (_component_unit_weight(components) for components in component_groups)
    )
    content_tokens = set(_recall_tokens(content_lower))
    content_tokens.update(_hyphen_fragment_tokens(content_lower))
    content_tokens.update(_symbolic_code_tokens(content_lower))
    content_tokens.update(_leading_hyphen_fragments(content_lower))
    expanded_content_tokens = set(content_tokens)
    for token in list(content_tokens):
        expanded_content_tokens.update(
            (
                part
                for part in re.split("[_:/.-]+", token)
                if _is_meaningful_recall_token(part)
            )
        )
    content_tokens = expanded_content_tokens
    short_stems = {
        token for token in query_tokens if len(token) < 3 and (not _has_hangul(token))
    }
    if short_stems:
        content_tokens.update(
            short_stems.intersection(_RECALL_TOKEN_RE.findall(content_lower))
        )
    if not content_tokens and (not query_cjk):
        return 0.0
    exact = 0.0
    partial = 0.0
    for token, components in zip(query_tokens, component_groups, strict=True):
        if token in content_tokens:
            exact += _component_unit_weight(components)
            continue
        if _has_hangul(token) and any(
            (ctoken.startswith(token) for ctoken in content_tokens)
        ):
            exact += _component_unit_weight(components) * _HANGUL_PREFIX_MATCH_WEIGHT
            continue
        component_hits = sum((part in content_tokens for part in components))
        if len(components) >= 2 and component_hits >= 2:
            exact += component_hits
            continue
        if components:
            continue
        synonyms = _RECALL_SYNONYMS.get(token, ())
        if synonyms and any((syn in content_tokens for syn in synonyms)):
            partial += 0.75
            continue
        if (
            len(token) >= 4
            and (not (_has_hangul(query_lower) and (not _has_hangul(token))))
            and any(
                (
                    token in ctoken or ctoken in token
                    for ctoken in content_tokens
                    if len(ctoken) >= 4
                )
            )
        ):
            partial += 0.4
    if _has_hangul(query_lower):
        full_match = (
            1.0 if query_tokens and set(query_tokens) <= content_tokens else 0.0
        )
    else:
        full_match = 1.0 if query_lower and query_lower in content_lower else 0.0
    score = (exact + partial + full_match) / max(lexical_unit_count, 1)
    if score == 0.0:
        if query_cjk:
            content_cjk = {
                ch
                for ch in content_lower
                if "一" <= ch <= "鿿"
                or "\u3040" <= ch <= "ヿ"
                or "가" <= ch <= "\ud7af"
            }
            score = len(query_cjk & content_cjk) / len(query_cjk)
        elif _has_cyrillic(query_lower):
            score = _cyrillic_score(query_lower, content_lower)
    return min(score, 1.0)


def _strict_fact_matches(query: str, fact_text: str) -> bool:
    """Conservative fact matching for natural-language recall queries.

    The legacy fact matcher accepts any query token as a substring of the
    fact. That makes stopwords like "where"/"the"/"use" retrieve unrelated
    facts. The strict matcher keeps exact phrase/path/domain matches, then
    requires multiple meaningful token overlaps (or one very distinctive
    path/domain-like token) before admitting a fact candidate.
    """
    query_lower = query.lower().strip()
    fact_lower = fact_text.lower().strip()
    if not query_lower or not fact_lower:
        return False
    if query_lower in fact_lower:
        return True
    query_tokens = _fact_match_tokens(query_lower)
    fact_tokens = _fact_match_tokens(fact_lower)
    if not query_tokens or not fact_tokens:
        return False
    overlap = query_tokens & fact_tokens
    if len(overlap) >= 2:
        return True
    if len(overlap) == 1:
        token = next(iter(overlap))
        if len(token) >= 8 and any((c in token for c in (".", "/", ":", "-", "_"))):
            return True
        if len(query_tokens) <= 2:
            return len(token) >= 5
        return False
    return False


def _find_memories_by_fact(beam: "BeamMemory", query: str) -> List[str]:
    """
    Find memory IDs that have extracted facts matching the query.
    Does simple keyword matching against stored fact annotations.
    Returns list of memory_id strings.

    Post-E6: reads from AnnotationStore. Memories with multiple extracted
    facts now all surface (silent-destruction bug fixed).
    """
    try:
        all_facts = beam.annotations.query_by_kind("fact")
        if not all_facts:
            return []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        strict_fact_match = not _env_truthy("MNEMOSYNE_LENIENT_FACT_MATCH")
        memory_ids: Set[str] = set()
        for fact_row in all_facts:
            fact_text = fact_row.get("value", "").lower()
            if strict_fact_match:
                if _strict_fact_matches(query_lower, fact_text):
                    memory_ids.add(fact_row["memory_id"])
            elif any((word in fact_text for word in query_words)):
                memory_ids.add(fact_row["memory_id"])
            elif query_lower in fact_text:
                memory_ids.add(fact_row["memory_id"])
        return list(memory_ids)
    except Exception:
        return []


def _dim_from_ddl(sql: str) -> Optional[int]:
    """Parse a vec0 table's declared embedding dimension from its DDL."""
    match = re.search("\\[(\\d+)\\]", sql)
    return int(match.group(1)) if match else None


def _wm_rowid(conn: sqlite3.Connection, memory_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    return int(row["rowid"]) if row else None


def _wm_vec_delete(conn: sqlite3.Connection, memory_id: str) -> None:
    return None


def _backfill_vec_working_from_memory_embeddings(conn: sqlite3.Connection) -> int:
    return 0


def vec_working_coverage(conn: sqlite3.Connection) -> Dict[str, Any]:
    return {"status": "not_applicable", "reason": "Recall uses Jev decisions"}


def repair_vec_working(
    conn: sqlite3.Connection, *, dry_run: bool = False
) -> Dict[str, Any]:
    raise NotImplementedError("PerfectRecall does not build vector indexes")


def _invalidate_query_cache_for_conn(conn, operation: str) -> None:
    return None


def reindex_vectors(
    conn: sqlite3.Connection,
    *,
    batch_size: int = 64,
    dry_run: bool = False,
    progress=None,
) -> Dict[str, Any]:
    raise NotImplementedError("PerfectRecall does not build vector indexes")


def _has_dim_mismatch_signal(exc: BaseException) -> bool:
    """True when the error's own text is sqlite-vec's dimension-mismatch
    signal. Shared by the early gate in ``_vec_search`` and the classifier so
    the two can never drift apart: a mismatch the classifier would confirm
    must always clear the gate, or confirmed mismatches would re-raise out of
    recall() instead of degrading."""
    return "dimension mismatch" in str(exc).lower()


def _is_query_dim_mismatch(
    exc: BaseException, query_dim: int, existing_dim: Optional[int]
) -> bool:
    """True only when sqlite-vec actually rejected the query vector for its
    dimension.

    An unrelated ``OperationalError`` (locked database, missing table) must not
    be dressed up as a dimension mismatch with self-heal guidance, even when
    the submitted and stored dimensions happen to disagree: classify on the
    error's own signal, not on the dimension coincidence alone.
    """
    return (
        existing_dim is not None
        and query_dim != existing_dim
        and _has_dim_mismatch_signal(exc)
    )


_KO_JOSA = (
    "으로부터",
    "에게서",
    "에서부터",
    "이라고",
    "라고",
    "에서",
    "에게",
    "한테",
    "부터",
    "까지",
    "으로",
    "이나",
    "라도",
    "처럼",
    "보다",
    "마다",
    "조차",
    "이란",
    "이든",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "로",
    "만",
    "랑",
    "야",
    "여",
    "나",
    "와",
    "과",
)


def _has_hangul(text: str) -> bool:
    """True if text contains at least one precomposed Hangul syllable."""
    return any(("가" <= ch <= "힣" for ch in text))


_KO_JOSA_LONGEST_FIRST = tuple(sorted(_KO_JOSA, key=len, reverse=True))
_HANGUL_PREFIX_MATCH_WEIGHT = 0.7


def _strip_ko_josa(token: str) -> str:
    """Strip trailing Korean particles (조사) from a whitespace token.

    Suffix trimming, not morphological analysis. Trimming repeats to a fixed
    point so the function is idempotent, which the recall path requires: a
    query token and the same word inflected inside a document must normalize
    to the same string or they can never match. A single pass does not
    guarantee that — ``바나나`` trims to ``바나`` because the final ``나`` is
    itself a particle, while ``바나나를`` trims only to ``바나나``.

    At least two syllables are always left behind, and a token carrying no
    particle is returned unchanged.

        갱신은 -> 갱신    회사에서 -> 회사    바나나를 -> 바나    백업 -> 백업

    Single-syllable particles are also verb endings, so ``갱신하나`` trims to
    ``갱신하``.
    """
    while True:
        for josa in _KO_JOSA_LONGEST_FIRST:
            if len(token) > len(josa) + 1 and token.endswith(josa):
                token = token[: -len(josa)]
                break
        else:
            return token


def _fts_precise_terms(query: str, *, widen: bool = True) -> List[str]:
    """FTS terms built from the raw query tokens, before particle stripping.

    ``_fts_query_terms()`` runs every term through ``_strip_ko_josa()`` first,
    and the trim is fixed-point, so the *surface form the user typed* is not
    reachable from its output: ``바나나`` leaves as ``"바나"*`` because the
    final ``나`` is itself a particle, and ``AI가`` leaves as ``"ai"*``. Both
    stems match a large family of unrelated rows -- ``바나00는``, ``air07`` --
    and with a bounded candidate pool that flood is admitted first, so the
    target is truncated before ranking ever sees it.

    Neither form alone is sufficient, which is why the caller asks for both.

    ``widen=True`` keeps the raw token as a prefix term. It narrows the stem
    flood -- ``"바나나"*`` no longer reaches ``바나00는``, ``"ai가"*`` no longer
    reaches ``air07`` -- and it is the only form that reaches the inflected
    ``바나나는`` a Korean document actually stores, because Korean inflects by
    suffixing. But it is still a prefix, so a token that shares the *whole*
    raw query surface stays reachable: ``바나나01`` matches ``"바나나"*``, and
    61 such rows exhaust a bounded pool exactly as the stem flood did. A
    widened term therefore cannot be the reservation for a literal token.

    ``widen=False`` emits the exact phrase. ``"바나나"`` matches only a
    document that carries ``바나나`` as its own index token, which no
    ``바나나NN`` row does -- unicode61 tokenizes those as single glued tokens.
    That makes it the one form a longer same-prefix token cannot displace,
    and it is why the caller spends its first slice of budget here.

    Tokens outside the widening rule keep the exact-phrase form either way,
    which makes both lists identical to ``_fts_query_terms()`` for any query
    without Hangul. The caller uses that equality to skip the extra stages
    entirely, so non-Korean recall pays nothing for this.

    Synonym expansion is deliberately absent: widening recall is the last
    stage's job.
    """
    terms: List[str] = []
    seen: Set[str] = set()
    symbolic = set(_symbolic_code_tokens(query))
    query_has_hangul = _has_hangul(query)
    for token in _RECALL_TOKEN_RE.findall(query.lower()):
        if not _is_meaningful_recall_token(token):
            continue
        if token in symbolic:
            continue
        token = token.replace('"', '""').strip()
        if not token or token in seen:
            continue
        seen.add(token)
        if widen and (_has_hangul(token) or query_has_hangul):
            terms.append(f'"{token}"*')
        else:
            terms.append(f'"{token}"')
    return terms


def _fts_query_terms(query: str) -> List[str]:
    """FTS-safe meaningful terms for natural-language recall queries.

    Hangul terms are emitted as ``stem*`` prefix terms rather than quoted
    phrases, because unicode61 keeps the particle glued to the stem and an
    exact phrase can therefore never match a differently-inflected stored
    form. Every other language keeps the quoted-phrase behaviour below.

    Terms are quoted so FTS5 treats them as literal phrases. A term must
    never start with ``-``: FTS5 parses a leading hyphen as the NOT /
    column-exclusion operator, so e.g. ``'"rm" OR "-rf"'`` raises
    ``no such column: rf`` and recall fails silently. Leading-hyphen
    fragments are therefore split into their components (``rm -rf`` ->
    ``"rf"``) and any hyphen-leading token is dropped. Symbolic code
    names (``C++``, ``C++20``) are also excluded: unicode61 tokenizes
    them down to bare characters, so an FTS term would only flood
    candidates with noise; they are matched exact-only by the lexical
    layer via ``_symbolic_code_tokens()``.
    """
    terms: List[str] = []
    seen: Set[str] = set()
    symbolic = set(_symbolic_code_tokens(query))
    query_has_hangul = _has_hangul(query)
    for term in _expanded_query_tokens(_recall_tokens(query)):
        if term.startswith("-"):
            continue
        if term in symbolic:
            continue
        term = term.replace('"', '""').strip()
        if not term:
            continue
        if _has_hangul(term) or query_has_hangul:
            stem = _strip_ko_josa(term)
            if len(stem) >= 2:
                if stem not in seen:
                    seen.add(stem)
                    terms.append(f'"{stem}"*')
                continue
        if term not in seen:
            seen.add(term)
            terms.append(f'"{term}"')
    for component in _hyphen_fragment_tokens(query):
        component = component.replace('"', '""').strip()
        if component and component not in seen:
            seen.add(component)
            terms.append(f'"{component}"')
    return terms


def _has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return any(
        (
            "一" <= ch <= "鿿" or "\u3040" <= ch <= "ヿ" or "가" <= ch <= "\ud7af"
            for ch in text
        )
    )


def _cjk_like_search(
    conn: sqlite3.Connection, query: str, k: int = 20, working: bool = False
) -> List[Dict]:
    """Fallback LIKE search for CJK text.

    The default unicode61 FTS5 tokenizer does not index CJK characters
    on this SQLite build. When a CJK query produces zero FTS results,
    fall back to scanning content via LIKE for each unique CJK character
    in the query. Rows matching more query characters rank higher.

    This is slower than FTS but correctness matters more than speed for
    underserved CJK users. Once FTS5 gets tokenchars or ICU support,
    this function can be removed.
    """
    cjk_chars = sorted(
        set(
            (
                ch
                for ch in query
                if "一" <= ch <= "鿿"
                or "\u3040" <= ch <= "ヿ"
                or "가" <= ch <= "\ud7af"
            )
        )
    )
    if not cjk_chars:
        return []
    if working:
        table = "working_memory"
        id_col = "id"
    else:
        table = "episodic_memory"
        id_col = "rowid"
    conditions = " OR ".join(("content LIKE ? ESCAPE '\\'" for _ in cjk_chars))
    params = [f"%{ch}%" for ch in cjk_chars]
    try:
        all_rows = conn.execute(
            f"SELECT {id_col}, content FROM {table} WHERE {conditions} LIMIT ?",
            params + [k * 5],
        ).fetchall()
    except Exception:
        return []
    if not all_rows:
        return []
    scored = []
    for row in all_rows:
        rid = row[id_col]
        content = row["content"]
        score = sum((1 for ch in cjk_chars if ch in content)) / max(len(cjk_chars), 1)
        if score > 0:
            scored.append((rid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    result_key = "id" if working else "rowid"
    return [{result_key: rid, "rank": -score} for rid, score in scored[:k]]


_CYRILLIC_CHAR_RE = re.compile("[\u0430-\u044f\u0400-\u04ff]")


def _has_cyrillic(text: str) -> bool:
    """True if text contains at least one Russian/Cyrillic character.

    not Serbian/Mongolian Cyrillic — see module-level note.
    """
    return bool(_CYRILLIC_CHAR_RE.search(text))


def _ngrams(s: str, n: int = 3) -> set:
    """Return the set of length-n sliding n-grams of s.

    For strings shorter than n, returns the whole string as a single
    n-gram to keep the Jaccard math well-defined.
    """
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _cyrillic_score(query: str, content: str, n: int = 3) -> float:
    """Trigram Jaccard score in [0, 1] for Russian/Cyrillic recall.

    For each query word, find the best-matching content word by Jaccard
    similarity of their character n-gram sets. Return the mean across
    query words. Words shorter than 3 characters are ignored to avoid
    noisy matches on stop words and punctuation.

    N-gram sets are computed once per unique word and cached in a dict
    to avoid redundant recomputation when the same content word is
    compared against multiple query words.
    """
    q_words = [
        w
        for w in re.findall("[\u0430-\u044f\u0451a-z0-9]+", query.lower())
        if len(w) >= 3
    ]
    c_words = [
        w
        for w in re.findall("[\u0430-\u044f\u0451a-z0-9]+", content.lower())
        if len(w) >= 3
    ]
    if not q_words or not c_words:
        return 0.0
    ng_cache: dict = {}
    total = 0.0
    for qw in q_words:
        if qw not in ng_cache:
            ng_cache[qw] = _ngrams(qw, n)
        q_ng = ng_cache[qw]
        best = 0.0
        for cw in c_words:
            if cw not in ng_cache:
                ng_cache[cw] = _ngrams(cw, n)
            c_ng = ng_cache[cw]
            union = q_ng | c_ng
            if not union:
                continue
            jacc = len(q_ng & c_ng) / len(union)
            if jacc > best:
                best = jacc
        total += best
    return total / len(q_words)


def _cyrillic_like_search(
    conn: sqlite3.Connection, query: str, k: int = 20, working: bool = False
) -> List[Dict]:
    """LIKE-based FTS5 fallback for Russian/Cyrillic text.

    Candidate generation: scan ``working_memory``/``episodic_memory`` for
    rows whose content contains any 4+ character word from the query as
    a substring. Re-rank the candidate set by trigram Jaccard similarity
    receive comparable scores regardless of which surface form appears
    in the stored text.

    The candidate-generation LIMIT is ``k * 5`` to keep the Python-side
    scoring bounded; this matches the CJK fallback's behaviour.
    """
    if not _has_cyrillic(query):
        return []
    if working:
        table, id_col = ("working_memory", "id")
    else:
        table, id_col = ("episodic_memory", "rowid")
    conn.create_function(
        "_py_lower", 1, lambda s: s.lower() if isinstance(s, str) else s
    )
    q_words = [
        w
        for w in re.findall("[\u0430-\u044f\u0451a-z0-9]+", query.lower())
        if len(w) >= 4
    ]
    if not q_words:
        return []
    conditions = " OR ".join(["_py_lower(content) LIKE ? ESCAPE '\\'"] * len(q_words))
    params = [f"%{w[:4].lower()}%" for w in q_words]
    try:
        all_rows = conn.execute(
            f"SELECT {id_col}, content FROM {table} WHERE {conditions} LIMIT ?",
            params + [k * 5],
        ).fetchall()
    except Exception:
        return []
    if not all_rows:
        return []
    scored = []
    for row in all_rows:
        rid = row[id_col]
        content = row["content"]
        score = _cyrillic_score(query, content)
        if score > 0:
            scored.append((rid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    result_key = "id" if working else "rowid"
    return [{result_key: rid, "rank": -score} for rid, score in scored[:k]]


def _fts_staged_rows(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    query: str,
    k: int,
    expansion_terms: List[str],
) -> List[Dict]:
    """Fill the bounded candidate budget with exact hits before expansions.

    The pool handed to Python is capped (``LIMIT k``), so a single MATCH over
    ``exact OR prefix`` lets the prefix branch spend the whole budget on rows
    that share nothing but an opening substring. Reserving part of the budget
    rather than raising it keeps recall cost independent of how many
    near-prefix rows the corpus happens to hold.

    The second stage is skipped when it would repeat the first: a query with
    no Hangul produces identical term lists, and running the same MATCH twice
    would be a pure cost regression for every non-Korean caller.

    Ranks come from one final pass over the expansion terms, never from the
    precise stage. bm25 scores two different MATCH expressions on two
    different scales, and the callers min/max-normalize ``rank`` across the
    whole candidate set -- mixing scales silently reweights every row.

    Staging decides *membership* only. The returned rows are re-sorted onto
    the single expansion-term scale, exactly the order a plain
    ``ORDER BY rank, {id_col}`` produced before, so reserving budget cannot
    quietly promote an exact hit over a better-ranked row.

    Every stage needs a cap of its own, including the first, because a term
    can flood at any width. A raw prefix term floods two ways: downward, as
    ``시스템의`` matches the family ``시스템`` does, and sideways, as
    ``"바나나"*`` matches 61 rows of ``바나나NN``. Even the exact phrase can
    flood, on a corpus where many documents legitimately carry the literal
    token -- letting it take every slot would starve the inflected forms that
    only the widened term reaches. Capping each stage below the total means
    no single failure mode can consume the pool.

    The budgets are cumulative, not per-stage, so a stage that returns fewer
    rows than its share gives the remainder to the next one. Exact and
    widened together still stop at half the pool, unchanged from when they
    were one stage, because exact matches are a strict subset of what the
    widened term reaches -- reserving for them shifts *membership* toward
    literal hits without enlarging the pool. Thirds and halves need no
    per-corpus constant.
    """
    match_sql = f"SELECT {id_col}, rank FROM {table} WHERE {table} MATCH ? ORDER BY rank, {id_col} LIMIT ?"
    precise_terms = _fts_precise_terms(query)
    if not precise_terms or precise_terms == expansion_terms:
        rows = conn.execute(match_sql, (" OR ".join(expansion_terms), k)).fetchall()
        return [{id_col: r[id_col], "rank": r["rank"]} for r in rows]
    exact_terms = _fts_precise_terms(query, widen=False)
    ordered_ids: List[Any] = []
    seen_ids: Set[Any] = set()
    for terms, budget in (
        (exact_terms, max(1, k // 3)),
        (precise_terms, max(1, k // 2)),
        (expansion_terms, k),
    ):
        if len(ordered_ids) >= budget:
            continue
        for r in conn.execute(match_sql, (" OR ".join(terms), budget)).fetchall():
            rid = r[id_col]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            ordered_ids.append(rid)
            if len(ordered_ids) >= budget:
                break
    if not ordered_ids:
        return []
    placeholders = ",".join("?" * len(ordered_ids))
    rank_rows = conn.execute(
        f"SELECT {id_col}, rank FROM {table} WHERE {table} MATCH ? AND {id_col} IN ({placeholders})",
        (" OR ".join(expansion_terms), *ordered_ids),
    ).fetchall()
    ranks = {r[id_col]: r["rank"] for r in rank_rows}
    worst = max(ranks.values()) if ranks else 0.0
    rows = [{id_col: rid, "rank": ranks.get(rid, worst)} for rid in ordered_ids]
    rows.sort(key=lambda r: (r["rank"], r[id_col]))
    return rows


def _fts_search(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 episodes and return rowids with ranks.

    Natural assistant queries contain stopwords that make phrase-style FTS
    miss exact memories. Use stopword-filtered OR terms by default; precision
    is restored later by lexical reranking and abstention thresholds.
    """
    terms = _fts_query_terms(query)
    if not terms:
        if _has_cjk(query):
            return _cjk_like_search(conn, query, k=k, working=False)
        if _has_cyrillic(query):
            return _cyrillic_like_search(conn, query, k=k, working=False)
        return []
    rows = _fts_staged_rows(conn, "fts_episodes", "rowid", query, k, terms)
    if not rows and _has_cjk(query):
        return _cjk_like_search(conn, query, k=k, working=False)
    if not rows and _has_cyrillic(query):
        return _cyrillic_like_search(conn, query, k=k, working=False)
    return rows


def _fts_search_working(
    conn: sqlite3.Connection, query: str, k: int = 20
) -> List[Dict]:
    """Search FTS5 working memory and return ids with ranks."""
    terms = _fts_query_terms(query)
    if not terms:
        if _has_cjk(query):
            return _cjk_like_search(conn, query, k=k, working=True)
        if _has_cyrillic(query):
            return _cyrillic_like_search(conn, query, k=k, working=True)
        return []
    rows = _fts_staged_rows(conn, "fts_working", "id", query, k, terms)
    if not rows and _has_cjk(query):
        return _cjk_like_search(conn, query, k=k, working=True)
    if not rows and _has_cyrillic(query):
        return _cyrillic_like_search(conn, query, k=k, working=True)
    return rows


class BeamMemory:
    """
    BEAM memory interface.
    """

    def __init__(
        self,
        session_id: str = "default",
        db_path: Path = None,
        author_id: str = None,
        author_type: str = None,
        channel_id: str = None,
        use_cloud: bool = False,
        event_emitter: "Optional[Callable[[Any], None]]" = None,
    ):
        from mnemosyne.core.config import get_config

        get_config()
        self.session_id = session_id
        self.author_id = author_id
        self.author_type = author_type
        self.channel_id = channel_id or session_id
        self.canonical_owner_id = "default"
        self.agent_context = "primary"
        if db_path is not None and (not isinstance(db_path, Path)):
            db_path = Path(db_path)
        self.db_path = db_path or _default_db_path()
        self.use_cloud = use_cloud
        self._extraction_client = None
        self._extraction_buffer = []
        self._event_emitter = event_emitter
        self.db_path = self.db_path.expanduser().resolve()
        self.init_result = init_beam(self.db_path)
        self.conn = _get_connection(self.db_path)
        try:
            from mnemosyne.core.triples import init_triples

            init_triples(db_path=self.db_path)
        except Exception:
            logger.info("Regex extraction failed, skipping", exc_info=True)
        self._ensure_e6_schema_with_migration()
        from mnemosyne.core.annotations import AnnotationStore

        self.annotations = AnnotationStore(db_path=self.db_path, conn=self.conn)
        from mnemosyne.core.canonical import CanonicalStore

        self.canonical = CanonicalStore(db_path=self.db_path, conn=self.conn)
        from mnemosyne.core.media import MediaStore

        self.media = MediaStore(db_path=self.db_path, conn=self.conn)
        self.episodic_graph = None
        if EpisodicGraph is not None:
            try:
                self.episodic_graph = EpisodicGraph(
                    conn=self.conn, db_path=self.db_path
                )
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)
        self.veracity_consolidator = None
        if VeracityConsolidator is not None:
            try:
                self.veracity_consolidator = VeracityConsolidator(
                    conn=self.conn, db_path=self.db_path
                )
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)

    def _ensure_e6_schema_with_migration(self) -> None:
        """Ensure the AnnotationStore schema exists; auto-migrate legacy
        TripleStore rows on first run with a pre-E6 database.

        Idempotent. Safe to call on fresh installs (no triples table to
        migrate) and on databases that have already been migrated.

        Respects ``MNEMOSYNE_AUTO_MIGRATE=0`` for operators who want
        explicit control over schema migrations. When auto-migration is
        disabled and a migration would have been required, log a clear
        warning pointing at the manual migration script -- the AnnotationStore
        schema is still created so downstream code can run, but legacy rows
        remain in the triples table until the operator runs the script.

        Failures are caught and logged; init does not raise. The provider
        layer's silent-fail pattern (C27) would mask any exception we
        raised here, so logging is the visible channel for now. The user-
        facing pattern is "migration ran (or didn't), continue with
        whatever schema state we have."
        """
        import os
        from mnemosyne.core.annotations import ANNOTATION_KINDS, init_annotations

        logger = logging.getLogger(__name__)
        try:
            init_annotations(self.db_path)
        except Exception as e:
            logger.error("E6: failed to initialize annotations schema: %s", e)
            return
        if os.environ.get("MNEMOSYNE_AUTO_MIGRATE", "1") == "0":
            try:
                cursor = self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='triples'"
                )
                if cursor.fetchone() is not None:
                    placeholders = ",".join("?" * len(ANNOTATION_KINDS))
                    cursor = self.conn.execute(
                        f"SELECT COUNT(*) FROM triples WHERE predicate IN ({placeholders})",
                        tuple(ANNOTATION_KINDS),
                    )
                    pending = cursor.fetchone()[0]
                    if pending > 0:
                        logger.warning(
                            "E6: MNEMOSYNE_AUTO_MIGRATE=0 and %d annotation rows remain in the legacy triples table. Run `python scripts/migrate_triplestore_split.py --db %s` to migrate manually.",
                            pending,
                            self.db_path,
                        )
            except Exception as e:
                logger.debug("E6: opt-out probe failed: %s", e)
            return
        try:
            from mnemosyne.migrations.e6_triplestore_split import (
                migrate as _e6_migrate,
                has_pending_migration as _e6_has_pending,
            )

            if not _e6_has_pending(self.conn):
                return
            try:
                self.conn.commit()
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)
            written = _e6_migrate(
                db_path=self.db_path,
                dry_run=False,
                backup=True,
                log_fn=lambda line: logger.info("E6 migrate: %s", line),
            )
            if written > 0:
                logger.warning(
                    "E6: auto-migrated %d annotation rows from triples → annotations. Backup is at %s.pre_e6_backup (from this run if newly created, or an earlier run if the file already existed). Set MNEMOSYNE_AUTO_MIGRATE=0 to disable auto-migration.",
                    written,
                    self.db_path,
                )
        except Exception as e:
            logger.error(
                "E6: auto-migration failed (continuing init with current schema state). Run `python scripts/migrate_triplestore_split.py --db %s` manually. Error: %s",
                self.db_path,
                e,
            )

    def _find_duplicate(self, content: str) -> Optional[str]:
        """Check if exact same content already exists in working_memory for this session.
        Returns the existing memory_id if found, else None."""
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            SELECT id FROM working_memory\n            WHERE session_id = ? AND content = ?\n            LIMIT 1\n        ",
            (self.session_id, content),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def _emit_event(
        self,
        event_type,
        memory_id: str,
        content: str = None,
        source: str = None,
        importance: float = None,
        metadata: Dict = None,
        delta: Dict = None,
    ) -> None:
        """Fire a streaming event if an emitter is registered."""
        if self._event_emitter is None:
            return
        try:
            from mnemosyne.core.streaming import MemoryEvent, EventType

            evt_type = (
                EventType[event_type] if isinstance(event_type, str) else event_type
            )
            event = MemoryEvent(
                event_type=evt_type,
                memory_id=memory_id,
                session_id=self.session_id,
                content=content,
                source=source,
                importance=importance,
                metadata=metadata,
                delta=delta,
            )
            self._event_emitter(event)
        except Exception:
            pass

    def remember(
        self,
        content: str,
        source: str = "conversation",
        importance: float = 0.5,
        metadata: Dict = None,
        valid_until: str = None,
        scope: str = "session",
        memory_id: str = None,
        extract_entities: bool = False,
        extract: bool = False,
        veracity: str = "unknown",
        trust_tier: str = None,
        memory_type: str = None,
        dedupe: bool = True,
    ) -> str:
        """Store into working_memory. Deduplicates exact content matches.

        When called from the legacy-compatible Mnemosyne.remember() path,
        memory_id is passed through so the legacy memories row and BEAM
        working_memory row stay addressable by the same ID. Direct BEAM calls
        still generate their own deterministic ID.

        Args:
            content: The text to remember
            source: Origin of the memory (e.g., "conversation", "document")
            importance: 0.0-1.0 relevance score
            metadata: Optional dict of additional fields
            valid_until: ISO timestamp when this memory expires
            scope: "session" or "global"
            memory_id: Optional pre-generated ID from legacy layer
            extract_entities: If True, extract and store entity mentions as triples
            extract: If True, extract structured facts from content using LLM
                and store as triples. Default False.
            veracity: Confidence level -- 'stated', 'inferred', 'tool', 'imported', 'unknown'.
                Non-canonical labels are clamped to 'unknown' with a WARNING
                (mirrors the C12.b clamp at the hermes_memory_provider boundary).
            memory_type: Optional explicit MemoryType value (e.g. 'artifact').
                Overrides the content classifier entirely -- the classifier is
                not consulted when this is supplied. Unrecognized labels log a
                WARNING and fall back to classification, so a typo degrades to
                default behaviour rather than stripping the type. Callers that
                *know* what they are writing (a media caption is an artifact,
                whatever its words look like) should set this.
            dedupe: When False, skip the exact-content duplicate check and
                always write a new row.

                Callers writing programmatically-generated text need this. The
                dedup key is (session_id, content), and generated text collides
                far more readily than prose -- "a black frame", "a screenshot
                of a terminal window", a repeated slide. Two such rows
                describing *different* sources would otherwise collapse into
                one, and any sidecar table binding to the returned id would
                bind the second source's row to the first source's memory.
                Nothing raises and the counts all look right, which is what
                makes it worth an explicit opt-out.

                Leaving dedupe on has a second effect worth knowing: the
                dedup-update path applies memory_type via COALESCE, so an
                explicit type on a colliding write retypes the existing row.
        """
        from . import jev
        from .filters import should_remember

        allowed, _ = should_remember(content)
        if not allowed:
            return None
        veracity = clamp_veracity(veracity, context="remember")
        valid_until = _normalize_valid_until(valid_until)
        from mnemosyne.core.content_sanitizer import sanitize_content as _sanitize

        sanitized_content, blob_meta = _sanitize(content)
        if blob_meta:
            metadata = (metadata or {}).copy()
            metadata["_blob"] = blob_meta
            content = sanitized_content
        if trust_tier is None:
            trust_tier = _source_to_trust_tier(source)
        if trust_tier not in ("STATED", "DERIVED", "EXTERNAL_WRITE", "IMPORTED"):
            trust_tier = "STATED"
        memory_type = _clamp_memory_type(memory_type)
        if memory_type is None and classify_memory is not None:
            try:
                result = classify_memory(content)
                memory_type = result.memory_type.value
            except Exception:
                pass
        existing_id = self._find_duplicate(content) if dedupe else None
        if existing_id:
            cursor = self.conn.cursor()
            cursor.execute(
                "\n                UPDATE working_memory\n                SET importance = MAX(importance, ?), timestamp = ?, source = ?,\n                    valid_until = COALESCE(?, valid_until),\n                    scope = COALESCE(?, scope),\n                    author_id = COALESCE(?, author_id),\n                    author_type = COALESCE(?, author_type),\n                    channel_id = COALESCE(?, channel_id),\n                    memory_type = COALESCE(?, memory_type),\n                    veracity = CASE WHEN ? != 'unknown' THEN ? ELSE veracity END,\n                    trust_tier = COALESCE(?, trust_tier),\n                    consolidated_at = NULL,\n                    consolidation_claimed_at = NULL\n                WHERE id = ? AND session_id = ?\n            ",
                (
                    importance,
                    datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                    source,
                    valid_until,
                    scope,
                    self.author_id,
                    self.author_type,
                    self.channel_id,
                    memory_type,
                    veracity,
                    veracity,
                    trust_tier,
                    existing_id,
                    self.session_id,
                ),
            )
            self.conn.commit()
            self._invalidate_query_cache_after_remember_commit()
            try:
                if extract_entities:
                    _extract_and_store_entities(self, existing_id, content)
                if extract:
                    _extract_and_store_facts(self, existing_id, content, source)
                try:
                    self.extract_and_store_facts(
                        content, message_idx=0, source_memory_id=existing_id
                    )
                except Exception:
                    pass
                self._ingest_graph_and_veracity(existing_id, content, source, veracity)
                self._emit_event(
                    "MEMORY_UPDATED",
                    existing_id,
                    content=content,
                    source=source,
                    importance=importance,
                    metadata=metadata,
                )
                return existing_id
            finally:
                self._invalidate_query_cache_after_remember_commit()
        memory_id = memory_id or _generate_id(content)
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            INSERT INTO working_memory\n            (id, content, source, timestamp, session_id, importance, metadata_json, valid_until, scope,\n             author_id, author_type, channel_id, veracity, memory_type, trust_tier)\n            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n        ",
            (
                memory_id,
                content,
                source,
                timestamp,
                self.session_id,
                importance,
                json.dumps(metadata or {}),
                valid_until,
                scope,
                self.author_id,
                self.author_type,
                self.channel_id,
                veracity,
                memory_type,
                trust_tier,
            ),
        )
        self.conn.commit()
        try:
            self._trim_working_memory()
        finally:
            self._invalidate_query_cache_after_remember_commit()
        try:
            self._add_temporal_triple(memory_id, timestamp, source, content)
            if extract_temporal is not None:
                try:
                    temporal_info = extract_temporal(content)
                    if temporal_info and temporal_info.get("event_date"):
                        import json as _json_tmp

                        cursor.execute(
                            "UPDATE working_memory SET event_date=?, event_date_precision=?, temporal_tags=? WHERE id=?",
                            (
                                temporal_info["event_date"],
                                temporal_info["event_date_precision"],
                                _json_tmp.dumps(temporal_info["temporal_tags"]),
                                memory_id,
                            ),
                        )
                        self.conn.commit()
                except Exception:
                    pass
            if extract_entities:
                _extract_and_store_entities(self, memory_id, content)
            if extract:
                _extract_and_store_facts(self, memory_id, content, source)
            try:
                self.extract_and_store_facts(
                    content, message_idx=0, source_memory_id=memory_id
                )
            except Exception:
                pass
            self._ingest_graph_and_veracity(memory_id, content, source, veracity)
            self._emit_event(
                "MEMORY_ADDED",
                memory_id,
                content=content,
                source=source,
                importance=importance,
                metadata=metadata,
            )
            return memory_id
        finally:
            self._invalidate_query_cache_after_remember_commit()

    def remember_media(self, ref: str, **kwargs):
        """Register a piece of media and, if configured, describe it.

        Delegates to :func:`mnemosyne.core.media.remember_media`, which owns the
        flow. Keeping the body here to one line is deliberate: `beam.py` is
        already ~8,000 lines, and the ingest path is far easier to test as a
        free function against a stub than as a method on this class.

        Returns a ``MediaIngestResult``, not a bare id -- the degradation ladder
        produces a status (``ok``/``partial``/``unavailable``/``refused``) that a
        string return would discard, and ``unavailable`` is a success.
        """
        from mnemosyne.core.media import remember_media as _remember_media

        return _remember_media(self, ref=ref, **kwargs)

    def remember_batch(
        self,
        items: List[Dict],
        *,
        veracity: Optional[str] = None,
        force_veracity: bool = False,
        trust_tier: str = "IMPORTED",
        extract_entities: bool = False,
        extract: bool = False,
    ) -> List[str]:
        """
        Batch insert into working_memory for high-throughput ingestion.
        Each item dict should have keys: content, source, importance,
        metadata (optional), veracity (optional).

        Legal veracity values: 'stated', 'inferred', 'tool', 'imported',
        'unknown'. None / empty / whitespace silently → 'unknown'.
        Non-canonical non-empty labels emit a WARNING and clamp to
        'unknown'.

        veracity (method-level kwarg): default applied to items that
            don't supply their own `veracity` key.

        force_veracity (default False): security knob. When True, the
            method-level `veracity` is applied to EVERY row uniformly
            and per-item `item["veracity"]` is IGNORED (warning logged
            per item if present so the operator sees the override).
            Use this when the caller is the authority on trust --
            e.g., an importer ingesting LLM-generated content that
            shouldn't be able to self-elevate its label. Pre-E4 the
            per-item override was harmless because veracity didn't
            affect ranking; post-E4 it gates a real ranking signal
            so callers consuming untrusted content need this knob.
            When False (default), per-item `veracity` keys override
            the method default -- preserves the legitimate use case
            of mixed-trust batches (e.g., user messages='stated',
            tool observations='tool').

        All values are clamped to the canonical allowlist via
        `clamp_veracity` (mirrors C12.b at the hermes_memory_provider
        trust boundary). remember_batch is the high-throughput path
        used by importers, the BEAM benchmark adapter, and batch
        ingest CLIs where label quality varies.

        Pre-E4 the column defaulted to 'unknown' for every batch row;
        recall's veracity multiplier collapsed to a constant 0.8
        (global scale factor instead of rank signal). The recall
        scorer at beam.py::recall now applies the multiplier to
        working_memory hits too, so per-row veracity differentiates
        scores at the experiment level.

        E2 -- Enrichment parity with `remember()`:
            Post-E2 this method runs the same post-insert enrichment
            pipeline `remember()` runs unconditionally:
              - `_add_temporal_triple` writes the row's date as an
                `occurred_on` annotation + the source kind as a
                `has_source` annotation (zero-LLM, just date string
                slicing).
              - `_ingest_graph_and_veracity` runs pattern-based gist +
                fact extraction via `EpisodicGraph` and consolidates
                the extracted facts into `consolidated_facts` weighted
                by per-row veracity (`VeracityConsolidator`). Zero LLM
                -- rule-based / regex pattern matching only.

            Without this fix any high-throughput ingest path bypassed
            the enrichment layer entirely, leaving the polyphonic
            engine's `graph` and `fact` voices with no data to fuse --
            E5's RRF over 4 voices collapsed to 2 voices in practice.

        extract_entities (default False): opt-in regex entity scan
            via `_extract_and_store_entities`. Cheap but generates
            additional annotation rows; off by default to keep batch
            ingest stable for non-experiment callers.

        extract (default False): opt-in LLM-based structured fact
            extraction via `_extract_and_store_facts`. Real cloud-API
            cost per row; off by default. The BEAM-recovery experiment
            arm that tests LLM enrichment sets this True.

        New behavior change for existing batch callers: the always-on
        pattern-based enrichment now adds ~ms-per-row CPU cost (regex
        + a few SQLite inserts). For typical importers (10k-100k
        rows) this is a few seconds of additional latency; for the
        BEAM benchmark's 250k-message ingest, ~minutes. Documented in
        CHANGELOG.
        """
        from . import jev
        from .filters import should_remember

        selected = []
        for item in items:
            if not should_remember(item["content"])[0]:
                continue
            item = dict(item)
            if (
                _clamp_memory_type(item.get("memory_type")) is None
                and classify_memory is not None
            ):
                try:
                    item["memory_type"] = classify_memory(
                        item["content"]
                    ).memory_type.value
                except jev.JevError:
                    item["memory_type"] = "unknown"
            selected.append(item)
        items = selected
        cursor = self.conn.cursor()
        ids = []
        meta_by_id: Dict[str, Tuple[str, str]] = {}
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        default_veracity = clamp_veracity(veracity, context="remember_batch.default")
        for item in items:
            from mnemosyne.core.content_sanitizer import sanitize_content as _sanitize

            raw_content = item["content"]
            sanitized_content, blob_meta = _sanitize(raw_content)
            if blob_meta:
                item["content"] = sanitized_content
                item_meta = item.get("metadata") or {}
                item["metadata"] = {**item_meta, "_blob": blob_meta}
            memory_id = _generate_id(item["content"])
            ids.append(memory_id)
            item_type = _clamp_memory_type(item.get("memory_type"))
            if item_type is None and classify_memory is not None:
                try:
                    result = classify_memory(item["content"])
                    item_type = result.memory_type.value
                except Exception:
                    logger.info("Regex extraction failed, skipping", exc_info=True)
            if force_veracity:
                if "veracity" in item:
                    logger.warning(
                        "remember_batch.force_veracity=True; ignoring per-item veracity %r in favor of method-level default %r",
                        item["veracity"],
                        default_veracity,
                    )
                item_veracity = default_veracity
            elif "veracity" in item:
                item_veracity = clamp_veracity(
                    item["veracity"], context="remember_batch.per_item"
                )
            else:
                item_veracity = default_veracity
            item_source = item.get("source", "conversation")
            meta_by_id[memory_id] = (item_source, item_veracity)
            cursor.execute(
                "\n                INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json,\n                author_id, author_type, channel_id, memory_type, veracity, trust_tier)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    memory_id,
                    item["content"],
                    item_source,
                    timestamp,
                    self.session_id,
                    item.get("importance", 0.5),
                    json.dumps(item.get("metadata") or {}),
                    item.get("author_id", self.author_id),
                    item.get("author_type", self.author_type),
                    item.get("channel_id", self.channel_id),
                    item_type,
                    item_veracity,
                    trust_tier,
                ),
            )
        self.conn.commit()
        for memory_id in ids:
            item_source, item_veracity = meta_by_id.get(
                memory_id, ("conversation", "unknown")
            )
            try:
                row = cursor.execute(
                    "SELECT content, timestamp FROM working_memory WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    continue
                row_content = row["content"] if hasattr(row, "keys") else row[0]
                row_timestamp = row["timestamp"] if hasattr(row, "keys") else row[1]
                self._add_temporal_triple(
                    memory_id, row_timestamp, item_source, row_content
                )
                self._ingest_graph_and_veracity(
                    memory_id, row_content, item_source, item_veracity
                )
                if extract_entities:
                    _extract_and_store_entities(self, memory_id, row_content)
                if extract:
                    _extract_and_store_facts(self, memory_id, row_content, item_source)
                try:
                    self.extract_and_store_facts(
                        row_content, message_idx=0, source_memory_id=memory_id
                    )
                except Exception:
                    pass
                self._emit_event(
                    "MEMORY_ADDED",
                    memory_id,
                    content=row_content,
                    source=item_source,
                    importance=0.5,
                    metadata=None,
                )
            except Exception as exc:
                logger.warning(
                    "remember_batch: per-row enrichment failed for %s (%s): %s",
                    memory_id,
                    type(exc).__name__,
                    exc,
                )
        self._trim_working_memory()
        return ids

    def _ingest_graph_and_veracity(
        self, memory_id: str, content: str, source: str, veracity: str = "unknown"
    ):
        """Phase 3-4: Extract gists + facts, store in graph, consolidate veracity.
        Non-blocking -- failures in graph/veracity don't affect memory storage."""
        gist = None
        facts = []
        if self.episodic_graph is not None:
            try:
                gist = self.episodic_graph.extract_gist(content, memory_id)
                self.episodic_graph.store_gist(gist, memory_id)
                facts = self.episodic_graph.extract_facts(content, memory_id)
                for fact in facts:
                    self.episodic_graph.store_fact(fact, memory_id)
                for fact in facts:
                    self.episodic_graph.add_edge(
                        GraphEdge(
                            source=gist.id,
                            target=fact.id,
                            edge_type="ctx",
                            weight=fact.confidence,
                            timestamp=datetime.now().isoformat(),
                        )
                    )
            except Exception:
                pass
        if self.veracity_consolidator is not None and facts:
            try:
                for fact in facts:
                    self.veracity_consolidator.consolidate_fact(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        veracity=veracity,
                        source=memory_id,
                    )
            except Exception:
                pass
        self._proactively_link(memory_id, content)

    def _proactively_link(self, memory_id: str, content: str):
        """Phase 5: Auto-create graph edges between new memory and related existing memories.

        Two zero-LLM strategies:
        1. Content similarity via recall() — top-K via FTS5 + vector
        2. Entity overlap via shared facts in the graph

        Gated behind MNEMOSYNE_PROACTIVE_LINKING=1 env var.
        Non-blocking — failures never affect memory storage.
        """
        import os

        if os.environ.get("MNEMOSYNE_PROACTIVE_LINKING", "0") != "1":
            return
        if self.episodic_graph is None:
            return
        from . import jev
        from .jev_recall import proactively_link

        try:
            proactively_link(self, memory_id, content)
        except jev.JevError:
            logger.warning("Jev linking unavailable; stored memory preserved")
        return

    def _add_temporal_triple(
        self, memory_id: str, timestamp: str, source: str, content: str
    ):
        """Auto-generate temporal annotations for a memory.

        Post-E6: writes occurred_on / has_source as annotations rather
        than triples. These are inherently single-valued per memory
        today, but `annotations` is the correct home -- they describe a
        memory rather than expressing a current-truth fact like
        "user prefers X". Method name kept for backward compat.
        """
        try:
            date_str = timestamp[:10]
            self.annotations.add(
                memory_id=memory_id, kind="occurred_on", value=date_str
            )
            if source and source not in ("conversation", "user", "assistant"):
                self.annotations.add(
                    memory_id=memory_id, kind="has_source", value=source
                )
        except Exception:
            pass

    def _trim_working_memory(self):
        """Keep working_memory within size/time limits.

        Post-E3: consolidated rows (consolidated_at IS NOT NULL) are
        exempt from trim. The "originals stay" contract means they
        remain queryable until explicit forget(); the TTL window only
        bounds NOT-YET-consolidated content. Without this exemption,
        the additive promise expires at WORKING_MEMORY_TTL_HOURS and
        the experiment Arm B's "ADD-only" guarantee collapses at 24h.
        """
        cutoff = _utc_cutoff_sql(
            (
                datetime.now(timezone.utc) - timedelta(hours=WORKING_MEMORY_TTL_HOURS)
            ).isoformat()
        )
        # datetime() normalizes timezone forms but truncates fractional seconds.
        # Break ties by precise time and insertion order so a write at capacity
        # cannot immediately evict itself among records from the same second.
        self.conn.execute(
            f"\n            DELETE FROM working_memory\n            WHERE session_id = ?\n              AND consolidated_at IS NULL\n              AND (pinned IS NULL OR pinned = 0)\n              AND (\n                {_SQL_CHRONO_TS} < ? OR\n                id NOT IN (\n                    SELECT id FROM working_memory\n                    WHERE session_id = ? AND consolidated_at IS NULL\n                      AND (pinned IS NULL OR pinned = 0)\n                    ORDER BY {_SQL_CHRONO_TS} DESC, julianday(timestamp) DESC, rowid DESC\n                    LIMIT ?\n                )\n              )\n        ",
            (self.session_id, cutoff, self.session_id, WORKING_MEMORY_MAX_ITEMS),
        )
        self.conn.commit()

    def get_context(self, limit: int = 10) -> List[Dict]:
        """Get working_memory for prompt injection.
        Global memories first, then sorted by importance (high first),
        then by recency. High-importance rules/bans surface reliably.

        Bumps recall_count and last_recalled on returned items, capped
        so a single read cannot extend the effective clock by more than
        WM_BUMP_CAP_HOURS. Prevents infinite extension of stale items
        while keeping hot items visible."""
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        select_cols = "id, content, source, timestamp, importance, scope, last_recalled"
        include_consolidated = _env_truthy("MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED")
        predicates = [
            "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
            "superseded_by IS NULL",
        ]
        if not include_consolidated:
            predicates.append("consolidated_at IS NULL")
        common_predicate = " AND ".join(predicates)
        cursor.execute(
            f"\n            SELECT {select_cols}\n            FROM working_memory\n            WHERE scope = 'global'\n              AND {common_predicate}\n            ORDER BY importance DESC, timestamp DESC\n            LIMIT ?\n        ",
            (now, limit),
        )
        global_rows = [dict(row) for row in cursor.fetchall()]
        session_rows = []
        if limit < 0 or len(global_rows) < limit:
            session_limit = limit if limit < 0 else limit - len(global_rows)
            cursor.execute(
                f"\n                SELECT {select_cols}\n                FROM working_memory\n                WHERE session_id = ?\n                  AND (scope IS NULL OR scope != 'global')\n                  AND {common_predicate}\n                ORDER BY importance DESC, timestamp DESC\n                LIMIT ?\n            ",
                (self.session_id, now, session_limit),
            )
            session_rows = [dict(row) for row in cursor.fetchall()]
        rows = global_rows + session_rows
        if not rows:
            return rows
        now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        bump_delta = timedelta(hours=WM_BUMP_CAP_HOURS)
        updates = {}
        for row in rows:
            old_ts = row.pop("last_recalled", None)
            if old_ts is None:
                new_last = now_dt
            else:
                try:
                    parsed = _parse_iso_datetime_utc(old_ts).replace(tzinfo=None)
                    bumped = parsed + bump_delta
                except (AttributeError, ValueError, TypeError, OverflowError):
                    new_last = now_dt
                else:
                    new_last = min(now_dt, bumped)
            ts = new_last.isoformat()
            updates.setdefault(ts, []).append(row["id"])
        owns_transaction = not self.conn.in_transaction
        with _guarded_transaction(self.conn):
            if owns_transaction:
                cursor.execute("BEGIN TRANSACTION")
            for ts, ids in updates.items():
                placeholders = ",".join(("?" for _ in ids))
                cursor.execute(
                    f"UPDATE working_memory SET recall_count = recall_count + 1, last_recalled = ? WHERE id IN ({placeholders})",
                    (ts, *ids),
                )
        return rows

    def _invalidate_query_cache(self) -> None:
        """Clear the existing enhanced-recall cache without creating an empty one."""
        cache = getattr(self, "_query_cache", None)
        if cache is not None:
            cache.invalidate()
            return
        _invalidate_query_cache_for_conn(self.conn, "beam")

    def _invalidate_query_cache_after_remember_commit(self) -> None:
        """Best-effort cache invalidation after ``remember()`` has committed."""
        try:
            self._invalidate_query_cache()
        except Exception as exc:
            logger.warning(
                "remember: query-cache invalidation failed after commit (%s): %s",
                type(exc).__name__,
                exc,
            )

    def _invalidate_query_cache_after_commit(self, operation: str) -> None:
        """Best-effort query-cache invalidation after a mutating commit.

        Consolidation / reclaim / sleep change dense-pool eligibility
        (``consolidated_at`` transitions, new episodic summaries, episodic
        degradation), so warmed enhanced-recall v3 entries must not be served
        stale after those mutations.
        """
        try:
            self._invalidate_query_cache()
        except Exception as exc:
            logger.warning(
                "%s: query-cache invalidation failed after commit (%s): %s",
                operation,
                type(exc).__name__,
                exc,
            )

    def invalidate(
        self,
        memory_id: str,
        replacement_id: str = None,
        *,
        defer_cache_invalidation: bool = False,
    ) -> bool:
        """
        Mark a memory as invalid/superseded.
        If replacement_id is provided, sets superseded_by.
        Otherwise sets valid_until to now (immediate expiry).
        With defer_cache_invalidation=True, the caller must invalidate the
        query cache after committing its complete logical operation.
        """
        cursor = self.conn.cursor()
        if replacement_id:
            if replacement_id == memory_id:
                return False
            owns_transaction = not self.conn.in_transaction

            def validate_and_invalidate() -> bool:
                replacement_found = False
                for table in ("working_memory", "episodic_memory"):
                    cursor.execute(
                        f"\n                        SELECT 1 FROM {table}\n                        WHERE id = ? AND (session_id = ? OR scope = 'global')\n                        LIMIT 1\n                        ",
                        (replacement_id, self.session_id),
                    )
                    if cursor.fetchone() is not None:
                        replacement_found = True
                        break
                if not replacement_found:
                    return False
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "\n                    UPDATE working_memory\n                    SET valid_until = ?, superseded_by = ?\n                    WHERE id = ? AND (session_id = ? OR scope = 'global')\n                ",
                    (now, replacement_id, memory_id, self.session_id),
                )
                if cursor.rowcount == 0:
                    cursor.execute(
                        "\n                        UPDATE episodic_memory\n                        SET valid_until = ?, superseded_by = ?\n                        WHERE id = ? AND (session_id = ? OR scope = 'global')\n                    ",
                        (now, replacement_id, memory_id, self.session_id),
                    )
                return cursor.rowcount > 0

            if owns_transaction:
                with _guarded_transaction(self.conn):
                    cursor.execute("BEGIN IMMEDIATE")
                    invalidated = validate_and_invalidate()
            else:
                invalidated = validate_and_invalidate()
            if invalidated and (not defer_cache_invalidation):
                if owns_transaction:
                    self._invalidate_query_cache_after_commit("invalidate")
                else:
                    self._invalidate_query_cache()
            return invalidated
        owns_transaction = not self.conn.in_transaction
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "\n            UPDATE working_memory\n            SET valid_until = ?, superseded_by = ?\n            WHERE id = ? AND (session_id = ? OR scope = 'global')\n        ",
            (now, replacement_id, memory_id, self.session_id),
        )
        if cursor.rowcount > 0:
            if owns_transaction:
                self.conn.commit()
                if not defer_cache_invalidation:
                    self._invalidate_query_cache_after_commit("invalidate")
            elif not defer_cache_invalidation:
                self._invalidate_query_cache()
            return True
        cursor.execute(
            "\n            UPDATE episodic_memory\n            SET valid_until = ?, superseded_by = ?\n            WHERE id = ? AND (session_id = ? OR scope = 'global')\n        ",
            (now, replacement_id, memory_id, self.session_id),
        )
        invalidated = cursor.rowcount > 0
        if owns_transaction:
            self.conn.commit()
        if invalidated and (not defer_cache_invalidation):
            if owns_transaction:
                self._invalidate_query_cache_after_commit("invalidate")
            else:
                self._invalidate_query_cache()
        return invalidated

    def _detect_conflicts(
        self, rows: List[Dict], similarity_threshold: float = 0.88
    ) -> List[tuple]:
        "Enumerate chronological pairs for bounded Jev contradiction validation."
        if len(rows) < 2:
            return []
        from . import jev

        ordered = sorted(
            (r for r in rows if not r.get("superseded_by")),
            key=lambda r: r["timestamp"],
        )
        return [
            (a["id"], b["id"])
            for i, a in enumerate(ordered)
            for b in ordered[i + 1 :]
            if a["content"] != b["content"] and a["timestamp"] < b["timestamp"]
        ]

    def get_working_stats(
        self, author_id: str = None, author_type: str = None, channel_id: str = None
    ) -> Dict:
        cursor = self.conn.cursor()
        where_clauses = []
        params = []
        if author_id:
            where_clauses.append("author_id = ?")
            params.append(author_id)
        if author_type:
            where_clauses.append("author_type = ?")
            params.append(author_type)
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cursor.execute(f"SELECT COUNT(*) FROM working_memory{where_str}", params)
        total = cursor.fetchone()[0]
        consolidated_where = (
            f"{where_str} AND consolidated_at IS NOT NULL"
            if where_str
            else " WHERE consolidated_at IS NOT NULL"
        )
        cursor.execute(
            f"SELECT COUNT(*) FROM working_memory{consolidated_where}", params
        )
        consolidated = cursor.fetchone()[0]
        unconsolidated = total - consolidated
        pinned_where = (
            f"{where_str} AND pinned = 1 AND consolidated_at IS NULL"
            if where_str
            else " WHERE pinned = 1 AND consolidated_at IS NULL"
        )
        cursor.execute(f"SELECT COUNT(*) FROM working_memory{pinned_where}", params)
        pinned_unconsolidated = cursor.fetchone()[0]
        cursor.execute(
            f"SELECT timestamp FROM working_memory{where_str} ORDER BY timestamp DESC LIMIT 1",
            params,
        )
        last = cursor.fetchone()
        return {
            "total": total,
            "consolidated": consolidated,
            "unconsolidated": unconsolidated,
            "pinned_unconsolidated": pinned_unconsolidated,
            "last": last[0] if last else None,
        }

    def _count_unconsolidated_before(self, cutoff: str) -> int:
        """Count working memories eligible for consolidation before cutoff.
        Used by _maybe_auto_sleep() to skip full sleep passes when nothing
        is eligible — avoids unnecessary database work on always-on agents
        with longer TTLs after a prior auto-sleep already consolidated everything."""
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM working_memory WHERE {_SQL_CHRONO_TS} < ? AND {_SQL_PLACEABLE_TS} AND consolidated_at IS NULL AND (pinned IS NULL OR pinned = 0)",
            (cutoff,),
        )
        return cursor.fetchone()[0]

    def get_global_working_stats(self) -> Dict:
        """DEPRECATED: Use get_working_stats() instead. Kept for backward compatibility."""
        return self.get_working_stats()

    def update_working(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        pinned: int = None,
        timestamp: str = None,
    ) -> bool:
        "Update stored content and metadata while preserving transaction and scope checks."
        cursor = self.conn.cursor()
        updates = []
        params = []
        content_changed = False
        if content is not None:
            updates.append("content = ?")
            params.append(content)
            content_changed = True
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if pinned is not None:
            updates.append("pinned = ?")
            params.append(1 if pinned else 0)
        if timestamp is not None:
            if not _import_timestamp_ok(timestamp):
                raise ValueError(
                    f"update_working: timestamp {timestamp!r} is not a parseable ISO-8601 value"
                )
            updates.append("timestamp = ?")
            params.append(
                _parse_iso_datetime_utc(timestamp.strip())
                .replace(tzinfo=None)
                .isoformat()
            )
        if not updates:
            return False
        params.extend([memory_id, self.session_id])
        cursor.execute(
            f"UPDATE working_memory SET {', '.join(updates)} WHERE id = ? AND session_id = ?",
            params,
        )
        affected = cursor.rowcount
        self.conn.commit()
        if affected > 0:
            self._invalidate_query_cache_after_commit("update_working")
        return affected > 0

    def get(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve a single memory by its primary key (id).
        Pure read -- no side effects, no recall_count bump, no FTS trigger.

        Checks working_memory first (faster, higher hit rate),
        then episodic_memory (fallback).

        Returns None if not found in either table.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            SELECT id, content, source, timestamp, session_id,\n                   importance, metadata_json, veracity, created_at\n            FROM working_memory\n            WHERE id = ? AND (session_id = ? OR scope = 'global')\n        ",
            (memory_id, self.session_id),
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "timestamp": row[3],
                "session_id": row[4],
                "importance": row[5],
                "metadata": row[6],
                "veracity": row[7],
                "created_at": row[8],
                "memory_store": "working",
            }
        cursor.execute(
            "\n            SELECT id, content, source, timestamp, session_id,\n                   importance, metadata_json, veracity, created_at,\n                   event_date, event_date_precision\n            FROM episodic_memory\n            WHERE id = ? AND (session_id = ? OR scope = 'global')\n        ",
            (memory_id, self.session_id),
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "timestamp": row[3],
                "session_id": row[4],
                "importance": row[5],
                "metadata": row[6],
                "veracity": row[7],
                "created_at": row[8],
                "event_date": row[9],
                "event_date_precision": row[10],
                "memory_store": "episodic",
            }
        return None

    def forget_working(self, memory_id: str) -> bool:
        """Delete a session-authorized working memory row and its cascade
        (vector, annotations, embeddings, gists) atomically."""
        cursor = self.conn.cursor()
        owns_transaction = not self.conn.in_transaction
        with _guarded_transaction(self.conn):
            authorized_row = cursor.execute(
                "SELECT rowid FROM working_memory WHERE id = ? AND (session_id = ? OR scope = 'global')",
                (memory_id, self.session_id),
            ).fetchone()
            cursor.execute(
                "DELETE FROM working_memory WHERE id = ? AND (session_id = ? OR scope = 'global')",
                (memory_id, self.session_id),
            )
            wm_rows = cursor.rowcount
            if wm_rows > 0:
                cursor.execute(
                    "DELETE FROM annotations WHERE memory_id = ?", (memory_id,)
                )
                cursor.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
                )
                gists_table = cursor.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'gists'"
                ).fetchone()
                if gists_table is not None:
                    cursor.execute(
                        "DELETE FROM gists WHERE memory_id = ?", (memory_id,)
                    )
        forgotten = wm_rows > 0
        if forgotten:
            if owns_transaction:
                self._invalidate_query_cache_after_commit("forget_working")
            else:
                self._invalidate_query_cache()
        return forgotten

    def consolidate_to_episodic(
        self,
        summary: str,
        source_wm_ids: List[str],
        source: str = "consolidation",
        importance: float = 0.6,
        metadata: Dict = None,
        valid_until: str = None,
        scope: str = "session",
        veracity: Optional[str] = None,
        event_timestamp: "Optional[str]" = None,
        event_date: "Optional[str]" = None,
        event_date_precision: "Optional[str]" = None,
        emit_event: bool = True,
    ) -> str:
        "Store a consolidated summary with source provenance and scope."
        _caller_owns_txn = self.conn.in_transaction
        _emitter_registered = self._event_emitter is not None
        if emit_event and _caller_owns_txn and _emitter_registered:
            raise MemoryTransactionStateError(
                "consolidate_to_episodic(): event emission requested while a caller-owned transaction is open; the MEMORY_CONSOLIDATED event would fire before the outer commit (phantom event on rollback). Pass emit_event=False, or commit before consolidating."
            )
        if event_timestamp is not None:
            if not isinstance(event_timestamp, str):
                raise ValueError(
                    f"event_timestamp must be a string, got {type(event_timestamp).__name__}: {event_timestamp!r}"
                ) from None
            if event_timestamp:
                try:
                    _parse_iso_datetime_utc(event_timestamp)
                except (OverflowError, ValueError, TypeError):
                    raise ValueError(
                        f"event_timestamp not a parseable ISO-8601 datetime: {event_timestamp!r}"
                    ) from None
        if event_date is not None:
            if not isinstance(event_date, str):
                raise ValueError(
                    f"event_date must be a string, got {type(event_date).__name__}: {event_date!r}"
                ) from None
            if event_date and (not _event_date_valid(event_date)):
                raise ValueError(
                    f"event_date not a real calendar date (strict YYYY-MM-DD): {event_date!r}"
                ) from None
        if event_date_precision is not None and (
            not isinstance(event_date_precision, str)
        ):
            raise ValueError(
                f"event_date_precision must be a string, got {type(event_date_precision).__name__}: {event_date_precision!r}"
            ) from None
        if event_date_precision and event_date_precision not in _EVENT_DATE_PRECISIONS:
            raise ValueError(
                f"event_date_precision must be one of {sorted(_EVENT_DATE_PRECISIONS)}, got {event_date_precision!r}"
            )
        memory_id = _generate_id(summary)
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        ep_type = None
        if classify_memory is not None:
            try:
                result = classify_memory(summary)
                ep_type = result.memory_type.value
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)
        if veracity is None:
            row_veracity = "unknown"
        else:
            row_veracity = clamp_veracity(
                veracity, context="consolidate_to_episodic.veracity"
            )
        valid_until = _normalize_valid_until(valid_until)
        cursor = self.conn.cursor()
        _owned_txn = not self.conn.in_transaction
        with _guarded_transaction(self.conn):
            cursor.execute(
                "\n                INSERT INTO episodic_memory\n                (id, content, source, timestamp, session_id, importance, metadata_json, summary_of, valid_until, scope,\n                 author_id, author_type, channel_id, memory_type, veracity)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    memory_id,
                    _sanitize_utf8(summary),
                    source,
                    timestamp,
                    self.session_id,
                    importance,
                    json.dumps(metadata or {}),
                    ",".join(source_wm_ids),
                    valid_until,
                    scope,
                    self.author_id,
                    self.author_type,
                    self.channel_id,
                    ep_type,
                    row_veracity,
                ),
            )
            rowid = cursor.lastrowid
            if event_timestamp:
                _norm_ts = _parse_iso_datetime_utc(event_timestamp)
                _norm_ts = _norm_ts.replace(tzinfo=None).isoformat()
                cursor.execute(
                    "UPDATE episodic_memory SET timestamp = ? WHERE id = ?",
                    (_norm_ts, memory_id),
                )
            if event_date:
                cursor.execute(
                    "UPDATE episodic_memory SET event_date = ?, event_date_precision = ? WHERE id = ?",
                    (event_date, event_date_precision or "unknown", memory_id),
                )
        try:
            if _owned_txn:
                self.conn.commit()
            if _owned_txn:
                self._ingest_graph_and_veracity(
                    memory_id, summary, source, veracity=row_veracity
                )
            if emit_event:
                self._emit_event(
                    "MEMORY_CONSOLIDATED",
                    memory_id,
                    content=summary,
                    source=source,
                    importance=importance,
                    metadata={"summary_of": source_wm_ids, **(metadata or {})},
                )
        finally:
            if _owned_txn:
                self._invalidate_query_cache_after_commit("consolidate_to_episodic")
        return memory_id

    def detect_language(self, text: str) -> str:
        """Ultra-fast language detection, zero deps, <0.1ms.
        Returns ISO 639-1 code ('en', 'de', 'ru', etc.). Extendable."""
        if not text or not isinstance(text, str):
            return "en"
        text_lower = text.lower()
        if sum(("\u0400" <= c <= "\u04ff" for c in text_lower)) >= 5:
            return "ru"
        if any((c in text_lower for c in "äöüß")):
            return "de"
        german_markers = {
            "ich",
            "du",
            "wir",
            "ist",
            "nicht",
            "für",
            "und",
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "kein",
            "keine",
            "mein",
            "meine",
            "dann",
            "auch",
            "immer",
            "nie",
            "niemals",
            "mag",
            "will",
            "möchte",
            "kann",
            "kannst",
            "können",
            "habe",
            "hast",
            "hat",
            "haben",
            "bin",
            "bist",
            "sind",
            "seid",
            "einen",
            "einer",
            "eines",
            "dem",
            "den",
            "beim",
            "zum",
            "zur",
            "nach",
            "mit",
            "von",
            "bei",
            "aus",
            "auf",
            "vor",
            "aber",
            "oder",
            "weil",
            "denn",
            "dass",
            "sehr",
            "schon",
            "noch",
            "mal",
            "man",
            "nur",
            "wenn",
            "wie",
            "als",
            "doch",
            "gerne",
            "gern",
            "lieber",
            "einfach",
            "eigentlich",
            "vielleicht",
            "natürlich",
            "genau",
            "bereits",
            "eben",
        }
        import re

        words = set(re.findall("\\w+", text_lower))
        if len(words & german_markers) >= 2:
            return "de"
        if any((c in text_lower for c in "ñáéíóúü¿¡")):
            return "es"
        es_markers = {
            "y",
            "de",
            "por",
            "con",
            "para",
            "que",
            "qué",
            "como",
            "el",
            "la",
            "lo",
            "los",
            "las",
            "un",
            "una",
            "del",
            "este",
            "esta",
            "esto",
            "ese",
            "esa",
            "eso",
            "aquel",
            "mi",
            "mis",
            "tu",
            "tus",
            "su",
            "sus",
            "es",
            "está",
            "son",
            "hay",
            "tiene",
            "puede",
            "más",
            "no",
            "también",
            "si",
            "ya",
            "nunca",
            "he",
            "se",
            "me",
            "te",
            "le",
            "a",
            "yo",
            "ante",
            "bajo",
            "contra",
            "desde",
            "en",
            "entre",
            "hacia",
            "hasta",
            "según",
            "sin",
            "sobre",
            "tras",
            "todo",
            "toda",
            "cada",
            "muy",
            "pero",
            "siempre",
            "usa",
            "hacer",
            "antes",
            "recuerda",
            "evita",
        }
        words = set(re.findall("\\w+", text_lower))
        if len(words & es_markers) >= 2:
            return "es"
        if any((c in text_lower for c in "àèéìòù")):
            italian_markers = {
                "e",
                "il",
                "la",
                "i",
                "le",
                "di",
                "che",
                "non",
                "un",
                "una",
                "per",
                "è",
                "in",
                "sono",
                "mi",
                "ha",
                "ma",
                "lo",
                "se",
                "su",
                "con",
                "da",
                "come",
                "questo",
                "quello",
                "anche",
                "o",
                "ho",
                "ci",
                "si",
                "perché",
                "perche",
                "quando",
                "chi",
                "dove",
                "molto",
                "del",
                "della",
                "delle",
                "dei",
                "degli",
                "nel",
                "nella",
                "sul",
                "sulla",
                "sui",
                "sulle",
                "al",
                "alla",
                "agli",
                "alle",
            }
            words = set(re.findall("\\w+", text_lower))
            if len(words & italian_markers) >= 2:
                return "it"
        return "en"

    MULTILINGUAL_PATTERNS = {
        "en": {
            "negation": "\\b(I(?: have|\\'ve)?\\s*(?:never|not)\\s+[^.,;!?\\n]{15,120})",
            "decision": "(?:decided to|chose to|opted for|selected|picked|switching to)\\s+([^.,;!?\\n]{10,120})",
            "entity": "(?:the|my|our|your)\\s+([a-z_]+(?:\\s+(?:table|model|schema|API|endpoint|function|module|route|handler|tool|plugin|script|config|setting|workflow|pipeline|process|system|server|client|service|database|query|file|repo|branch|PR|issue|task|job)))\\s+(?:needs?|requires?|should|could|would|will|has|have|uses?|runs?|handles?|processes?|supports?)\\s+([^.,;!?\\n]{10,80})",
            "sequence": "((?:first|second|third|fourth|fifth|finally|next|then|after that)[^.,;!?\\n]{15,120})",
            "instruction_false_positives": [
                "i think you should leave",
                "should behave",
                "their work style",
            ],
            "instruction_imperative": "always|never|remember|use|keep|avoid|ensure|check|verify|run|test|build|deploy|push|pull|merge|commit|close|open|update|install|configure|set|enable|disable|add|remove|create|delete|start|stop|restart|reload|reset|try|implement|write|read|switch|move|copy|rename|send|reply|respond",
            "instruction": "\\b(?:always|never|must|must not|should(?: not)?(?=\\s+(?:you|we|i|one)\\s+(?:IMPVERBS))|need(?:s)? to(?: not)?|required to|prefer(?: not)? to|want to(?: avoid| ensure| use| keep))\\s+([^.,;!?\\n]{10,200})",
            "preference": "(?:(?:I|You|you|YOU)(?: |\\')?(?:like|love|prefer|hate|dislike|enjoy|use|stick with|switched to|moved to|changed to|want|need|tend to|usually|would rather|don\\'t like|don\\'t want|not a fan of|am okay with|am comfortable with|am used to|am happy with|am tired of|am sick of|prefer not to|try to avoid|find it easier to|find it better to|find it useful to)|(?:Nathan|Bob|User|Amy|Zander|Zella)\\s+(?:likes?|loves?|prefers?|hates?|dislikes?|enjoys?|uses?|wants?|needs?|tends\\s+to|switches?\\s+to|changes?\\s+to|moves?\\s+to)|(?:(?<=^)|(?<=\\n)|(?<=\\-\\s)|(?<=—\\s))(?:Prefers|Likes|Loves|Hates|Dislikes|Wants|Needs|Tends to|Enjoys|Uses))\\s+([^.,;!?\\n]{10,200})",
            "event_keywords": [
                "meeting",
                "call",
                "scheduled",
                "happened",
                "occurred",
                "plan to",
                "will be on",
                "due on",
                "release",
                "deadline",
                "launched",
                "deployed",
                "released",
                "published",
                "posted",
                "started",
                "began",
                "finished",
                "completed",
                "ended",
                "event",
                "conference",
                "workshop",
                "appointment",
            ],
            "named_months": "((?:(?<!\\w)\\d{1,2}(?:st|nd|rd|th)?\\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\b(?:(?:,\\s*|\\s+)\\d{4}(?!\\w)|(?!,?\\s*\\d)(?!\\w))|(?<!\\w)(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\b\\s+\\d{1,2}(?:st|nd|rd|th)?(?:(?:,\\s*|\\s+)\\d{4}(?!\\w)|(?!,?\\s*\\d)(?!\\w))))",
        },
        "de": {
            "negation": "\\b(Ich(?: habe|\\'ve)?\\s+(?:nie|niemals|nicht)\\s+[^.,;!?\\n]{15,120})",
            "decision": "(?:entschied(?: mich|en)?|habe mich entschieden|wechselte zu|umgestellt auf|umgestiegen auf|gewählt habe|ausgesucht|ausgewählt|genommen habe)\\s+([^.,;!?\\n]{10,120})",
            "entity": "(?:der|die|das|mein|meine|dein|deine|unser|unsere|Ihr|Ihre)\\s+([a-z_]+(?:\\s+(?:Tabelle|Modell|Schema|API|Endpunkt|Funktion|Modul|Route|Handler|Tool|Plugin|Script|Konfiguration|Einstellung|Workflow|Pipeline|Prozess|System|Server|Client|Service|Datenbank|Query|Datei|Repo|Branch|PR|Issue|Task|Job)))\\s+(?:braucht|benötigt|sollte|könnte|würde|wird|hat|hat|nutzt|verwendet|läuft|bearbeitet|verarbeitet|unterstützt)\\s+([^.,;!?\\n]{10,80})",
            "sequence": "((?:zuerst|als erstes|als zweites|als drittes|als viertes|als fünftes|schließlich|als nächstes|dann|danach|daraufhin)[^.,;!?\\n]{15,120})",
            "instruction_false_positives": [
                "du solltest gehen",
                "ich denke du solltest",
                "sollte funktionieren",
                "sollte klappen",
                "sollte passen",
                "sollte sich",
            ],
            "instruction_imperative": "immer|nie|niemals|merke|denk|verwende|nutze|behalte|vermeide|stelle sicher|prüfe|überprüfe|teste|baue|implementiere|schreibe|lösche|installiere|konfiguriere|aktualisiere|erstelle|entferne|starte|stoppe|setze|aktiviere|deaktiviere|füge hinzu|benenne um|sende|antworte",
            "instruction": "\\b(?:immer|nie|niemals|muss|darf nicht|sollte(?: nicht)?(?=\\s+(?:du|wir|ich|man|ihr)\\s+(?:IMPVERBS))|braucht|benötigt|möchte(?: vermeiden|sicherstellen|nutzen|behalten)|will(?: nicht)?)\\s+([^.,;!?\\n]{10,200})",
            "preference": "(?:Ich(?: |\\')?(?:mag|liebe|bevorzuge|hasse|mag nicht|nutze|verwende|benutze|bin bei geblieben|habe gewechselt zu|bin umgestiegen auf|bin umgestellt auf|will|möchte|brauche|tendiere zu|normalerweise|würde lieber|finde es einfacher|finde es besser|finde es nützlich|bin zufrieden mit|bin okay mit|bin es leid|versuche zu vermeiden))\\s+([^.,;!?\\n]{10,200})",
            "event_keywords": [
                "treffen",
                "meeting",
                "termin",
                "anruf",
                "geplant",
                "passiert",
                "stattgefunden",
                "fällig",
                "release",
                "deadline",
                "veröffentlicht",
                "deployed",
                "gestartet",
                "begonnen",
                "beendet",
                "abgeschlossen",
                "konferenz",
                "workshop",
                "termin",
            ],
            "named_months": "((?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\\s+\\d{1,2}(?:\\.)?\\s*(?:\\d{4})?)",
        },
        "it": {
            "negation": "\\b((?:Non(?: |')?(?:ho|ho mai|mai|non)\\s+[^.,;!?\\n]{15,120}))",
            "decision": "(?:ho deciso|mi sono deciso|ho scelto|ho optato|ho cambiato|sono passato|sono passata|ho selezionato|scelto)\\s+([^.,;!?\\n]{10,120})",
            "entity": "(?:il|la|i|le|il mio|la mia|i miei|le mie|il tuo|la tua|il nostro|la nostra)\\s+([a-z_]+(?:\\s+(?:tabella|modello|schema|API|endpoint|funzione|modulo|route|handler|tool|plugin|script|config|impostazione|workflow|pipeline|processo|sistema|server|client|servizio|database|query|file|repo|branch|PR|issue|task|job|progetto)))\\s+(?:ha bisogno|richiede|dovrebbe|potrebbe|vorra|ha|hanno|usa|usano|funziona|gestisce|processa|supporta)\\s+([^.,;!?\\n]{10,80})",
            "sequence": "((?:primo|prima|secondo|seconda|terzo|terza|quarto|quinta|infine|poi|dopo|dopodiche|successivamente|quindi)[^.,;!?\\n]{15,120})",
            "instruction_false_positives": [
                "dovresti andare",
                "penso che dovresti",
                "dovrebbe funzionare",
                "dovrebbe andare",
                "dovrebbe andare bene",
                "dovrebbe essere",
                "dovrebbe bastare",
            ],
            "instruction_imperative": "sempre|mai|ricorda|usa|tieni|evita|assicurati|controlla|verifica|esegui|testa|costruisci|distribuisci|fai push|fai pull|fai merge|chiudi|apri|aggiorna|installa|configura|imposta|abilita|disabilita|aggiungi|rimuovi|crea|elimina|avvia|ferma|riavvia|resetta|prova|implementa|scrivi|leggi|passa|sposta|copia|rinomina|invia|rispondi",
            "instruction": "\\b(?:sempre|mai|non deve|non devono|dovrebbe(?: non)?(?=\\s+(?:tu|voi|noi|io|si)\\s+(?:IMPVERBS))|ha bisogno di|deve|devono|preferisci(?: non)?|vuole(?: evitare|assicurarsi|usare|tenere))\\s+([^.,;!?\\n]{10,200})",
            "preference": "(?:Io(?: |')?(?:mi piace|amo|preferisco|odio|non mi piace|uso|utilizzo|sono passato a|ho cambiato a|voglio|ho bisogno|tendo a|di solito|preferirei|non mi piace per niente|non voglio|non sono un fan di|mi va bene|mi trovo bene|sono abituato a|sono felice con|sono stanco di|cerco di evitare|trovo piu facile|trovo meglio|trovo utile))\\s+([^.,;!?\\n]{10,200})",
            "event_keywords": [
                "riunione",
                "chiamata",
                "incontro",
                "programmato",
                "successo",
                "accaduto",
                "pianifico",
                "sara il",
                "scadenza",
                "rilascio",
                "lancio",
                "pubblicato",
                "iniziato",
                "cominciato",
                "finito",
                "completato",
                "evento",
                "conferenza",
                "workshop",
                "appuntamento",
            ],
            "named_months": "((?:(?:Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre|gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\\s+\\d{1,2}(?:°)?,?\\s*(?:\\d{4})?))",
        },
        "es": {
            "negation": "\\b(nunca|jamás|tampoco|ni\\s+(?:siquiera|de coña|loc[ao]|de broma|hablar)|no\\s+(?:me\\s+(?:gusta|convence|interesa|molesta|duele)|lo\\s+(?:hag[ao]s|haré|haría)|hace\\s+falta|quiero|voy\\s+a|sé|sabía|puedo|debo|es\\s+(?:para\\s+tanto|plan|momento)|tiene\\s+sentido|estoy\\s+(?:de\\s+acuerdo|seguro)|hay\\s+(?:derecho|manera|tipo|quien)|teng[ao]\\s+(?:ni\\s+idea|claro)|pienso|creo|son|era|está|estaba|será|está\\s+mal|vamos\\s+mal))\\s+([^.,;!?¿¡\\n]{15,120})",
            "decision": "(?:decid(?:í|ió|imos|iste|isteis|ieron|o|es|e|en)|opt(?:é|ó|amos|aste|asteis|aron|o|a|an)\\s+por|cambi(?:é|ó|amos|aste|asteis|aron|o|a|an)\\s+(?:de|a)|eleg(?:í|ió|imos|iste|isteis|ieron|o|es|e|en)|seleccion(?:é|ó|amos|aste|asteis|aron|o|a|an)|me\\s+(?:pas|decant|escog)(?:é|ó|amos|o|a|an)\\s+(?:a|por)|migr(?:é|ó|amos|aste|asteis|aron|o|a|an)\\s+(?:de|a)|actualic(?:é|ó|amos|aste|asteis|aron|o|a|an)\\s+(?:de|a)|sustitu(?:í|yó|imos|iste|isteis|yeron|yo|yes|ye|yen)\\s+por|elimin(?:é|ó|amos|aste|asteis|aron|o|a|an)|descart(?:é|ó|amos|aste|asteis|aron|o|a|an)|y\\s+si\\s+[^.,;!?¿¡\\n]{10,200}|mejor\\s+(?:si|así))\\s+([^.,;!?¿¡\\n]{10,120})",
            "entity": "(el|la|mi|tu|su|nuestr[oa]|vuestr[oa]|mis|tus|sus|los|las)\\s+(servidor|maquina|vm|contenedor|docker|nodo|clúster|cluster|router|enrutador|gateway|puerta\\s+de\\s+enlace|switch|ap|punto\\s+de\\s+acceso|firewall|cortafuegos|vpn|vlan|dns|dhcp|api|endpoint|función|funcio|módulo|modulo|servicio|proceso|script|plugin|tool|skill|base\\s+de\\s+datos|bd|tabla|query|consulta|log|backup|snapshot|sensor|cámara|camara|luz|interruptor|alarma|estación\\s+meteorológica|estacion\\s+meteorologica|automatización|automatizacion|puerta|repo|repositorio|rama|branch|pr|issue|tarea|workflow|pipeline|config|configuración|configuracion|ajuste|carpeta|opciones|archivo|fichero|dashboard|interfaz|sistema|actualización|actualizacion|versión|versio|despliegue|deploy|release|entorno)(?:\\s+(?:\\w+))?\\s+(?:necesita|requiere|debería|deberia|podría|podria|puede|tiene\\s+que|usa|utiliza|ejecuta|gestiona|maneja|procesa|soporta|funciona\\s+con|depende\\s+de|contiene|implementa|despliega|actualiza|configura|corre\\s+(?:en|sobre)|monitoriza|notifica|está|esta)\\s+([^.,;!?¿¡\\n]{10,80})",
            "sequence": "((?:primero|primeramente|en\\s+primer\\s+lugar|segundo|en\\s+segundo\\s+lugar|tercero|en\\s+tercer\\s+lugar|para\\s+empezar|yo\\s+empezaría\\s+por|yo\\s+empezaria\\s+por|por\\s+mi\\s+parte|por\\s+otro\\s+lado|luego|después|despues|a\\s+continuación|a\\s+continuacion|mientras\\s+tanto|al\\s+mismo\\s+tiempo|finalmente|por\\s+último|por\\s+ultimo|para\\s+terminar|antes\\s+de|acto\\s+seguido|por\\s+una\\s+parte|por\\s+otra\\s+parte|posteriormente)[^.,;!?¿¡\\n]{15,120})",
            "instruction_false_positives": [
                "evita perón",
                "evita peron",
                "goma de borrar",
                "guarda silencio",
                "busca la paz",
                "no cambies nunca",
                "comprimido efervescente",
                "copia de seguridad",
                "prueba a ver",
                "prueba y error",
                "mira tú por donde",
                "mira tu por donde",
                "baja la cabeza",
                "no deberías preocuparte por eso ahora",
                "no deberias preocuparte por eso ahora",
                "debería funcionar sin problemas",
                "deberia funcionar sin problemas",
                "mejor lo dejamos así",
                "mejor lo dejamos asi",
                "habría que verlo primero",
                "habria que verlo primero",
                "igual deberías preguntar antes",
                "igual deberias preguntar antes",
                "tendrías que probarlo tú mismo",
                "tendrias que probarlo tu mismo",
                "puedes hacer lo que quieras",
                "no hace falta que hagas nada",
                "yo que tú lo dejaba correr",
                "yo que tu lo dejaba correr",
                "a veces es mejor no tocar nada",
                "nunca he usado",
                "nunca he probado",
                "nunca he visto",
                "nunca he tenido",
                "nunca he hecho",
                "nunca he sido",
                "nunca has usado",
                "nunca ha usado",
                "nunca hemos usado",
                "nunca lo he",
                "nunca lo había",
                "nunca había",
            ],
            "instruction_imperative": "siempre|nunca|recuerda|recordad|recuerde|recuerden|haz|haced|haga|hagan|usa|usad|use|usen|mantén|mantened|mantenga|mantengan|evita|evitad|evite|eviten|asegúrate|aseguraos|asegúrese|asegúrense|asegurate|aseguraos|asegurese|asegurense|verifica|verificad|verifique|verifiquen|comprueba|comprobad|compruebe|comprueben|revisa|revisad|revise|revisen|ejecuta|ejecutad|ejecute|ejecuten|prueba|probad|pruebe|prueben|pon|poned|ponga|pongan|configura|configurad|configure|configuren|instala|instalad|instale|instalen|actualiza|actualizad|actualice|actualicen|borra|borrad|borre|borren|guarda|guardad|guarde|guarden|busca|buscad|busque|busquen|despliega|desplegad|despliegue|desplieguen|crea|cread|cree|creen|memoriza|memorizad|memorice|memoricen|graba|grabad|grabe|graben|añade|añadid|añada|añadan|anade|anadid|anada|anadan|cambia|cambiad|cambie|cambien|arregla|arreglad|arregle|arreglen|sube|subid|suba|suban|baja|bajad|baje|bajen|carga|cargad|cargue|carguen|descarga|descargad|descargue|descarguen|comprime|comprimid|comprima|compriman|descomprime|descomprimid|descomprima|descompriman|copia|copiad|copie|copien|mueve|moved|mueva|muevan",
            "instruction": "\\b(?:siempre|nunca|hay\\s+que|deb(?:es|éis|e|en|o|emos|éis|en)\\s+|tienes\\s+que|tenéis\\s+que|tiene\\s+que|tienen\\s+que|es\\s+necesario|es\\s+importante|es\\s+mejor|es\\s+aconsejable|asegúrate\\s+de|asegurate\\s+de|record(?:ad|a|e|en)\\s+|no\\s+olvid(?:es|éis|e|en|ad)\\s+)([^.,;!?¿¡\\n]{10,200})",
            "preference": "(?:(?:yo|a mí|a mi)\\s+)?(?:me\\s+(?:gusta|encanta|mola|flipa|chifla|va\\s+bien|resulta\\s+(?:cómodo|comodo|útil|util|fácil|facil|mejor))|no\\s+me\\s+(?:gusta|mola|interesa|va|conviene)|prefiero|preferiría|preferiria|odian?|odio|detesto|no\\s+soporto|me\\s+molesta|me\\s+duele|no\\s+quiero|paso\\s+de|estoy\\s+(?:harto|cansado)\\s+de|estoy\\s+acostumbrado\\s+a|suelo\\s+usar|suelo\\s+trabajar|me\\s+siento\\s+cómodo|comodo\\s+con|no\\s+soy\\s+fan\\s+de|he\\s+(?:empezado|dejado|comenzado|terminado)\\s+(?:a|de)|dejé|deje|descarte|descarté|eliminé|elimine|cambié|cambie|me\\s+quedo\\s+con|me\\s+decanto\\s+por|disfruto|me\\s+hace\\s+feliz|estoy\\s+(?:a\\s+gusto|probando))\\s+([^.,;!?¿¡\\n]{10,200})",
            "event_keywords": [
                "reunión",
                "reunion",
                "llamada",
                "cita",
                "meeting",
                "daily",
                "sprint",
                "planning",
                "retro",
                "review",
                "revisión",
                "revision",
                "demo",
                "demostración",
                "demostracion",
                "evento",
                "conferencia",
                "taller",
                "workshop",
                "webinar",
                "seminario",
                "cumpleaños",
                "cumpleanos",
                "aniversario",
                "festivo",
                "vacaciones",
                "programado",
                "agendado",
                "planeado",
                "previsto",
                "pendiente",
                "deadline",
                "fecha límite",
                "fecha tope",
                "entrega",
                "lanzamiento",
                "release",
                "despliegue",
                "deploy",
                "publicación",
                "publicacion",
                "subida",
                "empezó",
                "empezo",
                "comenzó",
                "comenzo",
                "inició",
                "inicio",
                "arrancó",
                "arranco",
                "terminó",
                "termino",
                "finalizó",
                "finalizo",
                "acabó",
                "acabo",
                "completé",
                "complete",
                "lanzamos",
                "publicamos",
                "desplegamos",
                "implementamos",
                "tengo una cita",
                "tenemos una reunión",
                "vamos a vernos",
                "agendé",
                "agende",
                "ocurrió",
                "ocurrio",
                "sucedió",
                "sucedio",
                "pasó",
                "paso",
            ],
            "named_months": "((?:\\d{1,2})\\s*de\\s*(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\\s*(?:de\\s*(?:\\d{4}))?)",
        },
    }

    def extract_and_store_facts(
        self, content: str, message_idx: int = 0, source_memory_id: Optional[str] = None
    ) -> dict:
        """Extract structured facts from a message and store in facts/timelines/kg tables.
        Uses regex patterns matching the BEAM benchmark oracles.
        Language-aware: detects language and uses language-specific patterns.
        Returns dict of counts per fact_type."""
        import re as _re

        counts = {
            "metric": 0,
            "date": 0,
            "version": 0,
            "entity": 0,
            "sequence": 0,
            "timeline": 0,
            "negation": 0,
            "decision": 0,
        }
        session = self.session_id
        lang = self.detect_language(content)
        pat = self.MULTILINGUAL_PATTERNS.get(lang, self.MULTILINGUAL_PATTERNS["en"])
        _metric_re_iter = list(
            _re.finditer(
                "(\\d+(?:[.,]\\d+)?)\\s*(ms|sec|seconds?|minutes?|hours?|days?|weeks?|months?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)",
                content,
                _re.IGNORECASE,
            )
        )
        _TRANSIENT_KEYWORDS = (
            "forecast",
            "weather",
            "temperature",
            "rain",
            "snow",
            "wind",
            "humidity",
            "chance",
            "regenrisiko",
            "zum",
            "heute",
            "morgen",
            "gestern",
            "today",
            "tomorrow",
            "yesterday",
            "week",
            "month",
        )
        for m in _metric_re_iter[:10]:
            num = m.group(1)
            unit = m.group(2)
            unit_clean = unit.lower()
            if unit_clean.endswith("s") and (not unit_clean.endswith("ms")):
                unit_clean = unit_clean[:-1]
            pre_text = content[max(0, m.start() - 50) : m.start()]
            if any((kw in pre_text.lower() for kw in _TRANSIENT_KEYWORDS)):
                continue
            _clean_pre = _re.sub("`[^`]*`", " ", pre_text)
            _clean_pre = _re.sub("\\*\\*[^*]+\\*\\*", " ", _clean_pre)
            _clean_pre = _re.sub("[*_]{1,2}[^*_\\n]+[*_]{1,2}", " ", _clean_pre)
            _clean_pre = _re.sub("[=<>|&]", " ", _clean_pre)
            ctx_words = [
                w.strip(".,:;!?()[]\"'`*_")
                for w in _clean_pre.split()
                if len(w.strip(".,:;!?()[]\"'`*_")) > 2
                and w.lower()
                not in (
                    "the",
                    "and",
                    "for",
                    "was",
                    "of",
                    "to",
                    "a",
                    "an",
                    "in",
                    "on",
                    "at",
                    "by",
                    "is",
                    "are",
                    "has",
                    "had",
                    "not",
                    "but",
                    "or",
                )
                and (
                    not _re.search(
                        "^(pt-|lg:|pr-|pl-|pb-|px-|py-|mt-|mr-|mb-|ml-|mx-|my-)", w
                    )
                )
                and (not _re.search("^[`*\\]]", w))
            ][-3:]
            prefix = "_".join((w.lower() for w in ctx_words)) if ctx_words else ""
            key = f"{prefix}_{unit_clean}" if prefix else unit_clean
            if "`" in key or "**" in key or key.count("**") > 0:
                continue
            if _re.search("[*_]{2,}", key):
                continue
            if len(_re.findall("[`=<>|]", key)) > 2:
                continue
            if unit_clean == "%":
                _nonalpha = len(_re.findall("[^a-zA-Z0-9\\s]", _clean_pre))
                _words = len(_clean_pre.split())
                if _words > 0 and _nonalpha / _words > 0.6:
                    continue
            val = f"{num}{unit}"
            if unit_clean == "%":
                key = key.replace("_%", "_pct")
                if not key.endswith("_pct"):
                    key = f"{prefix}_pct" if prefix else "pct"
            self._insert_fact(
                session,
                message_idx,
                "metric",
                key,
                val,
                self._context_snippet(content, m.start()),
                0.65,
                source_memory_id=source_memory_id,
            )
            counts["metric"] += 1
        _EVENT_KEYWORDS = pat["event_keywords"]
        for m in _re.finditer("\\b(\\d{4}-\\d{2}-\\d{2})\\b", content):
            dt = m.group(1)
            ctx = self._context_snippet(content, m.start(), width=100)
            _ctx_lower = ctx.lower()
            _has_event_context = any((kw in _ctx_lower for kw in _EVENT_KEYWORDS))
            if not _has_event_context:
                self._insert_fact(
                    session,
                    message_idx,
                    "date",
                    "iso_date",
                    dt,
                    ctx,
                    0.5,
                    source_memory_id=source_memory_id,
                )
                counts["date"] += 1
            else:
                self._insert_fact(
                    session,
                    message_idx,
                    "date",
                    "iso_date",
                    dt,
                    ctx,
                    0.7,
                    source_memory_id=source_memory_id,
                )
                self._insert_timeline(
                    session,
                    dt,
                    message_idx,
                    ctx[:120],
                    "iso_date",
                    source_memory_id=source_memory_id,
                )
                counts["date"] += 1
                counts["timeline"] += 1
        for m in _re.finditer(pat["named_months"], content, _re.IGNORECASE):
            dt = m.group(1).strip()
            ctx = self._context_snippet(content, m.start())
            self._insert_fact(
                session,
                message_idx,
                "date",
                "named_date",
                dt,
                ctx,
                0.7,
                source_memory_id=source_memory_id,
            )
            counts["date"] += 1
        for m in _VERSION_STRING_RE.finditer(content):
            name = m.group(1).strip()
            ver = m.group(2)
            key = f"{name.lower().replace(' ', '_')}_version"
            self._insert_fact(
                session,
                message_idx,
                "version",
                key,
                ver,
                self._context_snippet(content, m.start()),
                0.7,
                source_memory_id=source_memory_id,
            )
            counts["version"] += 1
        _seen_versions = set()
        for m in _re.finditer(
            "([A-Z][a-zA-Z]+)\\s+version\\s+v?(\\d+\\.\\d+(?:\\.\\d+)?)",
            content,
            _re.IGNORECASE,
        ):
            name = m.group(1).strip()
            ver = m.group(2)
            if name.lower() in (
                "running",
                "using",
                "installed",
                "upgraded",
                "currently",
            ):
                continue
            key = f"{name.lower().replace(' ', '_')}_version"
            if ver not in _seen_versions:
                _seen_versions.add(ver)
                self._insert_fact(
                    session,
                    message_idx,
                    "version",
                    key,
                    ver,
                    self._context_snippet(content, m.start()),
                    0.7,
                    source_memory_id=source_memory_id,
                )
                counts["version"] += 1
        for m in _re.finditer(pat["negation"], content, _re.IGNORECASE):
            neg_text = m.group(1).strip()
            neg_lower = neg_text.lower()
            if lang == "de":
                split_words = ["nie", "niemals", "nicht"]
            else:
                split_words = ["never", "not"]
            obj = neg_text
            for sw in split_words:
                if sw in neg_lower:
                    parts = neg_text.split(sw, 1)
                    if len(parts) > 1:
                        obj = parts[-1].strip()
                        break
            self._insert_kg(
                session,
                "user",
                "negation",
                obj[:80],
                message_idx,
                0.75,
                source_memory_id=source_memory_id,
            )
            counts["negation"] += 1
        for m in _re.finditer(pat["decision"], content, _re.IGNORECASE):
            decision = m.group(1).strip()
            self._insert_kg(
                session,
                "user",
                "decision",
                decision,
                message_idx,
                0.65,
                source_memory_id=source_memory_id,
            )
            counts["decision"] += 1
        for m in _re.finditer(pat["entity"], content, _re.IGNORECASE):
            entity = m.group(1).strip()
            action = m.group(2).strip()
            self._insert_kg(
                session,
                entity,
                "requires",
                action,
                message_idx,
                0.65,
                source_memory_id=source_memory_id,
            )
            counts["decision"] += 1
        for m in _re.finditer(pat["sequence"], content, _re.IGNORECASE):
            seq = m.group(1).strip()
            first_word = seq.split()[0].lower()
            self._insert_fact(
                session,
                message_idx,
                "sequence",
                first_word,
                seq[:120],
                self._context_snippet(content, m.start()),
                0.6,
                source_memory_id=source_memory_id,
            )
            counts["sequence"] += 1
        _INSTRUCTION_FALSE_POSITIVES = pat["instruction_false_positives"]
        _INSTR_IMPERATIVE_VERBS = pat["instruction_imperative"]
        _instr_re = pat["instruction"].replace("IMPVERBS", _INSTR_IMPERATIVE_VERBS)
        for m in _re.finditer(_instr_re, content, _re.IGNORECASE):
            instr = m.group(0).strip()
            topic = m.group(1).strip()[:60]
            _instr_lower = instr.lower()
            if any((fp in _instr_lower for fp in _INSTRUCTION_FALSE_POSITIVES)):
                continue
            if _re.match(
                "^(?:should|sollte|dovrebbe|dovresti)\\s+(?:i|we|it|they|he|she|the|ich|wir|es|man|der|die|das|io|noi|lui|lei|loro)\\b",
                instr,
                _re.IGNORECASE,
            ):
                continue
            self.conn.execute(
                "INSERT INTO memoria_instructions (session_id, message_idx, instruction, topic, context_snippet, source_memory_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session,
                    message_idx,
                    instr[:200],
                    topic,
                    self._context_snippet(content, m.start()),
                    source_memory_id,
                ),
            )
            counts["instruction"] = counts.get("instruction", 0) + 1
        for m in _re.finditer(pat["preference"], content, _re.IGNORECASE):
            pref = m.group(0).strip()
            topic = m.group(1).strip()[:60]
            _topic_key = (
                " ".join(
                    (
                        w
                        for w in _re.findall("[a-zA-Z]{4,}", topic)
                        if w.lower() not in _FACT_MATCH_STOPWORDS
                    )
                )[:30]
                or topic[:20]
            )
            existing = self.conn.execute(
                "SELECT preference, topic FROM memoria_preferences WHERE session_id = ? AND (topic LIKE ? OR preference LIKE ?) ORDER BY message_idx DESC LIMIT 1",
                (session, f"%{_topic_key}%", f"%{_topic_key}%"),
            ).fetchone()
            evolution = None
            if existing:
                evolution = f"was: {existing[0][:120]}"
            self.conn.execute(
                "INSERT INTO memoria_preferences (session_id, message_idx, preference, topic, evolution, context_snippet, source_memory_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session,
                    message_idx,
                    pref[:200],
                    topic,
                    evolution,
                    self._context_snippet(content, m.start()),
                    source_memory_id,
                ),
            )
            counts["preference"] = counts.get("preference", 0) + 1
        self.conn.commit()
        counts["_lang"] = lang
        return counts

    def _insert_fact(
        self,
        session: str,
        msg_idx: int,
        ftype: str,
        key: str,
        value: str,
        ctx: str,
        importance: float,
        source_memory_id: Optional[str] = None,
    ):
        if ftype == "date":
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet, importance, valid_from_msg_idx, source_memory_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session,
                    msg_idx,
                    ftype,
                    key,
                    value,
                    ctx,
                    importance,
                    msg_idx,
                    source_memory_id,
                ),
            )
            return
        existing = self.conn.execute(
            "SELECT id, value FROM memoria_facts WHERE session_id = ? AND key = ? AND fact_type = ? AND valid_to_msg_idx IS NULL ORDER BY version_id DESC LIMIT 1",
            (session, key, ftype),
        ).fetchone()
        if existing and existing[1] != value:
            self.conn.execute(
                "UPDATE memoria_facts SET valid_to_msg_idx = ?, previous_value = value WHERE id = ?",
                (msg_idx, existing[0]),
            )
            prev_version = self.conn.execute(
                "SELECT version_id FROM memoria_facts WHERE id = ?", (existing[0],)
            ).fetchone()
            new_version = prev_version[0] + 1 if prev_version else 1
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet, importance, version_id, previous_value, updated_msg_idx, valid_from_msg_idx, source_memory_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session,
                    msg_idx,
                    ftype,
                    key,
                    value,
                    ctx,
                    importance,
                    new_version,
                    existing[1],
                    msg_idx,
                    msg_idx,
                    source_memory_id,
                ),
            )
        else:
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, context_snippet, importance, valid_from_msg_idx, source_memory_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session,
                    msg_idx,
                    ftype,
                    key,
                    value,
                    ctx,
                    importance,
                    msg_idx,
                    source_memory_id,
                ),
            )

    def _insert_timeline(
        self,
        session: str,
        date: str,
        msg_idx: int,
        desc: str,
        source: str = "extraction",
        source_memory_id: Optional[str] = None,
    ):
        self.conn.execute(
            "INSERT INTO memoria_timelines (session_id, date, message_idx, description, source, source_memory_id) VALUES (?, ?, ?, ?, ?, ?)",
            (session, date, msg_idx, desc, source, source_memory_id),
        )

    def _insert_kg(
        self,
        session: str,
        subject: str,
        predicate: str,
        obj: str,
        msg_idx: int,
        confidence: float = 0.7,
        source_memory_id: Optional[str] = None,
    ):
        self.conn.execute(
            "INSERT INTO memoria_kg (session_id, subject, predicate, object, message_idx, confidence, source_memory_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session, subject, predicate, obj, msg_idx, confidence, source_memory_id),
        )

    @staticmethod
    def _context_snippet(content: str, pos: int, width: int = 60) -> str:
        """Extract surrounding context around a position in content."""
        start = max(0, pos - width)
        end = min(len(content), pos + width)
        snippet = content[start:end].strip()
        if start > 0:
            snippet = f"...{snippet}"
        if end < len(content):
            snippet = f"{snippet}..."
        return snippet[:200]

    @staticmethod
    def _normalize_spanish_accent(text: str) -> str:
        """Strip Spanish diacritics so 'configuración' matches 'configuracion'."""
        _accent_map = {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "Á": "A",
            "É": "E",
            "Í": "I",
            "Ó": "O",
            "Ú": "U",
            "ü": "u",
            "Ü": "U",
            "ñ": "n",
            "Ñ": "N",
        }
        return "".join((_accent_map.get(c, c) for c in text))

    def memoria_retrieve(
        self, query: str, ability: str = None, top_k: int = 10
    ) -> dict:
        """Route a query to the appropriate MEMORIA specialist table.
        Returns dict with keys: context (str), facts (list), source (str).
        Falls back to {'context': '', 'facts': [], 'source': 'fallback'} when empty."""
        result = {"context": "", "facts": [], "source": "fallback"}
        if not ability:
            ability = self._classify_ability(query)
        if ability in ("IE", "KU"):
            return self._memoria_fact_retrieve(query, top_k)
        elif ability == "TR":
            return self._memoria_timeline_retrieve(query, top_k)
        elif ability == "CR":
            return self._memoria_negation_retrieve(query, top_k)
        elif ability == "MR":
            return self._memoria_entity_retrieve(query, top_k)
        elif ability == "EO":
            return self._memoria_chrono_retrieve(query, top_k)
        elif ability == "IF":
            return self._memoria_instruction_retrieve(query, top_k)
        elif ability == "PF":
            return self._memoria_preference_retrieve(query, top_k)
        else:
            return result

    @staticmethod
    def _classify_ability(query: str) -> str:
        """Classify a question into BEAM ability based on keywords.
        Returns ability string or empty for unclassified."""
        from . import jev

        label, p = jev.choose(
            query,
            "Choose the memory query task.",
            {
                "TR": "Date or duration",
                "EO": "Event ordering",
                "CR": "Contradictions or changed facts",
                "IE": "Concrete information extraction",
                "PF": "Preferences",
                "IF": "Instructions",
                "MR": "Relationships across facts",
                "ABS": "User background",
                "unknown": "No applicable task",
            },
        )
        return label if label != "unknown" and p >= 0.6 else ""

    def _memoria_fact_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_facts table for exact metric/version/entity matches.
        Uses multi-pass strategy:
          Pass 1: numbers from query → search fact values
          Pass 2: capitalized terms → search keys + values
          Pass 3: synonym map (latency→ms, version→version, etc.) → search by type+unit
          Pass 4: context_snippet fallback → search raw surrounding text
        Returns {'context': str, 'facts': list, 'source': str} or fallback."""
        import re as _re

        facts = []
        seen = set()
        cursor = self.conn
        q_lower = query.lower()
        numbers = _re.findall("\\b(\\d+)\\b", query)
        for num in numbers[:3]:
            rows = cursor.execute(
                "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts WHERE value LIKE ? AND session_id = ? LIMIT ?",
                (f"%{num}%", self.session_id, top_k),
            ).fetchall()
            for row in rows:
                fk = (row[1], row[2])
                if fk not in seen:
                    seen.add(fk)
                    facts.append(
                        dict(
                            zip(
                                [
                                    "type",
                                    "key",
                                    "value",
                                    "context",
                                    "previous_value",
                                    "updated_msg_idx",
                                    "version_id",
                                    "source_memory_id",
                                ],
                                row,
                            )
                        )
                    )
        terms = _re.findall("\\b[A-Z][a-z]+(?:[-][A-Z][a-z]+)*\\b", query)
        stop_words = {
            "Have",
            "Did",
            "Do",
            "Can",
            "Will",
            "Would",
            "Should",
            "What",
            "When",
            "Where",
            "Which",
            "Who",
            "How",
            "Why",
            "Is",
            "Are",
            "Was",
            "Were",
            "The",
            "A",
            "An",
            "This",
            "That",
            "My",
            "Me",
            "I",
            "You",
            "How",
            "Many",
            "Much",
        }
        terms = [t for t in terms if t not in stop_words]
        for term in terms[:5]:
            rows = cursor.execute(
                "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts WHERE (key LIKE ? OR value LIKE ?) AND session_id = ? LIMIT ?",
                (f"%{term}%", f"%{term}%", self.session_id, top_k),
            ).fetchall()
            for row in rows:
                fk = (row[1], row[2])
                if fk not in seen:
                    seen.add(fk)
                    facts.append(
                        dict(
                            zip(
                                [
                                    "type",
                                    "key",
                                    "value",
                                    "context",
                                    "previous_value",
                                    "updated_msg_idx",
                                    "version_id",
                                    "source_memory_id",
                                ],
                                row,
                            )
                        )
                    )
        _SYNONYM_MAP = [
            ("version", "version", None),
            ("latency", "metric", ["ms"]),
            ("speed", "metric", ["ms"]),
            ("response time", "metric", ["ms"]),
            ("how many", "metric", None),
            ("how much", "metric", None),
            ("what date", "date", None),
            ("what day", "date", None),
            ("deployed", "date", None),
            ("deploy", "date", None),
            ("released", "date", None),
            ("release", "date", None),
            ("launched", "date", None),
        ]
        if not facts:
            for phrase, ftype, unit_hints in _SYNONYM_MAP:
                if phrase in q_lower:
                    if unit_hints:
                        for unit in unit_hints:
                            rows = cursor.execute(
                                "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts WHERE fact_type = ? AND key LIKE ? AND session_id = ? LIMIT ?",
                                (ftype, f"%{unit}%", self.session_id, top_k),
                            ).fetchall()
                    else:
                        rows = cursor.execute(
                            "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts WHERE fact_type = ? AND session_id = ? LIMIT ?",
                            (ftype, self.session_id, top_k),
                        ).fetchall()
                    for row in rows:
                        fk = (row[1], row[2])
                        if fk not in seen:
                            seen.add(fk)
                            facts.append(
                                dict(
                                    zip(
                                        [
                                            "type",
                                            "key",
                                            "value",
                                            "context",
                                            "previous_value",
                                            "updated_msg_idx",
                                            "version_id",
                                            "source_memory_id",
                                        ],
                                        row,
                                    )
                                )
                            )
                    if facts:
                        break
        if not facts:
            q_stop = {
                "what",
                "when",
                "where",
                "which",
                "who",
                "how",
                "why",
                "is",
                "are",
                "was",
                "were",
                "do",
                "does",
                "did",
                "can",
                "will",
                "would",
                "should",
                "could",
                "may",
                "the",
                "a",
                "an",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "my",
                "me",
                "i",
                "you",
                "it",
                "its",
                "this",
                "that",
                "these",
                "those",
                "tell",
                "list",
                "describe",
                "explain",
                "walk",
                "me",
                "through",
            }
            q_words = [
                w for w in _re.findall("\\b[a-zA-Z]{3,}\\b", q_lower) if w not in q_stop
            ]
            for word in q_words[:5]:
                rows = cursor.execute(
                    "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts WHERE context_snippet LIKE ? AND session_id = ? LIMIT ?",
                    (f"%{word}%", self.session_id, top_k),
                ).fetchall()
                for row in rows:
                    fk = (row[1], row[2])
                    if fk not in seen:
                        seen.add(fk)
                        facts.append(
                            dict(
                                zip(
                                    [
                                        "type",
                                        "key",
                                        "value",
                                        "context",
                                        "previous_value",
                                        "updated_msg_idx",
                                        "version_id",
                                        "source_memory_id",
                                    ],
                                    row,
                                )
                            )
                        )
                if facts:
                    break
        if facts:
            from collections import defaultdict

            by_key: dict = defaultdict(list)
            for f in facts:
                by_key[f["key"]].append(f)
            latest: list = []
            for key, versions in by_key.items():
                versions.sort(key=lambda x: x.get("version_id", 0), reverse=True)
                newest = versions[0]
                if len(versions) > 1:
                    prevs = [v["value"] for v in versions[1:]]
                    newest["evolution"] = (
                        " -> ".join(reversed(prevs)) + f" -> {newest['value']}"
                    )
                latest.append(newest)
            latest.sort(key=lambda x: x.get("version_id", 0), reverse=True)
            ctx_lines = []
            for f in latest[:top_k]:
                line = f"[Fact {f['type']}] {f['key']}: {f['value']}"
                if f.get("evolution"):
                    line += f" (evolved: {f['evolution']})"
                elif f.get("previous_value") and f.get("version_id", 0) > 0:
                    line += f" (was: {f['previous_value']}, updated at msg_idx {f.get('updated_msg_idx', '?')})"
                ctx_lines.append(line)
            return {
                "context": "\n".join(ctx_lines),
                "facts": latest[:top_k],
                "source": "memoria_facts",
                "source_memory_ids": [
                    f["source_memory_id"]
                    for f in latest[:top_k]
                    if f.get("source_memory_id")
                ],
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_timeline_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_timelines for chronological events matching query terms."""
        import re as _re

        cursor = self.conn
        date_terms = _re.findall("\\b(\\d{4}-\\d{2}-\\d{2})\\b", query)
        month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        months_in_query = [m for m in month_names if m in query.lower()]
        if date_terms:
            rows = cursor.execute(
                "SELECT date, description, message_idx FROM memoria_timelines WHERE date LIKE ? AND session_id = ? ORDER BY date LIMIT ?",
                (f"%{date_terms[0]}%", self.session_id, top_k),
            ).fetchall()
        elif months_in_query:
            month = months_in_query[0][:3]
            rows = cursor.execute(
                "SELECT date, description, message_idx FROM memoria_timelines WHERE date LIKE ? AND session_id = ? ORDER BY date LIMIT ?",
                (f"{month}%", self.session_id, top_k),
            ).fetchall()
        else:
            rows = cursor.execute(
                "SELECT date, description, message_idx FROM memoria_timelines WHERE session_id = ? ORDER BY date DESC LIMIT ?",
                (self.session_id, top_k),
            ).fetchall()
        if rows:
            facts = [dict(zip(["date", "description", "msg_idx"], r)) for r in rows]
            ctx_lines = [f"[{r[0]}] {r[1][:120]}" for r in rows]
            return {
                "context": "\n".join(ctx_lines),
                "facts": facts,
                "source": "memoria_timelines",
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_negation_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_kg for negation predicates matching query terms."""
        import re as _re

        cursor = self.conn
        terms = _re.findall("\\b[A-Z][a-z]+\\b", query)
        stop_words = {"Have", "Did", "Do", "Can", "Will", "Would", "Should"}
        terms = [t for t in terms if len(t) > 3 and t not in stop_words]
        if not terms:
            terms = [w for w in query.split() if len(w) > 3][:3]
        for term in terms:
            rows = cursor.execute(
                "SELECT subject, object, message_idx FROM memoria_kg WHERE predicate='negation' AND (subject LIKE ? OR object LIKE ?) AND session_id = ? LIMIT ?",
                (f"%{term}%", f"%{term}%", self.session_id, top_k),
            ).fetchall()
            if rows:
                facts = [dict(zip(["subject", "object", "msg_idx"], r)) for r in rows]
                ctx_lines = [f"[Negation] user said never/not: {r[1]}" for r in rows]
                return {
                    "context": "\n".join(ctx_lines),
                    "facts": facts,
                    "source": "memoria_kg_negation",
                }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_entity_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_kg for entity-action pairs (predicates: requires, decision)."""
        import re as _re

        cursor = self.conn
        terms = _re.findall("\\b[A-Z][a-z]+\\b", query)
        stop_words = {
            "Have",
            "Did",
            "Do",
            "Can",
            "Will",
            "Would",
            "Should",
            "What",
            "When",
            "Where",
            "Which",
            "Who",
            "How",
            "Why",
        }
        entities = [t.lower() for t in terms if t not in stop_words and len(t) > 3]
        rows = []
        if entities:
            for entity in entities[:3]:
                rows = cursor.execute(
                    "SELECT subject, predicate, object, message_idx FROM memoria_kg WHERE (subject LIKE ? OR object LIKE ?) AND session_id = ? LIMIT ?",
                    (f"%{entity}%", f"%{entity}%", self.session_id, top_k),
                ).fetchall()
                if rows:
                    break
        if not rows:
            rows = cursor.execute(
                "SELECT subject, predicate, object, message_idx FROM memoria_kg WHERE session_id = ? ORDER BY message_idx LIMIT ?",
                (self.session_id, top_k),
            ).fetchall()
        if rows:
            facts = [
                dict(zip(["subject", "predicate", "object", "msg_idx"], r))
                for r in rows
            ]
            ctx_lines = [f"[{r[1]}] {r[0]} -> {r[2]}" for r in rows]
            return {
                "context": "\n".join(ctx_lines),
                "facts": facts,
                "source": "memoria_kg",
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_chrono_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_facts for sequence markers, ordered by message_idx."""
        cursor = self.conn
        rows = cursor.execute(
            "SELECT value, message_idx FROM memoria_facts WHERE fact_type='sequence' AND session_id = ? ORDER BY message_idx ASC LIMIT ?",
            (self.session_id, top_k),
        ).fetchall()
        if rows:
            facts = [dict(zip(["sequence", "msg_idx"], r)) for r in rows]
            ctx_lines = [f"[{i + 1}] {r[0]}" for i, r in enumerate(rows)]
            return {
                "context": "\n".join(ctx_lines),
                "facts": facts,
                "source": "memoria_sequences",
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_instruction_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_instructions for user constraints matching query terms."""
        import re as _re

        cursor = self.conn
        q_lower = query.lower()
        stop_words = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "why",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "will",
            "would",
            "should",
            "could",
            "may",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "my",
            "me",
            "i",
            "you",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "tell",
            "list",
            "describe",
            "explain",
            "have",
            "has",
            "had",
            "am",
        }
        q_words = [
            w for w in _re.findall("\\b[a-zA-Z]{3,}\\b", q_lower) if w not in stop_words
        ]
        rows = []
        if q_words:
            for word in q_words[:5]:
                rows = cursor.execute(
                    "SELECT instruction, topic, message_idx, context_snippet FROM memoria_instructions WHERE (instruction LIKE ? OR topic LIKE ?) AND session_id = ? AND active = 1 LIMIT ?",
                    (f"%{word}%", f"%{word}%", self.session_id, top_k),
                ).fetchall()
                if rows:
                    break
        if not rows:
            rows = cursor.execute(
                "SELECT instruction, topic, message_idx, context_snippet FROM memoria_instructions WHERE session_id = ? AND active = 1 ORDER BY message_idx DESC LIMIT ?",
                (self.session_id, top_k),
            ).fetchall()
        if rows:
            facts = [
                dict(zip(["instruction", "topic", "msg_idx", "context"], r))
                for r in rows
            ]
            ctx_lines = [f"[Instruction] {r[0][:120]}" for r in rows]
            return {
                "context": "\n".join(ctx_lines),
                "facts": facts,
                "source": "memoria_instructions",
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_preference_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_preferences for evolving user tastes matching query terms."""
        import re as _re

        cursor = self.conn
        q_lower = query.lower()
        stop_words = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "why",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "will",
            "would",
            "should",
            "could",
            "may",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "my",
            "me",
            "i",
            "you",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "tell",
            "list",
            "describe",
            "explain",
            "have",
            "has",
            "had",
            "am",
        }
        q_words = [
            w for w in _re.findall("\\b[a-zA-Z]{3,}\\b", q_lower) if w not in stop_words
        ]
        rows = []
        if q_words:
            for word in q_words[:5]:
                rows = cursor.execute(
                    "SELECT preference, topic, message_idx, evolution, context_snippet FROM memoria_preferences WHERE (preference LIKE ? OR topic LIKE ?) AND session_id = ? LIMIT ?",
                    (f"%{word}%", f"%{word}%", self.session_id, top_k),
                ).fetchall()
                if rows:
                    break
        if not rows:
            rows = cursor.execute(
                "SELECT preference, topic, message_idx, evolution, context_snippet FROM memoria_preferences WHERE session_id = ? ORDER BY message_idx DESC LIMIT ?",
                (self.session_id, top_k),
            ).fetchall()
        if rows:
            facts = [
                dict(zip(["preference", "topic", "msg_idx", "evolution", "context"], r))
                for r in rows
            ]
            ctx_lines = []
            for r in rows:
                line = f"[Preference] {r[0][:120]}"
                if r[3]:
                    line += f" ({r[3]})"
                ctx_lines.append(line)
            return {
                "context": "\n".join(ctx_lines),
                "facts": facts,
                "source": "memoria_preferences",
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def recall(
        self,
        query: str,
        top_k: int = 40,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        source: Optional[str] = None,
        topic: Optional[str] = None,
        author_id: Optional[str] = None,
        author_type: Optional[str] = None,
        channel_id: Optional[str] = None,
        veracity: Optional[str] = None,
        memory_type: Optional[str] = None,
        temporal_weight: float = 0.0,
        query_time: Optional[Any] = None,
        temporal_halflife: Optional[float] = None,
        vec_weight: float = None,
        fts_weight: float = None,
        importance_weight: float = None,
        explain: bool = False,
        _cross_session: Optional[bool] = None,
        _resolved_weights: Optional[_RecallWeightSnapshot] = None,
        exclude_captures: Optional[ExclusionSnapshot] = None,
        evidence_questions: Optional[List[str]] = None,
    ) -> List[Dict]:
        "Evaluate every eligible memory with Jev, applying scope and temporal filters. Legacy scoring weight parameters are accepted and ignored."
        cross_session = (
            _cross_session_enabled() if _cross_session is None else _cross_session
        )
        from . import jev
        from .jev_recall import recall

        return recall(
            self,
            query,
            top_k,
            explain=explain,
            cross_session=cross_session,
            evidence_questions=evidence_questions,
            exclude_captures=exclude_captures,
            from_date=from_date,
            to_date=to_date,
            source=source,
            topic=topic,
            author_id=author_id,
            author_type=author_type,
            channel_id=channel_id,
            veracity=veracity,
            memory_type=memory_type,
            temporal_weight=temporal_weight,
            query_time=query_time,
            temporal_halflife=temporal_halflife,
        )

    _ENHANCED_RECALL_CACHE_VERSION = 8

    def recall_enhanced(
        self,
        query: str,
        top_k: int = 40,
        *,
        use_cache: bool = True,
        use_weibull: bool = True,
        use_mmr: bool = True,
        use_intent: bool = True,
        use_synonyms: bool = True,
        use_associative: bool = False,
        associative_depth: int = 1,
        mmr_lambda: float = 0.7,
        **kwargs,
    ) -> List[Dict]:
        "Compatibility wrapper for Jev recall; legacy retrieval switches are ignored."
        import os as _os
        from . import jev

        return self.recall(query, top_k=top_k, **kwargs)

    def _dedup_cross_tier_summary_links(
        self, results: List[Dict], *, ep_summary_of_map: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """E3.a.3: drop the lower-scored side of any (episodic_summary,
        working_memory_sources) cluster where both surface in the same recall.

        Pre-E3, `sleep()` DELETEd source `working_memory` rows when creating
        a summary, so dual-surface duplication couldn't happen. Post-E3
        (additive sleep), sources survive alongside summaries by design.
        A recall whose query matches both raw and summary text ranks them
        side-by-side AND compounds `recall_count` twice for the same
        logical fact -- the row's history boost double-counts on every call.

        Dedup rule (per-cluster, not per-edge):
          - For each episodic row with non-empty `summary_of`, collect the
            wm_ids it covers that are also present in `results`.
          - If the ep's score is >= the score of EVERY covered wm in
            results, drop those wms and keep the ep (summary wins the
            whole cluster).
          - Otherwise -- some covered wm beats the ep -- drop the ep and
            keep all covered wms (sources win; the dropped ep no longer
            represents those wms in the result set).

        Per-cluster decisions avoid the per-edge bug where a wm could
        lose to a summary that itself was being dropped by a different
        wm. Example fixed by this shape: ep covers wm-1 (0.9) + wm-2 (0.3)
        with ep at 0.6. Per-edge would drop ep (lost to wm-1) AND wm-2
        (lost to ep) -- but wm-2's representative ep is itself gone, so
        wm-2 was being dropped against a phantom. Per-cluster correctly
        keeps both wms.

        Ties (ep_score == wm_score) keep the episodic side (later-stage
        representation; matches polyphonic engine's diversity-rerank
        posture). The comparison runs on the post-multiplier `score`
        field, so the dedup decision reflects the rank the user would
        have seen.

        Preserves input order on retained rows. Returns the input list
        unchanged (same object) if no episodic rows are present or no
        summary_of linkage exists.

        Args:
            results: scored row dicts; each carries `id`, `tier`, `score`.
            ep_summary_of_map: optional precomputed `{ep_id: summary_of_str}`
                from a caller that already SELECT-ed `episodic_memory`
                rows. When provided, skips the helper's own SELECT -- keeps
                a single source of truth and avoids one round-trip per
                recall on paths that have already fetched the data.

        Caller pattern: linear path passes the tier-lookup SELECT's
        precomputed `summary_of` rows; polyphonic path lets the helper
        do its own SELECT since it has no prior per-ep query.

        Caveats:
          - Does NOT dedup ep ↔ ep (two summaries covering overlapping
            wm sets). `sleep()` doesn't re-summarize already-consolidated
            rows by design (it skips `consolidated_at IS NOT NULL` per
            E3), so this is rare in practice. If it happens via external
            re-consolidation tooling, both summaries survive.
          - The summary_of SELECT and the subsequent recall_count UPDATE
            are NOT wrapped in a single transaction. A concurrent
            `sleep()` / `forget()` between them could yield stale linkage
            data. Acceptable under SQLite WAL + busy_timeout: the worst
            case is a one-call dedup miss, not data loss.

        A/B toggle: `MNEMOSYNE_CROSS_TIER_DEDUP=0` disables the dedup,
        returning the input list unchanged. Used by the BEAM-recovery
        Phase 4 ablation to isolate the dedup's contribution.
        """
        if _env_disabled("MNEMOSYNE_CROSS_TIER_DEDUP"):
            return results
        ep_ids = [r["id"] for r in results if r.get("tier") == "episodic"]
        if not ep_ids:
            return results
        summary_map: Dict[str, set] = {}
        if ep_summary_of_map is not None:
            for ep_id in ep_ids:
                raw = ep_summary_of_map.get(ep_id) or ""
                wm_ids = {s.strip() for s in raw.split(",") if s.strip()}
                if wm_ids:
                    summary_map[ep_id] = wm_ids
        else:
            placeholders = ",".join("?" * len(ep_ids))
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT id, summary_of FROM episodic_memory WHERE id IN ({placeholders})",
                tuple(ep_ids),
            )
            for row in cursor.fetchall():
                raw = row["summary_of"] or ""
                wm_ids = {s.strip() for s in raw.split(",") if s.strip()}
                if wm_ids:
                    summary_map[row["id"]] = wm_ids
        if not summary_map:
            return results
        wm_scores = {
            r["id"]: r.get("score", 0.0) for r in results if r.get("tier") == "working"
        }
        wm_keyword_scores = {
            r["id"]: r.get("keyword_score", 0.0)
            for r in results
            if r.get("tier") == "working"
        }
        ep_scores = {
            r["id"]: r.get("score", 0.0) for r in results if r.get("tier") == "episodic"
        }
        drop_wm_ids: set = set()
        drop_ep_ids: set = set()
        for ep_id, covered_wm_ids in summary_map.items():
            if ep_id not in ep_scores:
                continue
            ep_score = ep_scores[ep_id]
            present_wms = [w for w in covered_wm_ids if w in wm_scores]
            if not present_wms:
                continue
            exact_source_hit = any(
                (wm_keyword_scores.get(w, 0.0) >= 0.95 for w in present_wms)
            )
            ep_wins_cluster = not exact_source_hit and all(
                (ep_score >= wm_scores[w] for w in present_wms)
            )
            if ep_wins_cluster:
                drop_wm_ids.update(present_wms)
            else:
                drop_ep_ids.add(ep_id)
        if not (drop_wm_ids or drop_ep_ids):
            return results
        return [
            r
            for r in results
            if not (
                r.get("tier") == "working"
                and r["id"] in drop_wm_ids
                or (r.get("tier") == "episodic" and r["id"] in drop_ep_ids)
            )
        ]

    def _sandwich_order(self, results: List[Dict], top_k: int = 10) -> dict:
        """Sort by score and partition into high/medium/closing for sandwich ordering.

        U-shaped attention: LLMs pay most attention to first AND last items.
        High-scored facts go first, medium in the middle, high-scored again at end.
        """
        scored = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        high = [r for r in scored if r.get("score", 0) > 0.7][:3]
        medium = [r for r in scored if 0.3 < r.get("score", 0) <= 0.7][:5]
        closing_pool = [r for r in scored if r not in high][:3]
        closing = closing_pool if closing_pool else high[:2]
        return {"high": high, "medium": medium, "closing": closing}

    def _fact_line(self, result: Dict) -> str:
        """Clean one-line fact: 'User prefers dark mode (2026-05-09, user, c:0.9)'"""
        content = (result.get("content") or "")[:200].strip()
        ts_raw = result.get("timestamp") or ""
        ts = ts_raw[:10] if ts_raw else "?"
        source = result.get("source", "unknown")
        score = result.get("score") or result.get("importance") or 0
        return f"{content} ({ts}, {source}, c:{score:.1f})"

    def format_context(self, results: List[Dict], format: str = "bullet") -> str:
        """Format recall results as structured context for LLM injection.

        Args:
            results: List of recall result dicts (from recall() or polyphonic recall)
            format: 'bullet' (default) for markdown bullets, 'json' for structured JSON

        Returns:
            Formatted context string ready for LLM prompt injection.
        """
        sandwich = self._sandwich_order(results)
        if format == "json":
            return self._format_context_json(sandwich)
        return self._format_context_bullet(sandwich)

    def _format_context_json(self, sandwich: dict) -> str:
        """JSON structured context with sandwich ordering."""
        import json as _json

        context = {
            "top_facts": [self._fact_line(r) for r in sandwich["high"]],
            "supporting_context": [self._fact_line(r) for r in sandwich["medium"]],
            "recent_memories": [self._fact_line(r) for r in sandwich["closing"]],
            "total_memories": sum((len(v) for v in sandwich.values())),
        }
        return _json.dumps(context, indent=2, ensure_ascii=False)

    def _format_context_bullet(self, sandwich: dict) -> str:
        """Bullet-point context with sandwich ordering (U-shaped attention).

        Highest-scored first, medium middle, high-scored again at end.
        """
        lines = []
        lines.append("## Top Facts")
        for r in sandwich["high"]:
            lines.append(f"- {self._fact_line(r)}")
        if sandwich["medium"]:
            lines.append("")
            lines.append("## Supporting Context")
            for r in sandwich["medium"]:
                lines.append(f"- {self._fact_line(r)}")
        if sandwich["closing"]:
            lines.append("")
            lines.append("## Recent Signals")
            for r in sandwich["closing"]:
                lines.append(f"- {self._fact_line(r)}")
        total = sum((len(v) for v in sandwich.values()))
        lines.append(f"\n_({total} memories retrieved)_")
        return "\n".join(lines)

    def _polyphonic_row_passes_filters(
        self,
        row_dict: Dict,
        *,
        from_date: Optional[str],
        to_date: Optional[str],
        source: Optional[str],
        topic: Optional[str],
        author_id: Optional[str],
        author_type: Optional[str],
        channel_id: Optional[str],
        veracity: Optional[str],
        memory_type: Optional[str],
        now_iso: str,
        cross_session: bool,
    ) -> bool:
        """Mirror the linear path's filter set for the engine path.
        Always-on filters: session scope, valid_until, superseded_by.
        Conditional filters: caller-supplied kwargs.
        """
        if not cross_session:
            row_session = (
                row_dict.get("session_id") if "session_id" in row_dict else None
            )
            row_scope = row_dict.get("scope") or "session"
            channel_matches = channel_id and row_dict.get("channel_id") == channel_id
            if (
                not (author_id or author_type)
                and row_scope != "global"
                and (row_session is not None)
                and (row_session != self.session_id)
                and (not channel_matches)
            ):
                return False
        valid_until = row_dict.get("valid_until")
        if valid_until and (not _valid_until_active(valid_until, now_iso)):
            return False
        if row_dict.get("superseded_by"):
            return False
        if from_date and (row_dict.get("timestamp") or "") < from_date:
            return False
        if to_date and (row_dict.get("timestamp") or "") > f"{to_date}T23:59:59":
            return False
        if source and row_dict.get("source") != source:
            return False
        if topic and row_dict.get("source") != topic:
            return False
        if author_id and row_dict.get("author_id") != author_id:
            return False
        if author_type and row_dict.get("author_type") != author_type:
            return False
        if channel_id and row_dict.get("channel_id") != channel_id:
            return False
        if veracity and row_dict.get("veracity") != veracity:
            return False
        if memory_type and row_dict.get("memory_type") != memory_type:
            return False
        return True

    def _fetch_polyphonic_row(
        self, cursor, memory_id: str, tier=None
    ) -> Optional[Dict]:
        """Hydrate the producing tier, never replace an episodic hit with WM.

        Untyped legacy engines keep their existing episodic-first fallback.
        A typed result must not fall through to a different tier if deleted.
        """
        tiers = (tier,) if tier in ("working", "episodic") else ("episodic", "working")
        for label in tiers:
            table = "episodic_memory" if label == "episodic" else "working_memory"
            columns = "id, content, source, timestamp, session_id, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type"
            if label == "episodic":
                columns += ", tier"
            cursor.execute(f"SELECT {columns} FROM {table} WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if row is not None:
                return self._polyphonic_row_to_dict(row, tier_label=label)
        return None

    def _polyphonic_row_to_dict(self, row, *, tier_label: str) -> Dict:
        """Shared row → recall-dict mapper. /review caught the
        near-duplicate column mapping across episodic/working
        branches -- single helper now."""
        d = {
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "session_id": row["session_id"] if "session_id" in row.keys() else None,
            "importance": row["importance"],
            "recall_count": row["recall_count"] or 0,
            "last_recalled": row["last_recalled"],
            "scope": row["scope"] if "scope" in row.keys() else "session",
            "author_id": row["author_id"] if "author_id" in row.keys() else None,
            "author_type": row["author_type"] if "author_type" in row.keys() else None,
            "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
            "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
            "memory_type": row["memory_type"]
            if "memory_type" in row.keys()
            else "unknown",
            "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
            "superseded_by": row["superseded_by"]
            if "superseded_by" in row.keys()
            else None,
            "tier": tier_label,
        }
        if tier_label == "episodic":
            d["degradation_tier"] = row["tier"] if "tier" in row.keys() else 1
        return d

    def fact_recall(self, query: str, top_k: int = 30) -> List[Dict]:
        """Search both the facts table and consolidated_facts for structured knowledge.

        Returns facts as list of dicts with: content, score, fact_id, subject, predicate.

        Falls back gracefully if facts tables are empty or sqlite-vec unavailable.
        Also queries consolidated_facts (sleep-consolidated LLM fact triples) when
        the veracity consolidator is available — covers the polyphonic fact voice
        source without requiring MNEMOSYNE_POLYPHONIC_RECALL=1.
        """
        from . import jev
        from .jev_recall import fact_recall

        return fact_recall(self, query, top_k)

    def get_episodic_stats(
        self, author_id: str = None, author_type: str = None, channel_id: str = None
    ) -> Dict:
        cursor = self.conn.cursor()
        where_clauses = []
        params = []
        if author_id:
            where_clauses.append("author_id = ?")
            params.append(author_id)
        if author_type:
            where_clauses.append("author_type = ?")
            params.append(author_type)
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cursor.execute(f"SELECT COUNT(*) FROM episodic_memory{where_str}", params)
        total = cursor.fetchone()[0]
        cursor.execute(
            f"SELECT timestamp FROM episodic_memory{where_str} ORDER BY timestamp DESC LIMIT 1",
            params,
        )
        last = cursor.fetchone()
        vec_count = 0
        vec_type = "unused"
        return {
            "total": total,
            "last": last[0] if last else None,
            "vectors": vec_count,
            "vec_type": vec_type,
        }

    def get_memoria_stats(self) -> Dict:
        """Return MEMORIA structured-fact table counts."""
        cursor = self.conn.cursor()
        stats = {}
        for table in (
            "memoria_facts",
            "memoria_timelines",
            "memoria_kg",
            "memoria_instructions",
            "memoria_preferences",
        ):
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            except Exception:
                stats[table] = 0
        return stats

    def scratchpad_write(self, content: str) -> str:
        pad_id = _generate_id(content)
        ts = datetime.now().isoformat()
        self.conn.execute(
            "\n            INSERT INTO scratchpad (id, content, session_id, created_at, updated_at)\n            VALUES (?, ?, ?, ?, ?)\n            ON CONFLICT(id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at\n        ",
            (pad_id, content, self.session_id, ts, ts),
        )
        self.conn.commit()
        return pad_id

    def scratchpad_read(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            f"\n            SELECT id, content, created_at, updated_at\n            FROM scratchpad\n            WHERE session_id = ?\n            ORDER BY updated_at DESC\n            LIMIT {SCRATCHPAD_MAX_ITEMS}\n        ",
            (self.session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def scratchpad_clear(self):
        self.conn.execute(
            "DELETE FROM scratchpad WHERE session_id = ?", (self.session_id,)
        )
        self.conn.commit()

    def _extract_key_signal(self, content: str, max_chars: int = 300) -> str:
        """Extract the highest-signal sentences from content for tier 3 compression.

        Scores each sentence by entity/keyword density (proper nouns, technical
        terms, preference indicators) and keeps top-scoring sentences until the
        character budget is reached. Falls back to first-N-chars if content has
        no clear sentence boundaries.
        """
        import re

        if len(content) <= max_chars:
            return content
        from . import jev

        return jev.compress_extractively(content, max_chars)

    def _refresh_episodic_embedding(self, memory_id: str, rowid: int, new_content: str):
        return None

    def degrade_episodic(self, dry_run: bool = False) -> Dict:
        """Degrade old episodic memories through tier 1→2→3 compression.

        Tier 1 (0-TIER2_DAYS): Full detail, 1.0x recall weight
        Tier 2 (TIER2_DAYS-TIER3_DAYS): LLM-summarized, 0.5x weight
        Tier 3 (TIER3_DAYS+): Text extraction compressed, 0.25x weight

        Each tier transition that mutates content also refreshes the
        row's dense-recall embedding (or invalidates it if the embeddings
        provider is unavailable) so vec_episodes / memory_embeddings /
        binary_vector stay aligned with the displayed text. See C18.b.

        Returns summary of tier transitions performed.
        """
        from mnemosyne.core.config import get_config

        degrade_batch_size = get_config().get_int("degrade_batch", DEGRADE_BATCH_SIZE)
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        results = {
            "status": "dry_run" if dry_run else "degraded",
            "tier1_to_tier2": 0,
            "tier2_to_tier3": 0,
        }
        tier2_cutoff = (now - timedelta(days=TIER2_DAYS)).isoformat()
        tier3_cutoff = (now - timedelta(days=TIER3_DAYS)).isoformat()
        try:
            cursor.execute(
                "\n                SELECT id, rowid, content, importance FROM episodic_memory\n                WHERE tier = 1 AND created_at < ?\n                ORDER BY created_at ASC LIMIT ?\n            ",
                (tier2_cutoff, degrade_batch_size),
            )
            tier1_rows = cursor.fetchall()
        except Exception as exc:
            logger.warning(
                "degrade_episodic: tier1 SELECT failed (possibly corrupt UTF-8 in episodic_memory.content): %s. Skipping tier-1→2 degradation.",
                exc,
            )
            tier1_rows = []
        try:
            cursor.execute(
                "\n                SELECT id, rowid, content FROM episodic_memory\n                WHERE tier = 2 AND created_at < ?\n                ORDER BY created_at ASC LIMIT ?\n            ",
                (tier3_cutoff, degrade_batch_size // 2),
            )
            tier2_rows = cursor.fetchall()
        except Exception as exc:
            logger.warning(
                "degrade_episodic: tier2 SELECT failed (possibly corrupt UTF-8 in episodic_memory.content): %s. Skipping tier-2→3 degradation.",
                exc,
            )
            tier2_rows = []
        if dry_run:
            results["tier1_to_tier2"] = len(tier1_rows)
            results["tier2_to_tier3"] = len(tier2_rows)
            return results
        from mnemosyne.core import local_llm

        for row in tier1_rows:
            cursor.execute("SAVEPOINT degrade_row")
            try:
                compressed = row["content"]
                if local_llm.llm_available() and len(row["content"]) > 300:
                    summary = local_llm.summarize_memories([row["content"]])
                    if summary:
                        compressed = summary[:400]
                final_content = compressed[:800]
                cursor.execute(
                    "UPDATE episodic_memory SET content = ?, tier = 2, degraded_at = ? WHERE id = ?",
                    (_sanitize_utf8(final_content), now.isoformat(), row["id"]),
                )
                if final_content != row["content"]:
                    self._refresh_episodic_embedding(
                        row["id"], row["rowid"], final_content
                    )
                cursor.execute("RELEASE degrade_row")
                results["tier1_to_tier2"] += 1
            except Exception:
                try:
                    cursor.execute("ROLLBACK TO degrade_row")
                    cursor.execute("RELEASE degrade_row")
                except Exception:
                    logger.info("degrade_episodic: rollback failed", exc_info=True)
        for row in tier2_rows:
            cursor.execute("SAVEPOINT degrade_row")
            try:
                content = row["content"]
                if SMART_COMPRESS and len(content) > TIER3_MAX_CHARS:
                    compressed = self._extract_key_signal(
                        content, max_chars=TIER3_MAX_CHARS
                    )
                else:
                    compressed = content[:TIER3_MAX_CHARS]
                    if len(content) > TIER3_MAX_CHARS:
                        compressed += " [...]"
                cursor.execute(
                    "UPDATE episodic_memory SET content = ?, tier = 3, degraded_at = ? WHERE id = ?",
                    (_sanitize_utf8(compressed), now.isoformat(), row["id"]),
                )
                if compressed != row["content"]:
                    self._refresh_episodic_embedding(
                        row["id"], row["rowid"], compressed
                    )
                cursor.execute("RELEASE degrade_row")
                results["tier2_to_tier3"] += 1
            except Exception:
                try:
                    cursor.execute("ROLLBACK TO degrade_row")
                    cursor.execute("RELEASE degrade_row")
                except Exception:
                    logger.info("degrade_episodic: rollback failed", exc_info=True)
        self.conn.commit()
        self._invalidate_query_cache_after_commit("degrade_episodic")
        return results

    def get_contaminated(
        self, limit: int = 50, min_importance: float = 0.0
    ) -> List[Dict]:
        """Return potentially contaminated memories for review.

        Contaminated = veracity in ('inferred', 'tool', 'imported', 'unknown')
        -- i.e., anything not explicitly stated by the user. Sorted by
        importance descending so the highest-stakes items surface first.

        Args:
            limit: Max memories to return
            min_importance: Only return memories with importance >= this
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            SELECT id, content, source, veracity, tier, importance,\n                   created_at, degraded_at, session_id\n            FROM episodic_memory\n            WHERE veracity IN ('inferred', 'tool', 'imported', 'unknown')\n              AND importance >= ?\n            ORDER BY importance DESC, created_at DESC\n            LIMIT ?\n        ",
            (min_importance, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def health(self, stale_threshold_hours: float = 24.0) -> Dict:
        """Return consolidation health status for monitoring/alerting.

        Checks:
        - last successful consolidation timestamp (from consolidation_log)
        - error count in recent attempts (last 100 log entries)
        - stale threshold alert: no consolidation in `stale_threshold_hours`

        Returns a dict with keys: ``status`` ("healthy" | "stale" | "no_data"),
        ``last_successful_consolidation``, ``error_count``, ``stale_hours``,
        ``details``, and ``recommendation``.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            SELECT max(created_at) AS last_consolidation\n            FROM consolidation_log\n            WHERE items_consolidated > 0\n        "
        )
        row = cursor.fetchone()
        last_ts_str = (
            row["last_consolidation"] if row and row["last_consolidation"] else None
        )
        cursor.execute(
            "\n            SELECT count(*) AS err_count\n            FROM consolidation_log\n            WHERE created_at > datetime('now', '-7 days')\n              AND (\n                  items_consolidated = 0\n                  AND summary_preview LIKE '%error%'\n                  OR summary_preview LIKE '%fail%'\n              )\n        "
        )
        error_count = cursor.fetchone()["err_count"]
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if last_ts_str is None:
            status = "no_data"
            stale_hours = None
            recommendation = "No consolidation_log entries found with items_consolidated > 0. Either sleep() has never run, or all runs have produced zero summaries. Run sleep_all_sessions() or check logs."
        else:
            last_ts = datetime.fromisoformat(last_ts_str)
            stale_hours = round((now - last_ts).total_seconds() / 3600.0, 2)
            if stale_hours > stale_threshold_hours:
                status = "stale"
                recommendation = f"Last successful consolidation was {stale_hours:.1f} hours ago (threshold: {stale_threshold_hours:.0f}h). Run sleep_all_sessions() to catch up, and investigate why scheduled consolidation stopped (e.g. LLM unreachable, silent failures in summarize_memories, or cron/loop down)."
            else:
                status = "healthy"
                recommendation = "Consolidation is within the healthy window."
        return {
            "status": status,
            "last_successful_consolidation": last_ts_str,
            "error_count": error_count,
            "stale_hours": stale_hours,
            "stale_threshold_hours": stale_threshold_hours,
            "details": {
                "stale": status == "stale",
                "consolidation_log_entries_checked": "last 7 days",
            },
            "recommendation": recommendation,
        }

    def reclaim_orphans(
        self, dry_run: bool = False, stale_after_seconds: int = 3600, limit: int = 1000
    ) -> Dict:
        """Clear stale sleep claims that have no episodic summary.

        sleep() claims working rows by setting ``consolidated_at`` before it
        writes the episodic summary. A process crash in that narrow window can
        leave rows permanently skipped by later sleep() runs. This maintenance
        helper finds old claims whose id does not appear in any
        ``episodic_memory.summary_of`` CSV token and clears the marker so a
        later sleep pass can summarize them.
        """
        import math as _math

        if isinstance(stale_after_seconds, float) and (
            not _math.isfinite(stale_after_seconds)
        ):
            raise ValueError("stale_after_seconds must be a finite number")
        stale_after_seconds = max(0, min(int(stale_after_seconds), 10**9))
        limit = max(0, int(limit))
        cutoff = (
            datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        if limit == 0:
            return {
                "status": "dry_run" if dry_run else "no_op",
                "stale_after_seconds": stale_after_seconds,
                "cutoff": cutoff,
                "candidates": 0,
                "reclaimed": 0,
                "candidate_ids": [],
            }
        cursor = self.conn.cursor()
        orphan_query = "\n            SELECT wm.id\n            FROM working_memory wm\n            WHERE wm.consolidated_at IS NOT NULL\n              AND wm.consolidation_claimed_at IS NOT NULL\n              AND wm.consolidation_claimed_at < ?\n              AND NOT EXISTS (\n                  SELECT 1\n                  FROM episodic_memory em\n                  WHERE instr(',' || COALESCE(em.summary_of, '') || ',',\n                              ',' || wm.id || ',') > 0\n              )\n            ORDER BY wm.consolidated_at ASC\n            LIMIT ?\n        "
        cursor.execute(orphan_query, (cutoff, limit))
        candidate_ids = [row["id"] for row in cursor.fetchall()]
        if not candidate_ids:
            return {
                "status": "dry_run" if dry_run else "no_op",
                "stale_after_seconds": stale_after_seconds,
                "cutoff": cutoff,
                "candidates": 0,
                "reclaimed": 0,
                "candidate_ids": [],
            }
        if dry_run:
            return {
                "status": "dry_run",
                "stale_after_seconds": stale_after_seconds,
                "cutoff": cutoff,
                "candidates": len(candidate_ids),
                "reclaimed": 0,
                "candidate_ids": candidate_ids,
            }
        placeholders = ",".join("?" * len(candidate_ids))
        cursor.execute(
            f"\n            UPDATE working_memory\n            SET consolidated_at = NULL,\n                consolidation_claimed_at = NULL\n            WHERE id IN ({placeholders})\n              AND consolidated_at IS NOT NULL\n              AND consolidation_claimed_at IS NOT NULL\n              AND consolidation_claimed_at < ?\n              AND NOT EXISTS (\n                  SELECT 1\n                  FROM episodic_memory em\n                  WHERE instr(',' || COALESCE(em.summary_of, '') || ',',\n                              ',' || working_memory.id || ',') > 0\n              )\n            ",
            (*candidate_ids, cutoff),
        )
        reclaimed = cursor.rowcount
        self.conn.commit()
        self._invalidate_query_cache_after_commit("reclaim_orphans")
        logger.info(
            "reclaim_orphans: reclaimed=%d candidates=%d", reclaimed, len(candidate_ids)
        )
        return {
            "status": "reclaimed" if reclaimed else "no_op",
            "stale_after_seconds": stale_after_seconds,
            "cutoff": cutoff,
            "candidates": len(candidate_ids),
            "reclaimed": reclaimed,
            "candidate_ids": candidate_ids,
        }

    def sleep(self, dry_run: bool = False, force: bool = False) -> Dict:
        """
        Consolidate old working_memory for this session into episodic summaries.
        Uses a local lightweight LLM when available; falls back to aaak
        compression if the model is missing or inference fails.

        Post-E3 (additive): the source working_memory rows are NOT
        deleted. Instead they're marked with consolidated_at = NOW
        so the next sleep cycle skips them. Originals remain
        recallable alongside the new episodic summary.

        Note: this method intentionally remains session-scoped. Use
        sleep_all_sessions() for maintenance that consolidates eligible old
        working memories across inactive sessions.

        When force=True, skips the age cutoff and consolidates all
        non-consolidated working memories immediately regardless of age.
        """
        from mnemosyne.core.aaak import encode as aaak_encode
        from mnemosyne.core import local_llm

        cursor = self.conn.cursor()
        _cutoff_raw = (
            datetime.now(timezone.utc) - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)
        ).isoformat()
        if force:
            _cutoff_raw = datetime.max.isoformat()
        cutoff = _utc_cutoff_sql(_cutoff_raw)
        cursor.execute(
            f"\n            SELECT id, content, source, timestamp, importance, metadata_json, scope, valid_until, veracity, event_date, event_date_precision, superseded_by\n            FROM working_memory\n            WHERE COALESCE(session_id, 'default') = ?\n              AND {_SQL_CHRONO_TS} < ?\n              -- Round-7 R7-B1: exclude rows whose timestamp cannot be\n              -- placed on the timeline BEFORE the LIMIT window. Without\n              -- this, a backlog of unplaceable rows fills the whole\n              -- batch every pass (they sort first) and eligible rows\n              -- behind them never consolidate. The Python filter below\n              -- stays as a second validation gate.\n              AND {_SQL_PLACEABLE_TS}\n              AND consolidated_at IS NULL\n              AND (pinned IS NULL OR pinned = 0)\n            ORDER BY {_SQL_CHRONO_TS} ASC\n            LIMIT {SLEEP_BATCH_SIZE}\n        ",
            (self.session_id, cutoff),
        )
        rows = cursor.fetchall()
        rows = [
            r
            for r in rows
            if isinstance(r["timestamp"], str)
            and r["timestamp"].strip().lower() in ("now", "subsec")
            or _import_timestamp_ok(r["timestamp"])
        ]
        if not rows:
            cursor.execute(
                f"\n                SELECT COUNT(*) AS n\n                FROM working_memory\n                WHERE COALESCE(session_id, 'default') = ?\n                  AND NOT {_SQL_PLACEABLE_TS}\n                  AND consolidated_at IS NULL\n                  AND (pinned IS NULL OR pinned = 0)\n            ",
                (self.session_id,),
            )
            blocked = cursor.fetchone()["n"]
            result = {
                "status": "no_op",
                "message": "No old working memories to consolidate",
                "conflicts_resolved": 0,
                "conflicts_detected_only": 0,
            }
            if blocked:
                result["filtered_unplaceable"] = blocked
                result["message"] = (
                    f"No eligible rows; {blocked} row(s) have unparseable timestamps and are excluded from consolidation"
                )
            else:
                cursor.execute(
                    "\n                    SELECT COUNT(*) AS n\n                    FROM working_memory\n                    WHERE COALESCE(session_id, 'default') = ?\n                      AND pinned = 1\n                      AND consolidated_at IS NULL\n                ",
                    (self.session_id,),
                )
                pinned_exempt = cursor.fetchone()["n"]
                if pinned_exempt:
                    result["pinned_exempt"] = pinned_exempt
                    result["message"] = (
                        f"No eligible rows; {pinned_exempt} pinned row(s) are exempt from consolidation (import quarantine); re-date or unpin via update_working"
                    )
            return result
        if not dry_run:
            now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            ids_to_claim = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids_to_claim))
            cursor.execute(
                f"UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ? WHERE id IN ({placeholders}) AND consolidated_at IS NULL",
                (now_iso, now_iso, *ids_to_claim),
            )
            claimed_ids = set()
            if cursor.rowcount == len(ids_to_claim):
                claimed_ids = set(ids_to_claim)
            else:
                cursor.execute(
                    f"SELECT id FROM working_memory WHERE id IN ({placeholders}) AND consolidated_at = ?",
                    (*ids_to_claim, now_iso),
                )
                claimed_ids = {r["id"] for r in cursor.fetchall()}
            if not claimed_ids:
                self.conn.commit()
                return {
                    "status": "no_op",
                    "message": "All eligible rows claimed by concurrent sleep",
                    "conflicts_resolved": 0,
                    "conflicts_detected_only": 0,
                }
            rows = [r for r in rows if r["id"] in claimed_ids]
            self.conn.commit()
            self._invalidate_query_cache_after_commit("sleep.claim")
        grouped: Dict[str, List[Dict]] = {}
        for row in rows:
            grouped.setdefault(row["source"], []).append(dict(row))
        consolidated_ids = []
        summaries_created = 0
        llm_used_count = 0
        conflicts_resolved = 0
        conflicts_detected_only = 0
        model_refresh_proposals = 0
        model_refresh_applied = 0
        import time as _time

        conflict_deadline = _time.monotonic() + _CONFLICT_TIME_BUDGET_S
        conflict_calls = 0
        pairs_skipped_budget = 0
        superseded_older_ids = set()
        for source, items in grouped.items():
            lines = [item["content"] for item in items]
            ids = [item["id"] for item in items]
            aggregated_scope = "session"
            aggregated_valid_until = None
            for item in items:
                if item.get("scope") == "global":
                    aggregated_scope = "global"
                if item.get("valid_until"):
                    if aggregated_valid_until is None or _valid_until_active(
                        aggregated_valid_until, item["valid_until"]
                    ):
                        aggregated_valid_until = item["valid_until"]
            aggregated_veracity = aggregate_veracity(
                [item.get("veracity") for item in items]
            )
            if len(items) >= 2:
                conflicts = self._detect_conflicts(items)
                conflicts_detected_only += len(conflicts)
                from mnemosyne.core.llm_conflict_detector import (
                    LLM_CONFLICT_DETECTION_ENABLED,
                    conflict_endpoint_is_safe,
                    validate_conflict_pair,
                )
                from . import jev

                content_map = {item["id"]: item["content"] for item in items}
                for older_id, newer_id in conflicts:
                    if dry_run or older_id in superseded_older_ids:
                        continue
                    if (
                        conflict_calls >= _CONFLICT_PAIR_BUDGET
                        or _time.monotonic() >= conflict_deadline
                    ):
                        pairs_skipped_budget += 1
                        continue
                    conflict_calls += 1
                    try:
                        is_conflict, confidence, correct_fact = validate_conflict_pair(
                            content_map.get(older_id, ""),
                            content_map.get(newer_id, ""),
                            session_id=self.session_id,
                            db_path=self.db_path,
                            deadline=conflict_deadline,
                        )
                        if is_conflict is not True:
                            continue
                        with _guarded_transaction(self.conn):
                            if not self.conn.in_transaction:
                                self.conn.execute("BEGIN IMMEDIATE")
                            invalidated = self.invalidate(
                                older_id,
                                replacement_id=newer_id,
                                defer_cache_invalidation=True,
                            )
                            if invalidated:
                                validation_cursor = self.conn.execute(
                                    "INSERT INTO memory_validations (memory_id, validator, action, new_content, note) VALUES (?, ?, ?, ?, ?)",
                                    (
                                        older_id,
                                        "llm_conflict",
                                        "invalidated",
                                        correct_fact or "",
                                        json.dumps(
                                            {
                                                "confidence": confidence,
                                                "replacement_id": newer_id,
                                            }
                                        ),
                                    ),
                                )
                                if validation_cursor.rowcount != 1:
                                    raise sqlite3.IntegrityError(
                                        "Conflict provenance was not inserted"
                                    )
                    except Exception as exc:
                        logger.warning(
                            "Conflict validation/persistence failed (%s); pair left detected-only",
                            type(exc).__name__,
                        )
                        continue
                    if invalidated:
                        self._invalidate_query_cache_after_commit("sleep.conflict")
                        superseded_older_ids.add(older_id)
                        conflicts_resolved += 1
                        conflicts_detected_only -= 1
            summary = None
            llm_succeeded = False
            if not dry_run and local_llm.llm_available():
                compression_plugin = _plugins.get_manager().get_plugin("compression")
                if compression_plugin and compression_plugin.enabled:
                    lines = compression_plugin.compress_lines(lines)
                chunks = local_llm.chunk_memories_by_budget(lines, source=source)
                if chunks:
                    invalid_reasoning = False
                    if len(chunks) == 1:
                        summary = local_llm._summarize_memories(
                            chunks[0], source=source
                        )
                        invalid_reasoning = local_llm._is_invalid_reasoning_output(
                            summary
                        )
                    else:
                        chunk_summaries = []
                        for chunk in chunks:
                            chunk_summary = local_llm._summarize_memories(
                                chunk, source=source
                            )
                            if local_llm._is_invalid_reasoning_output(chunk_summary):
                                invalid_reasoning = True
                                break
                            if chunk_summary:
                                chunk_summaries.append(chunk_summary)
                        if not invalid_reasoning and chunk_summaries:
                            if len(chunk_summaries) == 1:
                                summary = chunk_summaries[0]
                            else:
                                summary = local_llm._summarize_memories(
                                    chunk_summaries, source=f"{source} (consolidated)"
                                )
                                invalid_reasoning = (
                                    local_llm._is_invalid_reasoning_output(summary)
                                )
                                if not invalid_reasoning and (not summary):
                                    summary = " | ".join(chunk_summaries)
                    if invalid_reasoning:
                        logger.warning(
                            "sleep: malformed reasoning trace for source=%r (items=%d) — falling back to AAAK compression",
                            source,
                            len(items),
                        )
                        summary = None
                    if summary:
                        llm_used_count += 1
                        llm_succeeded = True
            if summary is None:
                if not dry_run:
                    logger.warning(
                        "sleep: LLM summarization failed for source=%r (items=%d, llm_available=%s) — falling back to AAAK compression",
                        source,
                        len(items),
                        local_llm.llm_available(),
                    )
                combined = " | ".join(lines)
                compressed = aaak_encode(combined)
                summary = f"[{source}] {compressed}"
            proposals = []
            agent_context = (
                str(getattr(self, "agent_context", "") or "").strip().lower()
            )
            model_refresh_owner_id = (
                str(getattr(self, "canonical_owner_id", "") or "").strip() or "default"
            )
            if not dry_run and agent_context != "cron":
                try:
                    from mnemosyne.core import model_refresh

                    proposals = model_refresh.infer_model_update_proposals(items)
                except Exception:
                    proposals = []
            model_refresh_proposals += len(proposals)
            if not dry_run:
                _event_ts = None
                try:
                    _event_ts = _latest_iso_string(
                        (item.get("timestamp") for item in items), normalized=True
                    )
                except Exception:
                    _event_ts = None
                _agg_event_date = None
                _agg_event_date_precision = None

                def _wm_event_date_text(item):
                    v = item.get("event_date")
                    if v is None:
                        return ""
                    if not isinstance(v, str):
                        logger.warning(
                            "sleep: group row %r has non-text event_date %r; treated as undated",
                            item.get("id"),
                            v,
                        )
                        return ""
                    return v.strip()

                _dated = [item for item in items if _wm_event_date_text(item)]
                if len(_dated) == len(items) and len(items) > 0:
                    _distinct_dates = {_wm_event_date_text(item) for item in _dated}
                    if len(_distinct_dates) == 1:
                        _agg_event_date = _distinct_dates.pop()

                        def _row_sort_key(it):
                            try:
                                return _parse_iso_datetime_utc(str(it.get("timestamp")))
                            except (TypeError, ValueError):
                                return _parse_iso_datetime_utc("1970-01-01T00:00:00")

                        _newest = max(_dated, key=_row_sort_key)
                        _newest_precision = _newest.get("event_date_precision")
                        if isinstance(_newest_precision, str):
                            _agg_event_date_precision = (
                                _newest_precision.strip() or "unknown"
                            )
                        else:
                            _agg_event_date_precision = "unknown"
                if _agg_event_date:
                    if not _event_date_valid(_agg_event_date):
                        logger.warning(
                            "sleep: group event_date %r is not a real YYYY-MM-DD calendar date; storing summary without an event date",
                            _agg_event_date[:40],
                        )
                        _agg_event_date = None
                        _agg_event_date_precision = "unknown"
                if _agg_event_date_precision not in _EVENT_DATE_PRECISIONS:
                    _agg_event_date_precision = "unknown"
                self.consolidate_to_episodic(
                    summary=summary,
                    source_wm_ids=ids,
                    source="sleep_consolidation",
                    event_timestamp=_event_ts,
                    event_date=_agg_event_date,
                    event_date_precision=_agg_event_date_precision,
                    importance=0.6,
                    scope=aggregated_scope,
                    valid_until=aggregated_valid_until,
                    veracity=aggregated_veracity,
                    metadata={
                        "original_count": len(items),
                        "source": source,
                        "llm_used": llm_succeeded,
                    },
                )
                if proposals:
                    from mnemosyne.core import model_refresh

                    proposal_ts = (
                        datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                    )
                    for proposal in proposals:
                        metadata = model_refresh.prepare_proposal_metadata(
                            proposal, source_wm_ids=ids
                        )
                        proposal_id = self.remember(
                            model_refresh.proposal_to_memory_content(proposal),
                            source="sleep_model_refresh_proposal",
                            importance=model_refresh.coerce_confidence(
                                proposal.get("confidence"), 0.5
                            ),
                            metadata=metadata,
                            scope="session",
                            veracity="inferred",
                            trust_tier="DERIVED",
                        )
                        cursor.execute(
                            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = NULL WHERE id = ?",
                            (proposal_ts, proposal_id),
                        )
                        if model_refresh.maybe_auto_apply_model_refresh_proposal(
                            self, proposal_id, owner_id=model_refresh_owner_id
                        ):
                            model_refresh_applied += 1
                    self.conn.commit()
                group_placeholders = ",".join("?" * len(ids))
                cursor.execute(
                    f"UPDATE working_memory SET consolidation_claimed_at = NULL WHERE id IN ({group_placeholders})",
                    tuple(ids),
                )
                self.conn.commit()
            consolidated_ids.extend(ids)
            summaries_created += 1
        method = (
            "llm"
            if llm_used_count == summaries_created
            else "llm+aaak"
            if llm_used_count > 0
            else "aaak"
        )
        if not dry_run:
            cursor.execute(
                "\n                INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview, created_at)\n                VALUES (?, ?, ?, ?)\n            ",
                (
                    self.session_id,
                    len(consolidated_ids),
                    f"{summaries_created} summaries ({method}) from {len(consolidated_ids)} items",
                    datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                ),
            )
            self.conn.commit()
        if pairs_skipped_budget:
            logger.warning(
                "conflict validation budget reached: %d pair(s) skipped this sleep",
                pairs_skipped_budget,
            )
        degrade_result = self.degrade_episodic(dry_run=dry_run)
        if not dry_run:
            self._invalidate_query_cache_after_commit("sleep")
        logger.info(
            "sleep: consolidated=%d summaries=%d conflicts=%d llm=%s method=%s",
            len(consolidated_ids),
            summaries_created,
            conflicts_resolved,
            llm_used_count > 0,
            method,
        )
        return {
            "status": "dry_run" if dry_run else "consolidated",
            "items_consolidated": len(consolidated_ids),
            "summaries_created": summaries_created,
            "conflicts_resolved": conflicts_resolved,
            "conflicts_detected_only": conflicts_detected_only,
            "llm_used": llm_used_count,
            "method": method,
            "consolidated_ids": consolidated_ids,
            "degradation": degrade_result,
            "model_refresh": {
                "proposals": model_refresh_proposals,
                "applied": model_refresh_applied,
            },
        }

    def sleep_all_sessions(self, dry_run: bool = False, force: bool = False) -> Dict:
        """
        Consolidate eligible old working memories across all sessions.

        This is the maintenance-oriented counterpart to sleep(), which remains
        scoped to self.session_id. It prevents inactive sessions from leaving
        old working_memory rows stranded after they pass the sleep cutoff.

        When force=True, skips the age cutoff and consolidates all
        non-consolidated working memories across all sessions immediately.
        """
        cursor = self.conn.cursor()
        _cutoff_raw = (
            datetime.now(timezone.utc) - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)
        ).isoformat()
        if force:
            _cutoff_raw = datetime.max.isoformat()
        cutoff = _utc_cutoff_sql(_cutoff_raw)
        cursor.execute(
            f"\n            SELECT session_id, COUNT(*) AS eligible\n            FROM working_memory\n            WHERE {_SQL_CHRONO_TS} < ?\n              AND {_SQL_PLACEABLE_TS}\n              AND consolidated_at IS NULL\n              AND (pinned IS NULL OR pinned = 0)\n            GROUP BY session_id\n            ORDER BY MIN({_SQL_CHRONO_TS}) ASC\n        ",
            (cutoff,),
        )
        session_rows = cursor.fetchall()
        if not session_rows:
            return {
                "status": "no_op",
                "message": "No old working memories to consolidate",
                "conflicts_resolved": 0,
                "conflicts_detected_only": 0,
                "sessions_scanned": 0,
                "sessions_consolidated": 0,
                "items_consolidated": 0,
                "summaries_created": 0,
                "llm_used": 0,
                "errors": 0,
                "model_refresh": {"proposals": 0, "applied": 0},
                "session_results": [],
            }
        session_results = []
        sessions_consolidated = 0
        items_consolidated = 0
        summaries_created = 0
        llm_used = 0
        errors = []
        model_refresh_proposals = 0
        model_refresh_applied = 0
        conflicts_resolved = 0
        conflicts_detected_only = 0
        for row in session_rows:
            session_id = row["session_id"] if hasattr(row, "keys") else row[0]
            if session_id is None:
                session_id = "default"
            try:
                beam = (
                    self
                    if session_id == self.session_id
                    else BeamMemory(
                        session_id=session_id,
                        db_path=self.db_path,
                        author_id=self.author_id,
                        author_type=self.author_type,
                    )
                )
                result = beam.sleep(dry_run=dry_run, force=force)
                result = dict(result)
                result["session_id"] = session_id
                result["eligible"] = row["eligible"] if hasattr(row, "keys") else row[1]
                session_results.append(result)
                if result.get("status") in ("consolidated", "dry_run"):
                    sessions_consolidated += 1
                    items_consolidated += int(result.get("items_consolidated", 0) or 0)
                    summaries_created += int(result.get("summaries_created", 0) or 0)
                    llm_used += int(result.get("llm_used", 0) or 0)
                    conflicts_resolved += int(result.get("conflicts_resolved", 0) or 0)
                    conflicts_detected_only += int(
                        result.get("conflicts_detected_only", 0) or 0
                    )
                    refresh = result.get("model_refresh") or {}
                    model_refresh_proposals += int(refresh.get("proposals", 0) or 0)
                    model_refresh_applied += int(refresh.get("applied", 0) or 0)
            except Exception as exc:
                logger.error(
                    "sleep_all_sessions: session %r consolidation failed: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
                errors.append({"session_id": session_id, "error": repr(exc)})
        degrade_result = self.degrade_episodic(dry_run=dry_run)
        if not dry_run:
            self._deduplicate_memoria_cross_session()
        return {
            "status": "dry_run"
            if dry_run
            else "consolidated"
            if items_consolidated
            else "no_op",
            "sessions_scanned": len(session_rows),
            "sessions_consolidated": sessions_consolidated,
            "items_consolidated": items_consolidated,
            "summaries_created": summaries_created,
            "llm_used": llm_used,
            "errors": len(errors),
            "error_details": errors,
            "conflicts_resolved": conflicts_resolved,
            "conflicts_detected_only": conflicts_detected_only,
            "model_refresh": {
                "proposals": model_refresh_proposals,
                "applied": model_refresh_applied,
            },
            "session_results": session_results,
            "degradation": degrade_result,
        }

    def get_consolidation_log(self, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            SELECT id, session_id, items_consolidated, summary_preview, created_at\n            FROM consolidation_log\n            WHERE session_id = ?\n            ORDER BY created_at DESC\n            LIMIT ?\n        ",
            (self.session_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _deduplicate_memoria_cross_session(self) -> dict:
        """Cross-session dedup for MEMORIA tables.

        During sleep_all_sessions, redundant entries accumulate across
        sessions (same fact key, same instruction topic, same preference).
        This method:
        - Keeps the most recent entry per unique key+topic across all sessions
        - For instructions: deactivates duplicates (active=0)
        - For preferences: merges evolution chains
        Returns summary counts.
        """
        result = {}
        cursor = self.conn.cursor()
        cursor.execute(
            "\n            SELECT topic, COUNT(*) as cnt, MAX(message_idx) as latest\n            FROM memoria_instructions\n            GROUP BY topic\n            HAVING cnt > 1\n        "
        )
        dup_topics = cursor.fetchall()
        deactivated = 0
        for topic, cnt, latest_idx in dup_topics:
            cursor.execute(
                "\n                UPDATE memoria_instructions\n                SET active = 0\n                WHERE topic = ? AND message_idx < ? AND active = 1\n            ",
                (topic, latest_idx),
            )
            deactivated += cursor.rowcount
        result["instructions_deactivated"] = deactivated
        cursor.execute(
            "\n            SELECT topic, COUNT(*) as cnt, MAX(message_idx) as latest\n            FROM memoria_preferences\n            GROUP BY topic\n            HAVING cnt > 1\n        "
        )
        dup_prefs = cursor.fetchall()
        merged_evo = 0
        for topic, cnt, latest_idx in dup_prefs:
            prev = cursor.execute(
                "\n                SELECT preference FROM memoria_preferences\n                WHERE topic = ? AND message_idx < ?\n                ORDER BY message_idx DESC LIMIT 1\n            ",
                (topic, latest_idx),
            ).fetchone()
            if prev:
                cursor.execute(
                    "\n                    UPDATE memoria_preferences\n                    SET evolution = ?\n                    WHERE topic = ? AND message_idx = ?\n                ",
                    (f"was: {prev[0][:120]}", topic, latest_idx),
                )
                merged_evo += 1
        result["preference_evolutions_merged"] = merged_evo
        cursor.execute(
            "\n            DELETE FROM memoria_kg\n            WHERE rowid NOT IN (\n                SELECT MIN(rowid) FROM memoria_kg\n                GROUP BY subject, predicate, object\n            )\n        "
        )
        result["kg_duplicates_removed"] = cursor.rowcount
        self.conn.commit()
        result["status"] = "ok"
        return result

    def export_to_dict(self) -> Dict:
        """
        Export all BEAM data to a portable dictionary.
        Includes working_memory, episodic_memory, embeddings, scratchpad,
        and consolidation_log across ALL sessions (not just current).
        """
        cursor = self.conn.cursor()
        export = {
            "mnemosyne_export": {
                "version": "1.0",
                "export_date": datetime.now().isoformat(),
                "source_db": str(self.db_path),
                "component": "beam",
            }
        }
        cursor.execute(
            "\n            SELECT id, content, source, timestamp, session_id, importance,\n                   metadata_json, valid_until, superseded_by, scope,\n                   recall_count, last_recalled, created_at, veracity,\n                   consolidated_at, consolidation_claimed_at,\n                   event_date, event_date_precision, pinned\n            FROM working_memory\n            ORDER BY session_id, timestamp\n        "
        )
        export["working_memory"] = [dict(row) for row in cursor.fetchall()]
        cursor.execute(
            "\n            SELECT rowid, id, content, source, timestamp, session_id, importance,\n                   metadata_json, summary_of, valid_until, superseded_by, scope,\n                   recall_count, last_recalled, created_at,\n                   event_date, event_date_precision\n            FROM episodic_memory\n            ORDER BY session_id, timestamp\n        "
        )
        export["episodic_memory"] = [dict(row) for row in cursor.fetchall()]
        export["episodic_embeddings"] = []
        cursor.execute(
            "\n            SELECT id, content, session_id, created_at, updated_at\n            FROM scratchpad\n            ORDER BY session_id, updated_at\n        "
        )
        export["scratchpad"] = [dict(row) for row in cursor.fetchall()]
        cursor.execute(
            "\n            SELECT id, session_id, items_consolidated, summary_preview, created_at\n            FROM consolidation_log\n            ORDER BY session_id, created_at\n        "
        )
        export["consolidation_log"] = [dict(row) for row in cursor.fetchall()]
        return export

    def import_from_dict(self, data: Dict, force: bool = False) -> Dict:
        """
        Import BEAM data from a dictionary produced by export_to_dict().
        Idempotent by default: skips records whose id already exists.
        Set force=True to overwrite existing records.
        Returns import statistics.
        """
        stats = {
            "working_memory": {"inserted": 0, "skipped": 0, "overwritten": 0},
            "episodic_memory": {
                "inserted": 0,
                "skipped": 0,
                "overwritten": 0,
                "embeddings_inserted": 0,
            },
            "scratchpad": {"inserted": 0, "updated": 0},
            "consolidation_log": {"inserted": 0, "skipped": 0, "overwritten": 0},
        }
        cursor = self.conn.cursor()
        for item in data.get("working_memory", []):
            mid = item.get("id")
            _ts_for_insert = item.get("timestamp")
            _pin_for_insert = item.get("pinned", 0)
            if not isinstance(_pin_for_insert, int) or isinstance(
                _pin_for_insert, bool
            ):
                _pin_for_insert = 1 if _pin_for_insert else 0
            if not _import_timestamp_ok(item.get("timestamp")):
                _ts_for_insert = "1970-01-01T00:00:00"
                _pin_for_insert = 1
                stats["working_memory"]["imported_bad_timestamp"] = (
                    stats["working_memory"].get("imported_bad_timestamp", 0) + 1
                )
                logger.warning(
                    "import_from_dict: working row %r has unusable timestamp %r; preserved with epoch timestamp and pinned=1 (exempt from trim and sleep until re-dated/unpinned)",
                    mid,
                    item.get("timestamp"),
                )
            else:
                _ts_for_insert = (
                    _parse_iso_datetime_utc(item.get("timestamp").strip())
                    .replace(tzinfo=None)
                    .isoformat()
                )
            cursor.execute("SELECT 1 FROM working_memory WHERE id = ?", (mid,))
            exists = cursor.fetchone() is not None
            if exists and (not force):
                stats["working_memory"]["skipped"] += 1
                continue
            _existing_pinned = 0
            if exists and force:
                existing_row = cursor.execute(
                    "SELECT rowid, pinned FROM working_memory WHERE id = ?", (mid,)
                ).fetchone()
                if existing_row is not None:
                    _existing_pinned = existing_row["pinned"] or 0
                    _pin_for_insert = max(_pin_for_insert, _existing_pinned)
                cursor.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,)
                )
                cursor.execute("DELETE FROM working_memory WHERE id = ?", (mid,))
                stats["working_memory"]["overwritten"] += 1
            else:
                stats["working_memory"]["inserted"] += 1
            cursor.execute(
                "\n                INSERT INTO working_memory\n                (id, content, source, timestamp, session_id, importance, metadata_json,\n                 valid_until, superseded_by, scope, recall_count, last_recalled, created_at,\n                 veracity, consolidated_at, consolidation_claimed_at,\n                 event_date, event_date_precision, pinned)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    mid,
                    item.get("content"),
                    item.get("source"),
                    _ts_for_insert,
                    item.get("session_id", "default"),
                    item.get("importance", 0.5),
                    item.get("metadata_json", "{}"),
                    _normalize_valid_until(item.get("valid_until")),
                    item.get("superseded_by"),
                    item.get("scope", "session"),
                    item.get("recall_count", 0),
                    item.get("last_recalled"),
                    item.get("created_at"),
                    item.get("veracity"),
                    item.get("consolidated_at"),
                    item.get("consolidation_claimed_at"),
                    *_sanitize_import_event_date(
                        item.get("event_date"), item.get("event_date_precision")
                    ),
                    _pin_for_insert,
                ),
            )
        self.conn.commit()
        try:
            _backfill_vec_working_from_memory_embeddings(self.conn)
        except Exception:
            logger.info(
                "vec_working backfill skipped after working import", exc_info=True
            )
        vec_ok = False
        old_to_new_rowid = {}
        for item in data.get("episodic_memory", []):
            mid = item.get("id")
            cursor.execute("SELECT rowid FROM episodic_memory WHERE id = ?", (mid,))
            existing = cursor.fetchone()
            if existing and (not force):
                stats["episodic_memory"]["skipped"] += 1
                old_to_new_rowid[item.get("rowid")] = existing["rowid"]
                continue
            if existing and force:
                cursor.execute("DELETE FROM episodic_memory WHERE id = ?", (mid,))
                stats["episodic_memory"]["overwritten"] += 1
            else:
                stats["episodic_memory"]["inserted"] += 1
            _ts_for_insert = item.get("timestamp")
            if not _import_timestamp_ok(item.get("timestamp")):
                _ts_for_insert = "1970-01-01T00:00:00"
                stats["episodic_memory"]["imported_bad_timestamp"] = (
                    stats["episodic_memory"].get("imported_bad_timestamp", 0) + 1
                )
                logger.warning(
                    "import_from_dict: episodic row %r has unusable timestamp %r; preserved with epoch timestamp",
                    mid,
                    item.get("timestamp"),
                )
            else:
                _ts_for_insert = (
                    _parse_iso_datetime_utc(item.get("timestamp").strip())
                    .replace(tzinfo=None)
                    .isoformat()
                )
            cursor.execute(
                "\n                INSERT INTO episodic_memory\n                (id, content, source, timestamp, session_id, importance, metadata_json,\n                 summary_of, valid_until, superseded_by, scope, recall_count, last_recalled, created_at,\n                 event_date, event_date_precision)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ",
                (
                    mid,
                    item.get("content"),
                    item.get("source"),
                    _ts_for_insert,
                    item.get("session_id", "default"),
                    item.get("importance", 0.5),
                    item.get("metadata_json", "{}"),
                    item.get("summary_of", ""),
                    _normalize_valid_until(item.get("valid_until")),
                    item.get("superseded_by"),
                    item.get("scope", "session"),
                    item.get("recall_count", 0),
                    item.get("last_recalled"),
                    item.get("created_at"),
                    *_sanitize_import_event_date(
                        item.get("event_date"), item.get("event_date_precision")
                    ),
                ),
            )
            new_rowid = cursor.lastrowid
            old_to_new_rowid[item.get("rowid")] = new_rowid
        self.conn.commit()
        for item in data.get("scratchpad", []):
            pid = item.get("id")
            cursor.execute("SELECT 1 FROM scratchpad WHERE id = ?", (pid,))
            exists = cursor.fetchone() is not None
            if exists:
                cursor.execute(
                    "\n                    UPDATE scratchpad SET content=?, session_id=?, created_at=?, updated_at=?\n                    WHERE id=?\n                ",
                    (
                        item.get("content"),
                        item.get("session_id", "default"),
                        item.get("created_at"),
                        item.get("updated_at"),
                        pid,
                    ),
                )
                stats["scratchpad"]["updated"] += 1
            else:
                cursor.execute(
                    "\n                    INSERT INTO scratchpad (id, content, session_id, created_at, updated_at)\n                    VALUES (?, ?, ?, ?, ?)\n                ",
                    (
                        pid,
                        item.get("content"),
                        item.get("session_id", "default"),
                        item.get("created_at"),
                        item.get("updated_at"),
                    ),
                )
                stats["scratchpad"]["inserted"] += 1
        self.conn.commit()
        for item in data.get("consolidation_log", []):
            log_id = item.get("id")
            if log_id is not None:
                exists = (
                    cursor.execute(
                        "SELECT 1 FROM consolidation_log WHERE id = ?", (log_id,)
                    ).fetchone()
                    is not None
                )
                if not force:
                    cursor.execute(
                        "\n                        INSERT OR IGNORE INTO consolidation_log\n                            (id, session_id, items_consolidated, summary_preview, created_at)\n                        VALUES (?, ?, ?, ?, ?)\n                        ",
                        (
                            log_id,
                            item.get("session_id", "default"),
                            item.get("items_consolidated", 0),
                            item.get("summary_preview", ""),
                            item.get("created_at"),
                        ),
                    )
                    stats["consolidation_log"][
                        "skipped" if cursor.rowcount == 0 else "inserted"
                    ] += 1
                else:
                    cursor.execute(
                        "\n                        INSERT INTO consolidation_log\n                            (id, session_id, items_consolidated, summary_preview, created_at)\n                        VALUES (?, ?, ?, ?, ?)\n                        ON CONFLICT(id) DO UPDATE SET\n                            session_id=excluded.session_id,\n                            items_consolidated=excluded.items_consolidated,\n                            summary_preview=excluded.summary_preview,\n                            created_at=excluded.created_at\n                        ",
                        (
                            log_id,
                            item.get("session_id", "default"),
                            item.get("items_consolidated", 0),
                            item.get("summary_preview", ""),
                            item.get("created_at"),
                        ),
                    )
                    stats["consolidation_log"][
                        "overwritten" if exists else "inserted"
                    ] += 1
            else:
                cursor.execute(
                    "\n                    INSERT INTO consolidation_log\n                        (session_id, items_consolidated, summary_preview, created_at)\n                    VALUES (?, ?, ?, ?)\n                ",
                    (
                        item.get("session_id", "default"),
                        item.get("items_consolidated", 0),
                        item.get("summary_preview", ""),
                        item.get("created_at"),
                    ),
                )
                stats["consolidation_log"]["inserted"] += 1
        self.conn.commit()
        return stats
