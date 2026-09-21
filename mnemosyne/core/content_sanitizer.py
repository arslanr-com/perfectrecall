"""
Content sanitizer for Mnemosyne ingest paths.

Detects binary-shaped content (base64 data URIs, large payloads, encoded
blobs) and extracts it to content-addressed blob storage, replacing the
in-row content with a stub and a blob reference in metadata.

Blob storage: ~/.hermes/mnemosyne/blobs/<hash[0:2]>/<hash[0:4]>/<full-hash>
"""

import base64
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple


# --- Detection thresholds ---
SIZE_HARD_CAP = 1_000_000        # 1 MB — always extract regardless of content type
SIZE_BASE64_CHECK = 100_000      # 100 KB — run the base64 heuristic
ENTROPY_THRESHOLD = 5.0          # Shannon entropy bits-per-char > 5.0 suggests encoded data

DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;]+)?(?:;base64)?,(?P<payload>.*)",
    re.IGNORECASE,
)


def _blob_root() -> Path:
    """Root directory for content-addressed blobs."""
    root = os.environ.get("MNEMOSYNE_BLOB_DIR", "")
    if root:
        return Path(root)
    return Path.home() / ".hermes" / "mnemosyne" / "blobs"


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_data_uri(content: str) -> bool:
    """Check if content starts with a data: URI scheme."""
    return content.startswith("data:")


def _parse_data_uri(content: str) -> Optional[Tuple[str, bytes]]:
    """Parse a data: URI, returning (mime_type, raw_bytes) or None."""
    m = DATA_URI_RE.match(content)
    if not m:
        return None
    mime_type = m.group("mime") or "application/octet-stream"
    payload = m.group("payload")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return None
    return mime_type, raw


def _shannon_entropy(text: str) -> float:
    """Shannon entropy in bits per character. Higher = more random/uniform."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def _looks_like_base64_blob(content: str) -> bool:
    """
    Heuristic: does this string look like a base64-encoded binary blob?

    Uses Shannon entropy. Base64-encoded binary data has near-maximum entropy
    (~5.5-6.0 bits/char) because the character distribution is close to uniform.
    Natural language text (English, code, logs) sits at ~3.5-4.5 bits/char.

    Returns True if entropy exceeds ENTROPY_THRESHOLD.
    """
    if len(content) < SIZE_BASE64_CHECK:
        return False
    return _shannon_entropy(content) > ENTROPY_THRESHOLD


def _store_blob(raw_bytes: bytes) -> str:
    """Store raw bytes as a content-addressed blob. Returns sha256 hex hash."""
    sha256 = _compute_sha256(raw_bytes)
    blob_root = _blob_root()
    blob_dir = blob_root / sha256[:2] / sha256[:4]
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / sha256
    if not blob_path.exists():
        blob_path.write_bytes(raw_bytes)
    return sha256


def sanitize_content(content: str) -> Tuple[str, Dict]:
    """
    Inspect content for binary-shaped payloads and extract to blob storage.

    Returns (sanitized_content, blob_metadata).
    blob_metadata is empty dict if no extraction occurred; otherwise contains
    blob_ref, original_size, and (if known) mime.

    Detection rules (checked in order):
    1. data: URI prefix → decode base64 payload, extract to blob
    2. Size > 1 MB → extract entire content to blob
    3. Size > 100 KB AND entropy > 5.0 bits/char → likely encoded blob
    """
    blob_meta = {}
    original_size = len(content.encode("utf-8"))

    # Rule 1: data: URI
    if _is_data_uri(content):
        parsed = _parse_data_uri(content)
        if parsed:
            mime_type, raw_bytes = parsed
            sha256 = _store_blob(raw_bytes)
            blob_meta = {
                "blob_ref": f"blob://sha256/{sha256}",
                "original_size": len(raw_bytes),
                "mime": mime_type,
                "extraction_reason": "data_uri",
            }
            return (
                f"[Binary content extracted: {mime_type}, "
                f"{len(raw_bytes):,} bytes → blob://sha256/{sha256}]",
                blob_meta,
            )

    # Rule 2: size hard cap
    if original_size > SIZE_HARD_CAP:
        raw_bytes = content.encode("utf-8")
        sha256 = _store_blob(raw_bytes)
        blob_meta = {
            "blob_ref": f"blob://sha256/{sha256}",
            "original_size": original_size,
            "extraction_reason": "size_cap",
        }
        return (
            f"[Large content extracted: {original_size:,} bytes → "
            f"blob://sha256/{sha256}]",
            blob_meta,
        )

    # Rule 3: high-entropy (likely encoded blob)
    if original_size > SIZE_BASE64_CHECK and _looks_like_base64_blob(content):
        raw_bytes = content.encode("utf-8")
        sha256 = _store_blob(raw_bytes)
        entropy = round(_shannon_entropy(content), 2)
        blob_meta = {
            "blob_ref": f"blob://sha256/{sha256}",
            "original_size": original_size,
            "entropy": entropy,
            "extraction_reason": "high_entropy",
        }
        return (
            f"[Encoded content extracted: {original_size:,} bytes, "
            f"entropy {entropy:.1f} bits/char → blob://sha256/{sha256}]",
            blob_meta,
        )

    return content, blob_meta
