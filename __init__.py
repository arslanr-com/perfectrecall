"""Directory-plugin entry point for Hermes' pinned Git installations."""
from pathlib import Path
import sys

# Hermes loads directory plugins by file location rather than installing the
# project package. Resolve the implementation from this same pinned checkout.
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from perfectrecall.hermes import (  # noqa: E402, F401
    PerfectRecallMemoryProvider,
    register,
    register_memory_provider,
)
