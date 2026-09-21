"""PerfectRecall compatibility namespace for existing Mnemosyne integrations."""

__version__ = "0.1.0a3"
__author__ = "Abdias J"
__license__ = "MIT"

import os
for _key, _value in tuple(os.environ.items()):
    if _key.startswith("PERFECTRECALL_"):
        os.environ["MNEMOSYNE_" + _key[len("PERFECTRECALL_"):]] = _value
os.environ["MNEMOSYNE_DECISION_BACKEND"] = "jev"
os.environ.setdefault("MNEMOSYNE_WRITE_CLASSIFIER", "strict")

# Lazy imports to allow mnemosyne.install to run without heavy deps
# (e.g. numpy is not yet installed during first-time setup)
_lazy_exports = {
    "Mnemosyne": (".core.memory", "Mnemosyne"),
    "remember": (".core.memory", "remember"),
    "recall": (".core.memory", "recall"),
    "get_context": (".core.memory", "get_context"),
    "get_stats": (".core.memory", "get_stats"),
    "get": (".core.memory", "get"),
    "forget": (".core.memory", "forget"),
    "update": (".core.memory", "update"),
    "reclaim_orphans": (".core.memory", "reclaim_orphans"),
    "SyncEngine": (".core.sync", "SyncEngine"),
    "SyncEvent": (".core.sync", "SyncEvent"),
    "SyncEncryption": (".core.sync", "SyncEncryption"),
    "ConflictResolution": (".core.sync", "ConflictResolution"),
    "run_sync_server": (".core.sync_server", "run_sync_server"),
}


def __getattr__(name: str):
    if name in _lazy_exports:
        mod_path, attr_name = _lazy_exports[name]
        mod = __import__(f"mnemosyne{mod_path}", fromlist=[attr_name])
        return getattr(mod, attr_name)
    raise AttributeError(f"module 'mnemosyne' has no attribute '{name}'")


__all__ = list(_lazy_exports.keys())

# Conditionally expose MCP server if mcp package is installed
try:
    import mcp  # noqa: F401

    _lazy_exports["run_mcp_server"] = (".mcp_server", "run_mcp_server")
    __all__.append("run_mcp_server")
except ImportError:
    pass  # MCP is optional — core works without it
