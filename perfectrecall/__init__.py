"""PerfectRecall: agent memory powered by typed Jev decisions.

The inherited ``mnemosyne`` modules remain a compatibility implementation namespace.
"""
__version__ = "0.1.0a3"
__license__ = "MIT"


def configure():
    """Activate PerfectRecall before importing the inherited storage modules.

    PERFECTRECALL_* variables override matching legacy MNEMOSYNE_* variables.
    Jev is the only decision backend in PerfectRecall.
    Storage and persona paths retain upstream resolution so existing databases
    and banks work in place without copying or renaming.
    """
    import os
    for key, value in tuple(os.environ.items()):
        if key.startswith("PERFECTRECALL_"):
            os.environ["MNEMOSYNE_" + key[len("PERFECTRECALL_"):]] = value
    os.environ["MNEMOSYNE_DECISION_BACKEND"] = "jev"
    os.environ.setdefault("MNEMOSYNE_WRITE_CLASSIFIER", "strict")


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    configure()
    import mnemosyne
    if name == "PerfectRecall":
        return mnemosyne.Mnemosyne
    return getattr(mnemosyne, name)


configure()
from mnemosyne import __all__ as _compat_exports

__all__ = ["PerfectRecall", "configure", *_compat_exports]
