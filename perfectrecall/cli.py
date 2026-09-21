"""Independent PerfectRecall command; inherited CLI operations remain compatible."""
from . import __version__, configure


def main():
    configure()
    import sys
    if sys.argv[1:] in (["--version"], ["version"]):
        print("PerfectRecall " + __version__)
        return
    if sys.argv[1:] == ["jev-status"]:
        import json
        import os
        from mnemosyne.core import jev
        from mnemosyne.core.jev_evidence import worker_count
        from mnemosyne.cli import _default_data_dir
        config = jev.settings()
        print(json.dumps({"product": "PerfectRecall", "version": __version__,
                          "backend": "jev" if jev.enabled() else "baseline",
                          "provider": config["provider"], "model": config["model"],
                          "jev_workers": worker_count(),
                          "api_key_configured": bool(os.environ.get(config["key_env"], "").strip()),
                          "data_dir": _default_data_dir(),
                          "live_api_checked": False}, indent=2))
        return
    from mnemosyne.cli import run_cli
    if not sys.argv[1:] or sys.argv[1:] in (["--help"], ["-h"], ["help"]):
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_cli()
        print(output.getvalue().replace("Mnemosyne - Local AI Memory System",
              "PerfectRecall - Agent memory with Jev decisions").replace("mnemosyne", "perfectrecall"), end="")
        print("  recall supports --evidence-question <yes/no question> (repeat up to 3 times)")
        print("  jev-status                             Provider/configuration status (no network)")
        return
    run_cli()


if __name__ == '__main__':
    main()
