"""Configure PerfectRecall in an existing or empty Hermes profile."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile

import yaml


def configure_hermes(home: Path) -> dict:
    home = home.expanduser().resolve()
    config_path = home / "config.yaml"
    original = config_path.read_bytes() if config_path.exists() else None
    config = yaml.safe_load(original) if original else {}
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Hermes config must be a YAML mapping")
    memory = config.setdefault("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("Hermes memory config must be a YAML mapping")
    memory["provider"] = "perfectrecall"
    # Legacy memory.mnemosyne settings are read in place by the provider.
    # Do not change banks, scopes, persona files or the existing data directory.
    home.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = None
    if original is not None:
        backup = home / ("config.yaml.before-perfectrecall-" + stamp)
        with backup.open("xb") as handle:
            handle.write(original)
        backup.chmod(0o600)
    disabled = []
    # Old source plugins can register a second set of hooks independently of
    # memory.provider. Move them intact outside Hermes' discovery directory.
    for name in ("mnemosyne", "jevosyne"):
        plugin = home / "plugins" / name
        if plugin.exists() or plugin.is_symlink():
            destination = home / "plugins-disabled" / (name + "-" + stamp)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Preserve a relative symlink's target after moving it.
            if plugin.is_symlink():
                target = plugin.resolve()
                destination.symlink_to(target, target_is_directory=True)
                plugin.unlink()
            else:
                shutil.move(str(plugin), str(destination))
            disabled.append(str(destination))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=home, prefix=".perfectrecall-", delete=False) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(config, handle, sort_keys=False)
        temporary.chmod(0o600)
        os.replace(temporary, config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"config": str(config_path), "backup": str(backup) if backup else None,
            "disabled_legacy_plugins": disabled, "provider": "perfectrecall"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", type=Path,
                        default=Path(os.environ.get("HERMES_HOME", "~/.hermes")))
    args = parser.parse_args()
    import json
    print(json.dumps(configure_hermes(args.hermes_home), indent=2))
    print("Set OPENROUTER_API_KEY in the Hermes environment, then restart Hermes.")


if __name__ == "__main__":
    main()
