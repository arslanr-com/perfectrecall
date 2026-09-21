# Installation and migration

Install PerfectRecall in the Python environment running Hermes. Installing in an unrelated system Python does not make a provider discoverable by Hermes. The tested host commit is recorded in `benchmarks/results/hermes-lifecycle-offline.json`.

## First memory provider

Hermes must already be installed. PerfectRecall does not depend on an existing Mnemosyne installation, database, configuration file, or source plugin.

```sh
python -m pip install 'git+https://github.com/arslanr-com/perfectrecall.git'
python -m perfectrecall.install --hermes-home /path/to/hermes-profile
```

Use Hermes' Python for both commands. Set `OPENROUTER_API_KEY` in the environment that launches Hermes, or its supported secret configuration. Restart Hermes. A new SQLite database is created on first use, not by the installer. No embedding model is downloaded.

For source development, use `python -m pip install -e .` from the repository root.

## Replace Mnemosyne

1. Stop Hermes and back up the existing profile and SQLite database, including its WAL if copying files directly. An SQLite online backup is preferable to copying a live database.
2. In Hermes' Python environment, remove the old distributions if installed: `python -m pip uninstall mnemosyne-memory mnemosyne-hermes jevosyne`. This removes package files, not your memory database. The packages share the `mnemosyne` import namespace; do not leave two distributions owning the same files.
3. Install PerfectRecall with the command above, then run `python -m perfectrecall.install` using the same `HERMES_HOME` as before.
4. Keep your existing `MNEMOSYNE_DATA_DIR`, `MNEMOSYNE_DB_PATH`, bank settings, and persona path. Configure the Jev key and restart Hermes.

The installer backs up `config.yaml`, sets `memory.provider: perfectrecall`, and moves old `plugins/mnemosyne` or `plugins/jevosyne` directories or symlinks to `plugins-disabled/`. This avoids duplicate legacy capture hooks. It does not edit database contents or move banks. The YAML serializer preserves settings but normalizes formatting and comments; the original bytes are retained in the backup.

The default database location remains under `HERMES_HOME/mnemosyne/data` (normally `~/.hermes/mnemosyne/data`). Existing path resolution, bank identities, stored IDs, scopes, and persona files remain compatible. Historical tables are retained. New text does not require reindexing. Vector scores, vector maintenance, and old retrieval presets are intentionally unsupported.

Existing `from mnemosyne import Mnemosyne` imports and `mnemosyne_*` tool calls work. New code can use `from perfectrecall import PerfectRecall`. This is database and tool compatibility, not identical retrieval scores or offline behavior.

## Configuration

`PERFECTRECALL_*` variables take precedence over corresponding legacy `MNEMOSYNE_*` variables at process startup. Jev is always enabled: an old `MNEMOSYNE_DECISION_BACKEND=baseline` cannot activate a removed retrieval engine.

Provider configuration reads `memory.perfectrecall` first, then `memory.mnemosyne`. Existing scope and profile-isolation choices are retained. Configure `sync_roles: [user, assistant]` only if you want automatic assistant-message capture. An empty list disables automatic conversation capture while explicit memory tools remain usable.

`PERFECTRECALL_JEV_WORKERS` defaults to 128; accepted values are 1–256. More workers are not guaranteed to improve latency. Provider credentials, endpoint, and model resolution are reported without the key by `python -m perfectrecall jev-status`.

## Rollback

Stop Hermes. Restore the backed-up `config.yaml` and move the saved legacy plugin back to its original location if it was used. Uninstall PerfectRecall, then reinstall the previous pinned Mnemosyne distribution in Hermes' environment. Restore the pre-migration database backup if you need to undo later writes or schema additions. Keep the original backup until the migration is accepted.

## Verify

```sh
python -m perfectrecall --version
python -m perfectrecall jev-status
python scripts/verify_hermes.py --hermes-root /path/to/hermes-agent \
  --wheel dist/perfectrecall-0.1.0a1-py3-none-any.whl --output benchmarks/runs/hermes.json
```

The last command runs against Hermes' real loader and manager in a temporary profile. Add `--live` for bounded paid Jev requests with `OPENROUTER_API_KEY`. It checks automatic user capture and memory injection after reopening, without any manual memory tool call. It does not simulate a full interactive agent conversation.
