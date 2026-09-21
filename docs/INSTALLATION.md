# Installation and migration

## Native Hermes plugin installation

From version 0.1.0a5, the repository is also a directory plugin. In a Hermes installation with Git plugin support:

```sh
hermes plugins install https://github.com/arslanr-com/perfectrecall
hermes memory setup perfectrecall
```

The second command uses the same configuration backup and legacy-plugin deactivation as the Python installer. Existing databases and memory banks remain in place. Set `OPENROUTER_API_KEY` in Hermes' environment and restart. Eligible memory text and queries are sent to the paid Jev API.

The plugin loads its implementation from the installed checkout; it does not download or update itself at startup. Hermes installs the two required dependencies, PyYAML and httpx. The optional MCP and encrypted-sync extras still require their documented Python dependencies.

The official catalog submission pins an exact commit. Until maintainers accept it, use the Git URL above. To undo a directory installation, select your previous memory provider, restore the configuration and legacy-plugin backup if needed, and remove PerfectRecall through `hermes plugins uninstall perfectrecall`. Keep your database backup if you also want to undo writes made after switching.

To reproduce the directory-install contract check in a Python environment containing Hermes but no installed PerfectRecall package:

```sh
python scripts/verify_hermes.py --hermes-root /path/to/hermes-agent \
  --plugin-dir . --output directory-install.json
```

This copies the plugin into a temporary profile and exercises native setup, discovery, automatic capture and retrieval after reopening. Jev responses are scripted by default, so the check makes no paid API calls and does not measure model quality or live latency.

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

`PERFECTRECALL_JEV_WORKERS` defaults to 128; accepted values are 1–256. More workers are not guaranteed to improve latency. Native question batching and pooled HTTPS are enabled by default; see [performance settings and measurements](PERFORMANCE.md). Provider credentials, endpoint, and model resolution are reported without the key by `python -m perfectrecall jev-status`.

## Rollback

Stop Hermes. Restore the backed-up `config.yaml` and move the saved legacy plugin back to its original location if it was used. Uninstall PerfectRecall, then reinstall the previous pinned Mnemosyne distribution in Hermes' environment. Restore the pre-migration database backup if you need to undo later writes or schema additions. Keep the original backup until the migration is accepted.

## Verify

```sh
python -m perfectrecall --version
python -m perfectrecall jev-status
python scripts/verify_hermes.py --hermes-root /path/to/hermes-agent \
  --wheel dist/perfectrecall-0.1.0a2-py3-none-any.whl --output benchmarks/runs/hermes.json
```

The last command runs against Hermes' real loader and manager in a temporary profile. Add `--live` for bounded paid Jev requests with `OPENROUTER_API_KEY`. It checks automatic user capture and memory injection after reopening, without any manual memory tool call. It does not simulate a full interactive agent conversation.

## Conversation cost control

Economy prefetch is enabled by default from 0.1.0a4. To use strict prefetch instead, add this to your existing Hermes configuration and restart Hermes:

```yaml
memory:
  provider: perfectrecall
  perfectrecall:
    prefetch_mode: strict
```

Use `prefetch_mode: economy` to restore the default behavior. `PERFECTRECALL_PREFETCH_MODE=strict` is the environment equivalent for strict mode; an explicit provider configuration takes precedence. Existing databases, automatic capture, and tool names remain compatible. Explicit memory tools always search their supplied question. No migration or new database is required.

See [measurements, refresh rules, and reproduction](CONVERSATION_COST.md).
