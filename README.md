# PerfectRecall

**Jev-powered memory for AI agents.** Try it on Hermes. Replace Mnemosyne while keeping your existing SQLite database, memory banks, and tool calls.

An independent 120-question LongMemEval-S experiment reduced final-answer errors from **62 to 17 (72.6%)**, using the same GPT-5.6 Luna caller with high reasoning effort. Accuracy rose from **48.3% to 85.8%**. These are results for a frozen predecessor of this release, not a fresh validation of the renamed package or an official leaderboard score. [Methods, results, regressions, and reproduction](benchmarks/README.md).

PerfectRecall asks Jev to evaluate every eligible memory against short criteria supplied by the calling agent. It returns the original evidence for that agent to reason over. There is no embedding model, vector index, or hidden query-writing model in the production package.

## Install in Hermes

Python 3.10 or newer is required. Install into **the same Python environment that runs Hermes**. On a standard Hermes installation:

```sh
~/.hermes/hermes-agent/venv/bin/python -m pip install 'git+https://github.com/arslanr-com/perfectrecall.git'
~/.hermes/hermes-agent/venv/bin/python -m perfectrecall.install
```

For another Hermes location, use its Python executable. This repository is the installation source; a PyPI release is not assumed.

Set `OPENROUTER_API_KEY` in the environment used to start Hermes, or in your existing Hermes secret configuration. Restart Hermes. The installer selects `memory.provider: perfectrecall`, backs up an existing configuration, and disables legacy source plugins by moving them intact outside the plugin discovery directory. It also works with an empty Hermes profile and no previous memory provider. Hermes itself must already be installed.

```sh
python -m perfectrecall.install --hermes-home /path/to/profile
python -m perfectrecall jev-status
```

`jev-status` reports configuration and key presence; it does not print the key or make an API call. [Installation, migration, and rollback](docs/INSTALLATION.md).

## Automatic memory

Hermes calls the provider before and after turns. Prefetch injects relevant memory into context without an explicit search call. Post-turn synchronization saves useful user messages automatically. Assistant messages are excluded by default; `sync_roles` makes that choice configurable. Normal scope, profile isolation, write-approval, and self-echo protections remain in place.

```yaml
memory:
  provider: perfectrecall
  perfectrecall:
    sync_roles: [user]
    default_scope: global
```

`global` deliberately shares these memories across sessions within the same bank. Omit it to preserve the inherited session scope. Existing `memory.mnemosyne` settings are still honored; `memory.perfectrecall` takes precedence.

## Use from Python or MCP

```python
from perfectrecall import PerfectRecall

memory = PerfectRecall(session_id="project-cedar")
memory.remember("Project Cedar uses PostgreSQL.")
hits = memory.recall(
    "Which database does Cedar use?",
    evidence_questions=["Does this memory name Project Cedar's database?"],
)
```

The `mnemosyne` Python namespace and `mnemosyne_*` tool names remain as compatibility interfaces. To run the inherited MCP tool surface, install the `mcp` extra and use `perfectrecall mcp`. Optional encrypted synchronization uses the `sync` extra.

The tool description teaches the calling agent to ask up to three short, concrete yes/no questions. It should retrieve partial facts, follow links through subsequent searches, and perform comparisons itself. [Tool guidance](docs/TOOLS.md).

## Data handling and limits

SQLite remains local. **Eligible memory text and queries are sent to the configured Jev API** for decisions. Optional Hermes-assisted consolidation can also use the host's language model. This is not an offline or zero-cloud memory system. Keep credentials outside the repository. [Privacy and security](SECURITY.md).

Recall scans the full eligible corpus, so cost and latency grow with text volume. The default is 128 concurrent Jev requests, configurable with `PERFECTRECALL_JEV_WORKERS` from 1 to 256. Increasing concurrency can increase failures and latency. The measured 10,000-record cold scan did **not** fit Hermes' eight-second prefetch window; this release does not promise automatic context at that scale. Explicit recalls can also fail when a decision request fails. [Capacity evidence](benchmarks/README.md#capacity).

PerfectRecall is an alpha release. The name describes the goal; it is not a guarantee of perfect memory. Model availability, routing, prices, and results can change.

## Development and credits

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
python scripts/audit_release.py
```

PerfectRecall is an independent fork of [Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne), originally by Abdias J. The inherited storage, memory lifecycle, integrations, and much of the tool surface are upstream work. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [third-party notices](THIRD_PARTY_NOTICES.md). No affiliation or endorsement is implied.
