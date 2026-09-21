# Contributing

Use English for code comments, documentation, issues, and benchmark explanations. Preserve compatibility interfaces intentionally; do not rename stored paths or `mnemosyne_*` tools as cosmetic cleanup.

Install `.[dev]`, run `python -m pytest -q`, `ruff check`, and `python scripts/audit_release.py`. Build with `python -m build`. The Hermes lifecycle script requires a separate Hermes checkout and its Python environment. Live API tests are opt-in and incur costs.

Changes to filtering must test disclosure boundaries, not just returned rankings. Retrieval must scan all eligible evidence and fail explicitly on incomplete decision operations. Do not add embeddings, an alternative vector engine, or a hidden query-writing language model to the production package.

Keep datasets, credentials, live memory, and scratch reports out of commits. Benchmark comparisons must identify runtime and prompt hashes, case selection, models, budgets, retries, regressions, and whether cases were already used for development. Never present development results as a new independent confirmation. The historical reproduction environment is separate from the installed package.
