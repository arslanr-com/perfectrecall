# Changelog

## 0.1.0a2 — Batched recall and timeout recovery

- Batch complete independent Jev evidence questions and reuse pooled HTTPS connections.
- Overlap duplicate checks while preserving the actual ordered comparison policy.
- Reuse exact decisions across request repacking with a bounded 65,536-entry cache.
- Propagate a cooperative prefetch deadline and report per-call stage usage separately from API failures.
- Parallelize native batches for structured facts and optional semantic lifecycle work.
- Fix immediate eviction of newly captured memories at capacity when timestamps tie.
- Add reproducible latency, timeout, 10,000-record lifecycle, and paired quality checks.

Live development results and final-wheel offline verification are distinguished in [performance documentation](docs/PERFORMANCE.md). The 12-case paired development check had no observed answer regressions. Final-wheel live reliability and automatic-write validation remain pending after the test key reached its configured spending cap.

## 0.1.0a1 — PerfectRecall publication candidate

- Renamed the public package, CLI, and Hermes provider to PerfectRecall.
- Retained Mnemosyne Python, tool, storage, bank, and configuration compatibility.
- Removed production embedding backends, optional vector dependencies, and alternative vector recall paths.
- Added a first-provider Hermes installer with configuration backups and legacy-plugin deactivation.
- Unified Hermes and MCP descriptions around the measured short-criteria prompt.
- Preserved automatic user-message capture and pre-turn memory injection.
- Added page-based SQLite backup support for databases with unloaded legacy virtual tables.
- Included independent and development benchmark outcomes, conservative retry analysis, capacity failures, pinned historical reproduction patches, and verification scripts.

This candidate has not been published to PyPI and is not a fresh independent quality benchmark. Historical results are attributed to their original runtime snapshots.
