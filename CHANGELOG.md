# Changelog

## 0.1.0a3 — Optional conversation cost control

- Add `prefetch_mode: economy` for supported conversation follow-ups, using two short Jev decisions in one bounded request.
- Reuse the original search question with the existing exact decision cache; re-read eligible storage, evaluate new or changed records, and rebuild context each turn.
- Refresh on scope changes, edits to active evidence, expiry, resets, explicit recall, or after six reuses/five minutes. Gate errors fall back to a full search within the shared prefetch budget.
- Keep strict prefetch as the default and explicit recall independent of the conversation gate.
- Add conversation-cost and gate-probe scripts, real Hermes integration checks, and regression tests for privacy, mutations, and deadline recovery.

Measured outcomes and their limits are in [conversation cost documentation](docs/CONVERSATION_COST.md). These are small synthetic development checks, not an independent final-answer quality benchmark.

## 0.1.0a2 — Batched recall and timeout recovery

- Batch complete independent Jev evidence questions and reuse pooled HTTPS connections.
- Overlap duplicate checks while preserving the actual ordered comparison policy.
- Reuse exact decisions across request repacking with a bounded 65,536-entry cache.
- Propagate a cooperative prefetch deadline and report per-call stage usage separately from API failures.
- Parallelize native batches for structured facts and optional semantic lifecycle work.
- Fix immediate eviction of newly captured memories at capacity when timestamps tie.
- Add reproducible latency, timeout, 10,000-record lifecycle, and paired quality checks.

Live development results and final-wheel offline verification are distinguished in [performance documentation](docs/PERFORMANCE.md). The 12-case paired development check had no observed answer regressions. The final wheel passed a live 10,000-record automatic capture/reopen check and three-criterion recall for $0.37715. Automatic prefetch took 4.19 seconds; cold explicit three-criterion recall took 9.57 seconds. One run does not establish a production reliability guarantee.

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
