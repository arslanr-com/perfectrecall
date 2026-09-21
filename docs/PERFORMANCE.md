# Performance and timeout recovery

Conversation cost control in 0.1.0a3 has separate [sequential-dialogue measurements](CONVERSATION_COST.md). The results below describe the 0.1.0a2 retrieval work.

The original problem was reproduced through Hermes' actual `MemoryManager`, using synthetic data in an isolated profile. With 241 eligible records and 242 evidence spans, the original serial duplicate-removal loop took 6.50 seconds after a 2.40-second relevance scan. Hermes returned at its eight-second timeout; its worker finished at 8.90 seconds. An immediate following call was skipped while that worker was still alive.

## Changes in 0.1.0a2

- Native Jev batches carry a separate complete memory span and criterion in each question. All eligible spans and all supplied criteria are evaluated; there is no shortlist or generative query writer.
- Up to 16 duplicate comparisons overlap. Speculative results are used only when the complete ordered comparator list matches the records actually retained. Otherwise the original comparison is rerun. Duplicate-heavy inputs can still require sequential work and extra requests.
- HTTPS connections are pooled. The request ceiling remains 128 workers by default, with bounded queues and 24,000-byte OpenRouter batches. Raising the ceiling to 512 is not the solution.
- A bounded 65,536-entry decision cache keys each exact criterion/evidence pair independently of packing. Changed content or criteria require fresh decisions; unchanged evidence can reuse its result. No plaintext memory is retained in cache keys. Cache capacity is finite and arbitrary workloads can exceed it.
- Prefetch carries a 6.5-second cooperative budget through locks, workers, network operations, and retries. Deadline expiry is separate from an API error. The tested timeout path drains its worker and allows the next turn to retrieve context. External custom prefetch sources must cooperate with deadlines too; network behavior is not a hard real-time guarantee.
- Native fanout also overlaps batches for structured facts and optional semantic lifecycle operations. The working-memory retention order preserves subsecond recency and resolves ties by insertion order, preventing a new write at capacity from evicting itself.
- `mnemosyne_diagnose` includes per-call ranking, weighting, deduplication, finalization, and prefetch timings. Concurrent calls do not mix request counts. These added fields omit queries, criteria, and memory content; the inherited full diagnostic still contains database paths.

## Final-wheel live check

The published 0.1.0a2 wheel passed one live test through the actual Hermes loader and `MemoryManager`, starting with 10,000 mixed-length synthetic records. The test automatically captured a new user fact, reopened the provider, injected that fact before an answer, and ran an explicit search with all three supported criteria. It kept the normal retention policy; the final working set remained at 10,000 records.

| Operation | Observed time |
|---|---:|
| Background automatic capture, through completion | 2.76 s |
| Automatic prefetch after reopening | 4.19 s |
| Explicit three-criterion search, cold | 9.57 s |
| Same three-criterion search, warm | 0.73 s |

The cold explicit search exceeds eight seconds; the automatic prefetch, which uses its default single criterion, completed within Hermes' eight-second window. These are different operations. The warm search reused 41,049 span/criterion decisions and made zero API requests. Total test usage was 1,498 API requests and **$0.377145132**, with zero API failures and zero decision deadlines. Automatic synchronization dispatched asynchronously; 2.76 seconds is its completion time, not blocking dispatch latency.

The [aggregate report](../benchmarks/results/performance-a2/hermes-10000-live.json) identifies the tested wheel SHA-256 `b7e80fce81b4bb9ac1399572fd0cad96ca93823d21ba00c64933ded712bdc095` and resolved Jev model. This was one final-wheel check of the most demanding defined fixture, not every possible worst case. No repeated final-wheel reliability sweep was run, to keep API spending minimal. Earlier prototype repetitions are retained below; neither set proves a production latency percentile. No user memory or installed profile was used.

## Live development measurements

These are development prototype measurements on 2026-09-21, not a production service-level guarantee or a final-wheel reliability certification. Reports include wheel hashes and the actual host commit. No private memory database was used.

| Workload and mode | Cold full prefetch | Warm full prefetch | Cold API requests | Observed cold Jev cost |
|---|---:|---:|---:|---:|
| 241 records, original serial post-processing | Host timeout at 8.00 s; worker 8.90 s | 0.022 s | 257 | $0.00395 |
| Same fixture, parallel duplicate checks | 2.99 s | 0.027 s | 257 | $0.00395 |
| Same fixture, native batching | 1.58 s | 0.018 s | 19 | $0.00149 |
| 10,000 short records, native batching | 3.68 s | 0.186 s | 172 | $0.03677 |
| 10,000 mixed records, pooled transport, three runs | 4.93–5.96 s | 0.474–0.500 s | 436 per run | $0.10345 per run |

The mixed fixture contains 13,683 spans, including long documents and eight-turn dialogues. Each filler project has a unique ID. All three pooled cold runs delivered target context, with no unrelated project in the injected block, no API failures, no deadline expiry, and no lingering worker. “Cold” clears local decision/request caches; later iterations retain the connection pool, and provider-side caching is uncontrolled. “Warm” repeats the same query against the same corpus. Three samples do not establish a reliable p95.

A 48,000-byte experimental batch size completed two cold runs in 5.70 and 5.00 seconds, then the key reached its total spending cap. The failed cold and warm runs are retained. A separate tiny request confirmed HTTP 403 for the key limit. This experiment does not justify increasing the default batch size. Account-specific details and credentials are excluded from the published reports.

All [aggregate measurement files](../benchmarks/results/performance-a2/) are retained, including failures and the earlier unpooled mixed run. Cost can differ across workloads: batching reduces round trips, not necessarily token charges.

## Quality comparison

Both the original per-span format and the native question format answered **12/12 fixed LongMemEval-S development cases correctly**, using GPT-5.6 Luna with high reasoning effort and the same GPT-4.1 judge. There were no observed regressions in these 12 pairs. This small reused development set does not prove equivalence or replace the historical 120-case independent comparison.

The original format made 9,802 Jev requests and cost $0.22362; batching made 942 requests and cost $0.22642. The decision probabilities and selected evidence can change even when final answers remain correct. Caller/judge inference is additional. The reduced [paired report](../benchmarks/results/performance-a2/paired-quality-12.json) retains IDs, labels, runtime and harness hashes, and usage without source conversations or answer text.

## Reproduce

Install development dependencies and build the wheel. Run host integration scripts with the Python that runs Hermes. Set `OPENROUTER_API_KEY` securely outside the repository; `--live` enables paid calls. Every script below creates an isolated temporary profile.

```sh
python -m pip install -e '.[dev]'
python -m build
python scripts/measure_recall_performance.py --hermes-root /path/to/hermes-agent \
  --wheel dist/perfectrecall-0.1.0a2-py3-none-any.whl \
  --records 241 --batch-mode single --http-transport urllib --serial-dedup --legacy-prefetch \
  --output benchmarks/runs/serial.json --live
python scripts/measure_recall_performance.py --hermes-root /path/to/hermes-agent \
  --wheel dist/perfectrecall-0.1.0a2-py3-none-any.whl \
  --records 10000 --shape mixed --iterations 3 \
  --output benchmarks/runs/mixed.json --live
python scripts/verify_hermes.py --hermes-root /path/to/hermes-agent \
  --wheel dist/perfectrecall-0.1.0a2-py3-none-any.whl --records 10000 \
  --output benchmarks/runs/lifecycle-10000.json --live
python scripts/verify_timeout_recovery.py --hermes-root /path/to/hermes-agent \
  --wheel dist/perfectrecall-0.1.0a2-py3-none-any.whl \
  --output benchmarks/runs/timeout-recovery.json
```

`--serial-dedup` preserves the old selection loop. `--legacy-prefetch` reproduces the old unbudgeted callback only in this disposable synthetic profile and is restricted to at most 1,024 records. Hermes still enforces its eight-second host window, and the harness observes any lingering worker and skipped follow-up. This reproduces the old failure mechanism with the supplied wheel; it is not a byte-identical reconstruction of the old runtime. Production settings remain untouched.

The lifecycle script starts with up to 10,000 mixed records, automatically captures a new user fact, reopens the provider, verifies prefetch, and checks cold and warm explicit searches with three criteria. It preserves the normal retention policy: one old unconsolidated record may leave the working set when a write crosses the configured capacity. Without `--live`, it verifies integration with a deterministic transport; those timings are not API performance evidence.

For the paired quality development check, download the public cleaned dataset using the [benchmark instructions](../benchmarks/README.md), then run each arm sequentially:

```sh
python scripts/evaluate_runtime_quality.py \
  --dataset benchmarks/downloads/longmemeval_s_cleaned.json --batch-mode single \
  --per-type 2 --test-offset 10 --output benchmarks/runs/quality-single.json --live
python scripts/evaluate_runtime_quality.py \
  --dataset benchmarks/downloads/longmemeval_s_cleaned.json --batch-mode question \
  --per-type 2 --test-offset 10 --output benchmarks/runs/quality-question.json --live
```

The harness records the selected IDs and plan before obtaining answers and refuses to overwrite existing outputs. Its full local output includes public-dataset answers and retrieval traces; publish reduced views after review. Preserve failed attempts. The synthetic latency harness limits requests and cost, but parallel requests already in flight can exceed a local spending guard; also configure a provider-side key limit.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `PERFECTRECALL_JEV_WORKERS` | `128` | Concurrent requests, range 1–256 |
| `PERFECTRECALL_JEV_BATCH_MODE` | `question` | Native independent questions; `single` retains the per-span diagnostic format |
| `PERFECTRECALL_JEV_HTTP_TRANSPORT` | `pooled` | Persistent verified HTTPS; `urllib` is available for diagnosis |
| `PERFECTRECALL_PREFETCH_BUDGET_SECONDS` | `6.5` | Cooperative prefetch budget, greater than zero and at most 7 |

Legacy `MNEMOSYNE_*` equivalents remain accepted. New names take precedence at startup. The reported 10,000-record results are for one criterion and a small set of distinct relevant facts. More text, more criteria, many semantic duplicates, concurrent users, and slow providers can increase latency. An incomplete required scan is a failure; it must not be reported as successful complete retrieval.
