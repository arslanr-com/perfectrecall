# Conversation cost control

PerfectRecall 0.1.0a3 adds an optional economy mode for automatic Hermes prefetch. In one eight-message synthetic conversation, it reduced total Jev cost by **55.6%** against strict mode, including an automatic memory write. The experiment used separate banks and cold client caches for each mode.

Enable it in Hermes and restart:

```yaml
memory:
  provider: perfectrecall
  perfectrecall:
    prefetch_mode: economy
```

The equivalent environment variable is `PERFECTRECALL_PREFETCH_MODE=economy`. An explicit provider setting takes precedence. The default remains `strict`. No database migration is needed.

## How it saves money

The first message runs a normal search. On a subsequent message, Jev answers two short questions in one API request: is this a continuation of the previous subject, and does it require additional historical evidence? Reuse requires a continuation score of at least 0.80 and a new-evidence score of at most 0.20. These are model scores, not calibrated error probabilities.

If both checks permit reuse, recall uses the original search question. It still reads **every eligible memory** from SQLite, applies current access and validity filters, and rebuilds the result. The existing exact decision cache skips API evaluation for unchanged evidence under that identical question. Newly captured or edited records still receive Jev evaluation. No embedding index or generative query writer is introduced.

This avoids treating a changed question as an exact cache hit. It also avoids returning a frozen context block. Identity, canonical slots and external prefetch sources continue through their normal paths. Changes elsewhere in the database are seen when recall reads storage again.

## Measured conversations

Live calls used OpenRouter, resolving to `typesafe/jev-1.13-20260917`, with 128 workers. Tests ran through Hermes' actual `MemoryManager`, not a direct replacement for the host callback. Reports identify the tested wheel by SHA-256 and Hermes by commit.

| Eight messages, 256 initial mixed-length records | Strict | Economy |
| --- | ---: | ---: |
| Total Jev cost, including automatic capture | $0.024154 | $0.010736 |
| API requests, including capture | 126 | 66 |
| Turns with expected evidence in injected context | 6/8 | 8/8 |
| Reused search questions | 0 | 4 |
| Incorrect reuse on a new-evidence turn | 0 | 0 |

The strict misses were two follow-ups referring to the preceding database fact without naming the project. The economy gate was developed using this conversation; this is a development result, not an independent quality benchmark. The evidence check searches for an expected fact in injected context. It does not measure final answers from a calling model. [Paired raw report](../benchmarks/results/conversation-a3/live-256-paired.json).

A separate live run started with **10,000 mixed-length records**:

| Automatic retrieval | Time | Jev cost |
| --- | ---: | ---: |
| Initial cold search | 4.66 s | $0.102310 |
| Follow-up after automatic capture of a new fact | 2.08 s | $0.000123 |
| Next follow-up, with no intervening write | 0.82 s | $0.000025 |

All three calls returned the expected fact, scanned 10,000 eligible records and completed before Hermes' eight-second host limit. Normal retention kept the bank at 10,000 after capture. Total test cost, including the write, was $0.102577. These are single observations; cold searches and follow-ups after writes are not generally subsecond. [10,000-record raw report](../benchmarks/results/conversation-a3/live-10000-followups.json).

Sixteen separate synthetic gate probes accepted all five supported continuations and rejected all eleven requests for additional evidence, including new attributes, other projects, corrections, comparisons and explicit rechecks. The probes were fixed before their live run and did not lead to prompt changes. They exercise the gate with supplied synthetic evidence, not a complete memory bank or answer model. [Probe cases and raw outcomes](../benchmarks/results/conversation-a3/live-gate-probes.json).

## Refresh, privacy and limits

- Explicit `mnemosyne_recall` always uses the supplied query and clears conversational reuse state. Explicit criteria remain exact and independent.
- Active evidence is fetched and checked before it is sent to the gate. Deletion, content or metadata edits, expired validity and revoked access force a new search.
- Session, bank, channel, author or profile changes prevent reuse. Session callbacks, reset/rewind, reinitialization and shutdown clear active state.
- A new search is required after six reuses or five minutes. Oversized gate context also falls back to a new search. Active text and queries are bounded; no conversation cache is persisted.
- The gate has a 0.9-second budget inside the existing 6.5-second prefetch budget. Gate failure or uncertainty falls back to searching the new message. Required search failures return no bank context, preserving timeout recovery.
- New topics require fresh decisions and may cost more than strict mode because of the extra gate call. Cache eviction, restarting Hermes, long memories, frequent writes and explicit tools also reduce savings.

A model gate can incorrectly decide that no further evidence is needed. Strict mode preserves a fresh question for every new message and remains the default. The historical LongMemEval final-answer results do not validate economy mode. Savings depend on the conversation; the 55.6% figure is not a production-wide promise.

Diagnostics include `prefetch.conversation.action`, reason codes, gate scores, timing and usage. These new fields contain no query or memory text. Overall experiment reports include capture cost; per-turn costs include the gate and retrieval.

## Reproduce

Build a wheel with `python -m build`. Use the Python environment that runs Hermes for host integration. All profiles, banks and memory fixtures below are temporary. Nothing is read from your existing memory.

```sh
# Offline contracts through actual Hermes, with scripted Jev responses.
/path/to/hermes/venv/bin/python scripts/measure_conversation_cost.py \
  --hermes-root /path/to/hermes \
  --wheel dist/perfectrecall-0.1.0a3-py3-none-any.whl \
  --records 10000 --output conversation-offline.json

# Paid eight-turn comparison. OPENROUTER_API_KEY must be set in the environment.
/path/to/hermes/venv/bin/python scripts/measure_conversation_cost.py \
  --hermes-root /path/to/hermes \
  --wheel dist/perfectrecall-0.1.0a3-py3-none-any.whl \
  --records 256 --live --max-cost 0.10 --output conversation-live.json

# One cold search plus two follow-ups at 10,000 records.
/path/to/hermes/venv/bin/python scripts/measure_conversation_cost.py \
  --hermes-root /path/to/hermes \
  --wheel dist/perfectrecall-0.1.0a3-py3-none-any.whl \
  --records 10000 --turns 3 --mode economy --live --max-cost 0.20 \
  --output conversation-10000.json

# Gate-only cases; omit --live for a scripted contract check.
python scripts/verify_conversation_gate.py \
  --wheel dist/perfectrecall-0.1.0a3-py3-none-any.whl \
  --live --output gate-probes.json
```

The cost stop is checked before dispatch; concurrent requests already in flight can exceed it. The gate-only script has a $0.01 stop threshold. Offline results establish code contracts, not model quality or API latency. The final packaged code is also checked offline; live report hashes identify the measured builds rather than implying every documentation rebuild was retested with paid calls.
