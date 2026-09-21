# Verification

## 0.1.0a2

Verified locally on 2026-09-21. The following checks use the final candidate wheel or source. Live performance and quality results were collected on the preceding development prototype and are documented separately in [PERFORMANCE.md](PERFORMANCE.md).

| Check | Result |
|---|---|
| Automated regression suite | 141 passed on Python 3.13.12 |
| Runtime lint and diff checks | Passed |
| Clean installation | Passed with declared HTTPX/PyYAML dependencies; no prior memory provider or vector packages |
| Actual Hermes, 10,000 mixed records | Automatic capture survived capacity trimming; reopened prefetch and three-criterion recall passed with deterministic transport |
| Warm three-criterion scan | 41,049 decision-cache hits, zero API requests |
| Actual Hermes timeout recovery | Controlled timeout returned in 0.156 seconds for a 0.15-second budget; no lingering worker; immediate next call injected memory |
| Existing upstream databases | Two banks, stored IDs/text, vector shadow tables, page backups and restores preserved |
| MCP stdio | 29 tools, legacy names, stored-record read verified |
| Final-wheel live 10,000-record lifecycle/reliability | Pending: configured API key spending cap exhausted |

Tested wheel SHA-256: `b7e80fce81b4bb9ac1399572fd0cad96ca93823d21ba00c64933ded712bdc095`.

[Machine-readable reports](../benchmarks/results/performance-a2/) distinguish `live_api: false` integration checks from live development timings. The user’s actual Hermes profile and memories were not changed.

## Historical 0.1.0a1 verification

Verified locally for PerfectRecall 0.1.0a1 on 2026-09-21. These are release integration checks, not a new independent quality benchmark.

| Check | Result |
|---|---|
| Automated regression suite | 109 passed, Python 3.13.12 |
| Runtime lint and syntax checks | Passed |
| Fresh isolated installation | Only PerfectRecall, PyYAML, and pip installed |
| Empty Hermes profile | Provider discovered and initialized without Mnemosyne |
| Automatic lifecycle | User fact captured after a turn and injected before a later turn after reopening |
| Live Jev lifecycle | 8 requests, 0 failures; observed cost $0.000131502 |
| Existing upstream databases | Two banks reopened, original IDs/text and populated vector shadow tables preserved |
| Vector dependency isolation | Imports blocked while reading and recalling legacy databases |
| Legacy-database backup and restore | SQLite page snapshots restored without vector dependencies |
| MCP stdio | 29 tools listed; legacy tool names and existing record read verified |
| Historical source reconstruction | All three runtime hashes match their freezes |
| LongMemEval data selection | Exact dataset hash and all 120 selected IDs verified |
| PersonaMem adaptation | All 20 development cases reproduce the frozen dataset hash |
| Publication source and archives | Automated language, dependency, private-path, and credential scans; see audit report |

Hermes host commit: `4f22543509d1b91dc45bcb369447126c5eb14fb7`, tested using its Python 3.11 environment. The Hermes checks exercise the real plugin loader, MemoryProvider interface, and MemoryManager callbacks. They do not claim a full interactive caller conversation or a production deployment.

Tested wheel SHA-256:

```text
ca3ed594906dd33d8e76dac1327e6642dfb9f964e4a4ec742b4504fa8eb070f1
```

Machine-readable reports are in [benchmarks/results](../benchmarks/results/). The live check uses synthetic facts in a temporary profile. The user's installed Hermes configuration and real memories were not changed.

The CI workflow is prepared for Python 3.10, 3.11, and 3.13 on Linux. Hosted CI has not run as part of this local preparation. These checks preceded the initial GitHub publication; they did not publish a PyPI package.

Historical limitation of the per-span runtime: the measured 10,000-record cold prefetch fails to deliver useful context within Hermes' eight-second window. Larger-corpus performance remains work for a later release; the benchmark documentation retains the failed and incomplete runs.
