# Publication-candidate verification

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

The CI workflow is prepared for Python 3.10, 3.11, and 3.13 on Linux. Hosted CI has not run as part of this local preparation. No GitHub repository or package has been published by these checks.

Known limitation: the measured 10,000-record cold prefetch fails to deliver useful context within Hermes' eight-second window. Larger-corpus performance remains work for a later release; the benchmark documentation retains the failed and incomplete runs.
