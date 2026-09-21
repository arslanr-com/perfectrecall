# Security and data handling

PerfectRecall persists memory in local SQLite files. Recall sends eligible memory text, user queries, and caller-supplied criteria to the configured Jev endpoint. Scope and visibility filters run before disclosure. The calling agent receives selected original records. Optional consolidation can use the Hermes host language model.

The default endpoint is OpenRouter's Decisions API. Service retention and processing terms are controlled by the chosen provider; this project does not promise zero retention. Use only data you are authorized to send there. This package is not an offline memory system.

Set credentials through the process environment or Hermes' secret configuration. Never commit keys, `.env` files, live databases, profile backups, or real conversation exports. API keys are not included in status output. The client rejects redirects and endpoints containing URL credentials. Errors remain errors; failed decisions do not silently become irrelevant memories.

The repository contains synthetic integration fixtures and reduced benchmark result views. Download public evaluation datasets separately. Reproduction scripts can write public-dataset text and model answers into local results; review any additional data before sharing outputs.

Report vulnerabilities privately through the repository's GitHub Security Advisories if enabled. Otherwise open an issue requesting a private contact channel without including credentials, memory contents, or an exploit against another person's data. There is no published security audit or production security certification.

## Conversation gate

Economy prefetch, enabled by default from 0.1.0a4, sends the original search, the preceding message, the new message, and a bounded set of currently accessible memory text to Jev to decide whether the search question can be reused. Active evidence is checked against storage before this disclosure. This state stays in process memory and is cleared on session callbacks, reinitialization, explicit recall, and shutdown; it is eligible for reuse for at most five minutes or six follow-ups. No persistent conversation cache is added. The new diagnostic fields contain counters, timings, scores, and reason codes, not query or memory text.
