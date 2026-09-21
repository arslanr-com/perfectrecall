# Architecture

PerfectRecall retains Mnemosyne's SQLite storage and Hermes lifecycle. Memory banks, scopes, timestamps, validity, provenance, canonical facts, and persona metadata keep their existing representation.

For recall, the provider enumerates visible working and episodic records, applies scope and validity filters before disclosure, and splits long content into structural evidence spans. Jev scores each span against caller-supplied criteria. Scores select original records; the calling agent reads those records and produces the answer. There is no generative query-writing step inside PerfectRecall.

The request pool bounds concurrency and queued work. All eligible records still participate, so reducing top-k does not eliminate corpus-scanning cost. Repeated request payloads can reuse a bounded exact cache. A failed decision operation causes an explicit failure rather than silently admitting a partial corpus result.

Hermes prefetch calls this recall path automatically. Post-turn sync admits and stores useful user content; assistant capture is optional. The inherited lifecycle includes write approval, session isolation, and exclusion of newly captured text that is already visible in the active transcript. The memory tool names remain `mnemosyne_*` for compatibility.

Legacy vector tables and columns remain in existing databases as unused data. The package neither loads their extension nor computes replacement embeddings. Backups copy SQLite pages, preserving those tables without needing their runtime. JSON export/import retains the ordinary legacy data surfaces; page backups are the appropriate way to preserve all virtual-table internals.

Some inherited operations, such as optional consolidation and enrichment, can use Hermes' host language model. Jev owns semantic search and typed decisions. Credentials and processing terms belong to the configured external services; SQLite locality does not make recall offline.
