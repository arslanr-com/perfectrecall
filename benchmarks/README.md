# Benchmarks

These experiments measure **final answers produced by a calling agent**, not just retrieval recall. Both arms use `openai/gpt-5.6-luna` with `reasoning.effort=high`. The common judge is `openai/gpt-4.1`. PerfectRecall's predecessor was called Jevosyne; that historical name remains in frozen scripts, protocol identifiers, salts, and patches so their hashes and case selections stay reproducible.

## Results

| Experiment | Status | Cases | Mnemosyne errors | Jev errors | Fewer errors | Accuracy, baseline → Jev |
|---|---|---:|---:|---:|---:|---:|
| LongMemEval-S, frozen confirmation | Independent at execution | 120 | 62 | 17 | 72.6% | 48.3% → 85.8% |
| Same cases, revised short-criteria prompt | Development | 120 | 62 | 11 | 82.3% | 48.3% → 90.8% |
| PersonaMem-v2, corrected common caller policy | Development | 20 | 9 | 9 | 0% | 55.0% → 55.0% |

For the independent run, the accuracy increase is **37.5 percentage points**, or **77.6% more correct answers relative to baseline**. This does not mean 77.6% more correct memories. Error reduction is `1 - Jev errors / baseline errors`.

The independent run repaired 54 baseline errors and introduced 9 regressions. The development prompt repaired 56 and introduced 5. The PersonaMem run repaired 1 and introduced 1. We do not pool these different experiments or claim universal superiority.

Conservatively scoring every retried baseline case as correct and every retried Jev case as incorrect gives **61 → 18 errors (70.5%)** for the independent run. For the short-criteria development run, the corresponding result is **61 → 14 (77.0%)**. The original 80% independent error-reduction target was not met. Model judging, a custom answer policy, related question families, and remote-model nondeterminism limit generalization.

[Per-case result views](results/) contain every selected ID and both correctness labels. They omit source conversations, private filesystem paths, raw logs, and internal model reasoning. They retain source-report hashes and runtime plans. These reduced views can verify arithmetic; they are not a substitute for inspecting answers in a rerun. Original frozen scripts can produce full local answer and retrieval traces from the public datasets.

```sh
python benchmarks/summarize_results.py
```

## Independent protocol

The source is the cleaned LongMemEval-S dataset, deterministically stratified into 20 questions of each of six types. IDs are hash-ordered with the recorded historical salt; the test offset is 10. All 120 cases remain in the denominator, spanning 119 question families. Earlier development cases were excluded at freeze time. The later prompt experiment reuses these cases and is therefore development work.

Each history is split identically into 6,000-character memory records. Gold answers and answer-session labels never enter retrieval or caller prompts. The caller can make up to three searches, receive at most 20 records and 120,000 characters in total, and produce up to 16,384 tokens. The judge receives the question, reference, and final answer. This is a custom paired tool-use experiment, not the official LongMemEval evaluation score.

Baseline: original Mnemosyne commit `199d4bc6662bd51c18275331ddeb51ac07387acf`, with API `openai/text-embedding-3-small` and its inherited hybrid ranking. Jev confirmation runtime: `e76571a968f74bab5dabf420da6057cd0ec85b1a`. The public patch reconstructs the latter from upstream and verifies the exact runtime hash. The short-criteria development snapshot is `748bd2329149466d785b21cbd06b2bc76eef5039`.

The release includes the short-criteria guidance plus subsequent packaging, storage, and performance changes. **No table above is a fresh independent evaluation of PerfectRecall 0.1.0a1.**

Observed confirmation usage, including retained attempts: Jev 114,724 requests, 114,716 priced responses, approximately $2.69835; caller/judge approximately $0.20718 baseline and $0.29337 Jev. Baseline embedding usage was 369 calls, approximately $0.25721 upstream inference cost (account-billed BYOK cost can differ). Missing priced responses, routing changes, and future pricing mean these are observations, not price guarantees. Do not infer deployment latency from these batched quality runs.

## Reproduce the historical comparison

Run from this repository root. The historical environment is separate and may use embeddings for the original baseline. Those dependencies and the reconstructed source are not installed into PerfectRecall.

```sh
python3.13 -m venv benchmarks/work/venv
benchmarks/work/venv/bin/python -m pip install -r benchmarks/requirements-historical.txt
python benchmarks/prepare_historical.py
mkdir -p benchmarks/downloads benchmarks/runs
curl --fail --location \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  --output benchmarks/downloads/longmemeval_s_cleaned.json
python benchmarks/verify_selection.py benchmarks/downloads/longmemeval_s_cleaned.json
```

The dataset SHA-256 must be `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. The selection check stops on a changed download. `prepare_historical.py` verifies all three runtime hashes after applying the pinned patches; it also accepts `--upstream-local /path/to/clone` for offline source reconstruction.

Save your OpenRouter key in a permission-restricted file **outside the repository**. The following commands make paid API calls; replace `/path/to/key-file` with that path. Run arms sequentially. The four jobs inside each arm are the frozen batch scheduling, not Jev's internal request concurrency.

```sh
benchmarks/work/venv/bin/python benchmarks/run_isolated.py benchmarks/historical/confirmation/run_agentic_batches.py \
  --dataset benchmarks/downloads/longmemeval_s_cleaned.json \
  --source-root benchmarks/work/baseline --backend baseline \
  --api-key-file /path/to/key-file --output benchmarks/runs/baseline.json \
  --scope frozen-independent --test-offset 10 --per-type 20 --batch-size 10 --jobs 4 \
  --caller-budget 1.5 --embedding-budget 0.5 --jev-budget 6 --max-jev-requests 360000

benchmarks/work/venv/bin/python benchmarks/run_isolated.py benchmarks/historical/confirmation/run_agentic_batches.py \
  --dataset benchmarks/downloads/longmemeval_s_cleaned.json \
  --source-root benchmarks/work/confirmation --backend jev \
  --api-key-file /path/to/key-file --output benchmarks/runs/jev.json \
  --scope frozen-independent --test-offset 10 --per-type 20 --batch-size 10 --jobs 4 \
  --caller-budget 1.5 --embedding-budget 0.5 --jev-budget 6 --max-jev-requests 360000

python benchmarks/historical/confirmation/compare_agentic_memory.py \
  --baseline benchmarks/runs/baseline.json --jev benchmarks/runs/jev.json \
  --output benchmarks/runs/comparison.json
```

The isolation wrapper uses a temporary Hermes profile and clears inherited memory settings, leaving your real profile untouched. Frozen evaluator files remain byte-for-byte unchanged. Run `--help` on the scripts for all options. Existing completed batches are reused only when the frozen plan matches. Failed attempts are retained under `.parts/failures`; do not delete them or drop hard cases. Report conservative retry sensitivity along with primary grades. Reruns can differ because models and routing change; original requested/resolved model information is recorded in result plans and usage.

For the short-criteria experiment, use `benchmarks/work/atomic` as the Jev source, `--scope diagnostic`, and a new output filename. Keep the same baseline, caller, budgets, case selection, and judge. Do not call this another independent confirmation.

## PersonaMem-v2

The latest 20-case result is a tie. Earlier judge/policy variants are not the headline: the corrected policy permits general knowledge for new recommendations in both arms while requiring remembered evidence for personal facts. The judge checks the target preference, not every incidental detail in an example response. This binary adaptation is not the official PersonaMem metric.

Download [PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) at revision `ed956dea41521fc4499acbc63f966e0fd3c053ba`, including `benchmark/text/benchmark.csv` and referenced 128k chat histories. Use Hugging Face's snapshot download or Git LFS. The adapter validates the CSV hash and records history hashes.

```sh
python benchmarks/historical/personamem/prepare_personamem.py \
  --dataset-root benchmarks/downloads/PersonaMem-v2 \
  --reservation benchmarks/runs/personamem-reservation.json \
  --split development --output benchmarks/runs/personamem-development.json

benchmarks/work/venv/bin/python benchmarks/run_isolated.py benchmarks/historical/personamem/evaluate_agentic_memory.py \
  --dataset benchmarks/runs/personamem-development.json \
  --source-root benchmarks/work/baseline --backend baseline \
  --api-key-file /path/to/key-file --output benchmarks/runs/personamem-baseline.json \
  --scope diagnostic --split test --test-offset 0 --per-type 20 \
  --caller-budget 0.5 --embedding-budget 0.1 --jev-budget 2 --max-jev-requests 100000

benchmarks/work/venv/bin/python benchmarks/run_isolated.py benchmarks/historical/personamem/evaluate_agentic_memory.py \
  --dataset benchmarks/runs/personamem-development.json \
  --source-root benchmarks/work/atomic --backend jev \
  --api-key-file /path/to/key-file --output benchmarks/runs/personamem-jev.json \
  --scope diagnostic --split test --test-offset 0 --per-type 20 \
  --caller-budget 0.5 --embedding-budget 0.1 --jev-budget 2 --max-jev-requests 100000
```

Compare with the same `compare_agentic_memory.py` interface. The development dataset hash and selected IDs are in [personamem-freeze.json](results/personamem-freeze.json). The reserved confirmation cases were not used for the reported result.

## Capacity

Cold automatic prefetch on 1,024 synthetic short memories, three runs per concurrency setting:

| Workers | Median full-worker time | Range | Context delivered within Hermes' 8-second window |
|---:|---:|---:|---:|
| 128 | 4.323 s | 3.730–12.077 s | 2/3 |
| 256 | 5.194 s | 3.761–5.772 s | 3/3 |
| 512, diagnostic override | 6.500 s | 5.604–33.401 s | 2/3 |

For **10,000 synthetic short memories**, the final 128-worker run took about **144.24 seconds**, recorded **106 failed operations**, and injected no target context. The 512-worker diagnostic failed after about **71.25 seconds**, with **434 failed operations**, and also injected no target context. An earlier 128-worker observation ended before completion and is retained as incomplete. No exact transport cause was established. These are failures, not completed retrieval timings or quality scores.

Full-corpus work grows with bytes and evidence spans, not just record count. The runtime streams SQL reads but materializes eligible records for ranking; it is not constant-memory. The default remains 128, configurable up to 256. No production 512-worker default is justified by these measurements. Capacity artifacts retain the historical wheel hashes and distinguish scheduling overrides. A new run on the renamed release is a new measurement, not an exact repetition of those wheel bytes.

Use `scripts/measure_capacity.py` with an explicit wheel, Hermes checkout, synthetic record count, and `--live` to reproduce this kind of measurement. It records the host deadline separately from the background worker. Stop large sweeps when failures are observed; do not convert partial results to successful recall.

## Attribution

LongMemEval: Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu, *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, 2024 / ICLR 2025, [paper](https://arxiv.org/abs/2410.10813), [code](https://github.com/xiaowu0162/LongMemEval), [cleaned data](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), MIT.

PersonaMem-v2: Bowen Jiang and collaborators, *PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory*, [paper](https://arxiv.org/abs/2512.06688), [data](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), CC BY 4.0. We selected one query per persona, adapted histories into memory records, and applied a custom binary judge. See [third-party notices](../THIRD_PARTY_NOTICES.md).
