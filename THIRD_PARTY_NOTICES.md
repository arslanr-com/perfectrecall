# Third-party notices

## Mnemosyne

PerfectRecall is an independent derivative of [Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne), originally by Abdias J. The reference upstream commit is `199d4bc6662bd51c18275331ddeb51ac07387acf`. Copyright (c) 2026 Abdias J. The original MIT copyright, permission, and warranty notice is retained in [LICENSE](LICENSE), together with the PerfectRecall modification notice.

Substantial inherited components include SQLite schemas and storage, banks, transactions, scopes, temporal metadata, consolidation, synchronization, canonical memory, persona integration, MCP tools, and the Hermes provider lifecycle. Changes include Jev decision-based selection and classification, removal of production vector retrieval, the PerfectRecall package and installer, unified caller guidance, backup changes, tests, and benchmark tooling. The clean publication repository does not carry the private development history; that does not change the attribution of inherited code.

## LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval), by Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, 2024, ICLR 2025. [Paper](https://arxiv.org/abs/2410.10813).

The [cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) is labeled MIT by its publisher. Copyright (c) 2024 Di Wu; the MIT notice is included at [LICENSES/LongMemEval-MIT.txt](LICENSES/LongMemEval-MIT.txt). No full dataset is bundled. Published result views derive from deterministic case selection, identical history chunking for both arms, tool-using answer generation, and a custom binary correctness judge. They are not official benchmark scores.

```bibtex
@article{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
  year={2024},
  eprint={2410.10813},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.10813}
}
```

## PersonaMem-v2

[PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), by Bowen Jiang and collaborators; [project](https://github.com/bowen-upenn/PersonaMem-v2), [paper](https://arxiv.org/abs/2512.06688). The data is published under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), [legal text](https://creativecommons.org/licenses/by/4.0/legalcode.en). The benchmark uses dataset revision `ed956dea41521fc4499acbc63f966e0fd3c053ba`.

No full dataset is bundled. Adaptations select one query per persona, convert complete 128k histories into memory records, omit reference preferences from the caller's context, and apply a custom binary judge. The latest development run also uses a corrected common recommendation policy. These changes and reduced per-case result views are described in [benchmarks/README.md](benchmarks/README.md). To the extent the PersonaMem-derived result views are adaptations of licensed data, they retain CC BY 4.0 attribution and terms; they are not relicensed as MIT by the root software license.

## Dependencies and services

PyYAML is a production dependency; MCP, AnyIO, and cryptography are optional dependencies. Their own licenses apply to their distributions. The repository does not vendor those packages. Original-baseline dependencies are installed only in the historical benchmark environment and do not belong to the production wheel.

Jev, TypeSafe, OpenRouter, Hermes, OpenAI model names, and upstream names identify services, compatibility, and provenance. Their service terms and trademarks remain their owners'. No sponsorship, partnership, or endorsement is claimed.
