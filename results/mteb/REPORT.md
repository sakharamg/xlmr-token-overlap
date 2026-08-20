# MTEB Multilingual v2 tokenizer-overlap pass

This is a coverage-balanced tokenizer audit over pinned official tasks from
`MTEB(Multilingual, v2)` as resolved in `mteb==2.19.5`. It is not an MTEB
embedding-score run and it does not claim that overlap causes performance
changes.

## Coverage and sampling

- Frozen language inventory: 24
- Full-coverage families: classification, clustering, retrieval
- STS coverage: 16/24
- Reranking coverage: 12/24
- Requested family token budget: 20,000
- Effective family budgets: {"classification": 20000, "clustering": 20000, "reranking": 20000, "retrieval": 20000, "sts": 2951}
- Requested overall token budget: 30,000
- Effective overall token budget: 30,000
- Selection seed: 1729

Missing languages are represented by blank/NA cells in the `*_all_languages`
24×24 matrices. They are never imputed from another task or language.

## Strongest observed pair by condition

| Condition | Languages | Pair | Type IoU |
|---|---:|---|---:|
| overall | 24 | ind_Latn–zsm_Latn | 39.13% |
| sts | 16 | spa_Latn–ita_Latn | 7.09% |
| retrieval | 24 | ind_Latn–zsm_Latn | 38.48% |
| classification | 24 | ind_Latn–zsm_Latn | 34.11% |
| clustering | 24 | ind_Latn–zsm_Latn | 46.64% |
| reranking | 12 | ita_Latn–ron_Latn | 14.93% |

## Stability against FLORES

| Condition | Languages | Pairs | Spearman | Mean absolute difference |
|---|---:|---:|---:|---:|
| overall | 24 | 276 | 0.946 | 6.33 |
| sts | 16 | 120 | 0.780 | 7.75 |
| retrieval | 24 | 276 | 0.935 | 7.10 |
| classification | 24 | 276 | 0.749 | 6.91 |
| clustering | 24 | 276 | 0.982 | 5.36 |
| reranking | 12 | 66 | 0.780 | 7.12 |

## Validation

All 55 aggregate MTEB checks pass, in addition to the
28 native matrix invariants run independently inside each condition.

See `task_inventory.csv`, `sampling_manifest.json`,
`sampling_by_language_family.csv`, and `source_contributions.csv` for the exact
task revisions, coverage, and selected token counts. Raw benchmark text is not
committed.
