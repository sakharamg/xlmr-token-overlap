# overall descriptive report

## Run coverage

- Languages: 24
- Examples per language: 857–2,105
- Analysis tokens per language: 29,998–30,000
- Unique observed analysis tokens: 2,524–6,467
- Unknown-token occurrences: 25

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
11.48% within Latin-script languages,
2.36% for Latin/non-Latin pairs, and
1.88% within the non-Latin set.

The strongest observed pair is `ind_Latn`–`zsm_Latn`
at 39.13% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| ind_Latn – zsm_Latn | 39.13% | 3,006 |
| spa_Latn – por_Latn | 22.73% | 2,090 |
| zsm_Latn – tgl_Latn | 17.84% | 1,396 |
| ind_Latn – tgl_Latn | 16.85% | 1,382 |
| ita_Latn – ron_Latn | 16.64% | 1,686 |
| ita_Latn – por_Latn | 15.46% | 1,581 |
| deu_Latn – nld_Latn | 15.44% | 1,570 |
| spa_Latn – ita_Latn | 15.32% | 1,532 |
| eng_Latn – tgl_Latn | 15.22% | 1,253 |
| eng_Latn – fra_Latn | 14.76% | 1,330 |

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
