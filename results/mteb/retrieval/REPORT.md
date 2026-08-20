# retrieval descriptive report

## Run coverage

- Languages: 24
- Examples per language: 294–442
- Analysis tokens per language: 19,994–20,000
- Unique observed analysis tokens: 1,905–4,941
- Unknown-token occurrences: 5

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
10.01% within Latin-script languages,
2.08% for Latin/non-Latin pairs, and
1.78% within the non-Latin set.

The strongest observed pair is `ind_Latn`–`zsm_Latn`
at 38.48% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| ind_Latn – zsm_Latn | 38.48% | 2,508 |
| spa_Latn – por_Latn | 23.21% | 1,654 |
| ita_Latn – por_Latn | 16.73% | 1,275 |
| spa_Latn – ita_Latn | 15.83% | 1,202 |
| zsm_Latn – tgl_Latn | 15.14% | 993 |
| eng_Latn – fra_Latn | 14.45% | 1,006 |
| ind_Latn – tgl_Latn | 14.40% | 981 |
| deu_Latn – nld_Latn | 14.13% | 1,095 |
| eng_Latn – ita_Latn | 13.73% | 1,051 |
| eng_Latn – swe_Latn | 13.42% | 1,026 |

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
