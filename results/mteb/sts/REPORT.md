# sts descriptive report

## Run coverage

- Languages: 16
- Examples per language: 5–250
- Analysis tokens per language: 2,923–2,951
- Unique observed analysis tokens: 706–1,507
- Unknown-token occurrences: 7

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
3.95% within Latin-script languages,
0.73% for Latin/non-Latin pairs, and
0.62% within the non-Latin set.

The strongest observed pair is `spa_Latn`–`ita_Latn`
at 7.09% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| spa_Latn – ita_Latn | 7.09% | 150 |
| eng_Latn – fra_Latn | 5.62% | 126 |
| spa_Latn – fra_Latn | 5.59% | 121 |
| fra_Latn – ita_Latn | 5.28% | 118 |
| eng_Latn – guj_Gujr | 4.85% | 105 |
| spa_Latn – ind_Latn | 4.83% | 120 |
| deu_Latn – nld_Latn | 4.81% | 91 |
| fra_Latn – ind_Latn | 4.75% | 122 |
| eng_Latn – ind_Latn | 4.63% | 119 |
| deu_Latn – fra_Latn | 4.62% | 102 |

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
