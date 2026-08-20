# clustering descriptive report

## Run coverage

- Languages: 24
- Examples per language: 460–710
- Analysis tokens per language: 19,988–20,000
- Unique observed analysis tokens: 2,004–5,453
- Unknown-token occurrences: 12

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
13.20% within Latin-script languages,
2.75% for Latin/non-Latin pairs, and
2.27% within the non-Latin set.

The strongest observed pair is `ind_Latn`–`zsm_Latn`
at 46.64% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| ind_Latn – zsm_Latn | 46.64% | 3,027 |
| spa_Latn – por_Latn | 25.44% | 1,907 |
| ind_Latn – tgl_Latn | 18.79% | 1,322 |
| zsm_Latn – tgl_Latn | 18.45% | 1,285 |
| eng_Latn – nld_Latn | 18.08% | 1,422 |
| eng_Latn – fra_Latn | 17.80% | 1,365 |
| ita_Latn – por_Latn | 17.52% | 1,422 |
| ita_Latn – ron_Latn | 17.33% | 1,375 |
| spa_Latn – ita_Latn | 16.63% | 1,355 |
| eng_Latn – ron_Latn | 16.60% | 1,323 |

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
