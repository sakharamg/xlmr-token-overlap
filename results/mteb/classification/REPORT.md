# classification descriptive report

## Run coverage

- Languages: 24
- Examples per language: 919–2,568
- Analysis tokens per language: 20,000–20,000
- Unique observed analysis tokens: 1,506–3,209
- Unknown-token occurrences: 2

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
11.06% within Latin-script languages,
1.74% for Latin/non-Latin pairs, and
0.94% within the non-Latin set.

The strongest observed pair is `ind_Latn`–`zsm_Latn`
at 34.11% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| ind_Latn – zsm_Latn | 34.11% | 1,259 |
| spa_Latn – por_Latn | 23.19% | 1,005 |
| eng_Latn – tgl_Latn | 19.76% | 761 |
| eng_Latn – nld_Latn | 18.43% | 749 |
| deu_Latn – nld_Latn | 18.28% | 764 |
| eng_Latn – fra_Latn | 17.57% | 701 |
| eng_Latn – deu_Latn | 17.41% | 724 |
| ind_Latn – tgl_Latn | 15.70% | 637 |
| eng_Latn – swe_Latn | 15.50% | 687 |
| deu_Latn – swe_Latn | 15.46% | 703 |

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
