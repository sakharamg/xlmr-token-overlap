# reranking descriptive report

## Run coverage

- Languages: 12
- Examples per language: 389–2,320
- Analysis tokens per language: 19,996–20,000
- Unique observed analysis tokens: 3,266–5,287
- Unknown-token occurrences: 31

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
10.69% within Latin-script languages,
2.13% for Latin/non-Latin pairs, and
2.47% within the non-Latin set.

The strongest observed pair is `ita_Latn`–`ron_Latn`
at 14.93% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| ita_Latn – ron_Latn | 14.93% | 1,306 |
| ita_Latn – por_Latn | 13.94% | 1,230 |
| por_Latn – ron_Latn | 13.32% | 1,170 |
| deu_Latn – nld_Latn | 12.81% | 1,141 |
| deu_Latn – swe_Latn | 12.00% | 1,109 |
| nld_Latn – swe_Latn | 11.86% | 1,041 |
| eng_Latn – ron_Latn | 11.40% | 867 |
| swe_Latn – ron_Latn | 11.26% | 1,016 |
| nld_Latn – ron_Latn | 11.25% | 985 |
| eng_Latn – ita_Latn | 11.12% | 858 |

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
