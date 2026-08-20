# flores descriptive report

## Run coverage

- Languages: 24
- Examples per language: 2,009–2,009
- Analysis tokens per language: 56,004–85,423
- Unique observed analysis tokens: 2,652–9,625
- Unknown-token occurrences: 31

## Script-level structure

Across unordered off-diagonal pairs, mean token-type IoU is
22.25% within Latin-script languages,
5.34% for Latin/non-Latin pairs, and
4.43% within the non-Latin set.

The strongest observed pair is `ind_Latn`–`zsm_Latn`
at 61.82% type IoU. This is descriptive evidence
about tokenizer sharing under content-matched FLORES text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

| Pair | Type IoU | Shared types |
|---|---:|---:|
| ind_Latn – zsm_Latn | 61.82% | 5,974 |
| spa_Latn – por_Latn | 40.42% | 5,047 |
| zsm_Latn – tgl_Latn | 32.62% | 3,616 |
| ind_Latn – tgl_Latn | 31.73% | 3,603 |
| eng_Latn – tgl_Latn | 30.50% | 3,577 |
| ita_Latn – ron_Latn | 30.32% | 4,051 |
| eng_Latn – fra_Latn | 29.60% | 3,737 |
| ita_Latn – por_Latn | 29.12% | 3,973 |
| eng_Latn – nld_Latn | 28.82% | 3,661 |
| spa_Latn – ita_Latn | 27.97% | 3,896 |

## Interpretation boundary

These matrices should be frozen and compared with independent MTEB task-family
matrices before Stage-1 → Stage-2 deltas are joined. Script, family, tokenizer
coverage, training exposure, and task/domain distribution remain confounded at
this stage.
