# MTEB Multilingual v2 tokenizer-audit protocol

## Objective

Pass 2 asks whether the XLM-R overlap structure observed on controlled FLORES
text remains stable under downstream task distributions. It produces one
balanced overall condition plus independent STS, retrieval, classification,
clustering, and reranking conditions. It does not run an embedding model or
claim that token overlap causes a score change.

The task inventory is resolved from the official
[`MTEB(Multilingual, v2)` benchmark](https://docs.mteb.org/overview/available_benchmarks/)
in `mteb==2.19.5`, source revision
`0adc124417f74e56fd974eb3c64856b8a9e12196`. Every selected Hugging Face
dataset is additionally pinned by an immutable revision in `mteb_data.py`.

## Coverage-first task slice

Running every English-only dataset in the benchmark would make the observed
vocabulary primarily a measure of English task count. The tokenizer audit
therefore uses common multilingual anchor tasks for full-coverage families
and adds official tasks only where needed to maximize sparse-family coverage.

| Family | Selected official tasks | Requested-language coverage |
|---|---|---:|
| Classification | MassiveIntentClassification; GujaratiNewsClassification | 24/24 |
| Clustering | SIB200ClusteringS2S | 24/24 |
| Retrieval | BelebeleRetrieval | 24/24 |
| STS | STS22.v2; STS17; SemRel24STS; IndicCrosslingualSTS; JSICK | 16/24 |
| Reranking | WikipediaRerankingMultilingual; WebLINXCandidatesReranking; AlloprofReranking; VoyageMMarcoReranking; RuBQReranking; T2Reranking | 12/24 |

This is accurately described as a **coverage-balanced task slice of MTEB
Multilingual v2**, not as the complete text of every benchmark task.

## Text roles

- Classification and clustering: model-input text only; labels are excluded.
- STS: both sentence fields, assigned to their task-declared languages.
- Retrieval: deduplicated Belebele questions and passages; answer labels and
  relevance metadata are excluded.
- Reranking: query text, corpus title, and corpus text are independent audit
  records; qrels, candidate IDs, URLs, and scores are excluded.

All text is passed to the pinned XLM-R tokenizer unchanged. Complete records
are retained—there is no text truncation, lowercasing, transliteration, accent
stripping, or external Unicode normalization.

## Balancing

1. Deduplicate exact UTF-8 text within task family and language.
2. Count eligible XLM-R tokens after excluding special/control IDs.
3. Rank complete records by a stable SHA-256 key with seed `1729`.
4. Select without exceeding the common family budget. The effective budget is
   the lower of 20,000 tokens and the least available supported-language pool.
5. Construct `overall` from the already-balanced family records and apply a
   common 30,000-token budget across all 24 languages.

STS and reranking do not have official source coverage for all requested
languages. Their canonical matrices contain only observed languages; their
`*_all_languages` matrices are 24×24 and use NA cells for unsupported
languages. No data is translated, borrowed, or imputed to fill a gap.

## Outputs and validation

Each condition receives the same native outputs as FLORES: symmetric shared
type counts, symmetric type-IoU, directional frequency overlap, per-language
statistics, token-frequency audit data, manifests, and heatmaps. Additional
24-language masked views make coverage differences visually explicit.

Cross-pass outputs compare every available pair against FLORES using Spearman
rank correlation, mean absolute difference, and signed difference. Native
matrix invariants run independently for each condition; aggregate validation
also checks the expected coverage mask and all 24×24 views.

Raw MTEB text is cached under ignored `data/` paths and is never committed.
