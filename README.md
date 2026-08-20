# XLM-R cross-lingual token overlap

Reproducible 24-language matrices for testing whether XLM-R tokenizer sharing
is a plausible correlate of multilingual Stage-2 performance degradation.

The repository completes **Pass 1 on FLORES-200** and implements **Pass 2 on a
coverage-balanced task slice of MTEB Multilingual v2**. It does not claim that
overlap causes a performance change.

## Checked-in FLORES result

All 28 result invariants pass. Across the 276 unordered off-diagonal pairs,
mean token-type IoU is **22.25% within Latin-script languages**, compared with
**5.34% for Latin/non-Latin pairs** and **4.43% within the non-Latin set**.
Indonesian–Malay is the strongest pair at **61.82%**. This makes overlap a
plausible signal worth testing against Stage-2 deltas, but does not establish
that it caused those deltas.

![FLORES XLM-R token-type IoU](results/flores/heatmaps/type_iou_upper.png)

## What is measured

For language `i`, let `V_i` be the observed XLM-R token IDs and `c_i(t)` the
occurrence count after excluding tokenizer special/control IDs.

| Output | Definition | Shape |
|---|---|---|
| Shared types | `|V_i ∩ V_j|` | symmetric 24 × 24 |
| Token-type IoU | `100 × |V_i ∩ V_j| / |V_i ∪ V_j|` | symmetric; upper triangle supplied |
| Frequency overlap | `100 × Σ_{t∈V_i∩V_j} c_i(t) / Σ_{t∈V_i} c_i(t)` | directional 24 × 24 (`i → j`) |

Text is passed to the tokenizer exactly as released: there is no external
lowercasing, transliteration, accent stripping, or Unicode normalization.
Tokenizer-internal normalization remains active because it is part of genuine
XLM-R behavior. Encoding has no wrapper tokens, truncation, or padding.

## Reproduce the checked-in FLORES pass

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
xlmr-token-overlap all-flores \
  --source-dir data \
  --output-dir results/flores
```

`all-flores` downloads and verifies:

- the official [FLORES-200 archive](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz),
  SHA-256 `b8b0b767...4011f6`;
- the official [`FacebookAI/xlm-roberta-base` tokenizer](https://huggingface.co/FacebookAI/xlm-roberta-base),
  revision `42f548f3...68bf6`, SHA-256 `a898ea75...9d2e2c`.

Only the 24 requested languages are extracted. FLORES text is licensed
CC BY-SA 4.0 and is intentionally not committed here; the derived matrices are
reproducible from the pinned public source.

To use the exact tokenizer from a trained model instead, point the runner at
that model's `tokenizer.json`:

```bash
xlmr-token-overlap run-flores \
  --flores-root data/flores200_dataset \
  --tokenizer-json /path/to/model/tokenizer.json \
  --output-dir results/my-model-tokenizer
```

## Outputs

The checked-in [`results/flores`](results/flores) directory contains:

- full CSV and Parquet matrices plus an upper-triangular IoU matrix;
- `pairwise_long.{csv,parquet}` with all 576 ordered pairs;
- per-language and per-split statistics;
- observed token frequencies for audit/re-analysis;
- grouped heatmaps using linguistic/script ordering;
- source/output checksums and runtime metadata in `manifest.json`;
- `validation_report.json` and a descriptive `REPORT.md`.

The matrix row order is Germanic Latin, Romance Latin, Austronesian Latin,
other Latin-script languages, then the eight non-Latin-script languages.
`language_key.csv` is the authoritative label mapping.

## Pass 2: MTEB Multilingual v2

The pinned MTEB pass produces `overall` plus five independent task-family
conditions. Classification, clustering, and retrieval cover all 24 requested
languages. The official task inventory provides usable STS data for 16 and
reranking data for 12; unsupported cells remain NA in full 24×24 views rather
than being imputed.

| Condition | Language coverage | Main source design |
|---|---:|---|
| Overall | 24/24 | Rebalanced selected family records |
| STS | 16/24 | Five official STS tasks |
| Retrieval | 24/24 | Belebele questions and passages |
| Classification | 24/24 | MASSIVE plus Gujarati News |
| Clustering | 24/24 | SIB-200 |
| Reranking | 12/24 | Six official reranking tasks |

Reproduce the pass with:

```bash
pip install -r requirements-lock.txt
pip install -r requirements-mteb.txt
pip install -e . --no-deps
xlmr-token-overlap all-mteb \
  --source-dir data \
  --output-dir results/mteb \
  --flores-dir results/flores \
  --family-token-budget 20000 \
  --overall-token-budget 30000 \
  --seed 1729
```

The loader resolves the official benchmark at `mteb==2.19.5`, pins every
selected dataset revision, streams only selected configurations, applies
deterministic complete-record XLM-R token budgets, and commits no raw text.
See [`docs/MTEB_PROTOCOL.md`](docs/MTEB_PROTOCOL.md).

## Generic Pass 2/3 interface

The generic runner accepts JSON Lines with one text-bearing input per row:

```json
{"condition":"mteb/sts","language_code":"eng_Latn","text":"...","example_id":"task:id:side-a","split":"test"}
{"condition":"mteb/retrieval","language_code":"eng_Latn","text":"...","example_id":"task:id:query","split":"test"}
```

```bash
xlmr-token-overlap run-jsonl \
  --input mteb_texts.jsonl \
  --tokenizer-json data/tokenizers/xlm-roberta-base/tokenizer.json \
  --output-dir results/mteb
```

Every `condition` is tokenized and written independently, so custom STS,
retrieval, classification, clustering, and reranking corpora cannot be
accidentally pooled. See [`docs/PLAN.md`](docs/PLAN.md).

## Validate or test

```bash
python -m unittest discover -s tests -v
xlmr-token-overlap validate results/flores
```

Validation checks matrix dimensions/order, symmetric metrics, 100% diagonals,
directional bounds, upper-triangle masking, pair uniqueness, and token-count
identities.

## Data and model citations

- NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered
  Machine Translation*.
- Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning
  at Scale*.

Code is MIT licensed. FLORES-200 retains its CC BY-SA 4.0 license.
