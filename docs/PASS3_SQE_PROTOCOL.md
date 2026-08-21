# Local Pass-3 and SQE protocol

## Scope

This runner consumes the canonical local dataset cache directly. It performs a
tokenizer-distribution analysis, not an embedding-score evaluation.

The non-negotiable full-text policy is:

- every valid source row is retained;
- every accepted text cell is encoded as one complete string;
- no character clipping;
- tokenizer truncation and padding are disabled;
- no row sampling or XLM-R token-budget sampling;
- no lowercasing, transliteration, accent stripping, or external Unicode
  normalization;
- special/control IDs are diagnosed but excluded from overlap calculations.

Batching in `AuditedTokenizer.count` only bounds peak memory. It never divides
or truncates an input text.

## Pass 3

### Canonical inputs

| Condition | Files | Text roles |
|---|---|---|
| STS | `sts/{locale}/canonical/data.parquet` | `sentence1`, `sentence2` |
| Retrieval | `belebele_retrieval/{locale}/canonical/queries.parquet` and `corpus.parquet` | query `text`, corpus `text` |
| Overall | exact union of the two conditions above | all four roles |

`qrels.parquet` is used to validate query/corpus IDs. It does not repeat text
once per relevance judgment.

All 24 locale directories are required. The audited STS snapshot has different
row counts for Hindi, Japanese, Korean, and Thai and has no stable ID column.
Consequently, it is a full translated-corpus condition rather than a strictly
row-aligned parallel condition. This does not prevent tokenizer-overlap
measurement; it is explicitly recorded as a distributional limitation.

The output contains independent `sts`, `retrieval`, and `overall` directories,
each with native matrices, full-axis matrices, token statistics, validation,
and heatmaps.

## SQE

### Canonical inputs

Each domain/locale uses:

- `data.xlsx#data`: `id`, `text`, optional `remark`;
- `tc.xlsx#tc`: `query`, `gt_ids`;
- optional `contextual_tc.xlsx#tc`;
- optional `contextual_drop_time_tc.xlsx#tc`.

Only `data.text` and test-case `query` are tokenizer inputs. `remark` is
metadata. `gt_ids` is parsed and checked against `data.id` but is not tokenized.
Each domain and query variant is an independent condition. Data text is
included once within each condition alongside that variant's complete query
set.

Empty `gt_ids` cells and references absent from `data.id` are recorded as
metadata warnings in `corpus_manifest.json`; they do not remove or block valid
query text. Use `--strict-ground-truth` when a separate integrity-audit run
should fail on either anomaly.

### Frozen audited coverage

| Domain | Languages | Variants |
|---|---:|---|
| Settings | 24 | standard |
| Notes | 16 | standard, contextual, contextual-drop-time |
| Calendar | 1 (Korean) | all three |
| Call recording | 1 (Korean) | all three |
| Reminder | 1 (Korean) | all three |
| Voice recording | 1 (Korean) | all three |

There is deliberately no pooled SQE-overall matrix. Pooling would make Korean
represent six domains, 15 other languages represent two, and eight languages
represent settings alone. That would confound language with domain. The
`settings_standard` condition is the complete 24-language SQE comparison.
Notes is emitted on the frozen 24-language axes with eight languages masked.
Korean-only conditions retain full token and vocabulary diagnostics but have
no off-diagonal language pairs.

## Reproduction

Install the pinned environment and package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Run Pass 3:

```bash
xlmr-token-overlap all-pass3 \
  --datasets-root /group-volume/SSCore/User_Data/Sanskar/Benchmarking/SR_Gauss_24/model-orchestration/cache/datasets \
  --tokenizer-json data/tokenizers/xlm-roberta-base/tokenizer.json \
  --output-dir results/pass3 \
  --flores-dir results/flores
```

Run all SQE domains and variants:

```bash
xlmr-token-overlap all-sqe \
  --datasets-root /group-volume/SSCore/User_Data/Sanskar/Benchmarking/SR_Gauss_24/model-orchestration/cache/datasets \
  --tokenizer-json data/tokenizers/xlm-roberta-base/tokenizer.json \
  --output-dir results/sqe \
  --flores-dir results/flores
```

Omit `--tokenizer-json` and provide `--source-dir data` to download/reuse the
same repository-pinned XLM-R tokenizer. Use `--domains settings notes` only
when a quicker comparable-language subset is intentionally desired; the
default analyzes every domain.

`corpus_manifest.json` records source checksums, coverage, integrity, and the
explicit null limits. `source_contributions.csv` records examples and full
character totals by condition, language, source, and role. Raw source text is
never written into result artifacts.
