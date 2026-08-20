# Study plan

## Scope and guardrails

The project asks whether XLM-R token sharing is a plausible correlate of the
roughly one-point Stage-2 degradation seen across Latin-script languages.  It
does **not** treat overlap as causal.  Matrix construction and validation are
completed before any Stage-1/Stage-2 score is introduced.

The same frozen definitions are used in every pass:

- observed token types are XLM-R token IDs after the tokenizer's own normalizer;
- no external lowercasing, transliteration, accent stripping, or text cleanup;
- control/special IDs, including `<unk>`, are counted for diagnostics but
  excluded from the overlap vocabulary and occurrence denominator;
- type IoU and shared-type count are symmetric;
- frequency overlap is directional.

## Phase 1 — FLORES-200 (implemented and run)

1. Pin the official FLORES archive and official `xlm-roberta-base`
   `tokenizer.json` by immutable revision and SHA-256.
2. Extract only the 24 requested languages.
3. Require the same 997 `dev` and 1,012 `devtest` line positions in every
   language.  Do not balance away token-count differences because the examples
   are parallel.
4. Encode full sentences with truncation and padding disabled and with no
   wrapper tokens.
5. Emit full and upper-triangular matrices, long-form pairwise tables,
   per-language statistics, token-frequency audit data, grouped heatmaps,
   checksums, and a descriptive report.
6. Validate shape, order, bounds, symmetry, diagonals, upper-triangle masking,
   and count identities.

## Phase 2 — MTEB Multilingual v2 (implemented)

1. Resolve the exact MTEB v2 task inventory and map datasets independently to
   `mteb/sts`, `mteb/retrieval`, `mteb/classification`, `mteb/clustering`, and
   `mteb/reranking`.
2. Convert every text-bearing role (query, document, sentence, label text only
   when it is genuinely model input) to the repository's condition-aware JSONL
   interchange format.
3. Within each task family, choose a deterministic common XLM-R token budget
   across languages and record both available and selected counts.  Do not pool
   task families.
4. Run the unchanged metrics engine once per family.
5. Compare each family matrix with FLORES using pairwise Spearman correlation,
   mean absolute difference, signed difference, and cross-condition rank
   stability.

The implemented pass uses a coverage-balanced slice of official benchmark
tasks. Classification, clustering, and retrieval cover all 24 languages; STS
and reranking retain their official 16/24 and 12/24 coverage as explicit NA
cells in 24×24 masked views. See [`MTEB_PROTOCOL.md`](MTEB_PROTOCOL.md) for the
frozen source revisions, text roles, and sampling rules.

## Phase 3 — balanced STS/retrieval-like data

Use the same JSONL interface, keep STS-like and retrieval-like conditions
separate initially, verify the supplied balance, and run the same engine.

## Performance analysis (only after matrix validation)

Join language-level Stage-1 → Stage-2 deltas only after all three passes are
frozen.  Test descriptive associations with average type IoU, directional
frequency overlap, exposure-weighted overlap to Stage-2 languages, script, and
family.  Report uncertainty and confounds; do not infer causality from a
correlation.
