# Publish the private Pass 3 and SQE results

The README result slots can be filled entirely on the machine that holds the
generated outputs. Nothing from the private datasets needs to be pasted into
chat, and the helper reads only aggregate CSV/JSON summaries.

## 1. Identify the two generated result roots

If the documented commands were run from the repository root with their
default output paths, these are already `results/pass3` and `results/sqe` and
no relocation is needed. Otherwise, set the actual paths:

```bash
PASS3_RUN=/absolute/path/to/generated/results/pass3
SQE_RUN=/absolute/path/to/generated/results/sqe
REPO_ROOT=/absolute/path/to/xlmr-token-overlap
cd "$REPO_ROOT"
```

## 2. Fill every numeric README slot

```bash
python scripts/fill_readme_results.py \
  --readme README.md \
  --pass3-dir "$PASS3_RUN" \
  --sqe-dir "$SQE_RUN"
```

The script refuses to advertise a suite whose `validation_report.json` is not
passing. It fills values from these exact sources:

| README value | Generated source |
|---|---|
| Pass 3 validation count | `$PASS3_RUN/validation_report.json` → `checks_passed`, `checks_run` |
| Pass 3 coverage, Spearman, mean IoU delta | `$PASS3_RUN/cross_pass_summary.csv`, row keyed by `overall`, `sts`, or `retrieval` |
| Pass 3 strongest pair and IoU | `$PASS3_RUN/<condition>/summary.json` → `strongest_type_iou_pair` |
| SQE validation count | `$SQE_RUN/validation_report.json` → `checks_passed`, `checks_run` |
| SQE coverage, Spearman, mean IoU delta | `$SQE_RUN/cross_pass_summary.csv`, row keyed by the displayed condition |
| SQE strongest pair and IoU | `$SQE_RUN/<condition>/summary.json` → `strongest_type_iou_pair` |
| SQE warning/skip counts | `$SQE_RUN/run_summary.json` → `ground_truth_summary`, `data_id_summary`, `text_summary` |

The four SQE rows shown in the README are the cross-language conditions:
`settings_standard`, `notes_standard`, `notes_contextual`, and
`notes_contextual_drop_time`. Korean-only conditions stay in the detailed SQE
report because a one-language slice has no off-diagonal pair or rank
correlation.

## 3. Put each heatmap at its README destination

If the generated result roots are elsewhere, copy these seven files. The
helper also prints this same source-to-destination mapping for any missing
image.

| Copy from | Put in the repository at |
|---|---|
| `$PASS3_RUN/overall/heatmaps_all_languages/type_iou_upper.png` | `results/pass3/overall/heatmaps_all_languages/type_iou_upper.png` |
| `$PASS3_RUN/sts/heatmaps_all_languages/type_iou_upper.png` | `results/pass3/sts/heatmaps_all_languages/type_iou_upper.png` |
| `$PASS3_RUN/retrieval/heatmaps_all_languages/type_iou_upper.png` | `results/pass3/retrieval/heatmaps_all_languages/type_iou_upper.png` |
| `$SQE_RUN/settings_standard/heatmaps_all_languages/type_iou_upper.png` | `results/sqe/settings_standard/heatmaps_all_languages/type_iou_upper.png` |
| `$SQE_RUN/notes_standard/heatmaps_all_languages/type_iou_upper.png` | `results/sqe/notes_standard/heatmaps_all_languages/type_iou_upper.png` |
| `$SQE_RUN/notes_contextual/heatmaps_all_languages/type_iou_upper.png` | `results/sqe/notes_contextual/heatmaps_all_languages/type_iou_upper.png` |
| `$SQE_RUN/notes_contextual_drop_time/heatmaps_all_languages/type_iou_upper.png` | `results/sqe/notes_contextual_drop_time/heatmaps_all_languages/type_iou_upper.png` |

For example:

```bash
install -Dm644 "$PASS3_RUN/overall/heatmaps_all_languages/type_iou_upper.png" \
  results/pass3/overall/heatmaps_all_languages/type_iou_upper.png
install -Dm644 "$PASS3_RUN/sts/heatmaps_all_languages/type_iou_upper.png" \
  results/pass3/sts/heatmaps_all_languages/type_iou_upper.png
install -Dm644 "$PASS3_RUN/retrieval/heatmaps_all_languages/type_iou_upper.png" \
  results/pass3/retrieval/heatmaps_all_languages/type_iou_upper.png
install -Dm644 "$SQE_RUN/settings_standard/heatmaps_all_languages/type_iou_upper.png" \
  results/sqe/settings_standard/heatmaps_all_languages/type_iou_upper.png
install -Dm644 "$SQE_RUN/notes_standard/heatmaps_all_languages/type_iou_upper.png" \
  results/sqe/notes_standard/heatmaps_all_languages/type_iou_upper.png
install -Dm644 "$SQE_RUN/notes_contextual/heatmaps_all_languages/type_iou_upper.png" \
  results/sqe/notes_contextual/heatmaps_all_languages/type_iou_upper.png
install -Dm644 "$SQE_RUN/notes_contextual_drop_time/heatmaps_all_languages/type_iou_upper.png" \
  results/sqe/notes_contextual_drop_time/heatmaps_all_languages/type_iou_upper.png
```

## 4. Copy a minimal public audit bundle

The README needs only the seven PNGs above, but these aggregate files make the
published numbers independently auditable:

- `REPORT.md`, `cross_pass_summary.csv`, `validation_report.json`, and
  `run_summary.json` from each suite root;
- `summary.json` from each README-displayed condition.

Keep the same relative paths under `results/pass3` and `results/sqe`.
`run_summary.json` is required for the SQE warning counts; a Pass 3 copy is
recommended for symmetry.

When the run directories are outside the repository, the aggregate bundle can
be placed with:

```bash
for filename in REPORT.md cross_pass_summary.csv validation_report.json run_summary.json; do
  install -Dm644 "$PASS3_RUN/$filename" "results/pass3/$filename"
  install -Dm644 "$SQE_RUN/$filename" "results/sqe/$filename"
done
for condition in overall sts retrieval; do
  install -Dm644 "$PASS3_RUN/$condition/summary.json" \
    "results/pass3/$condition/summary.json"
done
for condition in settings_standard notes_standard notes_contextual notes_contextual_drop_time; do
  install -Dm644 "$SQE_RUN/$condition/summary.json" \
    "results/sqe/$condition/summary.json"
done
```

Do **not** blindly stage `corpus_manifest.json` or per-condition
`manifest.json`. A run made against the cluster dataset path can record that
absolute internal path in `dataset_root`. Either omit those manifests from the
public snapshot or replace only that path with a non-sensitive placeholder and
recompute any affected manifest checksum metadata before committing. The
runner does not write raw sentences or workbook cells to its results.

## 5. Verify and push to `main`

```bash
python scripts/fill_readme_results.py \
  --readme README.md \
  --pass3-dir "$PASS3_RUN" \
  --sqe-dir "$SQE_RUN" \
  --check
python -m unittest discover -s tests -v
git status --short
git diff -- README.md
```

Inspect the staged paths, especially for manifests or internal paths, then add
only the intended README, heatmaps, and aggregate summaries:

```bash
git add README.md \
  results/pass3/REPORT.md \
  results/pass3/cross_pass_summary.csv \
  results/pass3/validation_report.json \
  results/pass3/run_summary.json \
  results/pass3/overall/summary.json \
  results/pass3/overall/heatmaps_all_languages/type_iou_upper.png \
  results/pass3/sts/summary.json \
  results/pass3/sts/heatmaps_all_languages/type_iou_upper.png \
  results/pass3/retrieval/summary.json \
  results/pass3/retrieval/heatmaps_all_languages/type_iou_upper.png \
  results/sqe/REPORT.md \
  results/sqe/cross_pass_summary.csv \
  results/sqe/validation_report.json \
  results/sqe/run_summary.json \
  results/sqe/settings_standard/summary.json \
  results/sqe/settings_standard/heatmaps_all_languages/type_iou_upper.png \
  results/sqe/notes_standard/summary.json \
  results/sqe/notes_standard/heatmaps_all_languages/type_iou_upper.png \
  results/sqe/notes_contextual/summary.json \
  results/sqe/notes_contextual/heatmaps_all_languages/type_iou_upper.png \
  results/sqe/notes_contextual_drop_time/summary.json \
  results/sqe/notes_contextual_drop_time/heatmaps_all_languages/type_iou_upper.png
git diff --cached --stat
git commit -m "Add Pass 3 and SQE result snapshots"
git push origin main
```

If `git status` shows unrelated work, leave it unstaged. If the clone predates
the placeholder commit, run `git pull --ff-only origin main` before filling the
README.
