from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fill_readme_results.py"


class FillReadmeResultsTests(unittest.TestCase):
    def _write_suite(
        self,
        root: Path,
        conditions: list[tuple[str, int, str, str, float, float, float]],
        *,
        sqe: bool = False,
    ) -> None:
        root.mkdir(parents=True)
        (root / "validation_report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "checks_passed": 7,
                    "checks_run": 7,
                }
            ),
            encoding="utf-8",
        )
        with (root / "cross_pass_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "condition",
                    "languages",
                    "pairs",
                    "spearman_vs_flores",
                    "mean_absolute_difference_points",
                    "mean_signed_difference_points",
                ],
            )
            writer.writeheader()
            for condition, languages, _, _, _, spearman, delta in conditions:
                writer.writerow(
                    {
                        "condition": condition,
                        "languages": languages,
                        "pairs": languages * (languages - 1) // 2,
                        "spearman_vs_flores": spearman,
                        "mean_absolute_difference_points": abs(delta),
                        "mean_signed_difference_points": delta,
                    }
                )
        for condition, _, left, right, iou, _, _ in conditions:
            path = root / condition
            path.mkdir()
            (path / "summary.json").write_text(
                json.dumps(
                    {
                        "strongest_type_iou_pair": {
                            "language_i": left,
                            "language_j": right,
                            "type_iou_percent": iou,
                        }
                    }
                ),
                encoding="utf-8",
            )
        if sqe:
            (root / "run_summary.json").write_text(
                json.dumps(
                    {
                        "ground_truth_summary": {
                            "empty_gt_rows": 9,
                            "missing_gt_reference_occurrences": 69,
                        },
                        "data_id_summary": {
                            "duplicate_data_id_extra_rows": 8,
                        },
                        "text_summary": {
                            "skipped_empty_or_nontext_data_rows": 0,
                            "skipped_empty_or_nontext_query_rows": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

    def test_fills_both_blocks_and_check_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readme = root / "README.md"
            readme.write_text(
                "before\n<!-- BEGIN GENERATED: PASS3_RESULTS -->\nstale\n"
                "<!-- END GENERATED: PASS3_RESULTS -->\nmiddle\n"
                "<!-- BEGIN GENERATED: SQE_RESULTS -->\nstale\n"
                "<!-- END GENERATED: SQE_RESULTS -->\nafter\n",
                encoding="utf-8",
            )
            language_key = root / "results" / "flores"
            language_key.mkdir(parents=True)
            (language_key / "language_key.csv").write_text(
                "language_code,language\nind_Latn,Indonesian\n"
                "zsm_Latn,Malay\n",
                encoding="utf-8",
            )
            pass3 = root / "private" / "pass3"
            self._write_suite(
                pass3,
                [
                    ("overall", 24, "ind_Latn", "zsm_Latn", 41.234, 0.9124, -3.2),
                    ("sts", 24, "ind_Latn", "zsm_Latn", 20.0, 0.8, -5.0),
                    ("retrieval", 24, "ind_Latn", "zsm_Latn", 35.0, 0.9, -4.0),
                ],
            )
            sqe = root / "private" / "sqe"
            self._write_suite(
                sqe,
                [
                    ("settings_standard", 24, "ind_Latn", "zsm_Latn", 31.0, 0.7, -6.0),
                    ("notes_standard", 16, "ind_Latn", "zsm_Latn", 30.0, 0.6, -7.0),
                    ("notes_contextual", 16, "ind_Latn", "zsm_Latn", 29.0, 0.5, -8.0),
                    ("notes_contextual_drop_time", 16, "ind_Latn", "zsm_Latn", 28.0, 0.4, -9.0),
                ],
                sqe=True,
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--readme",
                str(readme),
                "--pass3-dir",
                str(pass3),
                "--sqe-dir",
                str(sqe),
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = readme.read_text(encoding="utf-8")
            self.assertIn("Indonesian–Malay | 41.23% | 0.912 | -3.20 pp", result)
            self.assertIn("Settings standard | 24/24", result)
            self.assertIn("**9** empty ground-truth row(s)", result)
            self.assertIn("**1** blank/non-text query cell(s)", result)
            checked = subprocess.run(
                [*command, "--check"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("current", checked.stdout)

    def test_refuses_failed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readme = root / "README.md"
            readme.write_text(
                "<!-- BEGIN GENERATED: PASS3_RESULTS -->\nx\n"
                "<!-- END GENERATED: PASS3_RESULTS -->\n"
                "<!-- BEGIN GENERATED: SQE_RESULTS -->\ny\n"
                "<!-- END GENERATED: SQE_RESULTS -->\n",
                encoding="utf-8",
            )
            pass3 = root / "pass3"
            pass3.mkdir()
            (pass3 / "validation_report.json").write_text(
                json.dumps(
                    {"status": "failed", "checks_passed": 1, "checks_run": 2}
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--readme",
                    str(readme),
                    "--pass3-dir",
                    str(pass3),
                    "--sqe-dir",
                    str(root / "sqe"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("non-passing result", completed.stderr)


if __name__ == "__main__":
    unittest.main()
