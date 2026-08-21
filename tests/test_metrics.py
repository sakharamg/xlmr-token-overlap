import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from xlmr_token_overlap.constants import LANGUAGES
from xlmr_token_overlap.metrics import compute_metrics
from xlmr_token_overlap.reporting import build_descriptive_outputs
from xlmr_token_overlap.tokenization import LanguageTokens


class _IdLookup:
    @staticmethod
    def id_to_token(token_id):
        return f"t{token_id}"


class _Tokenizer:
    eligible_vocab_size = 100
    tokenizer = _IdLookup()


def _language_tokens(code, counts):
    counts = Counter(counts)
    total = sum(counts.values())
    return LanguageTokens(
        language_code=code,
        examples=1,
        encoded_token_count=total,
        analysis_token_count=total,
        unique_analysis_token_count=len(counts),
        special_token_count=0,
        control_token_count=0,
        unknown_token_count=0,
        counts=counts,
        split_examples=Counter({"test": 1}),
        split_encoded_tokens=Counter({"test": total}),
        split_analysis_tokens=Counter({"test": total}),
    )


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.languages = LANGUAGES[:2]
        self.tokens = {
            self.languages[0].code: _language_tokens(self.languages[0].code, {10: 2, 11: 1}),
            self.languages[1].code: _language_tokens(self.languages[1].code, {11: 4, 12: 1}),
        }
        self.metrics = compute_metrics("test", self.languages, self.tokens, _Tokenizer())

    def test_type_metrics_are_symmetric(self):
        left, right = (language.code for language in self.languages)
        self.assertEqual(self.metrics.shared_count.loc[left, right], 1)
        self.assertAlmostEqual(self.metrics.type_iou.loc[left, right], 100 / 3)
        self.assertAlmostEqual(
            self.metrics.type_iou.loc[left, right], self.metrics.type_iou.loc[right, left]
        )

    def test_frequency_overlap_is_directional(self):
        left, right = (language.code for language in self.languages)
        self.assertAlmostEqual(self.metrics.frequency_overlap.loc[left, right], 100 / 3)
        self.assertAlmostEqual(self.metrics.frequency_overlap.loc[right, left], 80.0)

    def test_diagonals_and_upper_triangle(self):
        left, right = (language.code for language in self.languages)
        self.assertEqual(self.metrics.type_iou.loc[left, left], 100.0)
        self.assertEqual(self.metrics.frequency_overlap.loc[right, right], 100.0)
        self.assertTrue(self.metrics.type_iou_upper.loc[right, left] != self.metrics.type_iou_upper.loc[right, left])

    def test_single_language_condition_emits_diagnostics_without_pair(self):
        language = LANGUAGES[-1]
        tokens = {
            language.code: _language_tokens(language.code, {10: 2, 11: 1})
        }
        metrics = compute_metrics(
            "korean-only",
            (language,),
            tokens,
            _Tokenizer(),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = build_descriptive_outputs(metrics, output)
            persisted = json.loads((output / "summary.json").read_text())
            self.assertIsNone(summary["strongest_type_iou_pair"])
            self.assertIsNone(persisted["largest_frequency_asymmetry"])
            self.assertTrue((output / "REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
