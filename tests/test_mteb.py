import unittest

from xlmr_token_overlap.constants import LANGUAGE_CODES, MTEB_FAMILIES
from xlmr_token_overlap.mteb_data import (
    EXPECTED_FAMILY_LANGUAGES,
    Candidate,
    build_balanced_records,
    select_token_budget,
)


def _candidate(family, code, index, tokens=100):
    return Candidate(
        family=family,
        language_code=code,
        text=f"{family}-{code}-{index}",
        task=f"{family}-task",
        subset="test",
        split="test",
        role="text",
        example_id=f"{family}:{code}:{index}",
        analysis_tokens=tokens,
    )


class MtebSamplingTests(unittest.TestCase):
    def test_expected_coverage_matches_protocol(self):
        for family in ("classification", "clustering", "retrieval"):
            self.assertEqual(EXPECTED_FAMILY_LANGUAGES[family], frozenset(LANGUAGE_CODES))
        self.assertEqual(len(EXPECTED_FAMILY_LANGUAGES["sts"]), 16)
        self.assertEqual(len(EXPECTED_FAMILY_LANGUAGES["reranking"]), 12)

    def test_selection_is_deterministic_and_never_splits_records(self):
        candidates = [_candidate("classification", "eng_Latn", i, 60) for i in range(5)]
        first = select_token_budget(candidates, 180, seed=7, namespace="test")
        second = select_token_budget(reversed(candidates), 180, seed=7, namespace="test")
        self.assertEqual([row.example_id for row in first], [row.example_id for row in second])
        self.assertEqual(sum(row.analysis_tokens for row in first), 180)

    def test_balanced_conditions_and_overall_have_expected_languages(self):
        candidates = []
        for family in MTEB_FAMILIES:
            for code in EXPECTED_FAMILY_LANGUAGES[family]:
                candidates.extend(_candidate(family, code, index) for index in range(3))
        records, audit, selected = build_balanced_records(
            candidates,
            requested_family_tokens=100,
            requested_overall_tokens=200,
            seed=11,
        )
        by_condition = {}
        for record in records:
            by_condition.setdefault(record.condition, set()).add(record.language_code)
        self.assertEqual(by_condition["overall"], set(LANGUAGE_CODES))
        for family in MTEB_FAMILIES:
            self.assertEqual(by_condition[family], set(EXPECTED_FAMILY_LANGUAGES[family]))
            self.assertEqual(audit["effective_family_token_budgets"][family], 100)
        self.assertEqual(audit["effective_overall_token_budget"], 200)
        self.assertTrue(selected)


if __name__ == "__main__":
    unittest.main()
