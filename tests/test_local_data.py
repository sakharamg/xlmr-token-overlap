import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import Workbook

from xlmr_token_overlap.constants import LANGUAGE_CODES
from xlmr_token_overlap.local_data import (
    LABEL_TO_CODE,
    collect_pass3_records,
    collect_sqe_records,
    normalize_sqe_locale,
    parse_gt_ids,
)
from xlmr_token_overlap.io import Record
from xlmr_token_overlap.tokenization import AuditedTokenizer


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_xlsx(path: Path, sheet: str, headers, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    worksheet.append(list(headers))
    for row in rows:
        worksheet.append(list(row))
    workbook.save(path)


class Pass3LocalLoaderTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        for label in LABEL_TO_CODE:
            _touch(root / "sts" / label / "canonical" / "data.parquet")
            canonical = root / "belebele_retrieval" / label / "canonical"
            for filename in ("queries.parquet", "corpus.parquet", "qrels.parquet"):
                _touch(canonical / filename)

    @staticmethod
    def _read_parquet(path):
        path = Path(path)
        if "sts" in path.parts:
            return pd.DataFrame(
                {
                    "sentence1": ["s" * 10_000],
                    "sentence2": ["second sentence"],
                    "score": [4.5],
                }
            )
        if path.name == "queries.parquet":
            return pd.DataFrame({"id": ["q1"], "text": ["query text"]})
        if path.name == "corpus.parquet":
            return pd.DataFrame({"id": ["d1"], "text": ["document text"]})
        if path.name == "qrels.parquet":
            return pd.DataFrame(
                {"query-id": ["q1"], "corpus-id": ["d1"], "score": [1]}
            )
        raise AssertionError(path)

    def test_every_row_and_complete_text_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with patch("pandas.read_parquet", side_effect=self._read_parquet):
                collection, audit = collect_pass3_records(root)

            counts = Counter(
                (record.condition, record.language_code)
                for record in collection.records
            )
            for code in LANGUAGE_CODES:
                self.assertEqual(counts[("sts", code)], 2)
                self.assertEqual(counts[("retrieval", code)], 2)
                self.assertEqual(counts[("overall", code)], 4)
            long_rows = [
                record
                for record in collection.records
                if record.condition == "sts" and len(record.text) == 10_000
            ]
            self.assertEqual(len(long_rows), 24)
            self.assertEqual(audit["sampling"], "none")
            self.assertIsNone(audit["character_limit"])
            self.assertFalse(audit["tokenizer_truncation"])
            self.assertEqual(len(audit["source_files"]), 24 * 4)

    def test_orphan_qrels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)

            def broken(path):
                frame = self._read_parquet(path)
                if Path(path).name == "qrels.parquet":
                    frame["corpus-id"] = "missing"
                return frame

            with patch("pandas.read_parquet", side_effect=broken):
                with self.assertRaisesRegex(ValueError, "integrity failure"):
                    collect_pass3_records(root)


class SqeLocalLoaderTests(unittest.TestCase):
    def _locale(
        self,
        root: Path,
        label: str,
        gt_ids='["d1"]',
        data_rows=None,
    ) -> None:
        canonical = (
            root
            / "sqe"
            / "sqe_settings_search"
            / label
            / "canonical"
        )
        _write_xlsx(
            canonical / "data.xlsx",
            "data",
            ("id", "text", "remark"),
            data_rows or (("d1", "D" * 10_000, "metadata"),),
        )
        _write_xlsx(
            canonical / "tc.xlsx",
            "tc",
            ("query", "gt_ids"),
            (("Q" * 12_000, gt_ids),),
        )

    def test_sqe_uses_full_data_and_query_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._locale(root, "EN")
            self._locale(root, "KO")
            collection, audit, coverage = collect_sqe_records(
                root,
                domains=("settings",),
                allow_coverage_drift=True,
            )
            counts = Counter(record.language_code for record in collection.records)
            self.assertEqual(counts["eng_Latn"], 2)
            self.assertEqual(counts["kor_Hang"], 2)
            self.assertEqual(
                sorted(len(record.text) for record in collection.records),
                [10_000, 10_000, 12_000, 12_000],
            )
            self.assertEqual(
                coverage["settings_standard"],
                {"eng_Latn", "kor_Hang"},
            )
            self.assertEqual(audit["sampling"], "none")
            self.assertIsNone(audit["character_limit"])
            self.assertFalse(audit["tokenizer_truncation"])

    def test_empty_ground_truth_warns_without_dropping_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._locale(root, "KO", gt_ids=None)
            collection, audit, _ = collect_sqe_records(
                root,
                domains=("settings",),
                allow_coverage_drift=True,
            )
            queries = [
                record for record in collection.records
                if record.split.endswith("|query")
            ]
            integrity = audit["ground_truth_integrity"][
                "settings_standard:kor_Hang"
            ]
            self.assertEqual(len(queries), 1)
            self.assertEqual(integrity["status"], "warning")
            self.assertEqual(integrity["empty_gt_rows"], 1)
            self.assertEqual(
                audit["ground_truth_summary"]["conditions_with_warnings"],
                1,
            )

    def test_strict_ground_truth_rejects_missing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._locale(root, "KO", gt_ids='["missing"]')
            with self.assertRaisesRegex(ValueError, "ground-truth integrity"):
                collect_sqe_records(
                    root,
                    domains=("settings",),
                    allow_coverage_drift=True,
                    strict_ground_truth=True,
                )

    def test_missing_reference_warns_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._locale(root, "KO", gt_ids='["missing"]')
            collection, audit, _ = collect_sqe_records(
                root,
                domains=("settings",),
                allow_coverage_drift=True,
            )
            integrity = audit["ground_truth_integrity"][
                "settings_standard:kor_Hang"
            ]
            self.assertEqual(len(collection.records), 2)
            self.assertEqual(integrity["status"], "warning")
            self.assertEqual(
                integrity["missing_gt_reference_occurrences"],
                1,
            )

    def test_duplicate_and_empty_data_ids_do_not_drop_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._locale(
                root,
                "KO",
                data_rows=(
                    (None, "empty ID text", "metadata"),
                    ("d1", "first duplicate-ID text", "metadata"),
                    ("d1", "second duplicate-ID text", "metadata"),
                ),
            )
            collection, audit, _ = collect_sqe_records(
                root,
                domains=("settings",),
                allow_coverage_drift=True,
            )
            data_records = [
                record
                for record in collection.records
                if record.split.endswith("|data")
            ]
            integrity = audit["data_id_integrity"][
                "settings:kor_Hang"
            ]
            self.assertEqual(len(data_records), 3)
            self.assertEqual(len({record.example_id for record in data_records}), 3)
            self.assertEqual(integrity["status"], "warning")
            self.assertEqual(integrity["empty_data_id_rows"], 1)
            self.assertEqual(integrity["duplicate_data_id_values"], 1)
            self.assertEqual(integrity["duplicate_data_id_extra_rows"], 1)

    def test_strict_integrity_rejects_duplicate_data_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._locale(
                root,
                "KO",
                data_rows=(
                    ("d1", "first text", "metadata"),
                    ("d1", "second text", "metadata"),
                ),
            )
            with self.assertRaisesRegex(ValueError, "data-ID integrity"):
                collect_sqe_records(
                    root,
                    domains=("settings",),
                    allow_coverage_drift=True,
                    strict_ground_truth=True,
                )

    def test_gt_ids_and_locale_formats(self):
        self.assertEqual(parse_gt_ids('["d1", "d2"]'), ["d1", "d2"])
        self.assertEqual(parse_gt_ids("d1,d2"), ["d1", "d2"])
        self.assertEqual(normalize_sqe_locale("korean-republic_of_korea"), "kor_Hang")
        self.assertEqual(normalize_sqe_locale("ZH"), "zho_Hans")


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must stay disabled")
        return [
            _FakeEncoding([1] * len(text.split()))
            for text in texts
        ]


class FullTextTokenizationTests(unittest.TestCase):
    def test_memory_batching_does_not_clip_or_split_complete_texts(self):
        audited = AuditedTokenizer.__new__(AuditedTokenizer)
        audited.tokenizer = _FakeTokenizer()
        audited.special_ids = set()
        audited.unknown_id = None
        long_text = " ".join(["token"] * 7_500)
        records = [
            Record("full", "eng_Latn", "token", str(index), "test")
            for index in range(300)
        ]
        records.append(Record("full", "eng_Latn", long_text, "long", "test"))
        counts = audited.count("eng_Latn", records)
        self.assertEqual(counts.examples, 301)
        self.assertEqual(counts.analysis_token_count, 7_800)
        self.assertEqual(counts.encoded_token_count, 7_800)


if __name__ == "__main__":
    unittest.main()
