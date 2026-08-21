import tempfile
import unittest
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from xlmr_token_overlap.io import Record
from xlmr_token_overlap.tokenization import AuditedTokenizer


class TokenizationTests(unittest.TestCase):
    def test_special_and_unknown_ids_are_recorded_but_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer = Tokenizer(
                WordLevel(
                    {"<unk>": 0, "hello": 1, "world": 2, "<pad>": 3},
                    unk_token="<unk>",
                )
            )
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.add_special_tokens(
                [AddedToken("<unk>", special=True), AddedToken("<pad>", special=True)]
            )
            tokenizer.save(str(path))

            audited = AuditedTokenizer(path)
            counts = audited.count(
                "eng_Latn",
                [Record("test", "eng_Latn", "hello missing world", "1", "test")],
            )
            self.assertEqual(counts.encoded_token_count, 3)
            self.assertEqual(counts.analysis_token_count, 2)
            self.assertEqual(counts.unknown_token_count, 1)
            self.assertEqual(counts.special_token_count, 1)
            self.assertEqual(counts.counts, {1: 1, 2: 1})

    def test_long_text_is_encoded_completely_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer = Tokenizer(
                WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>")
            )
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.add_special_tokens([AddedToken("<unk>", special=True)])
            tokenizer.save(str(path))

            text = " ".join(["hello"] * 5_000)
            audited = AuditedTokenizer(path)
            counts = audited.count(
                "eng_Latn",
                [Record("test", "eng_Latn", text, "long", "test")],
            )
            self.assertEqual(counts.encoded_token_count, 5_000)
            self.assertEqual(counts.analysis_token_count, 5_000)
            self.assertEqual(counts.counts, {1: 5_000})


if __name__ == "__main__":
    unittest.main()
