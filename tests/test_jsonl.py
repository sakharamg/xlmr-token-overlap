import json
import tempfile
import unittest
from pathlib import Path

from xlmr_token_overlap.io import load_jsonl


class JsonlTests(unittest.TestCase):
    def test_conditions_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [
                {"condition": "mteb/sts", "language_code": "eng_Latn", "text": "a"},
                {"condition": "mteb/retrieval", "language_code": "deu_Latn", "text": "b"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            records, provenance = load_jsonl(path)
            self.assertEqual([record.condition for record in records], ["mteb/sts", "mteb/retrieval"])
            self.assertEqual(provenance["row_count"], 2)


if __name__ == "__main__":
    unittest.main()

