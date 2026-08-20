import unittest

from xlmr_token_overlap.constants import LANGUAGES, LANGUAGE_CODES


class LanguageInventoryTests(unittest.TestCase):
    def test_frozen_inventory_has_24_unique_languages(self):
        self.assertEqual(len(LANGUAGES), 24)
        self.assertEqual(len(set(LANGUAGE_CODES)), 24)

    def test_visual_order_exposes_requested_groups(self):
        groups = [language.visual_group for language in LANGUAGES]
        self.assertEqual(groups[:4], ["Germanic Latin"] * 4)
        self.assertEqual(groups[4:9], ["Romance Latin"] * 5)
        self.assertEqual(groups[9:12], ["Austronesian Latin"] * 3)
        self.assertEqual(groups[12:16], ["Other Latin"] * 4)
        self.assertEqual(groups[16:], ["Other scripts"] * 8)


if __name__ == "__main__":
    unittest.main()

