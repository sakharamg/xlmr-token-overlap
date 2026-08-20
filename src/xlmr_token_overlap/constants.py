"""Frozen language inventory and source revisions for the study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Language:
    code: str
    name: str
    label: str
    script: str
    family: str
    visual_group: str


# The order is intentionally linguistic, not alphabetical.  It exposes the
# block structure requested in the study protocol.
LANGUAGES: tuple[Language, ...] = (
    Language("eng_Latn", "English", "EN", "Latin", "Germanic", "Germanic Latin"),
    Language("deu_Latn", "German", "DE", "Latin", "Germanic", "Germanic Latin"),
    Language("nld_Latn", "Dutch", "NL", "Latin", "Germanic", "Germanic Latin"),
    Language("swe_Latn", "Swedish", "SV", "Latin", "Germanic", "Germanic Latin"),
    Language("spa_Latn", "Spanish", "ES", "Latin", "Romance", "Romance Latin"),
    Language("fra_Latn", "French", "FR", "Latin", "Romance", "Romance Latin"),
    Language("ita_Latn", "Italian", "IT", "Latin", "Romance", "Romance Latin"),
    Language("por_Latn", "Portuguese", "PT", "Latin", "Romance", "Romance Latin"),
    Language("ron_Latn", "Romanian", "RO", "Latin", "Romance", "Romance Latin"),
    Language("ind_Latn", "Indonesian", "ID", "Latin", "Austronesian", "Austronesian Latin"),
    Language("zsm_Latn", "Malay", "MS", "Latin", "Austronesian", "Austronesian Latin"),
    Language("tgl_Latn", "Filipino/Tagalog", "TL", "Latin", "Austronesian", "Austronesian Latin"),
    Language("pol_Latn", "Polish", "PL", "Latin", "Slavic", "Other Latin"),
    Language("hun_Latn", "Hungarian", "HU", "Latin", "Uralic", "Other Latin"),
    Language("tur_Latn", "Turkish", "TR", "Latin", "Turkic", "Other Latin"),
    Language("vie_Latn", "Vietnamese", "VI", "Latin", "Austroasiatic", "Other Latin"),
    Language("arb_Arab", "Arabic", "AR", "Arabic", "Semitic", "Other scripts"),
    Language("hin_Deva", "Hindi", "HI", "Devanagari", "Indo-Aryan", "Other scripts"),
    Language("guj_Gujr", "Gujarati", "GU", "Gujarati", "Indo-Aryan", "Other scripts"),
    Language("rus_Cyrl", "Russian", "RU", "Cyrillic", "Slavic", "Other scripts"),
    Language("tha_Thai", "Thai", "TH", "Thai", "Kra-Dai", "Other scripts"),
    Language("zho_Hans", "Simplified Chinese", "ZH", "Han", "Sinitic", "Other scripts"),
    Language("jpn_Jpan", "Japanese", "JA", "Japanese", "Japonic", "Other scripts"),
    Language("kor_Hang", "Korean", "KO", "Hangul", "Koreanic", "Other scripts"),
)

LANGUAGE_BY_CODE = {language.code: language for language in LANGUAGES}
LANGUAGE_CODES = tuple(language.code for language in LANGUAGES)

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES_SPLIT_SIZES = {"dev": 997, "devtest": 1012}

XLMR_REPOSITORY = "FacebookAI/xlm-roberta-base"
XLMR_REVISION = "42f548f32366559214515ec137cdd16002968bf6"
XLMR_TOKENIZER_URL = (
    f"https://huggingface.co/{XLMR_REPOSITORY}/resolve/{XLMR_REVISION}/tokenizer.json"
)
XLMR_TOKENIZER_SHA256 = "a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c"

