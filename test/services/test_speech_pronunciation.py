import json
import os
import tempfile
import unittest
from unittest import mock

from app.services.speech import pronunciation as pr
from app.services.speech.pronunciation import PronunciationProcessor


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"MPT_STORAGE_DIR": self.tmp})
        self.env.start()

    def tearDown(self):
        self.env.stop()


class TestNumbers(Case):
    def test_arabic_words(self):
        cases = {0: "صفر", 11: "أحد عشر", 12: "اثنا عشر", 21: "واحد وعشرون", 100: "مئة", 250: "مئتان وخمسون",
                 2026: "ألفان وستة وعشرون", 3000: "ثلاثة آلاف", 15000: "خمسة عشر ألفًا",
                 1250000: "مليون ومئتان وخمسون ألفًا"}
        for number, words in cases.items():
            self.assertEqual(pr.arabic_number(number), words)

    def test_english_words_and_years(self):
        self.assertEqual(pr.english_number(48), "forty-eight")
        self.assertEqual(pr.english_number(1250000), "one million two hundred fifty thousand")
        self.assertEqual(pr.english_year(2026), "twenty twenty-six")
        self.assertEqual(pr.english_year(2005), "two thousand five")

    def test_percent_decimal_and_arabic_digits(self):
        spoken = PronunciationProcessor("ar").process("دقة ٩٩٫٥٪ خلال 48 ساعة").spoken
        self.assertEqual(spoken, "دقة تسعة وتسعون فاصلة خمسة بالمئة خلال ثمانية وأربعون ساعة.")
        english = PronunciationProcessor("en").process("99.5% in 1,200 papers since 2026").spoken
        self.assertEqual(english, "ninety-nine point five percent in one thousand two hundred papers since twenty twenty-six.")

    def test_digits_can_be_left_to_the_engine(self):
        self.assertIn("48", PronunciationProcessor("ar", numbers="keep").process("خلال 48 ساعة").spoken)


class TestDictionary(Case):
    def test_brand_and_acronyms_display_untouched(self):
        display = "مع QAI-VO تتحقق من روابط DOI بالـ AI."
        result = PronunciationProcessor("ar").process(display)
        self.assertEqual(result.display, display)  # the shown text never changes
        self.assertEqual(result.spoken, "مع كيو إيه آي فو تتحقق من روابط دي أو آي بالـ إيه آي.")
        self.assertEqual([r["term"] for r in result.replacements], ["QAI-VO", "DOI", "AI"])
        english = PronunciationProcessor("en").process("QAI-VO finds DOI links with AI")
        self.assertEqual(english.spoken, "Q A I Vo finds D O I links with A I.")

    def test_ai_inside_words_is_not_replaced(self):
        self.assertEqual(PronunciationProcessor("en").process("Said the email").spoken, "Said the email.")

    def test_project_overrides_and_user_file(self):
        with open(pr.user_dictionary_path(), "w", encoding="utf-8") as fp:
            json.dump({"Academic Suite": {"ar": "الحقيبة الأكاديمية"}}, fp, ensure_ascii=False)
        processor = PronunciationProcessor("ar", overrides={"QAI-VO": "كاي فو"})
        spoken = processor.process("QAI-VO Academic Suite").spoken
        self.assertEqual(spoken, "كاي فو الحقيبة الأكاديمية.")

    def test_unknown_latin_terms_are_flagged(self):
        result = PronunciationProcessor("ar").process("جرّب ChatGPT وScopus.")
        self.assertEqual(result.unknown_terms, ["ChatGPT", "Scopus"])


class TestPausesAndTashkeel(Case):
    def test_pauses(self):
        self.assertEqual(PronunciationProcessor("ar").process("سريع — ودقيق | مضمون").spoken, "سريع، ودقيق، مضمون.")

    def test_tashkeel_accepted_only_if_letters_unchanged(self):
        good = PronunciationProcessor("ar", diacritizer=lambda t: t.replace("نتائج", "نَتَائِجَ")).process("نتائج ممتازة")
        self.assertTrue(good.diacritized)
        self.assertEqual(good.spoken, "نَتَائِجَ ممتازة.")
        changed = PronunciationProcessor("ar", diacritizer=lambda t: t.replace("نتائج", "نتيجة")).process("نتائج ممتازة")
        self.assertFalse(changed.diacritized)
        self.assertEqual(changed.spoken, "نتائج ممتازة.")

        def broken(text):
            raise RuntimeError("LLM offline")

        self.assertEqual(PronunciationProcessor("ar", diacritizer=broken).process("نتائج").spoken, "نتائج.")

    def test_llm_diacritizer_uses_configured_llm(self):
        with mock.patch("app.services.llm._generate_response", return_value="نَتَائِجُ مُمْتَازَةٌ") as ask:
            result = PronunciationProcessor("ar", diacritizer=pr.llm_diacritizer).process("نتائج ممتازة")
        self.assertTrue(result.diacritized)
        self.assertIn("Do not add, remove", ask.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
