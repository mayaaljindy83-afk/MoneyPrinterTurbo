import json
import os
import tempfile
import unittest
from unittest import mock

from app.config import config
from app.models.schema import VideoParams
from app.services import llm, long_script, task


class FakeLLM:
    """Scripted stand-in for llm._generate_response that records prompts."""

    def __init__(self, section_words=120, fail_first=0):
        self.prompts = []
        self.section_words = section_words
        self.fail_first = fail_first

    def __call__(self, prompt, app_config=None):
        self.prompts.append(prompt)
        if self.fail_first:
            self.fail_first -= 1
            return "Error: connection refused"
        if "JSON array of exactly" in prompt:
            count = int(prompt.split("exactly ")[1].split()[0])
            return "```json\n" + json.dumps([f"Title {i}" for i in range(count)]) + "\n```"
        if "stock-video search terms" in prompt:
            section = prompt.split("Narration:\n")[1].split()[0]
            return json.dumps([f"{section} city", f"{section} people", "مدينة"])
        index = prompt.split("Write section ")[1].split()[0]
        body = " ".join([f"s{index}w"] * self.section_words)
        return f"## Section {index}:\n**{body}.**"


class TestPlanning(unittest.TestCase):
    def test_words_per_minute_depends_on_language_and_rate(self):
        self.assertEqual(long_script.words_per_minute("en"), 150)
        self.assertEqual(long_script.words_per_minute("ar-SA"), 115)
        self.assertEqual(long_script.words_per_minute("", subject="الفضاء"), 115)
        self.assertEqual(long_script.words_per_minute("en", voice_rate=1.2), 180)

    def test_clamp_minutes(self):
        self.assertEqual(long_script.clamp_minutes(0), 0.0)
        self.assertEqual(long_script.clamp_minutes(None), 0.0)
        self.assertEqual(long_script.clamp_minutes(float("nan")), 0.0)
        self.assertEqual(long_script.clamp_minutes(0.1), 0.5)
        self.assertEqual(long_script.clamp_minutes(25), 10.0)

    def test_plan_sections_scales_with_duration(self):
        one = long_script.plan_sections(1, "en")
        ten = long_script.plan_sections(10, "en")
        self.assertEqual(one[0], 1)
        self.assertGreaterEqual(ten[0], 10)
        self.assertLessEqual(ten[0], long_script.MAX_SECTIONS)
        # Total words land close to the target narration length.
        self.assertAlmostEqual(ten[0] * ten[1], 1500, delta=60)


class TestCleaning(unittest.TestCase):
    def test_clean_section_removes_markdown_and_headings(self):
        text = "Section 2:\n**Bold** start\n- item\nNarrator: hello [music]\n# Heading"
        self.assertEqual(long_script.clean_section(text), "Bold start item hello")

    def test_clean_section_arabic_heading(self):
        text = "القسم الأول:\nالراوي: نص عربي جميل"
        self.assertEqual(long_script.clean_section(text), "نص عربي جميل")


class TestGenerateLongScript(unittest.TestCase):
    def test_sections_outline_and_ordered_terms(self):
        fake = FakeLLM()
        with mock.patch.object(llm, "_generate_response", fake):
            result = long_script.generate_long_script("Deep sea", "en", minutes=3)
        count, _ = long_script.plan_sections(3, "en")
        self.assertEqual(len(result.sections), count)
        self.assertEqual(result.script.count("\n\n"), count - 1)
        self.assertNotIn("#", result.script)
        self.assertNotIn("*", result.script)
        # Terms follow section order and never contain Arabic.
        self.assertEqual(result.terms[0], "s1w city")
        self.assertEqual(result.terms[-1], f"s{count}w people")
        self.assertTrue(all("مدينة" not in t for t in result.terms))
        # Every section after the first sees the end of the previous one.
        section_prompts = [p for p in fake.prompts if p.startswith("Write section")]
        self.assertNotIn("previous section ended", section_prompts[0])
        self.assertIn("previous section ended", section_prompts[1])
        self.assertIn("final section", section_prompts[-1])

    def test_arabic_prompt_asks_for_standard_arabic(self):
        fake = FakeLLM()
        with mock.patch.object(llm, "_generate_response", fake):
            long_script.generate_long_script("تاريخ دمشق", "", minutes=1, with_terms=False)
        self.assertIn("Modern Standard Arabic", fake.prompts[-1])

    def test_retries_transient_errors(self):
        fake = FakeLLM(fail_first=2)
        with mock.patch.object(llm, "_generate_response", fake):
            result = long_script.generate_long_script("Topic", "en", minutes=1, with_terms=False)
        self.assertTrue(result.script)

    def test_raises_when_llm_keeps_failing(self):
        with mock.patch.object(llm, "_generate_response", return_value="Error: down"):
            with self.assertRaises(RuntimeError):
                long_script.generate_long_script("Topic", "en", minutes=1)


class TestPipelineIntegration(unittest.TestCase):
    def test_generate_script_uses_long_mode(self):
        params = VideoParams(video_subject="Volcanoes", video_language="en", video_duration_minutes=2)
        with mock.patch.object(llm, "_generate_response", FakeLLM()), mock.patch.object(
            task.sm.state, "update_task"
        ):
            script = task.generate_script("task-id", params)
        self.assertGreater(len(script.split()), 150)
        self.assertTrue(params.match_materials_to_script)
        self.assertTrue(params.video_terms)

    def test_short_mode_is_unchanged(self):
        params = VideoParams(video_subject="Volcanoes")
        with mock.patch.object(llm, "generate_script", return_value="short") as generate:
            self.assertEqual(task.generate_script("task-id", params), "short")
        generate.assert_called_once()

    def test_order_videos_by_terms_groups_round_robin_downloads(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            task.utils, "task_dir", return_value=tmp
        ):
            files = [os.path.join(tmp, f"{name}.mp4") for name in ("a1", "b1", "a2", "b2")]
            with open(os.path.join(tmp, "script.json"), "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "material_sources": [
                            {"local_file": "a1.mp4", "search_term": "alpha"},
                            {"local_file": "b1.mp4", "search_term": "beta"},
                            {"local_file": "a2.mp4", "search_term": "alpha"},
                            {"local_file": "b2.mp4", "search_term": "beta"},
                        ]
                    },
                    fp,
                )
            ordered = task.order_videos_by_terms("t", files, ["alpha", "beta"])
        self.assertEqual([os.path.basename(f) for f in ordered], ["a1.mp4", "a2.mp4", "b1.mp4", "b2.mp4"])


class TestFallbackProvider(unittest.TestCase):
    def test_fallback_is_used_when_primary_fails(self):
        calls = []

        def once(prompt, app_config=None):
            provider = (app_config or config.app).get("llm_provider")
            calls.append(provider)
            return "Error: ollama is not running" if provider == "ollama" else "from gemini"

        cfg = {"llm_provider": "ollama", "llm_fallback_provider": "gemini"}
        with mock.patch.object(llm, "_generate_response_once", once):
            self.assertEqual(llm._generate_response("hi", app_config=cfg), "from gemini")
        self.assertEqual(calls, ["ollama", "gemini"])

    def test_no_fallback_configured_returns_primary_error(self):
        with mock.patch.object(llm, "_generate_response_once", return_value="Error: x") as once:
            response = llm._generate_response("hi", app_config={"llm_provider": "ollama"})
        self.assertEqual(response, "Error: x")
        once.assert_called_once()

    def test_success_does_not_touch_fallback(self):
        cfg = {"llm_provider": "ollama", "llm_fallback_provider": "gemini"}
        with mock.patch.object(llm, "_generate_response_once", return_value="ok") as once:
            self.assertEqual(llm._generate_response("hi", app_config=cfg), "ok")
        once.assert_called_once()


if __name__ == "__main__":
    unittest.main()
