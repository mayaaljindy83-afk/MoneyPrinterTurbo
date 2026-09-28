"""Voice Lab: the same QAI-VO paragraph through every available voice setup, side by side.

Nothing is chosen automatically: each WAV comes with its engine, settings, generation time,
spoken text and audio facts in ``benchmark.json`` so the user can listen and pick.

Engines:
* ``edge``        Edge TTS as it is today (the text as written);
* ``edge_fixed``  Edge TTS reading the PronunciationProcessor's spoken text;
* ``edge_tashkeel`` the same plus LLM tashkeel (Arabic only);
* ``silma``       SILMA TTS on the RunPod endpoint (own tashkeel + number reading), SILMA's reference voice;
* ``silma_lina``  SILMA copying the presenter's Edge voice from a short reference clip.

Files live in ``<data>/voice_lab/<time>/`` (E:, never C:).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time

from app.services import voice
from app.services.speech import narration
from app.services.speech.pronunciation import PronunciationProcessor, llm_diacritizer
from app.utils import utils

SAMPLES = {
    "ar": ("مع QAI-VO، ينجز الباحث مراجعته الأكاديمية خلال 48 ساعة فقط. نستخدم تقنيات AI للتحقق من كل "
           "مرجع ورابط DOI، لتحصل على نتائج دقيقة بنسبة 99%. ابدأ اليوم وانضم إلى آلاف الباحثين."),
    "en": ("With QAI-VO, researchers finish their academic review in just 48 hours. We use AI to check every "
           "reference and DOI link, with 99% accuracy. Start today and join thousands of researchers."),
}
VOICES = {"ar": "ar-SA-ZariyahNeural-Female", "en": "en-US-JennyNeural-Female"}
# Read by the presenter's Edge voice, then given to SILMA as the voice to copy (text must match the audio).
REFERENCE_TEXT = {"ar": "أهلاً بكم، أنا لينا، وسأرافقكم اليوم في جولة قصيرة داخل منصتنا.",
                  "en": "Hello, I am Lina, and today I will take you on a short tour of our platform."}
ENGINES = ("edge", "edge_fixed", "edge_tashkeel", "silma", "silma_lina")
SAMPLE_RATE = 24000


def lab_dir() -> str:
    folder = os.path.join(utils.storage_dir("voice_lab"), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    return folder


def _to_wav(src: str, dst: str) -> None:
    subprocess.run([utils.get_ffmpeg_binary(), "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar",
                    str(SAMPLE_RATE), dst], check=True)


def audio_facts(path: str) -> dict:
    """Duration, sample rate and loudness (EBU R128 integrated LUFS, true peak) of a WAV."""
    result = subprocess.run([utils.get_ffmpeg_binary(), "-hide_banner", "-i", path, "-af", "ebur128=peak=true",
                             "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = result.stderr
    facts = {"duration": round(voice.get_audio_duration(path) or 0.0, 2), "sample_rate": SAMPLE_RATE}
    import re

    summary = text[text.rfind("Summary:"):] if "Summary:" in text else ""
    lufs = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    if lufs:
        facts["loudness_lufs"] = float(lufs.group(1))
    if peak:
        facts["true_peak_dbfs"] = float(peak.group(1))
    return facts


def edge(text: str, language: str, out_wav: str, tts=None) -> None:
    tts = tts or voice.tts
    mp3 = out_wav[:-4] + ".mp3"
    if tts(text=text, voice_name=VOICES[language], voice_rate=1.0, voice_file=mp3) is None or not os.path.isfile(mp3):
        raise RuntimeError("Edge TTS failed (internet connection?)")
    _to_wav(mp3, out_wav)
    os.remove(mp3)


def silma(items: list[dict], agent=None) -> dict:
    """SILMA on the RunPod endpoint. ``items``: [{"id", "text", "ref_wav" (path), "ref_text"}]."""
    from app.services.presenter import runpod_agent

    agent = agent or runpod_agent.RunPodAgent()
    payload = []
    for item in items:
        entry = {"id": item["id"], "text": item["text"], "ref_text": item.get("ref_text"), "seed": 2026}
        if item.get("ref_wav"):
            with open(item["ref_wav"], "rb") as fp:
                entry["ref_wav"] = base64.b64encode(fp.read()).decode("ascii")
        payload.append(entry)
    return agent.call({"mode": "tts", "engine": "silma", "items": payload}, timeout=2400)


def spoken_text(engine: str, language: str) -> str:
    display = SAMPLES[language]
    if engine == "edge":
        return display
    if engine == "edge_tashkeel" and language == "ar":
        processor = PronunciationProcessor(language, numbers="keep", diacritizer=llm_diacritizer)
    else:
        # SILMA reads digits itself (with its own Arabic grammar), like Edge.
        processor = PronunciationProcessor(language, numbers="keep")
    return narration.prepare(display, processor)["spoken"]


def run(engines: list[str], languages: list[str], folder: str | None = None, tts=None, agent=None,
        progress=None) -> dict:
    """Make every requested engine x language sample; returns (and saves) the benchmark."""
    folder = folder or lab_dir()
    report = {"folder": folder, "samples": SAMPLES, "results": []}

    def record(engine, language, path, seconds, spoken, extra=None, error=""):
        entry = {"engine": engine, "language": language, "file": os.path.basename(path) if path else "",
                 "generation_seconds": round(seconds, 2), "spoken_text": spoken, "error": error}
        if path and os.path.isfile(path):
            entry.update(audio_facts(path))
        entry.update(extra or {})
        report["results"].append(entry)
        if progress:
            progress(entry)

    silma_items, silma_meta = [], {}
    for language in languages:
        for engine in engines:
            if engine == "edge_tashkeel" and language != "ar":
                continue
            path = os.path.join(folder, f"{engine}_{language}.wav")
            started = time.time()
            try:
                spoken = spoken_text(engine, language)
            except Exception as exc:  # e.g. the LLM for tashkeel is offline
                record(engine, language, "", time.time() - started, "", error=str(exc))
                continue
            if engine.startswith("edge"):
                try:
                    edge(spoken, language, path, tts)
                    record(engine, language, path, time.time() - started, spoken)
                except Exception as exc:
                    record(engine, language, "", time.time() - started, spoken, error=str(exc))
                continue
            item = {"id": f"{engine}_{language}", "text": spoken}
            if engine == "silma_lina":
                ref = os.path.join(folder, f"reference_{language}.wav")
                if not os.path.isfile(ref):
                    edge(REFERENCE_TEXT[language], language, ref, tts)
                item.update({"ref_wav": ref, "ref_text": REFERENCE_TEXT[language]})
            silma_items.append(item)
            silma_meta[item["id"]] = (engine, language, path, spoken)
    if silma_items:
        started = time.time()
        try:
            answer = silma(silma_items, agent)
        except Exception as exc:
            for engine, language, path, spoken in silma_meta.values():
                record(engine, language, "", time.time() - started, spoken, error=str(exc))
        else:
            elapsed = time.time() - started
            for item in answer.get("items", []):
                engine, language, path, spoken = silma_meta[item["id"]]
                with open(path, "wb") as fp:
                    fp.write(base64.b64decode(item["wav"]))
                record(engine, language, path, item.get("inference_seconds", 0.0), spoken,
                       {"round_trip_seconds": round(elapsed, 2), "model_load_seconds": answer.get("model_load_seconds"),
                        "gpu": answer.get("gpu", "")})
    with open(os.path.join(folder, "benchmark.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    return report


# --------------------------------------------------------------------------- background runs (for the page)
def start(engines: list[str], languages: list[str], tts=None, agent=None) -> str:
    """Run in the background (SILMA's first call downloads its model); returns the result folder.
    ``status.json`` there says running / done / error."""
    import threading

    folder = lab_dir()

    def status(state, **extra):
        with open(os.path.join(folder, "status.json"), "w", encoding="utf-8") as fp:
            json.dump({"state": state, "engines": engines, "languages": languages, **extra}, fp, ensure_ascii=False)

    def work():
        try:
            run(engines, languages, folder, tts=tts, agent=agent,
                progress=lambda entry: status("running", last=f"{entry['engine']} {entry['language']}"))
            status("done")
        except Exception as exc:
            status("error", error=str(exc))

    status("running")
    threading.Thread(target=work, name="voice-lab", daemon=True).start()
    return folder


def runs() -> list[str]:
    root = utils.storage_dir("voice_lab")
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in sorted(os.listdir(root), reverse=True)
            if os.path.isfile(os.path.join(root, d, "status.json"))]


def read(folder: str) -> tuple[dict, dict]:
    """(status, benchmark) of one run."""
    def load(name):
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            return {}

    return load("status.json"), load("benchmark.json")
