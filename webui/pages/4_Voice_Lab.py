"""Voice Lab: hear the same QAI-VO paragraph from every voice setup, then choose."""

import os
import sys
from pathlib import Path

import streamlit as st

root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.config import config  # noqa: E402
from app.services.presenter import runpod_agent  # noqa: E402
from app.services.speech import pronunciation, voice_lab  # noqa: E402

st.set_page_config(page_title="Voice Lab", page_icon="🎙️", layout="wide")

TEXT = {
    "title": ("Voice Lab", "مختبر الأصوات"),
    "intro": ("The same short QAI-VO paragraph (with a number, AI and DOI) from each voice setup. "
              "Nothing is chosen for you: listen and decide.",
              "نفس الفقرة القصيرة عن QAI-VO (فيها رقم وAI وDOI) بكل طريقة صوت. ما في شي بينختار لحاله: "
              "اسمعي وقرّري."),
    "engines": ("Voice setups", "طرق الصوت"),
    "languages": ("Languages", "اللغات"),
    "run": ("Make the samples", "اعملي العيّنات"),
    "running": ("Working… (SILMA's first run downloads its model: several minutes)",
                "عم يشتغل… (أول مرة SILMA بينزّل الموديل: كم دقيقة)"),
    "need_runpod": ("SILMA runs on RunPod: set RunPod up on the Presenter Video page.",
                    "SILMA بيشتغل على RunPod: جهّزي RunPod بصفحة Presenter Video."),
    "results": ("Results", "النتائج"),
    "seconds": ("generation", "وقت التوليد"),
    "spoken": ("What the voice read", "شو قرأ الصوت"),
    "tashkeel": ("Use LLM tashkeel in my videos (Edge voice)", "استعملي التشكيل بالذكاء الاصطناعي بفيديوهاتي (صوت Edge)"),
    "saved": ("Saved", "انحفظ"),
    "sample": ("Sample text", "النص"),
    "brand_title": ("How should the voice say QAI-VO?", "كيف لازم يلفظ الصوت QAI-VO؟"),
    "brand_help": ("The same sentence with QAI-VO written in different ways for the voice (the screen always "
                   "shows QAI-VO). Listen and keep the one that sounds like the official pronunciation.",
                   "نفس الجملة، وQAI-VO مكتوبة للصوت بكذا طريقة (على الشاشة دايماً QAI-VO). "
                   "اسمعي وخلّي اللي بيطلع متل اللفظ الرسمي."),
    "brand_make": ("Make the versions", "اعملي النسخ"),
    "brand_use": ("Use this one", "استعملي هاي"),
    "brand_current": ("In use", "المستعملة هلأ"),
}
LABELS = {
    "edge": ("Edge (as today)", "Edge (متل هلأ)"),
    "edge_fixed": ("Edge + pronunciation fixes", "Edge + تصحيح اللفظ"),
    "edge_tashkeel": ("Edge + fixes + tashkeel", "Edge + تصحيح + تشكيل"),
    "silma": ("SILMA (its own voice)", "SILMA (صوته)"),
    "silma_lina": ("SILMA with Lina's voice", "SILMA بصوت لينا"),
}

arabic = st.radio("Language / اللغة", ["العربية", "English"], horizontal=True, key="lab_ui_lang") == "العربية"


def t(key: str) -> str:
    english, arabic_text = TEXT[key]
    return arabic_text if arabic else english


def label(engine: str) -> str:
    english, arabic_text = LABELS[engine]
    return arabic_text if arabic else english


st.title(t("title"))
st.caption(t("intro"))
with st.expander(t("sample")):
    st.write(voice_lab.SAMPLES["ar"])
    st.write(voice_lab.SAMPLES["en"])

runpod_ready = bool(runpod_agent.configured_key() and runpod_agent.configured_endpoint())
engines = st.multiselect(t("engines"), list(voice_lab.ENGINES), default=list(voice_lab.ENGINES),
                         format_func=label, key="lab_engines")
languages = st.multiselect(t("languages"), ["ar", "en"], default=["ar", "en"], key="lab_languages")
if any(e.startswith("silma") for e in engines) and not runpod_ready:
    st.info(t("need_runpod"))
    engines = [e for e in engines if not e.startswith("silma")]
latest = voice_lab.runs()[:1]
busy = bool(latest) and voice_lab.read(latest[0])[0].get("state") == "running"
if st.button(t("run"), type="primary", disabled=busy or not engines or not languages):
    voice_lab.start(engines, languages)
    st.rerun()

tashkeel = st.checkbox(t("tashkeel"), value=bool(config.app.get("pronunciation_tashkeel", False)))
if tashkeel != bool(config.app.get("pronunciation_tashkeel", False)):
    config.app["pronunciation_tashkeel"] = tashkeel
    config.save_config()
    st.success(t("saved"))


st.subheader(t("brand_title"))
st.caption(t("brand_help"))
for code, name in (("ar", "العربية"), ("en", "English")):
    with st.expander(f"QAI-VO — {name}", expanded=code == "ar"):
        if st.button(t("brand_make"), key=f"brand_make_{code}"):
            with st.spinner("…"):
                voice_lab.brand_test(code)
        current = pronunciation.brand_spoken(code)
        for index, item in enumerate(voice_lab.saved_brand_test(code)):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{index + 1}.** `{item['variant']}`" + (f"  ✓ {t('brand_current')}"
                                                                    if item["variant"] == current else ""))
            c1.audio(item["file"])
            if c2.button(t("brand_use"), key=f"brand_use_{code}_{index}", disabled=item["variant"] == current):
                voice_lab.choose_brand(code, item["variant"])
                st.success(t("saved"))
                st.rerun()


@st.fragment(run_every="10s")
def results():
    folders = voice_lab.runs()
    if not folders:
        return
    status, report = voice_lab.read(folders[0])
    st.subheader(f"{t('results')} — {os.path.basename(folders[0])}")
    if status.get("state") == "running":
        st.info(t("running") + (f"  ({status.get('last')})" if status.get("last") else ""))
    if status.get("error"):
        st.error(status["error"])
    for entry in report.get("results", []):
        with st.container(border=True):
            facts = [f"{t('seconds')}: {entry['generation_seconds']}s"]
            for key, unit in (("duration", "s"), ("loudness_lufs", " LUFS"), ("true_peak_dbfs", " dBFS")):
                if entry.get(key) is not None:
                    facts.append(f"{key}: {entry[key]}{unit}")
            if entry.get("gpu"):
                facts.append(f"GPU: {entry['gpu']}")
            st.markdown(f"**{label(entry['engine'])} — {entry['language']}**  ·  " + "  ·  ".join(facts))
            if entry.get("error"):
                st.error(entry["error"])
            elif entry.get("file"):
                st.audio(os.path.join(folders[0], entry["file"]))
                st.caption(f"{t('spoken')}: {entry['spoken_text']}")


results()
