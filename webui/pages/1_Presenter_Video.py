"""Presenter video: a consistent AI presenter explains a topic (rendered on Kaggle's free GPU)."""

import os
import sys
from pathlib import Path

import streamlit as st

root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.config import config  # noqa: E402
from app.services import branding, subtitle_styles, voice  # noqa: E402
from app.services.presenter import kaggle_agent, profiles, studio  # noqa: E402
from app.services.presenter import package as job_package  # noqa: E402
from app.utils import utils  # noqa: E402

st.set_page_config(page_title="Presenter video", page_icon="🎤", layout="wide")

TEXT = {
    "title": ("🎤 Presenter video", "🎤 فيديو مع مقدّمة"),
    "intro": ("One presenter who always looks the same talks, walks and points while explaining your topic. "
              "The heavy AI work runs on Kaggle's free GPU; this laptop plans, records the voice and assembles.",
              "مقدّمة وحدة بنفس الشكل دايماً، بتحكي وبتمشي وبتأشّر وهي عم تشرح موضوعك. الشغل التقيل بيصير "
              "على كرت Kaggle المجاني، واللابتوب بيخطّط وبيسجّل الصوت وبيجمّع الفيديو."),
    "kaggle": ("1. Kaggle connection", "1. الربط مع Kaggle"),
    "token": ("Kaggle API token", "مفتاح Kaggle API"),
    "token_help": ("kaggle.com → Settings → API → Generate New Token. Saved only in your local config.toml.",
                   "من kaggle.com ← Settings ← API ← Generate New Token. بينحفظ بس بملف config.toml عندك."),
    "save_test": ("Save and test", "احفظ وجرّب"),
    "connected": ("Connected as", "متصلة باسم"),
    "presenter": ("2. Presenter", "2. المقدّمة"),
    "choose": ("Presenter", "المقدّمة"),
    "new": ("New presenter", "مقدّمة جديدة"),
    "name": ("Name", "الاسم"),
    "description": ("Look (English, used by the AI)", "الشكل (بالإنكليزي، بيستعمله الذكاء الاصطناعي)"),
    "voice": ("Voice", "الصوت"),
    "rate": ("Voice speed", "سرعة الصوت"),
    "photos": ("Reference photos (1-3, the first is the main one)", "صور مرجعية (1-3، الأولى هي الأساسية)"),
    "save_presenter": ("Save presenter", "احفظ المقدّمة"),
    "create_ai": ("No photo? Create her with AI on Kaggle", "ما عندك صورة؟ اعمليها بالذكاء الاصطناعي على Kaggle"),
    "create_btn": ("Create 4 candidate photos", "اعملي 4 صور مقترحة"),
    "use_this": ("Use this one", "اختاري هي"),
    "video": ("3. Video", "3. الفيديو"),
    "topic": ("Topic", "الموضوع"),
    "language": ("Language", "اللغة"),
    "minutes": ("Length (minutes)", "المدة (بالدقايق)"),
    "aspect": ("Format", "الشكل"),
    "script": ("Script (leave empty to write it with the local AI)",
               "النص (خليه فاضي ليكتبه الذكاء الاصطناعي المحلي)"),
    "places": ("Photos of places (optional, the presenter appears inside them)",
               "صور أماكن (اختياري، المقدّمة بتطلع جوّاتها)"),
    "use_screens": ("Use website screenshots from the branding folder", "استعملي لقطات الموقع من مجلد branding"),
    "plan": ("Plan the shots", "خطّطي اللقطات"),
    "test10": ("Quick 10-second test job", "مشروع تجربة سريع (10 ثواني)"),
    "shots": ("Shots (you can edit them)", "اللقطات (فيكي تعدّليها)"),
    "save_shots": ("Save changes", "احفظي التعديلات"),
    "render": ("4. Render", "4. التوليد"),
    "job": ("Job", "المشروع"),
    "run": ("Render on Kaggle (free) and assemble", "ولّدي على Kaggle (ببلاش) وجمّعي الفيديو"),
    "assemble": ("Assemble again", "جمّعي الفيديو من جديد"),
    "colab": ("Backup: run on Google Colab by hand", "بديل: شغّلي على Google Colab بإيدك"),
    "download_pkg": ("Download job package (package.zip)", "نزّلي ملف الشغل (package.zip)"),
    "import": ("Import results (.zip from Colab)", "استوردي النتائج (zip من Colab)"),
    "style": ("Subtitle style", "شكل الترجمة"),
    "logo": ("Show logo", "أظهري اللوغو"),
    "intro_outro": ("Add intro/outro", "أضيفي مقدمة وخاتمة"),
    "music": ("Background music", "موسيقى خلفية"),
    "status": ("Status", "الحالة"),
    "need_token": ("Add your Kaggle token in step 1 first.", "حطّي مفتاح Kaggle بالخطوة 1 أول."),
    "need_presenter": ("Create a presenter in step 2 first.", "اعملي مقدّمة بالخطوة 2 أول."),
    "ready": ("Your video is ready", "الفيديو جاهز"),
    "interrupted": ("This job stopped on the laptop (it was closed, slept or the drive was removed). The work on "
                    "Kaggle kept going. Press Continue to fetch it and finish.",
                    "هالشغل وقف عاللابتوب (انطفى، أو نام، أو انشال الهارد). الشغل على Kaggle ضل ماشي. "
                    "اضغطي كمّلي لحتى يجيبه ويخلّص."),
    "continue": ("Continue (fetch from Kaggle and finish)", "كمّلي (جيبي الشغل من Kaggle وخلّصي)"),
    "safe_off": ("You can turn the laptop off now. The work continues on Kaggle; later open this page and press "
                 "Continue.",
                 "فيكي تطفي اللابتوب هلأ. الشغل بيكمّل على Kaggle، وبعدين افتحي هالصفحة واضغطي كمّلي."),
    "stopped_jobs": ("Jobs waiting to be continued", "مشاريع ناطرة تكمّليها"),
}

arabic = st.radio("Language / اللغة", ["العربية", "English"], horizontal=True, key="presenter_ui_lang") == "العربية"


def t(key: str) -> str:
    english, arabic_text = TEXT[key]
    return arabic_text if arabic else english


st.title(t("title"))
st.caption(t("intro"))

waiting = [j for j in job_package.list_jobs() if studio.was_interrupted(j)]
if waiting:
    st.warning(f"{t('stopped_jobs')}: " + ", ".join(waiting))

# ----------------------------------------------------------------------------- 1. Kaggle
with st.expander(t("kaggle"), expanded=not kaggle_agent.configured_token()):
    token = st.text_input(t("token"), value=kaggle_agent.configured_token(), type="password", help=t("token_help"))
    if st.button(t("save_test")):
        config.app["kaggle_api_token"] = token.strip()
        config.save_config()
        try:
            st.success(f"{t('connected')} {kaggle_agent.KaggleAgent(token=token.strip()).check()}")
        except Exception as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------- 2. Presenter
st.subheader(t("presenter"))
voices = voice.get_all_azure_voices(["ar-", "en-US-", "en-GB-"]) or [profiles.DEFAULT_VOICE]
names = profiles.list_presenters()
selected = st.selectbox(t("choose"), names + [t("new")], index=0)
current = profiles.load_presenter(selected) if selected in names else profiles.Presenter(name="")
col_form, col_photo = st.columns([2, 1])
with col_form:
    name = st.text_input(t("name"), value=current.name)
    description = st.text_area(t("description"), value=current.description, height=80)
    voice_name = st.selectbox(t("voice"), voices,
                              index=voices.index(current.voice_name) if current.voice_name in voices else 0)
    rate = st.slider(t("rate"), 0.7, 1.3, float(current.voice_rate), 0.05)
    uploads = st.file_uploader(t("photos"), type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if st.button(t("save_presenter"), disabled=not name.strip()):
        temp_dir = utils.storage_dir("presenter_uploads", create=True)
        paths = []
        for upload in uploads or []:
            path = os.path.join(temp_dir, os.path.basename(upload.name))
            with open(path, "wb") as fp:
                fp.write(upload.getvalue())
            paths.append(path)
        saved = profiles.save_presenter(
            profiles.Presenter(name=name, description=description, voice_name=voice_name, voice_rate=rate), paths)
        st.success(f"✓ {saved.name}")
        st.rerun()
with col_photo:
    for image in current.reference_images[:3]:
        st.image(image, width=220)

with st.expander(t("create_ai")):
    if st.button(t("create_btn"), disabled=not kaggle_agent.configured_token()):
        st.session_state["create_job"] = studio.start_presenter_creation(description, kaggle_agent.configured_token())
    create_job = st.session_state.get("create_job") or next(iter(studio.creation_jobs()), None)
    if create_job:
        status = studio.read_status(create_job)
        st.write(f"{t('status')}: **{status.get('state')}** {status.get('kaggle', '')}")
        st.code("\n".join(status.get("log", [])[-8:]) or "...")
        if studio.was_interrupted(create_job):
            st.warning(t("interrupted"))
            if st.button(t("continue"), key=f"resume_{create_job}", disabled=not kaggle_agent.configured_token()):
                studio.start_resume(create_job, kaggle_agent.configured_token())
                st.rerun()
        elif studio.is_busy(create_job) and status.get("kaggle") in ("running", "queued"):
            st.caption(t("safe_off"))
        pictures = studio.candidates(create_job)
        for column, picture in zip(st.columns(max(1, len(pictures))), pictures):
            with column:
                st.image(picture)
                if st.button(t("use_this"), key=picture):
                    profiles.save_presenter(profiles.Presenter(
                        name=name or "Presenter", description=description, voice_name=voice_name, voice_rate=rate),
                        [picture])
                    st.rerun()
        if not pictures and st.button("↻"):
            st.rerun()

# ----------------------------------------------------------------------------- 3. Video
st.subheader(t("video"))
col1, col2, col3 = st.columns(3)
topic = col1.text_input(t("topic"))
language = col2.selectbox(t("language"), ["ar-SA", "en-US", "fr-FR", "tr-TR", "de-DE", "es-ES"])
aspect = col3.selectbox(t("aspect"), ["16:9", "9:16"])
minutes = st.slider(t("minutes"), 0.5, 10.0, 1.0, 0.5)
script = st.text_area(t("script"), height=150)
place_uploads = st.file_uploader(t("places"), type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
use_screens = st.checkbox(t("use_screens"), value=False)

if st.button(t("plan"), type="primary", disabled=not (topic.strip() and names)):
    places = []
    place_dir = utils.storage_dir("presenter_places", create=True)
    for upload in place_uploads or []:
        path = os.path.join(place_dir, os.path.basename(upload.name))
        with open(path, "wb") as fp:
            fp.write(upload.getvalue())
        places.append({"name": Path(upload.name).stem.replace("_", " "), "path": path, "screen": False})
    if use_screens:
        for path in branding.list_screenshots(portrait=aspect == "9:16"):
            places.append({"name": Path(path).stem.replace("_", " "), "path": path, "screen": True})
    presenter_name = selected if selected in names else names[0]
    with st.spinner("..."):
        text = script.strip() or studio.write_script(topic, language, minutes,
                                                     profiles.load_presenter(presenter_name).voice_rate)
        plan = studio.plan_video(topic, text, language, aspect, presenter_name, places)
    st.session_state["presenter_job"] = plan["job_id"]
if st.button(t("test10"), disabled=not names):
    st.session_state["presenter_job"] = studio.test_job(selected if selected in names else names[0],
                                                        language, aspect)["job_id"]
if not names:
    st.info(t("need_presenter"))

# ----------------------------------------------------------------------------- 4. Render
jobs = [j for j in job_package.list_jobs() if not studio.is_creation_job(j)]
if jobs:
    st.subheader(t("render"))
    current_job = st.session_state.get("presenter_job")
    job_id = st.selectbox(t("job"), jobs, index=jobs.index(current_job) if current_job in jobs else 0)
    plan = job_package.load_plan(job_id)
    st.markdown(f"**{t('shots')}**")
    edited = st.data_editor(
        [{k: s.get(k, "") for k in ("type", "narration", "location", "action", "camera", "place")}
         for s in plan["shots"]],
        num_rows="dynamic", use_container_width=True, key=f"shots_{job_id}",
        column_config={"type": st.column_config.SelectboxColumn(options=["TALK", "POINT", "WALK", "BROLL"])},
    )
    if st.button(t("save_shots")):
        studio.update_shots(job_id, list(edited))
        st.rerun()

    col_a, col_b, col_c, col_d = st.columns(4)
    styles = subtitle_styles.style_names()
    options = {
        "subtitle_style": col_a.selectbox(t("style"), styles, index=styles.index("boxed") if "boxed" in styles else 0),
        "logo": col_b.checkbox(t("logo"), value=bool(branding.find_part("logo"))),
        "intro_outro": col_c.checkbox(t("intro_outro"), value=False),
        "bgm_type": "random" if col_d.checkbox(t("music"), value=True) else "",
    }
    busy = studio.is_busy(job_id)
    col_run, col_again = st.columns(2)
    if col_run.button(t("run"), type="primary", disabled=busy or not kaggle_agent.configured_token()):
        studio.start_kaggle_render(job_id, kaggle_agent.configured_token(), options)
        st.rerun()
    if not kaggle_agent.configured_token():
        col_run.caption(t("need_token"))
    if col_again.button(t("assemble"), disabled=busy or not studio.rendered_shots(job_id)):
        studio.start_assembly(job_id, options)
        st.rerun()
    if studio.was_interrupted(job_id):
        st.warning(t("interrupted"))
        if st.button(t("continue"), type="primary", key=f"resume_{job_id}",
                     disabled=not kaggle_agent.configured_token()):
            studio.start_resume(job_id, kaggle_agent.configured_token(), options)
            st.rerun()

    @st.fragment(run_every="15s")
    def show_status():
        status = studio.read_status(job_id)
        st.write(f"{t('status')}: **{status.get('state', 'new')}** {status.get('kaggle', '')}")
        st.progress(min(1.0, float(status.get("progress", 0) or 0)))
        if studio.is_busy(job_id) and status.get("kaggle") in ("running", "queued"):
            st.caption(t("safe_off"))
        st.code("\n".join(status.get("log", [])[-12:]) or "...")
        final = status.get("final")
        if status.get("state") == "done" and final and os.path.isfile(final):
            st.success(f"{t('ready')}: {final}")
            st.video(final)

    show_status()

    with st.expander(t("colab")):
        if st.button(t("download_pkg")):
            if not os.path.isfile(os.path.join(job_package.job_dir(job_id), "package", "job.json")):
                with st.spinner("..."):
                    studio.prepare_package(job_id)
            st.session_state["package_zip"] = studio.package_zip(job_id)
        if st.session_state.get("package_zip"):
            st.download_button("package.zip", st.session_state["package_zip"], file_name="package.zip")
        results = st.file_uploader(t("import"), type=["zip"], key=f"results_{job_id}")
        if results is not None and st.button("OK", key=f"import_{job_id}"):
            st.success(f"✓ {studio.import_results(job_id, results.getvalue())}")
