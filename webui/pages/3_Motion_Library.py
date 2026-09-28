"""Motion Library: driving clips for the presenter's gestures (Wan Animate 2)."""

import os
import sys
from pathlib import Path

import streamlit as st

root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.services.presenter import cloud, kaggle_agent, motion_library, studio  # noqa: E402
from app.services.presenter import package as job_package  # noqa: E402
from app.utils import utils  # noqa: E402

st.set_page_config(page_title="Motion Library", page_icon="🕺", layout="wide")

TEXT = {
    "title": ("Motion Library", "مكتبة الحركات"),
    "intro": ("Each gesture of the presenter (point, look, welcome...) is copied from a short driving clip. "
              "A clip for one side is enough: the other side is its mirror. Keep only clips that look right.",
              "كل حركة للمقدّمة (تأشر، تتطلّع، ترحّب...) بتنقلد من فيديو قصير. بيكفي فيديو لجهة وحدة، "
              "والجهة التانية بتنعمل بالعكس. خلّي بس الفيديوهات اللي طالعة منيحة."),
    "status": ("Library", "المكتبة"),
    "motion": ("Motion", "الحركة"),
    "source": ("Clip", "الفيديو"),
    "tested": ("Tested", "مجرّبة"),
    "missing": ("missing", "ناقصة"),
    "own": ("own clip", "فيديو خاص"),
    "ai_title": ("B. Make clips with AI (uses the GPU service)", "ب. اعملي فيديوهات بالذكاء الاصطناعي (على خدمة الـ GPU)"),
    "ai_help": ("Lina's photo performs each gesture; you get a few versions and keep the best. "
                "Costs GPU time (RunPod is paid).",
                "صورة لينا بتعمل كل حركة، وبيطلعلك كم نسخة وإنتِ بتختاري الأحلى. بياخد وقت GPU (RunPod مدفوع)."),
    "which": ("Motions", "الحركات"),
    "variants": ("Versions per motion", "كم نسخة لكل حركة"),
    "make": ("Make AI clips", "اعملي الفيديوهات"),
    "need_cloud": ("Set up Kaggle or RunPod on the Presenter Video page first.",
                   "جهّزي Kaggle أو RunPod بصفحة Presenter Video أول."),
    "candidates": ("AI clips ready: keep the good ones", "الفيديوهات جاهزة: اختاري المنيحين"),
    "use": ("Use for", "استعمليه لـ"),
    "saved": ("Saved", "انحفظ"),
    "stock_title": ("C. Add a free stock clip (Pexels)", "ج. ضيفي فيديو مجاني من الإنترنت (Pexels)"),
    "stock_help": ("Open the search, download a clip (Free download), then upload it here. "
                   "Pick a person from the knees up, facing the camera, plain background.",
                   "افتحي البحث، نزّلي فيديو (Free download)، وبعدين ارفعيه هون. "
                   "اختاري شخص مبيّن من الركب لفوق، وجهه للكاميرا، والخلفية بسيطة."),
    "search": ("Search clips for this motion", "دوّري فيديوهات لهالحركة"),
    "upload": ("Upload the clip", "ارفعي الفيديو"),
    "add": ("Add to library", "ضيفيه للمكتبة"),
    "mark": ("Mark as tested (it gave a good result)", "علّميها مجرّبة (طلعت منيحة)"),
    "delete": ("Delete clip", "امسحي الفيديو"),
    "running": ("Making clips…", "عم يعمل الفيديوهات…"),
}

arabic = st.radio("Language / اللغة", ["العربية", "English"], horizontal=True, key="motion_ui_lang") == "العربية"


def t(key: str) -> str:
    english, arabic_text = TEXT[key]
    return arabic_text if arabic else english


st.title(t("title"))
st.caption(t("intro"))

# ----------------------------------------------------------------------------- status
status = motion_library.available()
rows = [{t("motion"): motion, t("source"): info["source"] or t("missing"), t("tested"): "✓" if info["tested"] else ""}
        for motion, info in status.items()]
st.subheader(t("status"))
st.dataframe(rows, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- B. AI candidates
st.subheader(t("ai_title"))
st.caption(t("ai_help"))
missing = [m for m, info in status.items() if not info["source"]]
col1, col2 = st.columns([3, 1])
chosen = col1.multiselect(t("which"), list(motion_library.CATALOG), default=missing[:4], key="motion_ai_pick")
variants = col2.number_input(t("variants"), 1, 4, 2, key="motion_ai_variants")
if not cloud.ready():
    st.info(t("need_cloud"))
if st.button(t("make"), type="primary", disabled=not chosen or not cloud.ready()):
    st.session_state["motion_job"] = studio.start_motion_candidates(chosen, int(variants),
                                                                   kaggle_agent.configured_token())
    st.rerun()

jobs = [j for j in job_package.list_jobs() if studio.is_motion_job(j)]
for job_id in sorted(jobs, reverse=True)[:3]:
    job_status = studio.read_status(job_id)
    with st.expander(f"{job_id} — {job_status.get('state', '')}", expanded=job_id == st.session_state.get("motion_job")):
        if studio.is_busy(job_id):
            st.info(t("running"))
        if job_status.get("message"):
            st.code(job_status["message"])
        if studio.can_cancel(job_id) and st.button("✕ RunPod", key=f"cancel_{job_id}"):
            studio.cancel_cloud_run(job_id)
            st.rerun()
        found = motion_library.candidates(job_id)
        if found:
            st.write(t("candidates"))
        for index, item in enumerate(found):
            c1, c2 = st.columns([2, 1])
            c1.video(item["path"])
            targets = [item["motion"]] + ([motion_library.mirror_name(item["motion"])]
                                          if motion_library.mirror_name(item["motion"]) else [])
            target = c2.selectbox(t("use"), targets, key=f"target_{job_id}_{index}")
            if c2.button(t("use") + f" {target}", key=f"use_{job_id}_{index}"):
                motion_library.use_candidate(target, item["path"])
                st.success(t("saved"))
                st.rerun()

# ----------------------------------------------------------------------------- C. stock clips + per motion
st.subheader(t("stock_title"))
st.caption(t("stock_help"))
motion = st.selectbox(t("motion"), list(motion_library.CATALOG), key="motion_pick")
st.link_button(t("search"), motion_library.search_link(motion))
uploaded = st.file_uploader(t("upload"), type=["mp4", "mov", "webm", "m4v"], key=f"upload_{motion}")
if uploaded and st.button(t("add"), key=f"add_{motion}"):
    upload_dir = utils.storage_dir("motions_upload")
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, os.path.basename(uploaded.name))
    with open(path, "wb") as fp:
        fp.write(uploaded.getbuffer())
    motion_library.import_clip(motion, path, source_note=f"stock: {uploaded.name}")
    st.success(t("saved"))
    st.rerun()

clip = motion_library.clip_path(motion)
if os.path.isfile(clip):
    st.video(clip)
    meta = motion_library.load_meta(motion)
    c1, c2 = st.columns(2)
    tested = c1.checkbox(t("mark"), value=bool(meta.get("tested")), key=f"tested_{motion}")
    if tested != bool(meta.get("tested")):
        motion_library.mark_tested(motion, tested)
        st.rerun()
    if c2.button(t("delete"), key=f"delete_{motion}"):
        motion_library.delete(motion)
        st.rerun()
