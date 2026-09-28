"""AI Website Video: a service page becomes an ad presented by the brand's virtual presenter."""

import os
import sys
from pathlib import Path

import streamlit as st

root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.config import config  # noqa: E402
from app.services import subtitle_styles  # noqa: E402
from app.services.marketing import director, pipeline, website, website_source  # noqa: E402
from app.services.presenter import cloud, kaggle_agent, profiles, studio  # noqa: E402

st.set_page_config(page_title="AI Website Video", page_icon="🎬", layout="wide")

TEXT = {
    "title": ("🎬 AI Website Video", "🎬 فيديو إعلاني من الموقع"),
    "intro": ("Give a service page. The AI reads it, writes an ad, and your presenter presents it inside a "
              "3D-like world made from the real website. Heavy AI work runs free on Kaggle.",
              "حطّي رابط صفحة الخدمة. الذكاء الاصطناعي بيقرأها وبيكتب إعلان، والمقدّمة تبعك بتقدّمه جوّا عالم "
              "شبه ثلاثي الأبعاد معمول من الموقع الحقيقي. الشغل التقيل ببلاش على Kaggle."),
    "url": ("Website / service URL", "رابط الموقع أو صفحة الخدمة"),
    "source_mode": ("Website source", "مصدر الموقع"),
    "mode_url": ("URL", "رابط"),
    "mode_local": ("Local source code", "كود الموقع على اللابتوب"),
    "folder": ("Website project folder", "مجلد مشروع الموقع"),
    "folder_help": ("The folder of the website project on this laptop, built once (npm run build). Nothing is "
                    "uploaded and no project command is run.",
                    "مجلد مشروع الموقع عاللابتوب، بعد ما يكون مبني مرة (npm run build). ما بينرفع شي، "
                    "وما بيشتغل ولا أمر من المشروع."),
    "page": ("Page (service)", "الصفحة (الخدمة)"),
    "public_url": ("Public website address (shown in the ad)", "عنوان الموقع الحقيقي (بيطلع بالإعلان)"),
    "found_pages": ("public pages found", "صفحة عامة لقيناها"),
    "language": ("Language", "اللغة"),
    "duration": ("Duration (seconds)", "المدة (بالثواني)"),
    "platform": ("Platform", "المنصة"),
    "goal": ("Marketing goal", "الهدف التسويقي"),
    "presenter": ("Presenter", "المقدّمة"),
    "aspect": ("Format", "الشكل"),
    "auto": ("Auto (from platform)", "تلقائي (حسب المنصة)"),
    "focus": ("Service to focus on (optional, for a homepage)", "الخدمة المقصودة (اختياري، إذا الرابط للصفحة الرئيسية)"),
    "analyze": ("ANALYZE", "حلّلي الموقع"),
    "make_default": ("Use this presenter for every video", "خلّي هالمقدّمة لكل الفيديوهات"),
    "no_presenter": ("Create your presenter first on the Presenter Video page.",
                     "اعملي المقدّمة أول شي من صفحة Presenter Video."),
    "project": ("Project", "المشروع"),
    "service": ("Service", "الخدمة"),
    "hook": ("Hook", "الجملة الافتتاحية"),
    "message": ("Message", "الرسالة"),
    "cta": ("Call to action", "الدعوة للاشتراك"),
    "script": ("Script", "النص"),
    "storyboard": ("Storyboard (edit and save)", "المشاهد (عدّلي واحفظي)"),
    "save": ("Save storyboard", "احفظي المشاهد"),
    "regenerate": ("Rewrite the script", "اكتبي النص من جديد"),
    "screens": ("Real screenshots from the page", "لقطات حقيقية من الصفحة"),
    "services": ("Services found on this homepage", "الخدمات الموجودة بالصفحة الرئيسية"),
    "warnings": ("Please check", "انتبهي"),
    "fallback": ("The AI writer was not reachable; this script is page text only. Check Gemini/Ollama settings.",
                 "ما قدرنا نوصل للذكاء الاصطناعي، فهالنص مأخوذ من الصفحة بس. شيّكي إعدادات Gemini/Ollama."),
    "preview": ("GENERATE 10 SECOND PREVIEW", "ولّدي معاينة 10 ثواني"),
    "full": ("GENERATE FULL VIDEO", "ولّدي الفيديو الكامل"),
    "full_locked": ("Make and watch the preview first.", "اعملي المعاينة وشوفيها أول شي."),
    "continue": ("Continue (fetch the GPU work and finish)", "كمّلي (جيبي الشغل من الـ GPU وخلّصي)"),
    "interrupted": ("This video stopped on the laptop. The GPU service kept working; press Continue.",
                    "هالفيديو وقف عاللابتوب، بس خدمة الـ GPU ضلّت شغّالة. اضغطي كمّلي."),
    "cancel": ("Cancel the RunPod run (stops GPU billing)", "إلغاء شغل RunPod (بيوقف الدفع)"),
    "need_token": ("Set up Kaggle or RunPod on the Presenter Video page (step 1).",
                   "جهّزي Kaggle أو RunPod بصفحة Presenter Video (الخطوة 1)."),
    "style": ("Subtitle style", "شكل الترجمة"),
    "music": ("Background music", "موسيقى خلفية"),
    "logo": ("Show logo", "أظهري اللوغو"),
    "intro_outro": ("Intro/outro (full video)", "مقدمة وخاتمة (للفيديو الكامل)"),
    "status": ("Status", "الحالة"),
    "ready": ("Ready", "جاهز"),
    "safe_off": ("You can turn the laptop off; press Continue later.",
                 "فيكي تطفي اللابتوب، وبعدين اضغطي كمّلي."),
}

arabic = st.radio("Language / اللغة", ["العربية", "English"], horizontal=True, key="web_ui_lang") == "العربية"


def t(key: str) -> str:
    english, arabic_text = TEXT[key]
    return arabic_text if arabic else english


st.title(t("title"))
st.caption(t("intro"))

presenters = profiles.list_presenters()
token = kaggle_agent.configured_token()
cloud_ready = cloud.ready()

# ----------------------------------------------------------------------------- inputs
mode = st.radio(t("source_mode"), [t("mode_url"), t("mode_local")], horizontal=True, key="web_source_mode")
local = mode == t("mode_local")
routes, folder = [], ""
if local:
    folder = st.text_input(t("folder"), value=str(config.app.get("website_source_folder", "") or ""),
                           placeholder=r"C:\Projects\qai-vo-launch", help=t("folder_help"))
    if folder.strip():
        try:
            routes = website_source.discover(folder.strip())["routes"]
            st.caption(f"{len(routes)} {t('found_pages')}")
        except website.WebsiteError as exc:
            st.error(str(exc))

with st.form("analyze_form"):
    if local:
        route = st.selectbox(t("page"), routes or ["/"],
                             index=routes.index("/products/academic") if "/products/academic" in routes else 0)
        url = st.text_input(t("public_url"), value=str(config.app.get("website_public_url", "") or ""),
                            placeholder="https://qai-vo.com")
    else:
        route = "/"
        url = st.text_input(t("url"), placeholder="https://qai-vo.com/products/academic")
    col1, col2, col3, col4 = st.columns(4)
    language = col1.selectbox(t("language"), ["Arabic", "English"])
    duration = col2.slider(t("duration"), 10, 90, 45, 5)
    platform = col3.selectbox(t("platform"), ["Facebook", "Instagram", "TikTok", "YouTube"])
    goal = col4.selectbox(t("goal"), ["Subscriptions", "Sign ups", "Sales", "Awareness", "Leads"])
    col5, col6, col7 = st.columns([1, 1, 2])
    default_presenter = profiles.default_presenter()
    presenter = col5.selectbox(t("presenter"), presenters or ["-"],
                               index=presenters.index(default_presenter) if default_presenter in presenters else 0)
    aspect = col6.selectbox(t("aspect"), [t("auto"), "16:9", "9:16"])
    focus = col7.text_input(t("focus"))
    submitted = st.form_submit_button(t("analyze"), type="primary",
                                      disabled=not presenters or (local and not routes))
if not presenters:
    st.info(t("no_presenter"))
elif presenter != default_presenter and st.button(t("make_default")):
    profiles.set_default_presenter(presenter)
    st.rerun()

if submitted and (url.strip() or local):
    if local and (folder.strip() != config.app.get("website_source_folder")
                  or url.strip() != config.app.get("website_public_url")):
        config.app["website_source_folder"] = folder.strip()  # remembered, stays in config.toml only
        config.app["website_public_url"] = url.strip()
        config.save_config()
    with st.spinner("..."):
        try:
            project = pipeline.analyze(url, language, duration, platform.lower(), goal.lower(),
                                       presenter=presenter, aspect="" if aspect == t("auto") else aspect,
                                       focus=focus, source_folder=folder.strip() if local else "", route=route)
            st.session_state["web_project"] = project["project_id"]
        except website.WebsiteError as exc:
            st.error(str(exc))
        except Exception as exc:  # never a crashed page: show what went wrong
            from loguru import logger

            logger.exception("website analysis failed")
            st.error(f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}")

# ----------------------------------------------------------------------------- analysis + storyboard
projects = pipeline.list_projects()
if projects:
    current = st.session_state.get("web_project")
    project_id = st.selectbox(t("project"), projects, index=projects.index(current) if current in projects else 0)
    project = pipeline.load_project(project_id)
    plan = project["plan"]
    site = website.load_website(project["website_dir"])

    st.subheader(f"{t('service')}: {plan.get('service_name', '')}")
    st.caption(f"{project['url']} · {project['language']} · {project['platform']} · {project['aspect']} · "
               f"{project['duration']:.0f}s · {project['presenter']}")
    if plan.get("fallback"):
        st.warning(t("fallback"))
    if plan.get("warnings"):
        st.warning(f"{t('warnings')}:\n\n" + "\n".join(f"- {w}" for w in plan["warnings"]))
    if site.get("services"):
        with st.expander(t("services")):
            for item in site["services"]:
                st.write(f"- [{item['name']}]({item['url']})")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**{t('hook')}**\n\n{plan.get('hook', '')}")
    c2.markdown(f"**{t('message')}**\n\n{plan.get('message', '')}")
    c3.markdown(f"**{t('cta')}**\n\n{plan.get('cta', '')}")
    with st.expander(t("script"), expanded=False):
        st.write(plan.get("voiceover", ""))

    st.markdown(f"**{t('storyboard')}**")
    asset_ids = [""] + [s["id"] for s in site["screenshots"] if s["kind"] != "logo"]
    edited = st.data_editor(
        [{k: s.get(k, "") for k in ("duration", "type", "voiceover", "presenter_action", "website_asset",
                                    "visual_prompt")} for s in plan["scenes"]],
        num_rows="dynamic", key=f"board_{project_id}",
        column_config={
            "type": st.column_config.SelectboxColumn(options=list(director.SCENE_TYPES)),
            "website_asset": st.column_config.SelectboxColumn(options=asset_ids),
            "duration": st.column_config.NumberColumn(min_value=2.5, max_value=12.0, step=0.5),
        })
    b1, b2 = st.columns(2)
    if b1.button(t("save")):
        rows = [dict(row, claims=plan["scenes"][i].get("claims", []) if i < len(plan["scenes"]) else [])
                for i, row in enumerate(edited)]
        pipeline.update_scenes(project_id, rows)
        st.rerun()
    if b2.button(t("regenerate")):
        with st.spinner("..."):
            new_plan = director.direct(site, project["language"], project["duration"], project["platform"],
                                       project["goal"], project.get("focus", ""))
            project["plan"] = new_plan
            project["jobs"].pop("full", None)
            pipeline.save_project(project)
        st.rerun()

    with st.expander(t("screens")):
        shots = [s for s in site["screenshots"] if s["kind"] not in ("page",)]
        for column, shot in zip(st.columns(4) * 10, shots):
            column.image(os.path.join(site["_dir"], shot["path"]), caption=f"{shot['id']}: {shot.get('text', '')[:40]}")

    # ------------------------------------------------------------------------- render
    options = {}
    o1, o2, o3, o4 = st.columns(4)
    styles = subtitle_styles.style_names()
    options["subtitle_style"] = o1.selectbox(t("style"), styles, index=styles.index("boxed") if "boxed" in styles else 0)
    options["bgm_type"] = "random" if o2.checkbox(t("music"), value=True) else ""
    options["logo"] = o3.checkbox(t("logo"), value=True)
    options["intro_outro"] = o4.checkbox(t("intro_outro"), value=False)
    if not cloud_ready:
        st.info(t("need_token"))

    def job_panel(mode: str, label: str, locked: bool = False):
        job_id = project["jobs"].get(mode)
        busy = bool(job_id) and studio.is_busy(job_id)
        interrupted = bool(job_id) and studio.can_continue(job_id)
        if interrupted:
            st.warning(t("interrupted"))
        if st.button(t("continue") if interrupted else label, type="primary", key=f"{mode}_{project_id}",
                     disabled=busy or locked or not cloud_ready):
            pipeline.start(project_id, mode, token, options)
            st.rerun()
        if locked:
            st.caption(t("full_locked"))
        if job_id and studio.can_cancel(job_id) and st.button(t("cancel"), key=f"cancel_{mode}_{project_id}"):
            studio.cancel_cloud_run(job_id)
            st.rerun()
        if job_id:
            status = studio.read_status(job_id)
            st.write(f"{t('status')}: **{status.get('state')}** {status.get('kaggle', '')}")
            st.progress(min(1.0, float(status.get("progress", 0) or 0)))
            st.code("\n".join(status.get("log", [])[-8:]) or "...")
            if busy and status.get("kaggle") in ("running", "queued"):
                st.caption(t("safe_off"))
            final = status.get("final")
            if status.get("state") == "done" and final and os.path.isfile(final):
                st.success(f"{t('ready')}: {final}")
                st.video(final)
        return job_id

    @st.fragment(run_every="15s")
    def render_panels():
        left, right = st.columns(2)
        with left:
            preview_id = job_panel("preview", t("preview"))
        preview_done = bool(preview_id) and studio.read_status(preview_id).get("state") == "done"
        with right:
            job_panel("full", t("full"), locked=not preview_done)

    render_panels()
