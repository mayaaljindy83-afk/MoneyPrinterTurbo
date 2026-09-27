"""Build the Kaggle and Google Colab notebooks from kaggle/wan_clips.py.

Both notebooks embed wan_clips.py with %%writefile so they work without
cloning this repository. Run after editing wan_clips.py:

    python kaggle/build_notebook.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "wan_clips.py")
NOTEBOOK = os.path.join(HERE, "ai_clips_wan22.ipynb")
COLAB_NOTEBOOK = os.path.join(HERE, "ai_clips_wan22_colab.ipynb")
COLAB_URL = (
    "https://colab.research.google.com/github/mayaaljindy83-afk/MoneyPrinterTurbo/"
    "blob/main/kaggle/ai_clips_wan22_colab.ipynb"
)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


INTRO = """
# 🎬 مقاطع فيديو بالذكاء الاصطناعي ببلاش (Wan 2.2 على Kaggle)

هاد الدفتر بيولّد مقاطع فيديو قصيرة بالذكاء الاصطناعي من وصف مكتوب، على كرت الشاشة المجاني تبع Kaggle، وبعدين بتحطيها بفيديوهاتك الطويلة بـ MoneyPrinterTurbo.

**قبل ما تبلشي (مرة وحدة بس):**
1. من اليمين، افتحي **Settings**:
   - **Accelerator** ← **GPU T4 x2**
   - **Internet** ← **On** (بدها تأكيد رقم الموبايل بحسابك على Kaggle)
2. الصقي الوصفات بالخلية التانية (**PROMPTS**)، سطر لكل مقطع. فيكي تنسخيهن من MoneyPrinterTurbo: **AI Clips (Kaggle)** ← **Create prompts for Kaggle**.
3. من فوق، اضغطي **Run All**.

**بالآخر:** من **Output** على اليمين نزّلي `ai_clips.zip`، وفكّيه بمجلد جديد جوّا `E:\\MoneyPrinterData\\ai_clips\\` (مثلاً `E:\\MoneyPrinterData\\ai_clips\\ocean`).

⏱️ أول تشغيل بياخد تقريباً 10–15 دقيقة للتثبيت وتنزيل النموذج (حوالي 19GB). بعد هيك كل مقطع مدته 5 ثواني بياخد عدة دقايق على T4، والدفتر بيكتب الوقت الحقيقي لكل مقطع. Kaggle بيعطي 30 ساعة GPU مجانية بالأسبوع.
"""

COLAB_INTRO = """
# 🎬 مقاطع فيديو بالذكاء الاصطناعي ببلاش (Wan 2.2 على Google Colab)

هاد الدفتر بيولّد مقاطع فيديو قصيرة بالذكاء الاصطناعي من وصف مكتوب، على كرت الشاشة المجاني تبع Google Colab، وبعدين بتحطيها بفيديوهاتك الطويلة بـ MoneyPrinterTurbo. **ما بدو تأكيد رقم موبايل**، بس حساب Google.

**قبل ما تبلشي:**
1. من القائمة فوق اختاري **Runtime** ← **Change runtime type** ← **T4 GPU** ← **Save**.
2. الصقي الوصفات بالخلية التانية (**PROMPTS**)، سطر لكل مقطع. فيكي تنسخيهن من MoneyPrinterTurbo: **AI Clips (Kaggle)** ← **Create prompts for Kaggle**.
3. من فوق، اضغطي **Runtime** ← **Run all**. إذا سألك "This notebook was not authored by Google"، اضغطي **Run anyway**.
4. **خلّي الصفحة مفتوحة** لحد ما يخلص. إذا سكّرتيها أو ضلّت فترة طويلة بدون حركة، Colab بيفصل الجلسة.

**بالآخر:** المتصفح بينزّل `ai_clips.zip` لحاله. فكّيه بمجلد جديد جوّا `E:\\MoneyPrinterData\\ai_clips\\` (مثلاً `E:\\MoneyPrinterData\\ai_clips\\ocean`).

⏱️ أول تشغيل بكل جلسة بياخد تقريباً 10–15 دقيقة للتثبيت وتنزيل النموذج، لأن Colab بيمسح كل شي لما تسكر الجلسة. فالأحسن تولّدي كل مقاطع الفيديو مرة وحدة. بعد هيك كل مقطع مدته 5 ثواني بياخد عدة دقايق، والدفتر بيكتب الوقت الحقيقي. وقت Colab المجاني محدود ومش ثابت (عادة كم ساعة باليوم)، بس بيكفي لكم فيديو.
"""

SETTINGS = '''
# ---- PROMPTS: one English prompt per line (one line = one clip) ----
PROMPTS = """
A slow aerial drone shot over a turquoise coral reef at sunrise, gentle waves, cinematic lighting, realistic.
Close-up of a glowing jellyfish drifting in the deep dark ocean, soft blue light, slow camera push-in.
"""

ASPECT = "landscape"   # "landscape" = YouTube 16:9, "portrait" = TikTok/Reels 9:16
QUALITY = "fast"       # "fast" = 960x544 (quicker), "hd" = 1280x704 (native 720p, ~2x slower)
SECONDS = 5            # clip length in seconds
FAST = True            # True = FastWan (about 8 steps). False = official 20 steps (much slower)

with open("{workdir}/prompts.txt", "w", encoding="utf-8") as f:
    f.write(PROMPTS.strip() + "\\n")
print(len([line for line in PROMPTS.splitlines() if line.strip()]), "prompts saved")
'''

RUN = """
import subprocess
import sys

command = [sys.executable, "{workdir}/wan_clips.py",
           "--prompts", "{workdir}/prompts.txt",
           "--out", "{workdir}/ai_clips",
           "--aspect", ASPECT, "--quality", QUALITY, "--seconds", str(SECONDS){extra}]
if not FAST:
    command.append("--no-fast")
subprocess.run(command, check=True)
"""

PREVIEW = """
# Preview the first clip here (optional)
from IPython.display import Video
import glob
clips = sorted(glob.glob("{workdir}/ai_clips/*.mp4"))
print(len(clips), "clips")
Video(clips[0], embed=True, width=640) if clips else None
"""

COLAB_DOWNLOAD = """
# Download ai_clips.zip to your computer
from google.colab import files
files.download("/content/ai_clips.zip")
"""

HELP = """
### مشاكل شائعة
- **`Could not resolve host` أو خطأ بالتنزيل:** الإنترنت مش مفعّل من Settings (بدها تأكيد رقم الموبايل).
- **`CUDA out of memory`:** خلّي `QUALITY = "fast"` أو نزّلي `SECONDS` لـ 3.
- **المقاطع فيها نويز أو ألوان غريبة:** جربي `FAST = False` (أبطأ بس أدق).
- **انقطعت الجلسة بالنص:** شغّلي **Run All** مرة تانية، والمقاطع اللي خلصت ما بتنعاد.
- سجل ComfyUI الكامل موجود بـ `/tmp/comfyui.log`.

النموذج: **Wan 2.2 TI2V-5B** (رخصة Apache-2.0)، ومعه FastWan LoRA للسرعة، عن طريق ComfyUI.
"""

COLAB_HELP = """
### مشاكل شائعة
- **`Cannot connect to GPU backend` أو ما في T4:** وقت Colab المجاني خلص لليوم، أو ما في كروت فاضية هلق. جربي بعد كم ساعة.
- **`No CUDA GPUs are available`:** نسيتي تختاري **T4 GPU** من **Runtime** ← **Change runtime type**.
- **الجلسة وقفت أو انقطعت (`Your session crashed` أو انفصلت):** غالباً خلصت الرام. خلّي `QUALITY = "fast"` و`SECONDS` على 3–5، وبعدين **Runtime** ← **Run all** مرة تانية.
- **`CUDA out of memory`:** نفس الحل: `QUALITY = "fast"` أو نزّلي `SECONDS`.
- **المقاطع فيها نويز أو ألوان غريبة:** جربي `FAST = False` (أبطأ بس أدق).
- **ما نزل الـ zip لحاله:** من الشريط الشمال افتحي 📁 **Files**، وكليك يمين على `ai_clips.zip` ← **Download**.
- سجل ComfyUI الكامل موجود بـ `/tmp/comfyui.log`.

النموذج: **Wan 2.2 TI2V-5B** (رخصة Apache-2.0)، ومعه FastWan LoRA للسرعة، عن طريق ComfyUI. على Colab بيتحمّل بدقة fp8 ليساع بالرام المجانية.
"""


def _script() -> str:
    with open(SCRIPT, encoding="utf-8") as fp:
        return fp.read()


def _cells(intro: str, workdir: str, extra_args: str, help_text: str, tail: list[dict]) -> list[dict]:
    return [
        markdown(intro),
        code(SETTINGS.replace("{workdir}", workdir)),
        markdown("### الكود (ما في داعي تعدّلي فيه)"),
        code(f"%%writefile {workdir}/wan_clips.py\n" + _script()),
        code(RUN.replace("{workdir}", workdir).replace("{extra}", extra_args)),
        code(PREVIEW.replace("{workdir}", workdir)),
        *tail,
        markdown(help_text),
    ]


def build() -> dict:
    """Kaggle notebook."""
    return {
        "cells": _cells(INTRO, "/kaggle/working", "", HELP, []),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_colab() -> dict:
    """Google Colab notebook: fp8 weights for the 12 GB RAM free tier."""
    return {
        "cells": _cells(
            COLAB_INTRO,
            "/content",
            ',\n           "--weight-dtype", "fp8_e4m3fn"',
            COLAB_HELP,
            [code(COLAB_DOWNLOAD)],
        ),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PRESENTER_SCRIPT = os.path.join(HERE, "presenter_worker.py")
PRESENTER_COLAB_NOTEBOOK = os.path.join(HERE, "presenter_colab.ipynb")
PRESENTER_COLAB_URL = (
    "https://colab.research.google.com/github/mayaaljindy83-afk/MoneyPrinterTurbo/"
    "blob/main/kaggle/presenter_colab.ipynb"
)

PRESENTER_INTRO = """
# 🎤 فيديو مع مقدّمة: التشغيل على Google Colab (البديل)

الطريقة الأساسية هي زر **Render on Kaggle** بصفحة **Presenter video** بالبرنامج، وهي أوتوماتيكية بالكامل.
استعملي هاد الدفتر بس إذا Kaggle مش متاح.

**الخطوات:**
1. بالبرنامج، بصفحة **Presenter video**، افتحي **Backup: run on Google Colab** ونزّلي **package.zip**.
2. هون، من القائمة: **Runtime** ← **Change runtime type** ← **T4 GPU** ← **Save**.
3. **Runtime** ← **Run all**. لما يطلب منك، اختاري ملف `package.zip`.
4. بالآخر بينزل `results.zip`. رجعي عالبرنامج واختاري **Import results**.

⚠️ **بصراحة:** رام Colab المجاني (حوالي 12GB) أقل من Kaggle (حوالي 29GB)، والنماذج هون كبيرة (14B).
ممكن تنجح، وممكن الجلسة توقف بنص الشغل. إذا وقفت، رجعي شغّلي **Run all**: اللقطات اللي خلصت محفوظة على Google Drive وما بتنعاد.
"""

PRESENTER_UPLOAD = """
import os

from google.colab import drive, files

USE_DRIVE = True  # keep finished shots on Google Drive so a crash loses nothing
if USE_DRIVE:
    drive.mount("/content/drive")
os.makedirs("/content/job_upload", exist_ok=True)
if not os.path.exists("/content/job_upload/package.zip"):
    uploaded = files.upload()  # choose package.zip
    name = next(iter(uploaded))
    with open("/content/job_upload/package.zip", "wb") as fp:
        fp.write(uploaded[name])
print("job package ready")
"""

PRESENTER_RUN = """
import subprocess
import sys

subprocess.run([sys.executable, "/content/presenter_worker.py", "--job", "/content/job_upload",
                "--work", "/content/presenter", "--lowvram"], check=True)
"""

PRESENTER_DOWNLOAD = """
import glob
import json
import shutil

from google.colab import files

summary = sorted(glob.glob("/content/**/output/*/summary.json", recursive=True)
                 + glob.glob("/content/drive/MyDrive/MoneyPrinterPresenter/output/*/summary.json"),
                 key=os.path.getmtime)[-1]
out_dir = os.path.dirname(summary)
print(json.load(open(summary)))
shutil.make_archive("/content/results", "zip", out_dir)
files.download("/content/results.zip")
"""


def build_presenter_colab() -> dict:
    with open(PRESENTER_SCRIPT, encoding="utf-8") as fp:
        worker = fp.read()
    return {
        "cells": [
            markdown(PRESENTER_INTRO),
            code(PRESENTER_UPLOAD),
            markdown("### الكود (ما في داعي تعدّلي فيه)"),
            code("%%writefile /content/presenter_worker.py\n" + worker),
            code(PRESENTER_RUN),
            code(PRESENTER_DOWNLOAD),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for path, notebook in ((NOTEBOOK, build()), (COLAB_NOTEBOOK, build_colab()),
                           (PRESENTER_COLAB_NOTEBOOK, build_presenter_colab())):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(notebook, fp, ensure_ascii=False, indent=1)
            fp.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
