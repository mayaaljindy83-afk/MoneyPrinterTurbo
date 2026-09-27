"""Build kaggle/ai_clips_wan22.ipynb from kaggle/wan_clips.py.

The notebook embeds wan_clips.py with %%writefile so it works without
cloning this repository on Kaggle. Run after editing wan_clips.py:

    python kaggle/build_notebook.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "wan_clips.py")
NOTEBOOK = os.path.join(HERE, "ai_clips_wan22.ipynb")


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

with open("/kaggle/working/prompts.txt", "w", encoding="utf-8") as f:
    f.write(PROMPTS.strip() + "\\n")
print(len([line for line in PROMPTS.splitlines() if line.strip()]), "prompts saved")
'''

RUN = """
import subprocess
import sys

command = [sys.executable, "/kaggle/working/wan_clips.py",
           "--prompts", "/kaggle/working/prompts.txt",
           "--out", "/kaggle/working/ai_clips",
           "--aspect", ASPECT, "--quality", QUALITY, "--seconds", str(SECONDS)]
if not FAST:
    command.append("--no-fast")
subprocess.run(command, check=True)
"""

PREVIEW = """
# Preview the first clip here (optional)
from IPython.display import Video
import glob
clips = sorted(glob.glob("/kaggle/working/ai_clips/*.mp4"))
print(len(clips), "clips")
Video(clips[0], embed=True, width=640) if clips else None
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


def build() -> dict:
    with open(SCRIPT, encoding="utf-8") as fp:
        script = fp.read()
    cells = [
        markdown(INTRO),
        code(SETTINGS),
        markdown("### الكود (ما في داعي تعدّلي فيه)"),
        code("%%writefile /kaggle/working/wan_clips.py\n" + script),
        code(RUN),
        code(PREVIEW),
        markdown(HELP),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    with open(NOTEBOOK, "w", encoding="utf-8") as fp:
        json.dump(build(), fp, ensure_ascii=False, indent=1)
        fp.write("\n")
    print(f"wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
