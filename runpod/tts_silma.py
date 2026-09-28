"""SILMA TTS runner (inside the RunPod image, in its own venv /opt/silma).

    /opt/silma/bin/python tts_silma.py request.json out_dir

request.json: {"items": [{"id", "text", "ref_wav" (path, optional), "ref_text", "speed", "seed"}]}
Writes <out_dir>/<id>.wav (24 kHz mono) and <out_dir>/result.json with timings.
SILMA adds its own tashkeel (CATT) and reads numbers through its text normaliser.
"""

from __future__ import annotations

import json
import os
import sys
import time
from importlib.resources import files

# SILMA's own Arabic reference voice (used when no presenter voice sample is given).
DEFAULT_REF = ("infer/ref_audio_samples/ar.ref.24k.wav",
               "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام.")


def main(request_path: str, out_dir: str) -> int:
    with open(request_path, encoding="utf-8") as fp:
        request = json.load(fp)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    from silma_tts.api import SilmaTTS

    tts = SilmaTTS(hf_cache_dir=os.environ.get("HF_HOME"))
    load_seconds = time.time() - started
    results = []
    for item in request.get("items", []):
        ref_wav = item.get("ref_wav") or str(files("silma_tts").joinpath(DEFAULT_REF[0]))
        ref_text = item.get("ref_text") or DEFAULT_REF[1]
        begin = time.time()
        target = os.path.join(out_dir, f"{item['id']}.wav")
        wav, sample_rate, _ = tts.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=item["text"],
                                        speed=float(item.get("speed") or 1.0), seed=item.get("seed", 2026),
                                        file_wave=target)
        results.append({"id": item["id"], "file": os.path.basename(target), "sample_rate": int(sample_rate),
                        "seconds": round(len(wav) / float(sample_rate), 3),
                        "inference_seconds": round(time.time() - begin, 2)})
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fp:
        json.dump({"engine": "silma", "model_load_seconds": round(load_seconds, 2), "items": results}, fp,
                  ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
