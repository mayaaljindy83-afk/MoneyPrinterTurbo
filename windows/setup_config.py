"""Create or update config.toml for the Windows laptop setup.

Called by windows/install.ps1 with the answers the user typed. Existing
settings in config.toml are kept; only the keys passed here are changed, so
running the installer again is safe. config.toml is ignored by git, so API
keys never end up in a commit.

Usage (all arguments optional):
    python windows/setup_config.py --data-dir E:/MoneyPrinterData \
        --pexels-key KEY --pixabay-key KEY --gemini-key KEY \
        --model gemma3:4b --resolution 720p

An empty key answer keeps the saved key; "-" removes it.
    python windows/setup_config.py --print-data-dir
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import toml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config.toml")
EXAMPLE = os.path.join(ROOT, "config.example.toml")

DEFAULT_MODEL = "gemma3:4b"
# Free Gemini tier model used only when the local model fails.
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"

DATA_SUBDIRS = ("music", "branding", "tasks", "temp", "ollama-models", "pip-cache")


def load_config(path: str = CONFIG) -> dict:
    if not os.path.isfile(path):
        shutil.copyfile(EXAMPLE, path)
    with open(path, encoding="utf-8-sig") as fp:
        return toml.load(fp)


def save_config(data: dict, path: str = CONFIG) -> None:
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fp:
        toml.dump(data, fp)
    os.replace(temp, path)


def normalize_path(path: str) -> str:
    # Forward slashes work everywhere in Python and avoid TOML escape issues.
    return os.path.abspath(os.path.expanduser(path)).replace("\\", "/")


def apply_settings(data: dict, args) -> dict:
    app = data.setdefault("app", {})
    ui = data.setdefault("ui", {})

    if args.data_dir:
        data_dir = normalize_path(args.data_dir)
        for sub in DATA_SUBDIRS:
            os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
        app["storage_dir"] = data_dir
        app["music_dir"] = f"{data_dir}/music"
        app["branding_dir"] = f"{data_dir}/branding"
        app["material_directory"] = ""

    # An empty answer keeps the saved key; "-" removes it.
    for arg_value, key in ((args.pexels_key, "pexels_api_keys"), (args.pixabay_key, "pixabay_api_keys")):
        value = (arg_value or "").strip()
        if value == "-":
            app[key] = []
        elif value:
            app[key] = [value]

    # Local LLM through Ollama is the default script writer.
    app["llm_provider"] = "ollama"
    app["ollama_base_url"] = "http://localhost:11434/v1"
    app["ollama_model_name"] = args.model or app.get("ollama_model_name") or DEFAULT_MODEL

    gemini_key = (args.gemini_key or "").strip()
    if gemini_key == "-":
        app["gemini_api_key"] = ""
        app["llm_fallback_provider"] = ""
    elif gemini_key:
        app["gemini_api_key"] = gemini_key
        app["gemini_model_name"] = GEMINI_FALLBACK_MODEL
        app["llm_fallback_provider"] = "gemini"

    # Laptop-friendly rendering defaults.
    if args.resolution:
        app["video_resolution"] = args.resolution
    app.setdefault("video_resolution", "720p")
    app["fast_clip_preparation"] = True
    app["video_encode_preset"] = "veryfast"
    app["bgm_prefer_user_music"] = True
    app["subtitle_provider"] = "edge"
    app["max_concurrent_tasks"] = 1
    app["hide_config"] = False

    # First-run WebUI defaults: Arabic voice, Arabic font, landscape video.
    ui.setdefault("language", "en")
    ui.setdefault("video_language", "ar-SA")
    ui.setdefault("tts_server", "azure-tts-v1")
    ui.setdefault("voice_name", "ar-SA-HamedNeural-Male")
    ui.setdefault("font_name", "Tajawal-Bold.ttf")
    ui.setdefault("subtitle_style", "classic")
    ui.setdefault("video_transition_mode", "Crossfade")
    ui.setdefault("video_aspect_pexels", "16:9")
    ui.setdefault("video_clip_duration", 5)
    ui.setdefault("video_duration_minutes", 3.0)
    ui.setdefault("bgm_volume", 0.15)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--pexels-key")
    parser.add_argument("--pixabay-key")
    parser.add_argument("--gemini-key")
    parser.add_argument("--model")
    parser.add_argument("--resolution", choices=["720p", "1080p"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--print-data-dir", action="store_true")
    args = parser.parse_args(argv)

    data = load_config(args.config)
    if args.print_data_dir:
        print(data.get("app", {}).get("storage_dir", "") or normalize_path(os.path.join(ROOT, "storage")))
        return 0

    save_config(apply_settings(data, args), args.config)
    app = data["app"]
    print("config.toml updated:")
    print(f"  data folder : {app.get('storage_dir') or 'storage (inside the project)'}")
    print(f"  music folder: {app.get('music_dir') or '-'}")
    print(f"  local model : {app['ollama_model_name']}")
    print(f"  Pexels key  : {'set' if app.get('pexels_api_keys') else 'NOT SET'}")
    print(f"  Gemini      : {'fallback enabled' if app.get('llm_fallback_provider') else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
