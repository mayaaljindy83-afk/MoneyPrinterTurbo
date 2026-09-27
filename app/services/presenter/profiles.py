"""Presenter profiles: reference images + description + voice.

A profile lives in ``<storage>/presenters/<name>/``::

    profile.json   {"name", "description", "voice_name", "voice_rate"}
    ref1.png ...   reference images (first = best, front-facing, full body)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field

from app.config import config
from app.utils import utils

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
MAX_REFERENCES = 3
DEFAULT_DESCRIPTION = (
    "a 25-year-old woman with shoulder-length dark brown hair, a warm friendly smile, "
    "wearing a fitted navy blazer over a white blouse, smart formal-youthful style"
)
DEFAULT_VOICE = "ar-SA-ZariyahNeural-Female"
DEFAULT_VOICE_EN = "en-US-JennyNeural-Female"


@dataclass
class Presenter:
    name: str
    description: str = DEFAULT_DESCRIPTION
    voice_name: str = DEFAULT_VOICE
    voice_rate: float = 1.0
    voice_name_en: str = DEFAULT_VOICE_EN  # the same presenter speaking English
    folder: str = ""
    reference_images: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("folder")
        data.pop("reference_images")
        return data


def presenters_root() -> str:
    custom = str(config.app.get("presenters_dir", "") or "").strip()
    root = custom or utils.storage_dir("presenters")
    os.makedirs(root, exist_ok=True)
    return root


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", str(name or ""), flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned or cleaned in (".", ".."):
        raise ValueError("presenter name is empty")
    return cleaned[:60]


def _references(folder: str) -> list[str]:
    files = [f for f in os.listdir(folder) if f.lower().startswith("ref") and f.lower().endswith(IMAGE_EXTENSIONS)]
    return [os.path.join(folder, f) for f in sorted(files)]


def list_presenters() -> list[str]:
    root = presenters_root()
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "profile.json")))


def load_presenter(name: str) -> Presenter:
    folder = os.path.join(presenters_root(), safe_name(name))
    with open(os.path.join(folder, "profile.json"), encoding="utf-8") as fp:
        data = json.load(fp)
    known = {k: data[k] for k in ("name", "description", "voice_name", "voice_rate", "voice_name_en") if k in data}
    presenter = Presenter(**known)
    presenter.folder = folder
    presenter.reference_images = _references(folder)
    return presenter


def save_presenter(presenter: Presenter, images: list[str] | None = None) -> Presenter:
    """Create or update a profile; ``images`` (paths) replace the reference images."""
    folder = os.path.join(presenters_root(), safe_name(presenter.name))
    os.makedirs(folder, exist_ok=True)
    if images:
        from PIL import Image

        for old in _references(folder):
            os.remove(old)
        for index, path in enumerate(images[:MAX_REFERENCES], start=1):
            Image.open(path).convert("RGB").save(os.path.join(folder, f"ref{index}.png"))
    with open(os.path.join(folder, "profile.json"), "w", encoding="utf-8") as fp:
        json.dump(presenter.to_json(), fp, ensure_ascii=False, indent=2)
    return load_presenter(presenter.name)


def voice_for(presenter: Presenter, language: str) -> str:
    """The presenter's voice for a video language (Arabic voice by default)."""
    if str(language or "").lower().startswith("en"):
        return presenter.voice_name_en or DEFAULT_VOICE_EN
    return presenter.voice_name


def default_presenter() -> str:
    """The brand presenter used for every video (``[app] default_presenter``), else the first one."""
    names = list_presenters()
    chosen = str(config.app.get("default_presenter", "") or "")
    if chosen in names:
        return chosen
    return names[0] if names else ""


def set_default_presenter(name: str) -> None:
    config.app["default_presenter"] = safe_name(name)
    config.save_config()


def delete_presenter(name: str) -> None:
    shutil.rmtree(os.path.join(presenters_root(), safe_name(name)), ignore_errors=True)
