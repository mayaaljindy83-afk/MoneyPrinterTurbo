"""AI presenter video worker for free cloud GPUs (Kaggle first, Colab as backup).

Reads a job package made by MoneyPrinterTurbo (``job.json`` + audio + images),
renders every shot with ComfyUI and open models, and writes one mp4 per shot:

- TALK / POINT: InfiniteTalk (Apache-2.0) on Wan 2.1 I2V 14B 480p. Lips, head,
  body and expressions follow the shot's narration audio. Long shots are made
  of chained 81-frame segments that continue from the previous frames.
- WALK / BROLL: Wan 2.1 I2V 14B 480p (Apache-2.0) + lightx2v 4-step LoRA,
  animating the shot's first frame with a motion prompt.
- First frames: Qwen-Image-Edit-2511 (Apache-2.0) + Lightning 4-step LoRA
  places the presenter (reference images) into the location photo, or into a
  described location, keeping face and outfit consistent.
- ``create_presenter`` jobs: Z-Image Turbo (Apache-2.0) makes candidate
  portraits of a new presenter.

Designed for unattended "Save & Run All" runs:
- every finished shot is written to ``<out>/shots/<id>.mp4`` (and its first
  frame to ``<out>/frames/<id>.png``) immediately;
- a rerun copies finished shots from the previous run's output (attached as
  input) and skips them, so a timed-out session loses nothing;
- work stops cleanly before the session limit (``--time-budget``);
- ``<out>/progress.json`` and ``<out>/worker.log`` record progress and timing.

Usage:  python presenter_worker.py --job /kaggle/input/<dataset>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

COMFY_TAG = "v0.37.4"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
PORT = 8188
TALK_FPS = 25  # InfiniteTalk works at 25 fps
MOTION_FPS = 16  # Wan 2.1 I2V native frame rate
SEGMENT_FRAMES = 81
MOTION_CONTEXT = 9  # frames InfiniteTalk re-uses when continuing a segment

HF = "https://huggingface.co"
# name -> (url, ComfyUI models sub-folder). URLs are the ones used by the
# official ComfyUI workflow templates for these models.
MODELS = {
    # --- video (Wan 2.1 I2V 14B 480p, InfiniteTalk) ---
    "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors": (
        f"{HF}/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
        "diffusion_models"),
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors": (
        f"{HF}/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "loras"),
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": (
        f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "text_encoders"),
    "Wan2_1_VAE_bf16.safetensors": (
        f"{HF}/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors", "vae"),
    "clip_vision_h.safetensors": (
        f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        "clip_vision"),
    "wan2.1_infiniteTalk_single_fp16.safetensors": (
        f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/model_patches/wan2.1_infiniteTalk_single_fp16.safetensors",
        "model_patches"),
    "wav2vec2-chinese-base_fp16.safetensors": (
        f"{HF}/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors", "audio_encoders"),
    # --- image (Qwen-Image-Edit-2511) ---
    "qwen_image_edit_2511_fp8mixed.safetensors": (
        f"{HF}/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "diffusion_models"),
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": (
        f"{HF}/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "text_encoders"),
    "qwen_image_vae.safetensors": (
        f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "vae"),
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": (
        f"{HF}/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "loras"),
    # --- presenter creation (Z-Image Turbo) ---
    "z_image_turbo_bf16.safetensors": (
        f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "diffusion_models"),
    "qwen_3_4b.safetensors": (
        f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    "ae.safetensors": (
        f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", "vae"),
}
IMAGE_MODELS = ["qwen_image_edit_2511_fp8mixed.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "qwen_image_vae.safetensors", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"]
VIDEO_MODELS = ["Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Wan2_1_VAE_bf16.safetensors", "clip_vision_h.safetensors"]
TALK_MODELS = ["wan2.1_infiniteTalk_single_fp16.safetensors", "wav2vec2-chinese-base_fp16.safetensors"]
CREATE_MODELS = ["z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"]

TALK_TYPES = {"TALK", "POINT"}
MOTION_TYPES = {"WALK", "BROLL", "AI_SCENE"}  # AI_SCENE: text-to-image first frame, no presenter
NEGATIVE = ("blurry, low quality, distorted face, deformed hands, extra fingers, extra limbs, text, watermark, "
            "logo, subtitles, static, frozen, jitter")


# --------------------------------------------------------------------------- platform
def detect_platform() -> str:
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    if "google.colab" in sys.modules or os.path.isdir("/content"):
        return "colab"
    return "local"


def default_paths(platform: str) -> dict:
    if platform == "kaggle":
        return {"out": "/kaggle/working", "work": "/tmp/presenter", "comfy": "/tmp/ComfyUI",
                "cache_dirs": ["/kaggle/input"]}
    if platform == "colab":
        drive = "/content/drive/MyDrive/MoneyPrinterPresenter"
        return {"out": f"{drive}/output" if os.path.isdir("/content/drive/MyDrive") else "/content/output",
                "work": "/content/presenter", "comfy": "/content/ComfyUI",
                "cache_dirs": [f"{drive}/models"] if os.path.isdir("/content/drive/MyDrive") else []}
    return {"out": os.path.abspath("presenter_output"), "work": os.path.abspath("presenter_work"),
            "comfy": os.path.abspath("ComfyUI"), "cache_dirs": []}


class Log:
    def __init__(self, path: str | None = None):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
            except OSError:
                pass  # the console line above is enough; never stop the work for a log file


log = Log()


def run(command: list[str], cwd: str | None = None) -> None:
    log("$ " + " ".join(command)[:300])
    subprocess.run(command, cwd=cwd, check=True)


# --------------------------------------------------------------------------- job
def find_job_json(search_dir: str, extract_dir: str) -> str:
    """job.json anywhere under ``search_dir``, or inside an uploaded ``package.zip``."""
    direct = os.path.join(search_dir, "job.json")
    if os.path.isfile(direct):
        return direct
    found = sorted(glob.glob(os.path.join(search_dir, "**", "job.json"), recursive=True), key=len)
    if found:
        return found[0]
    import zipfile

    for archive in sorted(glob.glob(os.path.join(search_dir, "**", "*.zip"), recursive=True)):
        with zipfile.ZipFile(archive) as zf:
            if "job.json" in zf.namelist():
                target = os.path.join(extract_dir, "job")
                zf.extractall(target)
                return os.path.join(target, "job.json")
    return ""


def load_job(job_dir: str, extract_dir: str | None = None) -> dict:
    """Load and check job.json from a folder (searched recursively) or a zip in it."""
    path = find_job_json(job_dir, extract_dir or job_dir)
    if not path:
        raise SystemExit(f"job.json not found in {job_dir}")
    with open(path, encoding="utf-8") as fp:
        job = json.load(fp)
    job["_root"] = os.path.dirname(path)
    validate_job(job)
    return job


def validate_job(job: dict) -> None:
    kind = job.get("kind", "video")
    if kind == "create_presenter":
        if not job.get("prompt"):
            raise SystemExit("create_presenter job needs a prompt")
        return
    shots = job.get("shots") or []
    if not shots:
        raise SystemExit("job has no shots")
    seen = set()
    for shot in shots:
        if shot.get("id") in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", str(shot.get("id", ""))):
            raise SystemExit(f"bad or duplicate shot id: {shot.get('id')!r}")
        seen.add(shot["id"])
        if shot.get("type") not in TALK_TYPES | MOTION_TYPES:
            raise SystemExit(f"shot {shot['id']}: unknown type {shot.get('type')}")
        if shot["type"] in TALK_TYPES and not shot.get("audio"):
            raise SystemExit(f"shot {shot['id']}: TALK/POINT shots need audio")
        if float(shot.get("duration", 0)) <= 0:
            raise SystemExit(f"shot {shot['id']}: duration must be positive")
    if not (job.get("presenter") or {}).get("reference_images"):
        needs_presenter = any(s["type"] not in ("BROLL", "AI_SCENE") for s in shots if not s.get("local"))
        if needs_presenter:
            raise SystemExit("presenter.reference_images is required")


def cloud_shots(job: dict) -> list[dict]:
    """Shots rendered here; ``local`` shots (e.g. website screenshots) are made on the laptop."""
    return [s for s in job.get("shots", []) if not s.get("local")]


def video_size(job: dict) -> tuple[int, int]:
    """Wan 480p canvas: 832x480 landscape or 480x832 portrait (multiples of 16)."""
    return (480, 832) if job.get("settings", {}).get("aspect") == "9:16" else (832, 480)


def talk_segments(duration: float) -> tuple[int, int]:
    """(segments, total frames) covering ``duration`` seconds at 25 fps."""
    needed = max(SEGMENT_FRAMES, math.ceil(duration * TALK_FPS) + 1)
    step = SEGMENT_FRAMES - MOTION_CONTEXT
    segments = 1 + max(0, math.ceil((needed - SEGMENT_FRAMES) / step))
    return segments, SEGMENT_FRAMES + (segments - 1) * step


def motion_chunks(duration: float) -> list[int]:
    """Frame counts (4k+1) of the I2V chunks that cover ``duration`` at 16 fps."""
    total = max(17, math.ceil(duration * MOTION_FPS) + 1)
    chunks = []
    remaining = total
    while remaining > 0:
        frames = min(SEGMENT_FRAMES, max(17, remaining))
        frames = (frames - 1) // 4 * 4 + 1
        chunks.append(frames)
        remaining -= frames - 1  # consecutive chunks share the joining frame
        if remaining <= 1:
            break
    return chunks


# --------------------------------------------------------------------------- workflows
def build_edit_workflow(prompt: str, image1: str, image2: str | None, image3: str | None,
                        prefix: str, seed: int) -> dict:
    """Qwen-Image-Edit-2511 + Lightning 4-step (official template wiring)."""
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "img1": {"class_type": "LoadImage", "inputs": {"image": image1}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["img1", 0]}},
        "sampling": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["sampling", 0], "strength": 1.0}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["cfgnorm", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "pos_text": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0], "image1": ["scale", 0]}},
        "neg_text": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": "", "vae": ["vae", 0], "image1": ["scale", 0]}},
        "pos": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos_text", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg_text", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "sampler": {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    for index, name in ((2, image2), (3, image3)):
        if name:
            g[f"img{index}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            for node in ("pos_text", "neg_text"):
                g[node]["inputs"][f"image{index}"] = [f"img{index}", 0]
    return g


def build_talk_workflow(image: str, audio: str, prompt: str, width: int, height: int, segments: int,
                        prefix: str, seed: int, steps: int = 6) -> dict:
    """InfiniteTalk single speaker, ``segments`` chained 81-frame windows.

    Every continuation segment re-decodes its ``MOTION_CONTEXT`` overlap
    frames first; they are dropped with ImageFromBatch so frames stay in step
    with the audio (the official template keeps them, adding ~0.36 s drift
    per join).
    """
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "patch": {"class_type": "ModelPatchLoader", "inputs": {"name": "wan2.1_infiniteTalk_single_fp16.safetensors"}},
        "audio_enc": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "wav2vec2-chinese-base_fp16.safetensors"}},
        "audio": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "audio_feat": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["audio_enc", 0], "audio": ["audio", 0]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["text", 0]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
        "start": {"class_type": "LoadImage", "inputs": {"image": image}},
        # The start frame's CLIP-vision embedding keeps the face consistent across segments.
        "clip_vision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "vision": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["clip_vision", 0], "image": ["start", 0], "crop": "none"}},
        "sampler_select": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    frames_so_far = None
    for k in range(segments):
        s = f"s{k}"
        inputs = {
            "mode": "single_speaker",
            "model": ["lora", 0], "model_patch": ["patch", 0],
            "positive": ["text", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "width": width, "height": height, "length": SEGMENT_FRAMES,
            "start_image": ["start", 0], "clip_vision_output": ["vision", 0],
            "audio_encoder_output_1": ["audio_feat", 0],
            "motion_frame_count": MOTION_CONTEXT, "audio_scale": 1.0,
        }
        if frames_so_far is not None:
            inputs["previous_frames"] = frames_so_far
        g[f"{s}_talk"] = {"class_type": "WanInfiniteTalkToVideo", "inputs": inputs}
        g[f"{s}_guider"] = {"class_type": "CFGGuider", "inputs": {"model": [f"{s}_talk", 0], "positive": [f"{s}_talk", 1], "negative": [f"{s}_talk", 2], "cfg": 1.0}}
        g[f"{s}_sched"] = {"class_type": "BasicScheduler", "inputs": {"model": [f"{s}_talk", 0], "scheduler": "normal", "steps": steps, "denoise": 1.0}}
        g[f"{s}_noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + k}}
        g[f"{s}_sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [f"{s}_noise", 0], "guider": [f"{s}_guider", 0], "sampler": ["sampler_select", 0], "sigmas": [f"{s}_sched", 0], "latent_image": [f"{s}_talk", 3]}}
        g[f"{s}_decode"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"{s}_sample", 0], "vae": ["vae", 0]}}
        if frames_so_far is None:
            frames_so_far = [f"{s}_decode", 0]
        else:
            g[f"{s}_trim"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"{s}_decode", 0], "batch_index": MOTION_CONTEXT, "length": 4096}}
            g[f"{s}_all"] = {"class_type": "BatchImagesNode", "inputs": {"images.image0": frames_so_far, "images.image1": [f"{s}_trim", 0]}}
            frames_so_far = [f"{s}_all", 0]
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": frames_so_far, "filename_prefix": prefix}}
    return g


def build_motion_workflow(image: str, prompt: str, width: int, height: int, length: int,
                          prefix: str, seed: int, steps: int = 4) -> dict:
    """Wan 2.1 I2V 14B 480p + lightx2v: one motion chunk from a start frame."""
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "sampling": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora", 0], "shift": 8.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEGATIVE}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
        "start": {"class_type": "LoadImage", "inputs": {"image": image}},
        "clip_vision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "vision": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["clip_vision", 0], "image": ["start", 0], "crop": "none"}},
        "i2v": {"class_type": "WanImageToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": width, "height": height, "length": length, "batch_size": 1, "clip_vision_output": ["vision", 0], "start_image": ["start", 0]}},
        "sampler": {"class_type": "KSampler", "inputs": {"model": ["sampling", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["i2v", 2], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


def build_portrait_workflow(prompt: str, width: int, height: int, prefix: str, seed: int) -> dict:
    """Z-Image Turbo text-to-image (official template settings: 8 steps, cfg 1)."""
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "sampling": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sampler": {"class_type": "KSampler", "inputs": {"model": ["sampling", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


# --------------------------------------------------------------------------- prompts
def placement_prompt(shot: dict, presenter: dict, has_location_photo: bool) -> str:
    look = presenter.get("description", "the presenter")
    action = shot.get("action") or "standing naturally, looking at the camera"
    camera = shot.get("camera") or "medium shot"
    keep = ("Keep her face, hairstyle, skin tone and outfit exactly the same as in the reference image. "
            "Photorealistic, natural light, sharp focus, realistic proportions.")
    if shot.get("green"):
        # Keyed out and composited on the laptop (website world scenes).
        return (f"Keep the person from image 1 ({look}) and replace the whole background with a flat, evenly lit, "
                f"pure green (#00FF00) chroma key studio backdrop without shadows or objects. She is {action}. "
                f"Knees-up framing, centred, facing the camera, with space above her head. {keep}")
    if has_location_photo and shot.get("screen"):
        return (f"Show the person from image 2 ({look}) in a bright modern presentation room, standing next to "
                f"a large wall screen that displays the website from image 1. She is {action}. {camera}. "
                f"Keep the website on the screen unchanged and readable. {keep}")
    if has_location_photo:
        return (f"Place the person from image 2 ({look}) into the scene of image 1. She is {action}. "
                f"{camera}. Keep the place in image 1 unchanged. {keep}")
    place = shot.get("location") or "a modern bright interior"
    return (f"Replace the plain background of image 1 with {place}. The person ({look}) is {action}. "
            f"{camera}. {keep}")


def broll_prompt(shot: dict) -> str:
    place = shot.get("location") or "the place"
    return f"Remove the person completely and show only {place}. Photorealistic, natural light, no people."


def motion_prompt(shot: dict) -> str:
    parts = [shot.get("action") or "", shot.get("camera") or "", shot.get("location") or ""]
    text = ", ".join(p for p in parts if p)
    return f"{text}. Natural smooth motion, realistic, cinematic, stable face." if text else "Natural smooth motion."


def talk_prompt(shot: dict) -> str:
    action = shot.get("action") or "talks to the camera with natural hand gestures"
    background = " Plain flat green studio background that never changes." if shot.get("green") else ""
    return f"A woman {action}, {shot.get('camera') or 'medium shot'}, natural expressions, realistic.{background}"


def ai_scene_prompt(shot: dict) -> str:
    place = shot.get("location") or "a cinematic modern technology space"
    return (f"{place}, cinematic lighting, photorealistic, high detail, shallow depth of field, "
            "no text, no letters, no logos, no user interface, no people")


# --------------------------------------------------------------------------- ComfyUI
class Comfy:
    def __init__(self, comfy_dir: str, port: int = PORT):
        self.dir = comfy_dir
        self.port = port
        self.process = None

    @property
    def input_dir(self) -> str:
        return os.path.join(self.dir, "input")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.dir, "output")

    def api(self, path: str, payload: dict | None = None, timeout: float = 60):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                         headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")

    def start(self, extra_args: list[str] | None = None) -> None:
        self.stop()
        log_file = open(os.path.join(os.path.dirname(self.dir.rstrip("/")), "comfyui.log"), "a", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port),
             "--disable-auto-launch", *(extra_args or [])],
            cwd=self.dir, stdout=log_file, stderr=subprocess.STDOUT)
        for _ in range(240):
            if self.process.poll() is not None:
                raise SystemExit("ComfyUI stopped while starting (see comfyui.log)")
            try:
                self.api("/system_stats")
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(2)
        raise SystemExit("ComfyUI did not start")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def run(self, graph: dict, timeout: float = 4 * 3600) -> list[str]:
        """Queue a graph, wait, return saved image paths (sorted)."""
        try:
            prompt_id = self.api("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"workflow rejected: {exc.read().decode(errors='replace')[:2000]}")
        started = time.time()
        while time.time() - started < timeout:
            history = self.api(f"/history/{prompt_id}")
            if history and prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status.get("messages", []))[-2000:])
                images = []
                for output in entry.get("outputs", {}).values():
                    for img in output.get("images", []):
                        images.append(os.path.join(self.output_dir, img.get("subfolder", ""), img["filename"]))
                return sorted(images)
            time.sleep(5)
        raise RuntimeError("generation timed out")


def install_comfyui(comfy_dir: str) -> None:
    if not os.path.isdir(comfy_dir):
        run(["git", "clone", "--depth", "1", "--branch", COMFY_TAG, COMFY_REPO, comfy_dir])
    with open(os.path.join(comfy_dir, "requirements.txt"), encoding="utf-8") as fp:
        requirements = filter_requirements(fp.read().splitlines())
    run([sys.executable, "-m", "pip", "install", "-q", *requirements])
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        import torch

        run([sys.executable, "-m", "pip", "install", "-q", f"torchaudio=={torch.__version__.split('+')[0]}", "--no-deps"])


PRESERVED_PACKAGES = {"torch", "torchvision", "torchaudio"}


def filter_requirements(lines: list[str]) -> list[str]:
    kept = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[\s<>=!~;\[]", line, maxsplit=1)[0].lower().replace("_", "-")
        if name not in PRESERVED_PACKAGES:
            kept.append(line)
    return kept


def find_cached(name: str, cache_dirs: list[str]) -> str:
    for directory in cache_dirs:
        if not os.path.isdir(directory):
            continue
        for path in glob.glob(os.path.join(directory, "**", name), recursive=True):
            if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
                return path
    return ""


def ensure_models(names: list[str], comfy_dir: str, cache_dirs: list[str]) -> None:
    """Link cached weights (Kaggle dataset / Google Drive) or download from Hugging Face."""
    for name in names:
        url, folder = MODELS[name]
        target_dir = os.path.join(comfy_dir, "models", folder)
        target = os.path.join(target_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
            continue
        cached = find_cached(name, cache_dirs)
        if cached:
            log(f"model from cache: {name}")
            if os.path.lexists(target):
                os.remove(target)
            os.symlink(cached, target)
            continue
        log(f"downloading {name} ...")
        started = time.time()
        partial = target + ".part"
        download(url, partial)
        os.replace(partial, target)
        log(f"downloaded {name}: {os.path.getsize(target) / 1e9:.1f} GB in {(time.time() - started) / 60:.1f} min")


DOWNLOAD_ATTEMPTS = 8


def remote_size(url: str) -> int:
    """Size of the file behind ``url`` (Hugging Face sends it as X-Linked-Size), 0 if unknown."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers.get("X-Linked-Size") or response.headers.get("Content-Length") or 0)
    except (urllib.error.URLError, ValueError, OSError):
        return 0


def download(url: str, partial: str) -> None:
    """Download a big model file, resuming after any network drop.

    Big files from Hugging Face regularly lose the connection (curl exit 92: "HTTP/2 stream was
    not closed cleanly"), which curl's own --retry does not cover. Use HTTP/1.1, retry on every
    error, abort a stalled transfer, and resume the partial file until it is complete.
    """
    expected = remote_size(url)
    command = ["curl", "-L", "--fail", "--http1.1", "--retry", "5", "--retry-all-errors", "--retry-delay", "10",
               "--connect-timeout", "30", "--speed-limit", "1000000", "--speed-time", "120",
               "-C", "-", "-sS", "-o", partial, url]
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        code = subprocess.run(command).returncode
        size = os.path.getsize(partial) if os.path.exists(partial) else 0
        if (code == 0 and (not expected or size >= expected)) or (expected and size == expected):
            return
        log(f"download interrupted (curl exit {code}, {size / 1e9:.2f}"
            f"{f' of {expected / 1e9:.2f}' if expected else ''} GB); resuming, attempt {attempt + 1}")
        time.sleep(min(60, 10 * attempt))
    raise RuntimeError(f"download failed after {DOWNLOAD_ATTEMPTS} attempts: {url}")


def free_models(names: list[str], comfy_dir: str) -> None:
    """Delete downloaded (not cached/linked) weights to free disk between stages."""
    for name in names:
        path = os.path.join(comfy_dir, "models", MODELS[name][1], name)
        if os.path.isfile(path) and not os.path.islink(path):
            os.remove(path)


# --------------------------------------------------------------------------- media helpers
def stage_input(comfy: Comfy, path: str, name: str) -> str:
    os.makedirs(comfy.input_dir, exist_ok=True)
    target = os.path.join(comfy.input_dir, name)
    shutil.copyfile(path, target)
    return name


def fit_image(src: str, dst: str, width: int, height: int, pad_color=(128, 128, 128)) -> str:
    """Crop (photos) or pad (reference portraits) to the video aspect."""
    from PIL import Image

    img = Image.open(src).convert("RGB")
    target_ratio = width / height
    ratio = img.width / img.height
    if abs(ratio - target_ratio) < 0.01:
        out = img
    elif pad_color is None:
        if ratio > target_ratio:
            new_w = int(img.height * target_ratio)
            left = (img.width - new_w) // 2
            out = img.crop((left, 0, left + new_w, img.height))
        else:
            new_h = int(img.width / target_ratio)
            top = (img.height - new_h) // 3
            out = img.crop((0, top, img.width, top + new_h))
    else:
        canvas_w = max(img.width, int(img.height * target_ratio))
        canvas_h = max(img.height, int(img.width / target_ratio))
        out = Image.new("RGB", (canvas_w, canvas_h), pad_color)
        out.paste(img, ((canvas_w - img.width) // 2, canvas_h - img.height))
    scale = 1024 / max(out.size)
    out = out.resize((max(16, int(out.width * scale)), max(16, int(out.height * scale))), Image.LANCZOS)
    out.save(dst)
    return dst


def frames_to_mp4(frames: list[str], output: str, fps: int, audio: str | None = None,
                  duration: float | None = None) -> None:
    if not frames:
        raise RuntimeError("no frames")
    seq = output + ".frames"
    os.makedirs(seq, exist_ok=True)
    try:
        for i, frame in enumerate(frames, start=1):
            shutil.move(frame, os.path.join(seq, f"{i:06d}.png"))
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", os.path.join(seq, "%06d.png")]
        if audio:
            cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
        if duration:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", output + ".tmp.mp4"]
        run(cmd)
        os.replace(output + ".tmp.mp4", output)
    finally:
        shutil.rmtree(seq, ignore_errors=True)


def last_frame(video: str, dst: str) -> str:
    run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.1", "-i", video, "-frames:v", "1", dst])
    return dst


def concat_mp4(parts: list[str], output: str) -> None:
    listing = output + ".txt"
    with open(listing, "w", encoding="utf-8") as fp:
        for part in parts:
            fp.write(f"file '{part}'\n")
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", output])
    os.remove(listing)


# --------------------------------------------------------------------------- progress
class Progress:
    def __init__(self, out_dir: str, job: dict):
        self.path = os.path.join(out_dir, "progress.json")
        self.data = {"job_id": job.get("job_id", ""), "shots": {}, "started": time.time()}
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding="utf-8") as fp:
                    previous = json.load(fp)
                if previous.get("job_id") == self.data["job_id"]:
                    self.data["shots"] = previous.get("shots", {})
            except (OSError, ValueError):
                pass

    def set(self, shot_id: str, **fields) -> None:
        self.data["shots"].setdefault(shot_id, {}).update(fields)
        self.data["updated"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, indent=2)
        os.replace(tmp, self.path)

    def seconds_per_video_second(self) -> float | None:
        rates = [s["render_seconds"] / s["duration"] for s in self.data["shots"].values()
                 if s.get("status") == "done" and s.get("render_seconds") and s.get("duration")]
        return sum(rates) / len(rates) if rates else None


def restore_previous_outputs(out_dir: str, cache_dirs: list[str], job_id: str) -> int:
    """Copy finished shots from an earlier run's output attached as input."""
    restored = 0
    for directory in cache_dirs:
        for progress_path in glob.glob(os.path.join(directory, "**", "progress.json"), recursive=True):
            root = os.path.dirname(progress_path)
            if os.path.abspath(root) == os.path.abspath(out_dir):
                continue
            try:
                with open(progress_path, encoding="utf-8") as fp:
                    if json.load(fp).get("job_id") != job_id:
                        continue
            except (OSError, ValueError):
                continue
            for sub in ("shots", "frames"):
                for src in glob.glob(os.path.join(root, sub, "*")):
                    dst = os.path.join(out_dir, sub, os.path.basename(src))
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copyfile(src, dst)
                        restored += sub == "shots"
            if not os.path.exists(os.path.join(out_dir, "progress.json")):
                shutil.copyfile(progress_path, os.path.join(out_dir, "progress.json"))
    return restored


# --------------------------------------------------------------------------- stages
class Worker:
    def __init__(self, job: dict, out_dir: str, work_dir: str, comfy_dir: str, cache_dirs: list[str],
                 time_budget: float, comfy_args: list[str] | None = None):
        self.job = job
        self.root = job["_root"]
        self.out = out_dir
        self.work = work_dir
        self.cache_dirs = cache_dirs
        self.deadline = time.time() + time_budget
        self.comfy = Comfy(comfy_dir)
        self.comfy_args = comfy_args or []
        self.width, self.height = video_size(job)
        self.seed = int(job.get("settings", {}).get("seed", 2026))
        for sub in ("shots", "frames"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
        self.progress = Progress(out_dir, job)

    def time_left(self) -> float:
        return self.deadline - time.time()

    def shot_path(self, shot: dict) -> str:
        return os.path.join(self.out, "shots", f"{shot['id']}.mp4")

    def frame_path(self, shot: dict) -> str:
        return os.path.join(self.out, "frames", f"{shot['id']}.png")

    def pending(self) -> list[dict]:
        return [s for s in cloud_shots(self.job) if not os.path.isfile(self.shot_path(s))]

    # -- stage A: first frames ------------------------------------------------
    def make_frames(self) -> None:
        shots = [s for s in self.pending() if not os.path.isfile(self.frame_path(s))]
        self.make_ai_frames([s for s in shots if s["type"] == "AI_SCENE"])
        shots = [s for s in shots if s["type"] != "AI_SCENE"]
        if not shots:
            return
        ensure_models(IMAGE_MODELS, self.comfy.dir, self.cache_dirs)
        self.comfy.start(self.comfy_args)
        presenter = self.job.get("presenter") or {}
        refs = [os.path.join(self.root, p) for p in presenter.get("reference_images", [])]
        try:
            for shot in shots:
                if self.time_left() < 600:
                    log("time budget reached during first frames; stopping")
                    return
                started = time.time()
                prefix = f"frame_{shot['id']}"
                location = shot.get("location_image")
                location_path = os.path.join(self.root, location) if location else ""
                if shot["type"] == "BROLL" and location_path:
                    # A real photo of the place is the best first frame.
                    fit_image(location_path, self.frame_path(shot), self.width, self.height, pad_color=None)
                    self.progress.set(shot["id"], frame="photo")
                    continue
                if location_path:
                    img1 = fit_image(location_path, os.path.join(self.work, f"{prefix}_loc.png"),
                                     self.width, self.height, pad_color=None)
                    names = [stage_input(self.comfy, img1, f"{prefix}_1.png")]
                    names += [stage_input(self.comfy, p, f"{prefix}_ref{i}.png") for i, p in enumerate(refs[:2])]
                    prompt = placement_prompt(shot, presenter, True)
                else:
                    img1 = fit_image(refs[0], os.path.join(self.work, f"{prefix}_ref.png"), self.width, self.height)
                    names = [stage_input(self.comfy, img1, f"{prefix}_1.png")]
                    names += [stage_input(self.comfy, p, f"{prefix}_ref{i}.png") for i, p in enumerate(refs[1:3])]
                    prompt = broll_prompt(shot) if shot["type"] == "BROLL" else placement_prompt(shot, presenter, False)
                graph = build_edit_workflow(prompt, *(names + [None, None])[:3], prefix=prefix,
                                            seed=self.seed + int(re.sub(r"\D", "", shot["id"]) or 0))
                try:
                    images = self.comfy.run(graph)
                except RuntimeError as exc:
                    log(f"[{shot['id']}] first frame FAILED: {exc}")
                    self.progress.set(shot["id"], status="failed", error=str(exc)[-500:])
                    continue
                shutil.move(images[-1], self.frame_path(shot))
                self.progress.set(shot["id"], frame="generated", frame_seconds=round(time.time() - started, 1))
                log(f"[{shot['id']}] first frame ready in {time.time() - started:.0f}s")
        finally:
            self.comfy.stop()
            free_models(IMAGE_MODELS, self.comfy.dir)

    def make_ai_frames(self, shots: list[dict]) -> None:
        """First frames of AI_SCENE shots: Z-Image Turbo text-to-image (no presenter, no text)."""
        if not shots:
            return
        ensure_models(CREATE_MODELS, self.comfy.dir, self.cache_dirs)
        self.comfy.start(self.comfy_args)
        width, height = (1280, 720) if self.width > self.height else (720, 1280)
        try:
            for shot in shots:
                if self.time_left() < 600:
                    log("time budget reached during AI scene frames; stopping")
                    return
                started = time.time()
                try:
                    images = self.comfy.run(build_portrait_workflow(
                        ai_scene_prompt(shot), width, height, f"aiframe_{shot['id']}",
                        self.seed + int(re.sub(r"\D", "", shot["id"]) or 0)))
                except RuntimeError as exc:
                    log(f"[{shot['id']}] AI scene frame FAILED: {exc}")
                    self.progress.set(shot["id"], status="failed", error=str(exc)[-500:])
                    continue
                fit_image(images[-1], self.frame_path(shot), self.width, self.height, pad_color=None)
                os.remove(images[-1])
                self.progress.set(shot["id"], frame="text-to-image", frame_seconds=round(time.time() - started, 1))
        finally:
            self.comfy.stop()
            free_models(CREATE_MODELS, self.comfy.dir)

    # -- stage B: video ---------------------------------------------------------
    def make_videos(self) -> None:
        shots = [s for s in self.pending() if os.path.isfile(self.frame_path(s))]
        if not shots:
            return
        needs_talk = any(s["type"] in TALK_TYPES for s in shots)
        ensure_models(VIDEO_MODELS + (TALK_MODELS if needs_talk else []), self.comfy.dir, self.cache_dirs)
        self.comfy.start(self.comfy_args)
        try:
            for shot in shots:
                estimate = self.estimate(shot)
                if self.time_left() < max(900, (estimate or 0) * 1.3):
                    log(f"not enough session time left for shot {shot['id']}; stopping (rerun to continue)")
                    return
                started = time.time()
                log(f"[{shot['id']}] {shot['type']} {float(shot['duration']):.1f}s"
                    + (f" (estimate {estimate / 60:.0f} min)" if estimate else ""))
                self.progress.set(shot["id"], status="running", duration=float(shot["duration"]))
                try:
                    if shot["type"] in TALK_TYPES:
                        self.render_talk(shot)
                    else:
                        self.render_motion(shot)
                except RuntimeError as exc:
                    log(f"[{shot['id']}] FAILED: {exc}")
                    self.progress.set(shot["id"], status="failed", error=str(exc)[-500:])
                    continue
                spent = time.time() - started
                self.progress.set(shot["id"], status="done", render_seconds=round(spent, 1))
                log(f"[{shot['id']}] done in {spent / 60:.1f} min -> {self.shot_path(shot)}")
        finally:
            self.comfy.stop()

    def estimate(self, shot: dict) -> float | None:
        rate = self.progress.seconds_per_video_second()
        return rate * float(shot["duration"]) if rate else None

    def render_talk(self, shot: dict) -> None:
        audio_src = os.path.join(self.root, shot["audio"])
        audio_name = stage_input(self.comfy, audio_src, f"{shot['id']}{os.path.splitext(audio_src)[1]}")
        image_name = stage_input(self.comfy, self.frame_path(shot), f"start_{shot['id']}.png")
        segments, _ = talk_segments(float(shot["duration"]))
        graph = build_talk_workflow(image_name, audio_name, talk_prompt(shot), self.width, self.height,
                                    segments, f"talk_{shot['id']}", self.seed)
        frames = self.comfy.run(graph)
        frames_to_mp4(frames, self.shot_path(shot), TALK_FPS, audio=audio_src, duration=float(shot["duration"]))

    def render_motion(self, shot: dict) -> None:
        parts = []
        start = self.frame_path(shot)
        for index, length in enumerate(motion_chunks(float(shot["duration"]))):
            image_name = stage_input(self.comfy, start, f"start_{shot['id']}_{index}.png")
            graph = build_motion_workflow(image_name, motion_prompt(shot), self.width, self.height, length,
                                          f"motion_{shot['id']}_{index}", self.seed + index)
            frames = self.comfy.run(graph)
            if index:
                frames = frames[1:]  # the joining frame is already in the previous part
            part = os.path.join(self.work, f"{shot['id']}_{index}.mp4")
            frames_to_mp4(frames, part, MOTION_FPS)
            parts.append(part)
            start = last_frame(part, os.path.join(self.work, f"{shot['id']}_{index}_last.png"))
        if len(parts) == 1:
            shutil.move(parts[0], self.shot_path(shot) + ".tmp.mp4")
        else:
            concat_mp4(parts, self.shot_path(shot) + ".tmp.mp4")
        # Trim to the planned length and keep the narration if the shot has one.
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", self.shot_path(shot) + ".tmp.mp4"]
        audio = shot.get("audio")
        if audio:
            cmd += ["-i", os.path.join(self.root, audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac"]
        cmd += ["-t", f"{float(shot['duration']):.3f}", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                self.shot_path(shot)]
        run(cmd)
        os.remove(self.shot_path(shot) + ".tmp.mp4")

    # -- presenter creation -----------------------------------------------------
    def create_presenter(self) -> None:
        count = int(self.job.get("count", 4))
        ensure_models(CREATE_MODELS, self.comfy.dir, self.cache_dirs)
        self.comfy.start(self.comfy_args)
        out = os.path.join(self.out, "candidates")
        os.makedirs(out, exist_ok=True)
        try:
            for i in range(count):
                target = os.path.join(out, f"candidate_{i + 1}.png")
                if os.path.isfile(target):
                    continue
                # One knees-up photo per candidate: face and outfit both stay clear, and a
                # second, separately generated close-up would show a different person.
                images = self.comfy.run(build_portrait_workflow(
                    f"{self.job['prompt']}, three-quarter shot from the knees up, standing, facing the camera, "
                    "plain light grey studio background, photorealistic, natural skin texture, soft studio light",
                    832, 1216, f"cand{i}", self.seed + i))
                shutil.move(images[-1], target)
                log(f"candidate {i + 1}/{count} ready")
        finally:
            self.comfy.stop()


def summarize(worker: Worker) -> dict:
    shots = cloud_shots(worker.job)
    done = [s for s in shots if os.path.isfile(worker.shot_path(s))]
    summary = {"job_id": worker.job.get("job_id"), "total": len(shots), "done": len(done),
               "remaining": [s["id"] for s in shots if s not in done],
               "complete": len(done) == len(shots)}
    with open(os.path.join(worker.out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def main(argv=None) -> int:
    global log
    platform = detect_platform()
    paths = default_paths(platform)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="", help="folder containing job.json (default: search the inputs)")
    parser.add_argument("--out", default=paths["out"])
    parser.add_argument("--work", default=paths["work"])
    parser.add_argument("--comfy", default=paths["comfy"])
    parser.add_argument("--cache", action="append", default=list(paths["cache_dirs"]),
                        help="extra folder with cached model weights (repeatable)")
    parser.add_argument("--time-budget", type=float, default=(11.0 if platform == "kaggle" else 10.5) * 3600,
                        help="seconds of work before stopping cleanly")
    parser.add_argument("--lowvram", action="store_true", help="pass --lowvram to ComfyUI")
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args(argv)

    job_dir = args.job or ("/kaggle/input" if platform == "kaggle" else "")
    if not job_dir:
        raise SystemExit("No job found. Attach the job dataset or pass --job.")
    job = load_job(job_dir, args.work)
    args.cache.append(job["_root"])  # package/previous holds finished shots of an earlier run
    if platform != "kaggle" and args.out == paths["out"]:
        # A kept folder (Google Drive) holds many jobs; never mix their shots.
        args.out = os.path.join(args.out, job.get("job_id") or "job")
    os.makedirs(args.out, exist_ok=True)
    log = Log(os.path.join(args.out, "worker.log"))
    log(f"platform={platform} job={job.get('job_id')} kind={job.get('kind', 'video')}")

    if not args.skip_setup:
        install_comfyui(args.comfy)
    comfy_args = ["--lowvram"] if args.lowvram else []
    if job.get("kind") == "create_presenter":
        Worker(job, args.out, args.work, args.comfy, args.cache, args.time_budget, comfy_args).create_presenter()
        return 0

    # Before Worker() so the restored progress.json (timings) is loaded.
    restored = restore_previous_outputs(args.out, args.cache, job.get("job_id", ""))
    if restored:
        log(f"restored {restored} finished shots from the previous run")
    worker = Worker(job, args.out, args.work, args.comfy, args.cache, args.time_budget, comfy_args)
    worker.make_frames()
    worker.make_videos()
    summary = summarize(worker)
    log(f"{summary['done']}/{summary['total']} shots ready"
        + ("" if summary["complete"] else f"; rerun to continue: {summary['remaining']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
