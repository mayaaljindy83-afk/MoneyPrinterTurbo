"""Generate AI video clips on a free Kaggle GPU with ComfyUI + Wan 2.2 TI2V-5B.

Used by kaggle/ai_clips_wan22.ipynb (the notebook embeds this file).
One text prompt per line becomes one clip: 001.mp4, 002.mp4, ...

Model choice (September 2026, free Kaggle T4 16 GB / 29 GB RAM):
- Wan 2.2 TI2V-5B (Apache-2.0): best open model that fits a 16 GB card
  without heavy quantisation, native 24 fps, good motion and realism.
- FastWan LoRA (Kijai/WanVideo_comfy): cuts sampling from ~20 to ~8 steps
  at CFG 1, which is what makes a T4 usable (roughly 3-5x faster).
The 14B Wan models and LTX-2 need more VRAM/RAM than Kaggle offers.

Works on Kaggle (kaggle/ai_clips_wan22.ipynb) and on Google Colab
(kaggle/ai_clips_wan22_colab.ipynb). Colab's free tier has only ~12 GB RAM,
so the Colab notebook loads the video model with fp8 weights.

Run:  python wan_clips.py --prompts prompts.txt --out /kaggle/working/ai_clips
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

COMFY_TAG = "v0.37.4"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
# Models live outside /kaggle/working (its 20 GB are kept as notebook output).
COMFY_DIR = "/tmp/ComfyUI"
PORT = 8188

MODELS = {
    # (repo_id, file in repo, ComfyUI models sub-folder)
    "unet": ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
    "text_encoder": ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    "vae": ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors", "vae"),
    "fastwan_lora": ("Kijai/WanVideo_comfy", "FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors", "loras"),
}

SIZES = {
    # Multiples of 32. "hd" is the model's native 720p; "fast" is ~1.8x quicker.
    ("landscape", "hd"): (1280, 704),
    ("landscape", "fast"): (960, 544),
    ("portrait", "hd"): (704, 1280),
    ("portrait", "fast"): (544, 960),
}
FPS = 24

NEGATIVE = (
    "blurry, low quality, distorted, deformed, extra limbs, bad anatomy, text, "
    "watermark, logo, subtitles, jpeg artifacts, static, frozen frame, split screen"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(command: list[str], cwd: str | None = None) -> None:
    log("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def read_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fp:
        prompts = [line.strip() for line in fp if line.strip() and not line.strip().startswith("#")]
    if not prompts:
        raise SystemExit("No prompts found. Paste one prompt per line in the PROMPTS cell.")
    return prompts


def frames_for(seconds: float) -> int:
    """Wan needs 4*k+1 frames; round up so 5 s gives the model's native 121."""
    frames = max(5, int(round(seconds * FPS)))
    return -(-(frames - 1) // 4) * 4 + 1


WEIGHT_DTYPES = ("default", "fp8_e4m3fn")


def build_workflow(
    prompt: str,
    prefix: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    fast: bool = True,
    steps: int | None = None,
    weight_dtype: str = "default",
) -> dict:
    """ComfyUI API graph for Wan 2.2 TI2V-5B text-to-video (frames saved as PNG)."""
    unet = os.path.basename(MODELS["unet"][1])
    graph = {
        # fp8 halves the model's memory (needed on Colab's 12 GB RAM); the T4
        # still computes in fp16.
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": weight_dtype}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": os.path.basename(MODELS["text_encoder"][1]), "type": "wan"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["4", 0]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": os.path.basename(MODELS["vae"][1])}},
        "8": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["7", 0], "width": width, "height": height, "length": length, "batch_size": 1}},
        # Tiled decoding keeps the VAE inside 16 GB for 121 frames.
        "10": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["9", 0], "vae": ["7", 0], "tile_size": 512, "overlap": 64, "temporal_size": 32, "temporal_overlap": 8}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }
    model_ref = ["1", 0]
    if fast:
        graph["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_ref, "lora_name": os.path.basename(MODELS["fastwan_lora"][1]), "strength_model": 1.0}}
        model_ref = ["2", 0]
    graph["3"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model_ref, "shift": 8.0}}
    graph["9"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["3", 0],
            "seed": seed,
            # FastWan: ~8 steps, no CFG. Official template: 20 steps, CFG 5.
            "steps": steps or (8 if fast else 20),
            "cfg": 1.0 if fast else 5.0,
            "sampler_name": "euler" if fast else "uni_pc",
            "scheduler": "simple",
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["8", 0],
            "denoise": 1.0,
        },
    }
    return graph


# --------------------------------------------------------------------------- setup
def install_comfyui() -> None:
    if not os.path.isdir(COMFY_DIR):
        run(["git", "clone", "--depth", "1", "--branch", COMFY_TAG, COMFY_REPO, COMFY_DIR])
    with open(os.path.join(COMFY_DIR, "requirements.txt"), encoding="utf-8") as fp:
        requirements = filter_requirements(fp.read().splitlines())
    run([sys.executable, "-m", "pip", "install", "-q", *requirements, "huggingface_hub>=0.24"])
    ensure_torchaudio()


# Kaggle already has a CUDA build of PyTorch; never replace these.
PRESERVED_PACKAGES = {"torch", "torchvision", "torchaudio"}


def filter_requirements(lines: list[str]) -> list[str]:
    """Drop only the exact PyTorch packages (keep torchsde, torch-related extras...)."""
    import re

    kept = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[\s<>=!~;\[]", line, maxsplit=1)[0].lower().replace("_", "-")
        if name not in PRESERVED_PACKAGES:
            kept.append(line)
    return kept


def ensure_torchaudio() -> None:
    """ComfyUI imports torchaudio at start-up; add the build matching Kaggle's torch."""
    try:
        import torchaudio  # noqa: F401
        return
    except ImportError:
        pass
    import torch

    version = torch.__version__.split("+")[0]
    log(f"torchaudio missing, installing torchaudio=={version}")
    run([sys.executable, "-m", "pip", "install", "-q", f"torchaudio=={version}", "--no-deps"])


def download_models() -> None:
    from huggingface_hub import hf_hub_download

    for name, (repo, filename, folder) in MODELS.items():
        target_dir = os.path.join(COMFY_DIR, "models", folder)
        target = os.path.join(target_dir, os.path.basename(filename))
        if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
            log(f"model ready: {name}")
            continue
        log(f"downloading {name} from {repo} (first run only, several GB)...")
        cached = hf_hub_download(repo_id=repo, filename=filename, local_dir="/tmp/hf-download")
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(cached, target)
    shutil.rmtree("/tmp/hf-download", ignore_errors=True)


def start_server() -> subprocess.Popen:
    log_file = open("/tmp/comfyui.log", "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch"],
        cwd=COMFY_DIR, stdout=log_file, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
    )
    for _ in range(180):
        if process.poll() is not None:
            raise SystemExit("ComfyUI stopped while starting. See /tmp/comfyui.log")
        try:
            api("/system_stats")
            log("ComfyUI is running")
            return process
        except (urllib.error.URLError, ConnectionError):
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in 6 minutes. See /tmp/comfyui.log")


def api(path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read() or b"null")


def check_nodes(fast: bool) -> None:
    info = api("/object_info")
    needed = {"UNETLoader", "CLIPLoader", "CLIPTextEncode", "VAELoader", "Wan22ImageToVideoLatent",
              "ModelSamplingSD3", "KSampler", "VAEDecodeTiled", "SaveImage"}
    if fast:
        needed.add("LoraLoaderModelOnly")
    missing = sorted(needed - set(info))
    if missing:
        raise SystemExit(f"This ComfyUI version lacks nodes: {missing}")


# --------------------------------------------------------------------------- generation
def queue_and_wait(graph: dict, timeout: float = 3600) -> dict:
    client_id = uuid.uuid4().hex
    try:
        prompt_id = api("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI rejected the workflow: {exc.read().decode(errors='replace')[:1500]}")
    started = time.time()
    while time.time() - started < timeout:
        history = api(f"/history/{prompt_id}")
        if history and prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status.get("messages", []))[-1500:])
            return entry
        time.sleep(5)
    raise RuntimeError("generation timed out")


def frames_to_mp4(entry: dict, output_file: str) -> None:
    images = []
    for node_output in entry.get("outputs", {}).values():
        images.extend(node_output.get("images", []))
    if not images:
        raise RuntimeError("ComfyUI returned no frames")
    frames = sorted(
        os.path.join(COMFY_DIR, "output", img.get("subfolder", ""), img["filename"]) for img in images
    )
    # Renumber the frames so FFmpeg's image sequence input gives every frame
    # exactly 1/FPS seconds (a concat list drops the last frame's duration).
    sequence_dir = output_file + ".frames"
    os.makedirs(sequence_dir, exist_ok=True)
    for number, frame in enumerate(frames, start=1):
        shutil.move(frame, os.path.join(sequence_dir, f"{number:06d}.png"))
    try:
        run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS),
             "-i", os.path.join(sequence_dir, "%06d.png"),
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", output_file])
    finally:
        shutil.rmtree(sequence_dir, ignore_errors=True)


def lora_warning() -> None:
    try:
        with open("/tmp/comfyui.log", encoding="utf-8", errors="replace") as fp:
            unloaded = sum(1 for line in fp if "lora key not loaded" in line)
    except OSError:
        return
    if unloaded:
        log(f"WARNING: {unloaded} FastWan LoRA keys were not applied. If clips look noisy, set FAST = False.")


def generate(prompts: list[str], out_dir: str, aspect: str, quality: str, seconds: float,
             fast: bool, seed: int, steps: int | None, weight_dtype: str = "default") -> list[str]:
    width, height = SIZES[(aspect, quality)]
    length = frames_for(seconds)
    os.makedirs(out_dir, exist_ok=True)
    done = []
    log(f"{len(prompts)} clips, {width}x{height}, {length} frames ({length / FPS:.1f}s), fast={fast}")
    for index, prompt in enumerate(prompts, start=1):
        target = os.path.join(out_dir, f"{index:03d}.mp4")
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            log(f"[{index}/{len(prompts)}] already done, skipping")
            done.append(target)
            continue
        started = time.time()
        log(f"[{index}/{len(prompts)}] {prompt[:90]}")
        graph = build_workflow(
            prompt, f"clip_{index:03d}", width, height, length, seed + index, fast, steps, weight_dtype
        )
        try:
            entry = queue_and_wait(graph)
            frames_to_mp4(entry, target)
        except RuntimeError as exc:
            log(f"[{index}] FAILED: {exc}")
            continue
        if index == 1 and fast:
            lora_warning()
        done.append(target)
        log(f"[{index}/{len(prompts)}] done in {(time.time() - started) / 60:.1f} min -> {target}")
    return done


def default_work_dir() -> str:
    """Kaggle keeps /kaggle/working as output; Colab's files panel shows /content."""
    for folder in ("/kaggle/working", "/content"):
        if os.path.isdir(folder):
            return folder
    return os.getcwd()


def make_zip(files: list[str], zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
        for file in files:
            archive.write(file, os.path.basename(file))
    log(f"ZIP ready: {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", default=os.path.join(default_work_dir(), "ai_clips"))
    parser.add_argument("--aspect", choices=["landscape", "portrait"], default="landscape")
    parser.add_argument("--quality", choices=["hd", "fast"], default="fast")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--no-fast", action="store_true", help="official 20-step sampling (slower)")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight-dtype", choices=WEIGHT_DTYPES, default="default",
                        help="fp8_e4m3fn for low-RAM machines such as free Google Colab")
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args(argv)

    prompts = read_prompts(args.prompts)
    fast = not args.no_fast
    if not args.skip_setup:
        install_comfyui()
        download_models()
    server = start_server()
    try:
        check_nodes(fast)
        files = generate(prompts, args.out, args.aspect, args.quality, args.seconds, fast,
                         args.seed, args.steps, args.weight_dtype)
    finally:
        server.terminate()
    if not files:
        log("No clip was generated. Read the errors above and /tmp/comfyui.log")
        return 1
    make_zip(files, os.path.join(os.path.dirname(args.out.rstrip("/")) or ".", "ai_clips.zip"))
    log(f"{len(files)}/{len(prompts)} clips ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
