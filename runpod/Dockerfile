# RunPod Serverless image for the presenter worker (build context: the repository root).
# Models are not baked in: they download once to the endpoint's network volume.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 MPT_COMFY_DIR=/comfy
RUN apt-get update && apt-get install -y --no-install-recommends git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY kaggle/presenter_worker.py runpod/handler.py /app/
# Same ComfyUI version and packages the Kaggle run uses (torch from the base image is kept).
RUN python -c "import presenter_worker as w; w.install_comfyui('/comfy')" \
    && pip install --no-cache-dir runpod==1.12.0 \
    && rm -rf /root/.cache/pip

CMD ["python", "-u", "/app/handler.py"]
