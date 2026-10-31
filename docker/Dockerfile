# syntax=docker/dockerfile:1
# ============================================================================
# VRM Hair Maker — fully self-contained image
#
# Everything is baked in at BUILD time; the container needs no network:
#   - HairDeck code + checkpoints (image/chain/card models) + data
#   - third-party weights: DINOv3, SAM 3, BiRefNet-matting (+ their sources)
#   - Blender 4.2.23 LTS (linux-x64) + the VRM Add-on
#   - the built web frontend (three-vrm preview)
#   - the default VRM avatar + idle animation + example portraits
#   - ffmpeg + all python deps (torch cu128 wheels bundle their own CUDA runtime)
#
# Build (from the HairDeck folder, the parent of hairmaker/):
#   docker build -t vrm-hairmaker -f hairmaker/docker/Dockerfile .
#
# Run (needs NVIDIA Container Toolkit for the GPU):
#   docker run --gpus all -p 8300:8300 vrm-hairmaker
# then open http://localhost:8300
# ============================================================================

############################  stage 1: web frontend  #########################
FROM node:22-bookworm-slim AS web
WORKDIR /build
COPY hairmaker/web/package.json hairmaker/web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY hairmaker/web/ ./
RUN npm run build

############################  stage 2: runtime  #############################
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HAIRDECK_ROOT=/opt/HairDeck/model \
    HAIRDECK_BLENDER=/opt/HairDeck/hairmaker/tools/blender/blender-4.2.23-linux-x64/blender \
    BLENDER_USER_EXTENSIONS=/opt/HairDeck/hairmaker/tools/blender/user/extensions \
    BLENDER_USER_CONFIG=/opt/HairDeck/hairmaker/tools/blender/user/config \
    BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION=true

# ffmpeg (head-turn video), GL/EGL libs (offscreen renders), X libs (headless blender)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libegl1 libopengl0 libglib2.0-0 libxrender1 \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libice6 \
        curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/HairDeck

# ---- HairDeck source tree (hairdeck package + the blender/ scripts it drives) ----
COPY code/ /opt/HairDeck/code/

# ---- the workspace: checkpoints, data, third-party weights and sources ----
COPY model/ /opt/HairDeck/model/

# ---- the app itself ----
COPY hairmaker/server/ /opt/HairDeck/hairmaker/server/
COPY hairmaker/blender/ /opt/HairDeck/hairmaker/blender/
COPY hairmaker/data/ /opt/HairDeck/hairmaker/data/
COPY hairmaker/static/ /opt/HairDeck/hairmaker/static/
COPY --from=web /build/dist/ /opt/HairDeck/hairmaker/web/dist/

# ---- Blender 4.2.23 LTS (linux-x64) + the VRM Add-on shipped with the workspace ----
RUN curl -fsSL -o /tmp/blender.tar.xz \
        https://download.blender.org/release/Blender4.2/blender-4.2.23-linux-x64.tar.xz \
    && tar -xJf /tmp/blender.tar.xz -C /opt/HairDeck/hairmaker/tools/blender/ \
    && rm /tmp/blender.tar.xz
COPY hairmaker/tools/blender/user/ /opt/HairDeck/hairmaker/tools/blender/user/

# ---- python dependencies (torch cu128 wheels bundle their own CUDA runtime) ----
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir \
        numpy==2.5.4 scipy==1.16.3 trimesh==5.1.1 pillow==12.0.0 \
        opencv-python-headless==5.0.0.93 safetensors==0.8.0 "huggingface_hub>=1.16" \
        setuptools==80.9.0 timm==1.0.22 iopath==0.1.10 ftfy==6.1.1 regex==2025.11.3 \
        pycocotools==2.0.10 einops==0.8.1 transformers==5.20.2 kornia==0.8.3 \
        torchmetrics==1.9.0 omegaconf==2.4.0 submitit==1.5.4 termcolor==3.3.0 \
        PyOpenGL==3.1.7 freetype-py==2.5.1 pyglet==2.1.19 networkx==3.6.1 \
        six==1.17.0 imageio==2.37.2 scikit-image==0.25.2 scikit-learn==1.8.0 \
        matplotlib==3.10.8 psutil==7.1.3 tqdm==4.67.1 PyYAML==6.0.3 \
        fastapi "uvicorn[standard]" python-multipart \
 && pip install --no-cache-dir --no-deps pyrender==0.1.45
# (triton, required by SAM 3's EDT kernel, is installed automatically as a torch dependency on linux)

EXPOSE 8300
WORKDIR /opt/HairDeck/hairmaker
CMD ["python", "-m", "server.main", "--host", "0.0.0.0", "--port", "8300"]
