# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# VA-AI-CAC – Dockerfile
# 
# Container CUDA: 11.8.0 + cuDNN 8
# Python: 3.10 (Ubuntu 22.04)
# PyTorch: 2.0.1 (compiled for CUDA 11.8)
# Flask API: port 25000
#
# Compatible with NVIDIA driver 535+ (supports CUDA 12.x via backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Virtual environment ───────────────────────────────────────────────────────
ENV VENV_PATH=/app/.venv
RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

# ── Python dependencies ───────────────────────────────────────────────────────
# Pin setuptools<81 to avoid pkg_resources deprecation warning
# Cache buster to force rebuild: 2026-07-02-v3
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    # PyTorch 2.0.1 for CUDA 11.8 (CRITICAL - must match driver compatibility)
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1 \
        torchvision==0.15.2 \
        torchmetrics==1.5.2 && \
    # Core numerical libraries, medical imaging, MONAI, and web framework in one layer
    pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        scikit-learn==1.3.2 \
        pandas==2.0.3 \
        matplotlib==3.7.2 \
        ipywidgets==7.8.1 \
        Pillow==10.1.0 \
        itk==5.3.0 \
        SimpleITK==2.4.1 \
        nibabel==5.3.2 \
        pydicom==2.4.4 \
        python-gdcm==3.0.24.1 \
        opencv-python==4.8.1.78 \
        "monai==1.4.0" \
        "einops>=0.6.0" \
        tensorboard \
        psutil \
        cucim \
        openslide-python \
        tqdm \
        lmdb \
        Flask>=2.0.0 \
        werkzeug>=2.0.0 \
        boto3>=1.35.0 && \
    # CRITICAL: Force reinstall correct PyTorch version (MONAI may have upgraded it)
    pip install --no-cache-dir --force-reinstall --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1 \
        torchvision==0.15.2

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p \
    storage/incoming \
    storage/outputs/masks \
    storage/outputs/debug \
    model

# ── Model weights ─────────────────────────────────────────────────────────────
# Copy model if it exists locally, otherwise download it
RUN if [ -f model/va_non_gated_ai_cac_model.pth ]; then \
        echo "Model already exists in build context: $(du -sh model/va_non_gated_ai_cac_model.pth)"; \
    else \
        MODEL_URL="https://github.com/Raffi-Hagopian/AI-CAC/releases/download/v1.0.0/va_non_gated_ai_cac_model.pth" && \
        echo "Model not found, downloading from GitHub..." && \
        wget -q --show-progress -O /app/model/va_non_gated_ai_cac_model.pth "$MODEL_URL" && \
        echo "Model downloaded: $(du -sh /app/model/va_non_gated_ai_cac_model.pth)"; \
    fi

# ── Environment ───────────────────────────────────────────────────────────────
ENV MODEL_CHECKPOINT_FILE=model/va_non_gated_ai_cac_model.pth \
    MPLBACKEND=Agg \
    FLASK_ENV=production

EXPOSE 25000

CMD ["python", "app.py"]