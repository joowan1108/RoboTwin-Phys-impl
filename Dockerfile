# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST=7.5
ARG MAX_JOBS=4
ARG ROBOTWIN_UID=1001
ARG ROBOTWIN_GID=1001

# Driver libraries are injected by NVIDIA Container Toolkit at runtime.
# These apt operations run inside the image, never on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv build-essential cmake ninja-build \
    git curl ca-certificates pkg-config ffmpeg libvulkan1 vulkan-tools \
    libgl1 libegl1 libglvnd0 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

ENV ROBOTWIN_PHYS_ROOT=/home/guest/robotwin-workspace \
    VIRTUAL_ENV=/home/guest/robotwin-workspace/.env \
    CUDA_HOME=/usr/local/cuda \
    HOME=/home/guest \
    PYTHONPATH=/home/guest/robotwin-workspace/src \
    PATH=/home/guest/robotwin-workspace/.env/bin:/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    SAPIEN_RENDER_DEVICE=cuda:0 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/home/guest/.cache \
    TORCH_EXTENSIONS_DIR=/home/guest/.cache/torch_extensions \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /home/guest/robotwin-workspace
COPY docker/requirements.lock docker/requirements.lock
RUN python3.10 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir -c docker/requirements.lock pip setuptools wheel \
    && python -m pip install --no-cache-dir -r docker/requirements.lock \
       --extra-index-url https://download.pytorch.org/whl/cu121

# Rebuild CuRobo CUDA extensions; host .so/.o/build artifacts are excluded.
# Explicit architecture selection allows docker build without GPU access.
COPY vendor/curobo/ vendor/curobo/
RUN test "$(git -C vendor/curobo rev-parse HEAD)" = d64c4b005459db10c5dd867d8b30a87d5bda9bdb \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
       python -m pip install --no-cache-dir --no-build-isolation --no-deps -e vendor/curobo

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY tools/ tools/
COPY tests/ tests/
COPY docs/ docs/
COPY docker/ docker/
COPY cluster/ cluster/
COPY references/*-info.json references/
COPY references/assets-tree.json references/
COPY vendor/RoboTwin/ vendor/RoboTwin/
# Adapter source is included for config validation; learned model environments
# and all checkpoints remain separate from the simulator image.
COPY vendor/GalaxeaVLA/ vendor/GalaxeaVLA/
COPY vendor/FastWAM/ vendor/FastWAM/
COPY vendor/FACT/ vendor/FACT/
COPY vendor/Motus/ vendor/Motus/

RUN mkdir -p references reports runs manifests checkpoints vendor/RoboTwin/assets /opt/robotwin-build \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps -e . \
    && python tools/patch_robotwin.py \
    && python -m pip check \
    && python docker/verify_image.py --output /opt/robotwin-build/environment.json \
    && python -m pytest -q -p no:cacheprovider \
    && python -m pip freeze > /opt/robotwin-build/pip-freeze.txt \
    && dpkg-query -W > /opt/robotwin-build/apt-packages.txt \
    && chmod +x docker/entrypoint.sh \
    && groupadd --gid "$ROBOTWIN_GID" robotwin \
    && useradd --uid "$ROBOTWIN_UID" --gid "$ROBOTWIN_GID" --home-dir /home/guest --no-create-home --shell /bin/bash robotwin \
    && mkdir -p /home/guest/.cache \
    && chown "$ROBOTWIN_UID:$ROBOTWIN_GID" /home/guest /home/guest/.cache reports runs manifests checkpoints \
    && git config --system --add safe.directory "$ROBOTWIN_PHYS_ROOT/vendor/RoboTwin" \
    && git config --system --add safe.directory "$ROBOTWIN_PHYS_ROOT/vendor/curobo"

LABEL org.opencontainers.image.title="RoboTwin-Phys independent reproduction simulator" \
      org.opencontainers.image.description="CUDA 12.1 / torch 2.4.1 / SAPIEN 3.0.0b1; datasets and learned policies external"
USER robotwin
ENTRYPOINT ["/home/guest/robotwin-workspace/docker/entrypoint.sh"]
CMD ["robotwin-phys", "tasks"]
