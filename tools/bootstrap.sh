#!/usr/bin/env bash
set -euo pipefail
ROBOTWIN_INSTALL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROBOTWIN_INSTALL_ROOT"
mkdir -p reports vendor
MAMBA_BIN="${MAMBA_BIN:-$HOME/.local/bin/micromamba}"
if [[ ! -x "$MAMBA_BIN" ]]; then
  mkdir -p "$(dirname "$MAMBA_BIN")"
  curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/robotwin-micromamba.tar.bz2
  tar -xjf /tmp/robotwin-micromamba.tar.bz2 -C "$(dirname "$MAMBA_BIN")" --strip-components=1 bin/micromamba
fi
if [[ ! -x .env/bin/python ]]; then
  "$MAMBA_BIN" create -y -p "$ROBOTWIN_INSTALL_ROOT/.env" -c conda-forge python=3.10 pip ffmpeg cmake ninja numpy=1.26
fi
"$MAMBA_BIN" install -y -p "$ROBOTWIN_INSTALL_ROOT/.env" -c nvidia/label/cuda-12.1.1 -c conda-forge cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1
source tools/activate.sh
if [[ ! -d vendor/RoboTwin/.git ]]; then
  git clone --branch stable_2.0 https://github.com/RoboTwin-Platform/RoboTwin.git vendor/RoboTwin
  git -C vendor/RoboTwin checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
fi
if [[ ! -d vendor/curobo/.git ]]; then
  git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git vendor/curobo
  git -C vendor/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
fi
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r configs/requirements-sim.txt
python -m pip install -e vendor/curobo --no-build-isolation -c configs/requirements-sim.txt
python -m pip install -e . --no-build-isolation
python tools/patch_robotwin.py
python tools/download_assets.py
(cd vendor/RoboTwin && python script/update_embodiment_config_path.py)
python tools/doctor.py
python -m pytest -q
python -m pip freeze > reports/pip-freeze.txt
