#!/usr/bin/env bash
# Source this file. Do not alter the system NVIDIA driver or system CUDA.
ROBOTWIN_PHYS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOTWIN_PHYS_ROOT
export PATH="$ROBOTWIN_PHYS_ROOT/.env/bin:$PATH"
if [[ -f /.dockerenv && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda
else
  export CUDA_HOME="$ROBOTWIN_PHYS_ROOT/.env"
fi
export TORCH_CUDA_ARCH_LIST="7.5"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="$ROBOTWIN_PHYS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
