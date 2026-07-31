#!/usr/bin/env bash
# Create conda env, install cu128 PyTorch + deps, build MultiScaleDeformableAttention.
# NOTE: Do not pass -y to conda (manual Accept required on newer conda).
set -euo pipefail

ENV_NAME="xpose"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo " X-Pose / UniPose setup for RTX 5090 (CUDA 12.8 / sm_120)"
echo " Repo: ${REPO_ROOT}"
echo "============================================================"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda and retry." >&2
  exit 1
fi

# Ensure conda activate works in non-interactive shells.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Skipping create."
else
  echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  echo ">>> Accept the conda prompt manually (no -y flag)."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
echo "Active Python: $(which python)"
python --version

echo ""
echo "Checking NVIDIA driver / GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || nvidia-smi
else
  echo "WARNING: nvidia-smi not found. GPU driver may be missing."
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo ""
  echo "WARNING: nvcc not found on PATH."
  echo "CUDA toolkit >= 12.8 is required to build MultiScaleDeformableAttention."
  echo "If needed, run (accept prompt manually, no -y):"
  echo "  conda install -c nvidia cuda-nvcc=12.8"
  echo "Then re-run this script."
  if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "ERROR: CUDA_HOME is also unset. Set CUDA_HOME to your CUDA toolkit and retry." >&2
    exit 1
  fi
else
  echo "nvcc: $(nvcc --version | tail -n 1)"
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  # Common layouts when nvcc is on PATH.
  NVCC_BIN="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
  echo "Inferred CUDA_HOME=${CUDA_HOME}"
fi

echo ""
echo "Installing PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} (cu128)..."
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo ""
echo "Installing remaining Python dependencies from requirements.txt..."
pip install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo "Building MultiScaleDeformableAttention CUDA ops for arch 12.0 (Blackwell)..."
export TORCH_CUDA_ARCH_LIST="12.0"
OPS_DIR="${REPO_ROOT}/models/UniPose/ops"
cd "${OPS_DIR}"
python setup.py build install

echo ""
echo "Running ops unit test (expect all checks True)..."
python test.py
cd "${REPO_ROOT}"

echo ""
echo "Verifying torch CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    print(f"capability: sm_{major}{minor}")
else:
    raise SystemExit("ERROR: torch.cuda.is_available() is False")
PY

echo ""
echo "Setup complete."
echo "Next: bash scripts/02_download_models.sh"
