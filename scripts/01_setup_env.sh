#!/usr/bin/env bash
# Create/activate env, ensure cu128 PyTorch, install deps, build MultiScaleDeformableAttention.
# Works on bare Ubuntu lab (RTX 5090) and RunPod (e.g. pytorch:1.0.2-cu1281-torch280 + A4500).
# NOTE: Do not pass -y to conda (manual Accept required on newer conda).
#
# Optional env vars:
#   XPOSE_USE_BASE_ENV=1   Skip conda; use current Python (recommended on RunPod images
#                          that already ship torch 2.8 + cu128).
#   FORCE_TORCH_REINSTALL=1  Always reinstall torch/torchvision from cu128 index.
set -euo pipefail

ENV_NAME="xpose"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
XPOSE_USE_BASE_ENV="${XPOSE_USE_BASE_ENV:-0}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo " X-Pose / UniPose setup (CUDA 12.8 / PyTorch 2.8)"
echo " Repo: ${REPO_ROOT}"
echo "============================================================"

activate_python() {
  if [[ "${XPOSE_USE_BASE_ENV}" == "1" ]]; then
    echo "XPOSE_USE_BASE_ENV=1 → using current Python (no conda env)."
    echo "Active Python: $(which python)"
    python --version
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. On RunPod, either install Miniconda or re-run with:" >&2
    echo "  XPOSE_USE_BASE_ENV=1 bash scripts/01_setup_env.sh" >&2
    exit 1
  fi

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
}

activate_python

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
  echo "A CUDA toolkit (nvcc) is required to build MultiScaleDeformableAttention."
  echo "On conda envs, you can install (accept prompt manually, no -y):"
  echo "  conda install -c nvidia cuda-nvcc=12.8"
  echo "On RunPod devel images, ensure the CUDA toolkit is on PATH / set CUDA_HOME."
  if [[ -z "${CUDA_HOME:-}" ]]; then
    # Common RunPod / container locations
    for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12; do
      if [[ -x "${candidate}/bin/nvcc" ]]; then
        export CUDA_HOME="${candidate}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        echo "Found nvcc via ${CUDA_HOME}"
        break
      fi
    done
  fi
  if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDA_HOME:-}" ]]; then
    echo "ERROR: CUDA_HOME unset and nvcc not found. Fix toolkit, then re-run." >&2
    exit 1
  fi
else
  echo "nvcc: $(nvcc --version | tail -n 1)"
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  NVCC_BIN="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
  echo "Inferred CUDA_HOME=${CUDA_HOME}"
fi

torch_ok() {
  python - <<'PY'
import sys
try:
    import torch
    import torchvision
except Exception as e:
    print("missing:", e)
    sys.exit(1)
ver = torch.__version__
print("found torch", ver, "torchvision", torchvision.__version__)
if not ver.startswith("2.8."):
    print("need torch 2.8.x")
    sys.exit(1)
if not torch.cuda.is_available():
    print("cuda not available in this python")
    sys.exit(1)
# Prefer cu128 builds; accept any 2.8.x with working CUDA on this GPU.
print("cuda", torch.version.cuda)
sys.exit(0)
PY
}

echo ""
if [[ "${FORCE_TORCH_REINSTALL}" == "1" ]]; then
  echo "FORCE_TORCH_REINSTALL=1 → installing PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} (cu128)..."
  pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
elif torch_ok; then
  echo "Compatible torch already present — skipping torch/torchvision reinstall."
  echo "(Set FORCE_TORCH_REINSTALL=1 to force a cu128 reinstall.)"
else
  echo "Installing PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} (cu128)..."
  pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
fi

echo ""
echo "Installing remaining Python dependencies from requirements.txt..."
pip install -r "${REPO_ROOT}/requirements.txt"

# Detect GPU arch for TORCH_CUDA_ARCH_LIST (A4500=8.6, 5090=12.0, ...)
ARCH_LIST="$(
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("8.6")
    raise SystemExit
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${ARCH_LIST}}"
echo ""
echo "Building MultiScaleDeformableAttention with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}..."
OPS_DIR="${REPO_ROOT}/models/UniPose/ops"
# Clean stale build artifacts from previous arch / torch versions
rm -rf "${OPS_DIR}/build" "${OPS_DIR}/dist" "${OPS_DIR}"/*.egg-info 2>/dev/null || true
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
echo ""
echo "Tips:"
echo "  RunPod (reuse image Python/torch): XPOSE_USE_BASE_ENV=1 bash scripts/01_setup_env.sh"
echo "  Force torch reinstall:             FORCE_TORCH_REINSTALL=1 bash scripts/01_setup_env.sh"
