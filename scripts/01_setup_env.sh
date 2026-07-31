#!/usr/bin/env bash
# Create/activate env, ensure cu128 PyTorch, install deps, build MultiScaleDeformableAttention.
# Works on bare Ubuntu lab (RTX 5090) and RunPod (e.g. pytorch:…-torch280 + A4500).
# NOTE: Do not pass -y to conda (manual Accept required on newer conda).
#
# Optional env vars:
#   XPOSE_USE_BASE_ENV=1     Prefer current Python if it is 3.10–3.12.
#                            If base is 3.13+ (common on newer RunPod images), the script
#                            falls back to conda env `xpose` with Python 3.11.
#   FORCE_TORCH_REINSTALL=1  Always reinstall torch/torchvision from cu128 index.
#   TORCH_VERSION / TORCHVISION_VERSION  Override pins (optional).
set -euo pipefail

ENV_NAME="xpose"
PYTHON_VERSION="3.11"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
XPOSE_USE_BASE_ENV="${XPOSE_USE_BASE_ENV:-0}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo " X-Pose / UniPose setup (CUDA 12.8 / PyTorch 2.8+)"
echo " Repo: ${REPO_ROOT}"
echo "============================================================"

python_mm() {
  python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

select_torch_pins() {
  # torch 2.8 supports Python <=3.13; cu128 wheels for 3.14 start at 2.9+/2.11.
  local mm
  mm="$(python_mm)"
  if [[ -n "${TORCH_VERSION:-}" && -n "${TORCHVISION_VERSION:-}" ]]; then
    echo "Using override TORCH_VERSION=${TORCH_VERSION} TORCHVISION_VERSION=${TORCHVISION_VERSION}"
    return
  fi
  case "${mm}" in
    3.10|3.11|3.12)
      TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
      ;;
    *)
      # Python 3.13+/3.14 on RunPod base: use current cu128 stable pair
      TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
      ;;
  esac
  echo "Python ${mm} → torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} (cu128)"
}

ensure_conda_xpose() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found, and current Python is unsuitable." >&2
    echo "Install Miniconda or use a Python 3.10–3.12 interpreter." >&2
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
}

activate_python() {
  if [[ "${XPOSE_USE_BASE_ENV}" == "1" ]]; then
    local mm
    mm="$(python_mm)"
    case "${mm}" in
      3.10|3.11|3.12)
        echo "XPOSE_USE_BASE_ENV=1 → using current Python ${mm}."
        echo "Active Python: $(which python)"
        python --version
        return
        ;;
      *)
        echo "WARNING: XPOSE_USE_BASE_ENV=1 but current Python is ${mm}."
        echo "torch==2.8.0 has no wheels for this Python; Gradio 4.44 is also happier on 3.11."
        echo "Falling back to conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})."
        ensure_conda_xpose
        echo "Active Python: $(which python)"
        python --version
        return
        ;;
    esac
  fi

  ensure_conda_xpose
  echo "Active Python: $(which python)"
  python --version
}

activate_python
select_torch_pins

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
ver = torch.__version__.split("+")[0]
print("found torch", torch.__version__, "torchvision", torchvision.__version__)
parts = [int(x) for x in ver.split(".")[:2]]
if parts[0] < 2 or (parts[0] == 2 and parts[1] < 8):
    print("need torch >= 2.8")
    sys.exit(1)
if not torch.cuda.is_available():
    print("cuda not available in this python")
    sys.exit(1)
print("cuda", torch.version.cuda)
sys.exit(0)
PY
}

install_torch() {
  echo "Installing PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} (cu128)..."
  if ! pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${TORCH_INDEX_URL}"; then
    echo "Pinned install failed; falling back to latest cu128 torch/torchvision..."
    pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
  fi
}

echo ""
if [[ "${FORCE_TORCH_REINSTALL}" == "1" ]]; then
  echo "FORCE_TORCH_REINSTALL=1"
  install_torch
elif torch_ok; then
  echo "Compatible torch already present — skipping torch/torchvision reinstall."
  echo "(Set FORCE_TORCH_REINSTALL=1 to force a cu128 reinstall.)"
else
  install_torch
fi

echo ""
echo "Installing remaining Python dependencies from requirements.txt..."
pip install -r "${REPO_ROOT}/requirements.txt"

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
rm -rf "${OPS_DIR}/build" "${OPS_DIR}/dist" "${OPS_DIR}"/*.egg-info 2>/dev/null || true
cd "${OPS_DIR}"
python setup.py build install

echo ""
echo "Running ops unit test (forward + small-channel gradcheck)..."
echo "Note: large D=1025+ double gradcheck is skipped by default (VRAM-heavy on A4500)."
# Avoid fragmenting VRAM before tests; large gradcheck opt-in via MSDA_TEST_LARGE_GRAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
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
echo "  Preferred on this RunPod image: unset XPOSE_USE_BASE_ENV  # use conda xpose (py3.11)"
echo "  Force torch reinstall: FORCE_TORCH_REINSTALL=1 bash scripts/01_setup_env.sh"
