#!/usr/bin/env bash
# Download UniPose Swin-T checkpoint and optionally warm CLIP cache.
set -euo pipefail

ENV_NAME="xpose"
GDRIVE_FILE_ID="13gANvGWyWApMFTAtC3ntrMgx0fOocjIa"
CHECKPOINT_NAME="unipose_swint.pth"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
  echo "Loaded .env (HF_HOME=${HF_HOME:-unset})"
else
  echo "WARNING: ${REPO_ROOT}/.env not found. Using defaults / existing env vars."
fi

HF_HOME="${HF_HOME:-/backup/data/art-gen}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}"
mkdir -p "${REPO_ROOT}/weights"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda activate "${ENV_NAME}"
  else
    echo "WARNING: conda env '${ENV_NAME}' not found. Using current Python: $(which python)"
  fi
fi

if ! python -c "import gdown" >/dev/null 2>&1; then
  echo "Installing gdown..."
  pip install gdown
fi

ROOT_CKPT="${REPO_ROOT}/${CHECKPOINT_NAME}"
WEIGHTS_CKPT="${REPO_ROOT}/weights/${CHECKPOINT_NAME}"

if [[ -f "${ROOT_CKPT}" ]]; then
  echo "Checkpoint already present: ${ROOT_CKPT}"
else
  echo "Downloading Swin-T checkpoint from Google Drive (${GDRIVE_FILE_ID})..."
  gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "${ROOT_CKPT}"
fi

if [[ ! -f "${ROOT_CKPT}" ]]; then
  echo "ERROR: download failed; ${ROOT_CKPT} missing." >&2
  exit 1
fi

# Gradio uses ./unipose_swint.pth; CLI README expects weights/unipose_swint.pth
if [[ -e "${WEIGHTS_CKPT}" || -L "${WEIGHTS_CKPT}" ]]; then
  echo "weights/ copy already present: ${WEIGHTS_CKPT}"
else
  ln -s "${ROOT_CKPT}" "${WEIGHTS_CKPT}" 2>/dev/null || cp "${ROOT_CKPT}" "${WEIGHTS_CKPT}"
  echo "Linked/copied checkpoint to ${WEIGHTS_CKPT}"
fi

echo ""
echo "Pre-warming OpenAI CLIP ViT-B/32 (used by UniPose text encoding)..."
python - <<'PY'
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
print("CLIP ViT-B/32 ready on", device)
del model
PY

echo ""
echo "Model download complete."
echo "  Gradio path : ${ROOT_CKPT}"
echo "  CLI path    : ${WEIGHTS_CKPT}"
echo "Next: bash scripts/03_run_gradio.sh"
