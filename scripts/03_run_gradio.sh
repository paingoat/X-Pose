#!/usr/bin/env bash
# Launch the UniPose Gradio demo with lab HF cache settings.
set -euo pipefail

ENV_NAME="xpose"
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
  echo "WARNING: ${REPO_ROOT}/.env not found."
fi

HF_HOME="${HF_HOME:-/backup/data/art-gen}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

CKPT="${REPO_ROOT}/${CHECKPOINT_NAME}"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint missing: ${CKPT}" >&2
  echo "Run: bash scripts/02_download_models.sh" >&2
  exit 1
fi

echo "Starting Gradio (share=True) ..."
echo "  python: $(which python)"
echo "  ckpt  : ${CKPT}"
echo "  HF_HOME=${HF_HOME}"
python "${REPO_ROOT}/app.py"
