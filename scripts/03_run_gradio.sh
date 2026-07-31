#!/usr/bin/env bash
# Launch the UniPose Gradio demo with HF cache settings.
# Optional: XPOSE_USE_BASE_ENV=1 to skip conda activate (RunPod base Python).
set -euo pipefail

ENV_NAME="xpose"
XPOSE_USE_BASE_ENV="${XPOSE_USE_BASE_ENV:-0}"
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
# Prefer HF_HOME only (TRANSFORMERS_CACHE is deprecated in transformers v5)
unset TRANSFORMERS_CACHE || true
mkdir -p "${HF_HOME}"

if [[ "${XPOSE_USE_BASE_ENV}" == "1" ]]; then
  echo "XPOSE_USE_BASE_ENV=1 → using current Python: $(which python)"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. On RunPod, re-run with:" >&2
    echo "  XPOSE_USE_BASE_ENV=1 bash scripts/03_run_gradio.sh" >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
fi

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
