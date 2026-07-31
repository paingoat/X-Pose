# X-Pose / UniPose — Setup Guide (Lab / RunPod)

End-to-end: environment → CUDA ops → checkpoint download → Gradio.

Supported targets:

| Target | Image / host | GPU | Notes |
|--------|--------------|-----|--------|
| RunPod (current) | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | RTX A4500 (`sm_86`) | Prefer `XPOSE_USE_BASE_ENV=1` to reuse image torch |
| Ubuntu lab | bare metal + conda | RTX 5090 (`sm_120`) | Create conda env `xpose` |

Stack:

- PyTorch **2.8.0** + torchvision **0.23.0**, CUDA **12.8** (`cu128`)
- Conda env (lab): `xpose`, Python `3.11`
- HF cache: `HF_HOME=/backup/data/art-gen` (see `.env`)

> Do **not** use conda `-y`. Accept prompts manually.

---

## A. RunPod (recommended while lab is unavailable)

Template: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`  
GPU example: **RTX A4500** (`sm_86`)

**Important:** On current pods, conda `base` is often **Python 3.14** with **no torch**.  
`torch==2.8.0` has **no** wheels for 3.14 (cu128 only lists 2.9+). Prefer the conda env `xpose` with **Python 3.11**.

```bash
cd /workspace/X-Pose   # or your clone path
chmod +x scripts/*.sh
mkdir -p /backup/data/art-gen

# Do NOT use base py3.14 — leave XPOSE_USE_BASE_ENV unset
unset XPOSE_USE_BASE_ENV
bash scripts/01_setup_env.sh       # creates/uses conda env xpose (py3.11); accept prompts
bash scripts/02_download_models.sh
bash scripts/03_run_gradio.sh
```

If you still export `XPOSE_USE_BASE_ENV=1` on py3.14, the script **auto-falls back** to conda `xpose` (py3.11) and installs `torch==2.8.0+cu128`.

What the setup does on RunPod:

- Uses conda env `xpose` / Python 3.11 (unless base is 3.10–3.12 with working torch).
- Installs torch 2.8 + torchvision 0.23 from cu128 (or skips if already OK).
- Auto-sets `TORCH_CUDA_ARCH_LIST` from the GPU (A4500 → `8.6`).

If you already have env `xpose` from an earlier attempt:

```bash
unset XPOSE_USE_BASE_ENV
bash scripts/01_setup_env.sh   # reuse xpose; skip torch if OK; fix Pillow via requirements
```

Force a clean torch reinstall (only if needed):

```bash
FORCE_TORCH_REINSTALL=1 bash scripts/01_setup_env.sh
```

Expose Gradio: use the share URL printed by `app.py`, or map the pod HTTP port.

---

## B. Ubuntu lab (RTX 5090)

### 0. Prerequisites

1. Driver R570+ (`nvidia-smi`; capability `12.0` for 5090).
2. CUDA toolkit **≥ 12.8** with `nvcc`, or install via conda later.
3. Conda, `git`, `build-essential`.
4. Disk for `/backup/data/art-gen`.

### 1. Repo + `.env`

```bash
cd /path/to/repo
cat .env   # HF_HOME=/backup/data/art-gen
mkdir -p /backup/data/art-gen
chmod +x scripts/*.sh
```

### 2. Setup

```bash
bash scripts/01_setup_env.sh      # accept conda prompts manually
bash scripts/02_download_models.sh
bash scripts/03_run_gradio.sh
```

`01_setup_env.sh` will:

1. Create/activate conda env `xpose` (Python 3.11).
2. Install or reuse torch 2.8 + torchvision 0.23 (cu128).
3. `pip install -r requirements.txt`.
4. Build ops with detected arch (`12.0` on 5090).
5. Run `python test.py`.

If `nvcc` is missing:

```bash
eval "$(conda shell.bash hook)"
conda activate xpose
conda install -c nvidia cuda-nvcc=12.8   # no -y
bash scripts/01_setup_env.sh
```

---

## Download models

```bash
bash scripts/02_download_models.sh
# RunPod:
# XPOSE_USE_BASE_ENV=1 bash scripts/02_download_models.sh
```

Places:

- `./unipose_swint.pth` — Gradio (`app.py`)
- `weights/unipose_swint.pth` — CLI

Also pre-warms CLIP `ViT-B/32`.

If Google Drive rate-limits `gdown`, download manually from the README link into `./unipose_swint.pth` and re-run.

---

## Gradio

```bash
bash scripts/03_run_gradio.sh
# RunPod:
# XPOSE_USE_BASE_ENV=1 bash scripts/03_run_gradio.sh
```

---

## Optional CLI inference

```bash
# lab
eval "$(conda shell.bash hook)" && conda activate xpose
set -a; source .env; set +a

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
  -c config_model/UniPose_SwinT.py \
  -p weights/unipose_swint.pth \
  -i /path/to/image.jpg \
  -o outputs \
  -t "person"
```

Config path in this repo is `config_model/UniPose_SwinT.py` (not `config/`).

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `Pillow==11.1.0` vs `gradio ... pillow<11.0` | Pull latest `requirements.txt` (`Pillow==10.4.0`) and re-run setup. |
| `ResolutionImpossible` / dependency conflict | Use updated `requirements.txt`; avoid manually pinning Pillow ≥11 with Gradio 4.44.1. |
| `No matching distribution for torch==2.8.0` on RunPod | conda `base` is Python 3.14. Unset `XPOSE_USE_BASE_ENV` and use conda env `xpose` (py3.11). |
| Re-downloading multi‑GB nvidia-* wheels | Script skips reinstall when torch ≥2.8 + CUDA already works in the active env. |
| `CUDA capability sm_120 is not compatible` | Need PyTorch ≥2.7 **cu128**. Reinstall with `FORCE_TORCH_REINSTALL=1`. |
| Ops build fails / wrong arch | Script auto-detects GPU arch. Override with `TORCH_CUDA_ARCH_LIST=8.6` (A4500) or `12.0` (5090). Clean `models/UniPose/ops/build` and rebuild. |
| `test.py` CUDA OOM on `check_gradient_numerical(D=1025)` | Expected on 16–24GB GPUs: double `gradcheck` is huge. Ops already OK if forward checks are `True`. Pull latest `ops/test.py` (skips large D by default). Re-run only the test: `cd models/UniPose/ops && python test.py` |
| `nvcc: command not found` | Install toolkit / `cuda-nvcc=12.8`, or set `CUDA_HOME` (RunPod often `/usr/local/cuda`). |
| Checkpoint missing | `bash scripts/02_download_models.sh` |
| HF cache not under `/backup/data/art-gen` | `source .env` / use the run scripts; `echo $HF_HOME`. |

---

## Quick checklists

**RunPod**

```bash
unset XPOSE_USE_BASE_ENV
chmod +x scripts/*.sh
bash scripts/01_setup_env.sh      # conda xpose / py3.11 (accept prompts)
bash scripts/02_download_models.sh
bash scripts/03_run_gradio.sh
```

**Lab (5090 + conda)**

```bash
chmod +x scripts/*.sh
bash scripts/01_setup_env.sh
bash scripts/02_download_models.sh
bash scripts/03_run_gradio.sh
```
