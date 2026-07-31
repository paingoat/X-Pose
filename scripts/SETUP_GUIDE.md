# X-Pose / UniPose — Ubuntu Lab Setup (RTX 5090)

End-to-end guide: conda environment → CUDA ops → checkpoint download → Gradio.

Target stack:

- GPU: NVIDIA RTX 5090 (Blackwell, `sm_120`)
- Driver: R570 or newer
- CUDA toolkit: **12.8+** (needed to compile `MultiScaleDeformableAttention`)
- Conda env: `xpose`, Python `3.11`
- PyTorch: `2.8.0` + torchvision `0.23.0` from the **cu128** wheel index
- HF cache: `HF_HOME=/backup/data/art-gen` (see repo `.env`)

> Do **not** use conda `-y`. Newer conda blocks it; accept prompts manually in the terminal.

---

## 0. Prerequisites

On the Ubuntu lab machine:

1. NVIDIA driver installed (`nvidia-smi` works; compute capability should show `12.0` for 5090).
2. CUDA toolkit **≥ 12.8** with `nvcc` on `PATH`, or install later via conda (see step 2 notes).
3. Conda (Miniconda/Anaconda).
4. `git`, build tools (`build-essential`), and enough disk under `/backup/data/art-gen`.

Confirm GPU:

```bash
nvidia-smi
```

---

## 1. Get the repo and check `.env`

```bash
cd /path/to/MVP2   # this repository
cat .env
```

Expected:

```bash
HF_HOME=/backup/data/art-gen
HUGGINGFACE_HUB_CACHE=/backup/data/art-gen
TRANSFORMERS_CACHE=/backup/data/art-gen
```

Create the cache directory if needed:

```bash
mkdir -p /backup/data/art-gen
```

Make scripts executable once:

```bash
chmod +x scripts/01_setup_env.sh scripts/02_download_models.sh scripts/03_run_gradio.sh
```

---

## 2. Create env, install deps, build CUDA ops

```bash
bash scripts/01_setup_env.sh
```

What it does:

1. Creates conda env `xpose` with Python 3.11 (**manual Accept**).
2. Activates the env.
3. Installs `torch==2.8.0` and `torchvision==0.23.0` from `https://download.pytorch.org/whl/cu128`.
4. Installs the rest from `requirements.txt`.
5. Builds `models/UniPose/ops` with `TORCH_CUDA_ARCH_LIST=12.0`.
6. Runs `python test.py` (all checks should be `True`).
7. Prints torch / device / capability verification.

If `nvcc` is missing, the script stops with a hint. Install toolkit NVCC (manual Accept, no `-y`):

```bash
eval "$(conda shell.bash hook)"
conda activate xpose
conda install -c nvidia cuda-nvcc=12.8
# then re-run:
bash scripts/01_setup_env.sh
```

Also ensure `CUDA_HOME` points at the toolkit root if it is not inferred automatically.

---

## 3. Download models

```bash
bash scripts/02_download_models.sh
```

What it does:

1. Sources `.env` and ensures `HF_HOME=/backup/data/art-gen` exists.
2. Downloads Swin-T checkpoint from Google Drive (`13gANvGWyWApMFTAtC3ntrMgx0fOocjIa`).
3. Places:
   - `./unipose_swint.pth` — used by Gradio (`app.py`)
   - `weights/unipose_swint.pth` — symlink/copy for CLI (README path)
4. Pre-warms OpenAI CLIP `ViT-B/32`.

If Google Drive rate-limits `gdown`, download manually from the README link and copy the file to `./unipose_swint.pth`, then re-run the script (it will skip the download and still create the `weights/` link).

---

## 4. Run Gradio

```bash
bash scripts/03_run_gradio.sh
```

This activates `xpose`, loads `.env`, checks the checkpoint, and runs:

```bash
python app.py
```

Gradio launches with `share=True` (public share URL printed in the terminal). Open the local URL or the share link, upload an image, set an instance prompt (e.g. `person`), and click **Run**.

---

## 5. Optional: CLI inference

With the env active and `.env` sourced:

```bash
eval "$(conda shell.bash hook)"
conda activate xpose
set -a; source .env; set +a

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
  -c config_model/UniPose_SwinT.py \
  -p weights/unipose_swint.pth \
  -i /path/to/image.jpg \
  -o outputs \
  -t "person"
```

Use a keypoint skeleton from `predefined_keypoints.py` via `-k` when needed.

> Note: the upstream README mentions `config/UniPose_SwinT.py`; in this repo the config is `config_model/UniPose_SwinT.py`.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `CUDA capability sm_120 is not compatible` | Wrong torch wheel. Reinstall from cu128 index (script step 2). Need PyTorch ≥ 2.7 built with CUDA 12.8. |
| Ops build fails / `THC/...` errors | Ensure you have the updated ops in this repo and CUDA toolkit ≥ 12.8; clean `models/UniPose/ops/build` and rebuild. |
| `Cuda is not availabel` during ops setup | `torch.cuda.is_available()` is false or `CUDA_HOME` unset. Fix driver / toolkit first. |
| `nvcc: command not found` | Install `cuda-nvcc=12.8` (conda) or system CUDA 12.8+. |
| Checkpoint missing when starting Gradio | Run `bash scripts/02_download_models.sh`. |
| Gradio Image / Button API errors | Use Gradio `4.44.1` from `requirements.txt` (do not upgrade to Gradio 5/6 without migrating UI). |
| HF downloads go to home instead of lab path | Confirm `.env` is sourced; `echo $HF_HOME` should be `/backup/data/art-gen`. |

---

## Quick command checklist

```bash
chmod +x scripts/*.sh
bash scripts/01_setup_env.sh      # accept conda prompts manually
bash scripts/02_download_models.sh
bash scripts/03_run_gradio.sh
```
