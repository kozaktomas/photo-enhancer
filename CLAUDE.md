# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REST API for AI-powered photo enhancement (colorization, restoration, face restoration, upscaling) built with FastAPI and PyTorch. Runs on CUDA GPU or CPU.

## Commands

### Run locally
```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-gpu.txt   # or requirements-cpu.txt for CPU-only

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Run with Docker (recommended)
```bash
docker compose up --build
```

### Test endpoints
```bash
curl -X POST http://localhost:8000/colorize -F "file=@photo.jpg" -o colorized.png
```

API docs are auto-generated at `/docs` (Swagger) and `/redoc`.

## Architecture

### Request flow
`POST /endpoint` → `read_image` → `validate_and_resize` (max 2048px) → model wrapper `.predict()` → `encode_image` → `Response`

### Key modules

- **`main.py`** — FastAPI app with lifespan startup/shutdown. Holds the global `model_registry` dict and `device` string. All 4 POST endpoints follow the same pattern: decode image, validate, run model, encode output.
- **`models/wrappers.py`** — Wrapper classes (`DDColorWrapper`, `NAFNetWrapper`, `CodeFormerWrapper`, `RealESRGANWrapper`). Each takes `(model_path, device, variant)` and exposes `predict(image, **kwargs)` returning a numpy BGR array. NAFNet and RealESRGAN infer their architecture (block counts, width, scale) from checkpoint keys — no hard-coded configs per variant. CodeFormer uses `facexlib` for face detection/alignment.
- **`models/archs/`** — Vendored PyTorch `nn.Module` architecture definitions (`RRDBNet`, `NAFNet`, `VQAutoEncoder`, `CodeFormer`). These avoid depending on `basicsr>=1.4.2` which would conflict with the DDColor dependency.
- **`utils/downloader.py`** — `ensure_model_exists(category, variant)` downloads weights from HuggingFace/GitHub/Google Drive into `/app/weights/<category>/` with `.part` file handling. Google Drive URLs are resolved via `drive.usercontent.google.com` to bypass virus-scan interstitials. Downloaded files are validated to reject corrupt HTML responses.
- **`utils/image_ops.py`** — `read_image(bytes)`, `validate_and_resize(img)`, `encode_image(img, format)`. All operate on numpy arrays via OpenCV.

### Error handling pattern

All endpoints follow the same error convention:
- **400** — `ValueError` from image decoding (`read_image`), validation (`validate_and_resize`), encoding (`encode_image`), or model-specific input issues. Returns `{"detail": "<message>"}`.
- **503** — Model not present in `model_registry` (failed to load at startup). Returns `{"detail": "<Model> model not loaded"}`.
- **500** — Unexpected exception during `model.predict()`. Logged with full traceback, returns `{"detail": "Internal processing error"}`.

### Model loading

Models load during FastAPI lifespan startup. Variants are selected via environment variables (`MODEL_COLORIZE`, `MODEL_RESTORE`, `MODEL_FACE`, `MODEL_UPSCALE`). `FORCE_CPU=true` overrides CUDA detection. Models that fail to load are skipped (endpoint returns 503).

### Adding a new model
1. Add a download entry in `utils/downloader.py` model registry (`MODEL_URLS` dict)
2. Add the PyTorch architecture module in `models/archs/` if not available via pip
3. Create a wrapper class in `models/wrappers.py` with `__init__(model_path, device, variant)` and `predict(image, **kwargs)` — prefer inferring architecture params from checkpoint keys over hard-coding per variant
4. Add config entry to `MODEL_CONFIG` in `main.py`
5. Add the POST endpoint in `main.py` following the existing pattern

## Dependencies

- **PyTorch**: Split into `requirements-cpu.txt` (CPU wheel index) and `requirements-gpu.txt` (CUDA 12.1 index). Docker uses CPU; local dev uses GPU.
- **DDColor**: Installed from git (`pip install --no-deps git+https://github.com/piddnad/DDColor.git`) with `--no-deps` because DDColor's pip package pulls `basicsr==1.3.4.6`.
- **Vendored architectures**: NAFNet, RealESRGAN (RRDBNet), CodeFormer, and VQ-GAN architectures are vendored in `models/archs/` to avoid requiring `basicsr>=1.4.2`, which would conflict with the DDColor dependency.
- **facexlib**: Used by CodeFormer for face detection (RetinaFace) and alignment. Downloads its own detection model weights on first use (stored in `TORCH_HOME`).

## Docker

- **Base image**: `python:3.11-slim` with system deps for OpenCV (`libgl1-mesa-glx`, `libglib2.0-0`) and `git` (for DDColor install).
- **Layer strategy**: `requirements.txt` and `requirements-cpu.txt` are copied and installed before `COPY .` to leverage Docker layer caching for dependencies.
- **Compose volume**: `model-weights` named volume mounted at `/app/weights` persists downloaded model weights across container rebuilds.
- **GPU reservations**: Compose `deploy.resources.reservations.devices` reserves all NVIDIA GPUs via the NVIDIA Container Toolkit.
- **Environment variables**: `TORCH_HOME` and `HF_HOME` are pointed to `/app/weights/.torch` and `/app/weights/.huggingface` so all cached files live in the persistent volume.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_COLORIZE` | `modelscope` | DDColor variant: `modelscope`, `paper_tiny`, `artistic` |
| `MODEL_RESTORE` | `denoise` | NAFNet variant: `denoise`, `deblur` |
| `MODEL_FACE` | `v0.1` | CodeFormer variant: `v0.1` |
| `MODEL_UPSCALE` | `x4plus` | Real-ESRGAN variant: `x4plus`, `x4anime`, `x2plus` |
| `FORCE_CPU` | `false` | Force CPU even if CUDA is available |
| `TORCH_HOME` | `/app/weights/.torch` | PyTorch cache directory (set in Dockerfile) |
| `HF_HOME` | `/app/weights/.huggingface` | HuggingFace cache directory (set in Dockerfile) |

## Extended Documentation

See [`docs/`](docs/) for detailed guides: API reference, architecture deep dive, deployment, model guide, and development/contributing.
