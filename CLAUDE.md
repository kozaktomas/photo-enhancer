# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REST API for AI-powered photo enhancement (colorization, restoration, face restoration, upscaling, old photo restoration) built with FastAPI and PyTorch. Runs on CUDA GPU or CPU.

## Workflow

**Always run `make check` after every change.** This runs both linting and tests. Do not consider a change complete until `make check` passes.

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

### Run tests
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Lint
```bash
ruff check . && ruff format --check .
```

### Test endpoints
```bash
curl -X POST http://localhost:8000/v1/colorize -F "file=@photo.jpg" -o colorized.png
```

API docs are auto-generated at `/docs` (Swagger) and `/redoc`.

## Architecture

### Request flow
`POST /v1/endpoint` → `check_file_size` (max 32 MB) → `read_image` → `validate_and_resize` (max 2048px) → model wrapper `.predict()` → `encode_image` → `Response`

### Key modules

- **`main.py`** — FastAPI app with lifespan startup/shutdown. Holds the global `model_registry` dict and `device` string. All 6 POST endpoints live on a `/v1` router and follow the same pattern: size check, decode image, validate, run model, encode output. Legacy un-prefixed paths redirect with 307. Includes `RequestLoggingMiddleware` for JSON request logging and Prometheus instrumentation via `prometheus-fastapi-instrumentator`.
- **`models/wrappers.py`** — Wrapper classes (`DDColorWrapper`, `NAFNetWrapper`, `CodeFormerWrapper`, `RealESRGANWrapper`, `OldPhotoRestoreWrapper`). Each takes `(model_path, device, variant)` and exposes `predict(image, **kwargs)` returning a numpy BGR array. NAFNet and RealESRGAN infer their architecture (block counts, width, scale) from checkpoint keys — no hard-coded configs per variant. CodeFormer uses `facexlib` for face detection/alignment. OldPhotoRestoreWrapper uses dlib and is a multi-file model (model_path is a directory).
- **`models/archs/`** — Vendored PyTorch `nn.Module` architecture definitions (`RRDBNet`, `NAFNet`, `VQAutoEncoder`, `CodeFormer`, `UNet` scratch detection, `GlobalGenerator_DCDCv2`/`Mapping_Model_with_mask_2` VAE+mapping, `SPADEGenerator` face enhancement). These avoid depending on `basicsr>=1.4.2` which would conflict with the DDColor dependency.
- **`utils/downloader.py`** — `ensure_model_exists(category, variant)` downloads single-file weights from HuggingFace/GitHub/Google Drive into `/app/weights/<category>/` with `.part` file handling. `ensure_model_files_exist(category, variant)` handles multi-file models (e.g. old photo restore with 6 files) and returns a directory path. Google Drive URLs are resolved via `drive.usercontent.google.com` to bypass virus-scan interstitials. Downloaded files are validated to reject corrupt HTML responses.
- **`utils/image_ops.py`** — `read_image(bytes)`, `validate_and_resize(img)`, `encode_image(img, format)`. All operate on numpy arrays via OpenCV.
- **`utils/logging.py`** — `setup_logging()` configures root logger with `python-json-logger` for JSON structured output to stdout. Fields: `timestamp`, `level`, `logger`, `message`, plus extras.

### Error handling pattern

All endpoints follow the same error convention:
- **413** — File exceeds 32 MB size limit. Returns `{"detail": "File too large (X.X MB). Maximum allowed size is 32 MB."}`.
- **400** — `ValueError` from image decoding (`read_image`), validation (`validate_and_resize`), encoding (`encode_image`), or model-specific input issues. Returns `{"detail": "<message>"}`.
- **503** — Model not present in `model_registry` (failed to load at startup). Returns `{"detail": "<Model> model not loaded"}`.
- **500** — Unexpected exception during `model.predict()`. Logged with `logger.exception()`, returns `{"detail": "Internal processing error"}`.

### Model loading

Models load during FastAPI lifespan startup. Variants are selected via environment variables (`MODEL_COLORIZE`, `MODEL_RESTORE`, `MODEL_FACE`, `MODEL_UPSCALE`, `MODEL_OLD_PHOTO`). `FORCE_CPU=true` overrides CUDA detection. Models with `"multi_file": True` in their config use `ensure_model_files_exist()` instead of `ensure_model_exists()`. Models that fail to load are skipped (endpoint returns 503).

### Adding a new model
1. Add a download entry in `utils/downloader.py` model registry (`MODEL_URLS` dict)
2. Add the PyTorch architecture module in `models/archs/` if not available via pip
3. Create a wrapper class in `models/wrappers.py` with `__init__(model_path, device, variant)` and `predict(image, **kwargs)` — prefer inferring architecture params from checkpoint keys over hard-coding per variant
4. Add config entry to `MODEL_CONFIG` in `main.py`
5. Add the POST endpoint to `v1_router` in `main.py` following the existing pattern (include `check_file_size`, `FileTooLargeError` handling)
6. Optionally add the legacy redirect path to `_LEGACY_PATHS`

## Dependencies

- **PyTorch**: Split into `requirements-cpu.txt` (CPU wheel index) and `requirements-gpu.txt` (CUDA 12.1 index). Docker uses CPU; local dev uses GPU.
- **DDColor**: Installed from git (`pip install --no-deps git+https://github.com/piddnad/DDColor.git`) with `--no-deps` because DDColor's pip package pulls `basicsr==1.3.4.6`.
- **Vendored architectures**: NAFNet, RealESRGAN (RRDBNet), CodeFormer, VQ-GAN, and Old Photo Restore (UNet, GlobalGenerator, MappingNet, SPADE) architectures are vendored in `models/archs/` to avoid requiring `basicsr>=1.4.2`, which would conflict with the DDColor dependency.
- **facexlib**: Used by CodeFormer for face detection (RetinaFace) and alignment. Downloads its own detection model weights on first use (stored in `TORCH_HOME`).
- **dlib**: Used by OldPhotoRestoreWrapper for face detection and 68-point landmark extraction. Requires `cmake` at build time.
- **python-json-logger**: JSON structured logging formatter.
- **prometheus-fastapi-instrumentator**: Auto-instruments routes and exposes `/metrics` endpoint.
- **Dev deps** (`requirements-dev.txt`): `pytest`, `httpx`, `ruff`.

## Docker

- **Base image**: `python:3.11-slim` with system deps for OpenCV (`libgl1-mesa-glx`, `libglib2.0-0`), `git` (for DDColor install), and `cmake` (for dlib compilation).
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
| `MODEL_OLD_PHOTO` | `v1` | Old Photo Restore variant: `v1` |
| `FORCE_CPU` | `false` | Force CPU even if CUDA is available |
| `TORCH_HOME` | `/app/weights/.torch` | PyTorch cache directory (set in Dockerfile) |
| `HF_HOME` | `/app/weights/.huggingface` | HuggingFace cache directory (set in Dockerfile) |

## Extended Documentation

See [`docs/`](docs/) for detailed guides: API reference, architecture deep dive, deployment, model guide, and development/contributing.
