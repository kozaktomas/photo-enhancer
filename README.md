# Photo AI Processing Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

REST API for AI-powered photo enhancement: colorization, restoration, face restoration, and upscaling.

Built with FastAPI and PyTorch. Runs on CUDA GPU or CPU. Vibecoded with [Claude Code](https://claude.ai/code).

## Table of Contents

- [Models](#models)
- [Quick Start](#quick-start)
- [API](#api)
  - [Health Check](#health-check)
  - [Colorize](#colorize)
  - [Restore](#restore)
  - [Face Restore](#face-restore)
  - [Upscale](#upscale)
  - [Pipeline](#pipeline)
  - [HTTP Status Codes](#http-status-codes)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Notes](#notes)
- [Documentation](#documentation)

## Models

| Endpoint | Model | Variants |
|---|---|---|
| `/colorize` | [DDColor](https://github.com/piddnad/DDColor) | `modelscope` (default), `paper_tiny`, `artistic` |
| `/restore` | [NAFNet](https://github.com/megvii-research/NAFNet) | `denoise` (default), `deblur` |
| `/face-restore` | [CodeFormer](https://github.com/sczhou/CodeFormer) | `v0.1` (default) |
| `/upscale` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `x4plus` (default), `x4anime`, `x2plus` |

Model weights are downloaded automatically on first startup.

## Quick Start

### Docker (recommended)

```bash
docker compose up --build
```

GPU support requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). To run on CPU only, set `FORCE_CPU=true` in `docker-compose.yml`.

### Local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-gpu.txt   # or requirements-cpu.txt for CPU-only
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive API docs are auto-generated at [`/docs`](http://localhost:8000/docs) (Swagger UI) and [`/redoc`](http://localhost:8000/redoc) (ReDoc).

## API

All image endpoints accept a file upload via `multipart/form-data` and return the processed image directly in the response body.

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": ["colorize", "restore", "face", "upscale"],
  "cuda_info": {
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "vram_total_gb": 24.0,
    "vram_used_gb": 2.31
  }
}
```

### Colorize

Colorize a grayscale photo.

```bash
curl -X POST http://localhost:8000/colorize \
  -F "file=@photo.jpg" \
  -o colorized.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/colorize?render_factor=50&output_format=webp" \
  -F "file=@photo.jpg" \
  -o colorized.webp
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image |
| `render_factor` | int (1-100) | 35 | Controls colorization intensity |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Restore

Remove noise or blur from a photo.

```bash
curl -X POST http://localhost:8000/restore \
  -F "file=@noisy_photo.jpg" \
  -o restored.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/restore?tile_size=256&output_format=jpg" \
  -F "file=@noisy_photo.jpg" \
  -o restored.jpg
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image |
| `tile_size` | int (>=0) | 0 | Tile size for processing (0 = no tiling). Use tiling to reduce VRAM usage on large images. |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Face Restore

Enhance and restore faces in a photo.

```bash
curl -X POST http://localhost:8000/face-restore \
  -F "file=@portrait.jpg" \
  -o face_restored.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/face-restore?fidelity=0.7&upscale=2&output_format=png" \
  -F "file=@portrait.jpg" \
  -o face_restored.png
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image |
| `fidelity` | float (0.0-1.0) | 0.5 | Balance between quality and fidelity. Lower = better quality, higher = closer to input. |
| `upscale` | int (1-4) | 2 | Upscale factor for the output |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Upscale

Upscale an image using super-resolution.

```bash
curl -X POST http://localhost:8000/upscale \
  -F "file=@small_photo.jpg" \
  -o upscaled.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/upscale?scale=4&tile_size=512&output_format=webp" \
  -F "file=@small_photo.jpg" \
  -o upscaled.webp
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image |
| `scale` | int (1-8) | 4 | Upscale factor |
| `tile_size` | int (>=0) | 512 | Tile size for processing (0 = no tiling). Use tiling to reduce VRAM usage on large images. |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Pipeline

Run multiple processing steps in a single request. Order: colorize → restore → upscale → face restore.

```bash
curl -X POST "http://localhost:8000/pipeline?width=4000&output_format=png" \
  -F "file=@photo.jpg" \
  -o pipeline_output.png
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image |
| `colorize` | bool | true | Enable colorization step |
| `restore` | bool | true | Enable restoration step |
| `face_restore` | bool | true | Enable face restoration step |
| `upscale` | bool | true | Enable upscale step |
| `width` | int (>=1) | 2400 | Target output width |
| `height` | int (>=1) | 2400 | Target output height |
| `render_factor` | int (1-100) | 35 | Colorization intensity |
| `restore_tile_size` | int (>=0) | 0 | Tile size for restoration |
| `fidelity` | float (0.0-1.0) | 0.7 | Face restoration fidelity |
| `upscale_tile_size` | int (>=0) | 512 | Tile size for upscaling |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### HTTP Status Codes

| Code | Meaning | When |
|---|---|---|
| `200` | Success | Image processed and returned successfully |
| `400` | Bad Request | Invalid image file, unsupported format, or invalid parameter values |
| `500` | Internal Error | Unexpected error during model inference |
| `503` | Service Unavailable | Requested model failed to load at startup |

Error responses return JSON:

```json
{"detail": "Could not decode image — invalid or unsupported format"}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_COLORIZE` | `modelscope` | DDColor variant (`modelscope`, `paper_tiny`, `artistic`) |
| `MODEL_RESTORE` | `denoise` | NAFNet variant (`denoise`, `deblur`) |
| `MODEL_FACE` | `v0.1` | CodeFormer variant (`v0.1`) |
| `MODEL_UPSCALE` | `x4plus` | Real-ESRGAN variant (`x4plus`, `x4anime`, `x2plus`) |
| `FORCE_CPU` | `false` | Force CPU inference even if CUDA is available |

## Project Structure

```
photo-enhancer/
├── main.py                  # FastAPI app, endpoints, model lifecycle
├── models/
│   ├── wrappers.py          # Model wrapper classes (predict interface)
│   └── archs/               # Vendored PyTorch architecture definitions
│       ├── rrdbnet_arch.py   #   Real-ESRGAN (RRDBNet)
│       ├── nafnet_arch.py    #   NAFNet
│       ├── codeformer_arch.py#   CodeFormer
│       └── vqgan_arch.py     #   VQ-GAN autoencoder (used by CodeFormer)
├── utils/
│   ├── downloader.py        # Model weight downloader (HF/GitHub/GDrive)
│   └── image_ops.py         # Image read/validate/resize/encode
├── requirements.txt         # Shared deps (no torch)
├── requirements-cpu.txt     # CPU torch (used by Docker)
├── requirements-gpu.txt     # CUDA torch (used for local dev)
├── Dockerfile
├── docker-compose.yml
└── docs/                    # Extended documentation
    ├── api-reference.md
    ├── architecture.md
    ├── deployment.md
    ├── models.md
    └── development.md
```

## Architecture

All four models run real AI inference using vendored PyTorch architectures in `models/archs/`. Model architectures (block counts, widths, scale factors) are inferred from checkpoint keys at load time — no hard-coded config per variant. This means dropping in a different weight file for the same model family will just work.

Dependencies:
- **DDColor**: installed via pip (`ddcolor` package, uses bundled `basicsr==1.3.4.6`)
- **NAFNet, Real-ESRGAN, CodeFormer**: architectures vendored locally in `models/archs/` to avoid requiring `basicsr>=1.4.2` (which would conflict with DDColor)
- **CodeFormer** additionally uses `facexlib` for face detection and alignment

## Notes

- Images larger than 2048px on either side are automatically resized before processing.
- Model weights are stored in `/app/weights` (Docker) and cached across restarts via a named volume.
- Google Drive downloads (NAFNet weights) are handled via `drive.usercontent.google.com` to bypass virus-scan interstitials.

## Documentation

For deeper reading, see the [`docs/`](docs/) folder:

- **[API Reference](docs/api-reference.md)** — Full endpoint documentation with request/response examples
- **[Architecture](docs/architecture.md)** — System design, module breakdown, processing pipelines
- **[Deployment](docs/deployment.md)** — Docker setup, local dev, environment variables, troubleshooting
- **[Models](docs/models.md)** — Per-model deep dives, variants, papers, weight sources
- **[Development](docs/development.md)** — Contributing guide, adding new models, code conventions
