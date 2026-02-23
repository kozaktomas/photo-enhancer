# Photo AI Processing Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

REST API for AI-powered photo enhancement: colorization, restoration, face restoration, upscaling, old photo restoration, and inpainting.

Built with FastAPI and PyTorch. Runs on CUDA GPU or CPU. Vibecoded with [Claude Code](https://claude.ai/code).

## Table of Contents

- [Models](#models)
- [Quick Start](#quick-start)
- [API](#api)
  - [Health Check](#health-check)
  - [Metrics](#metrics)
  - [Colorize](#colorize)
  - [Restore](#restore)
  - [Face Restore](#face-restore)
  - [Upscale](#upscale)
  - [Old Photo Restore](#old-photo-restore)
  - [Inpaint](#inpaint)
  - [Pipeline](#pipeline)
  - [HTTP Status Codes](#http-status-codes)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Notes](#notes)
- [Documentation](#documentation)

## Models

| Endpoint | Model | Variants |
|---|---|---|
| `/v1/colorize` | [DDColor](https://github.com/piddnad/DDColor) | `modelscope` (default), `paper_tiny`, `artistic` |
| `/v1/restore` | [NAFNet](https://github.com/megvii-research/NAFNet) | `denoise` (default), `deblur` |
| `/v1/face-restore` | [CodeFormer](https://github.com/sczhou/CodeFormer) | `v0.1` (default) |
| `/v1/upscale` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `x4plus` (default), `x4anime`, `x2plus` |
| `/v1/old-photo-restore` | [Old Photo Restoration](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) | `v1` (default) |
| `/v1/inpaint` | [LaMa](https://github.com/advimman/lama) | `big` (default) |

Model weights are downloaded automatically on first startup.

## Quick Start

### Docker (recommended)

A pre-built image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/kozaktomas/photo-enhancer:main
```

Or build locally:

```bash
docker compose up --build
```

The Docker image uses CPU-only PyTorch. For GPU support, you'll need to modify the `Dockerfile` to install `requirements-gpu.txt` instead and use a CUDA base image.

**Estimated disk usage:**

| Component | Size |
|---|---|
| Docker image (Python + PyTorch CPU + deps) | ~3.6 GB |
| Model weights (downloaded on first startup) | ~2.6 GB |
| **Total** | **~6.2 GB** |

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

All image endpoints accept a file upload via `multipart/form-data` and return the processed image directly in the response body. Maximum upload size is **32 MB**.

Endpoints are versioned under `/v1`. Legacy un-prefixed paths (`/colorize`, etc.) redirect to their `/v1` equivalents with a 307 status.

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": ["colorize", "restore", "face", "upscale", "old_photo_restore", "inpaint"],
  "cuda_info": {
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "vram_total_gb": 24.0,
    "vram_used_gb": 2.31
  }
}
```

### Metrics

Prometheus metrics in text exposition format:

```bash
curl http://localhost:8000/metrics
```

### Colorize

Colorize a grayscale photo.

```bash
curl -X POST http://localhost:8000/v1/colorize \
  -F "file=@photo.jpg" \
  -o colorized.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/v1/colorize?render_factor=50&output_format=webp" \
  -F "file=@photo.jpg" \
  -o colorized.webp
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `render_factor` | int (1-100) | 35 | Controls colorization intensity |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Restore

Remove noise or blur from a photo.

```bash
curl -X POST http://localhost:8000/v1/restore \
  -F "file=@noisy_photo.jpg" \
  -o restored.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/v1/restore?tile_size=256&output_format=jpg" \
  -F "file=@noisy_photo.jpg" \
  -o restored.jpg
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `tile_size` | int (>=0) | 0 | Tile size for processing (0 = no tiling). Use tiling to reduce VRAM usage on large images. |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Face Restore

Enhance and restore faces in a photo.

```bash
curl -X POST http://localhost:8000/v1/face-restore \
  -F "file=@portrait.jpg" \
  -o face_restored.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/v1/face-restore?fidelity=0.7&upscale=2&output_format=png" \
  -F "file=@portrait.jpg" \
  -o face_restored.png
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `fidelity` | float (0.0-1.0) | 0.5 | Balance between quality and fidelity. Lower = better quality, higher = closer to input. |
| `upscale` | int (1-4) | 2 | Upscale factor for the output |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Upscale

Upscale an image using super-resolution.

```bash
curl -X POST http://localhost:8000/v1/upscale \
  -F "file=@small_photo.jpg" \
  -o upscaled.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/v1/upscale?scale=4&tile_size=512&output_format=webp" \
  -F "file=@small_photo.jpg" \
  -o upscaled.webp
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `scale` | int (1-8) | 4 | Upscale factor |
| `tile_size` | int (>=0) | 512 | Tile size for processing (0 = no tiling). Use tiling to reduce VRAM usage on large images. |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Old Photo Restore

Restore old or damaged photos. Detects and repairs scratches, restores global quality, and optionally enhances faces.

```bash
curl -X POST http://localhost:8000/v1/old-photo-restore \
  -F "file=@old_photo.jpg" \
  -o restored.png
```

With parameters:

```bash
curl -X POST "http://localhost:8000/v1/old-photo-restore?with_scratch=true&with_face=true&scratch_threshold=0.4&output_format=png" \
  -F "file=@old_photo.jpg" \
  -o restored.png
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `with_scratch` | bool | true | Enable automatic scratch detection and repair |
| `with_face` | bool | true | Enable face enhancement |
| `scratch_threshold` | float (0.0-1.0) | 0.4 | Sensitivity for scratch detection |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Inpaint

Fill in a polygon-shaped region of an image (remove objects, repair damage). Define the region to inpaint as polygon points.

```bash
curl -g -X POST "http://localhost:8000/v1/inpaint?points=[[100,100],[400,100],[400,400],[100,400]]" \
  -F "file=@photo.jpg" \
  -o inpainted.png
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `points` | string (JSON) | required | JSON array of `[x,y]` points defining the polygon to inpaint (at least 3 points) |
| `output_format` | string | png | Output format: `png`, `jpg`, `jpeg`, `webp` |

### Pipeline

Run multiple processing steps in a single request. Order: old_photo_restore -> colorize -> restore -> upscale -> face restore.

```bash
curl -X POST "http://localhost:8000/v1/pipeline?width=4000&output_format=png" \
  -F "file=@photo.jpg" \
  -o pipeline_output.png
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Input image (max 32 MB) |
| `old_photo_restore` | bool | false | Enable old photo restoration step |
| `colorize` | bool | true | Enable colorization step |
| `restore` | bool | true | Enable restoration step |
| `face_restore` | bool | true | Enable face restoration step |
| `upscale` | bool | true | Enable upscale step |
| `width` | int (>=1) | 2400 | Target output width |
| `height` | int (>=1) | 2400 | Target output height |
| `with_scratch` | bool | true | Enable scratch detection (old photo restore) |
| `scratch_threshold` | float (0.0-1.0) | 0.4 | Scratch detection sensitivity (old photo restore) |
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
| `413` | Payload Too Large | Uploaded file exceeds the 32 MB limit |
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
| `MODEL_OLD_PHOTO` | `v1` | Old Photo Restore variant (`v1`) |
| `MODEL_INPAINT` | `big` | LaMa inpainting variant (`big`) |
| `FORCE_CPU` | `false` | Force CPU inference even if CUDA is available |

## Testing

```bash
# Using make (recommended)
make test          # run tests
make lint          # check linting + formatting
make check         # both lint and test

# Or manually
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check . && ruff format --check .
```

## Project Structure

```
photo-enhancer/
├── main.py                  # FastAPI app, v1 router, endpoints, middleware
├── models/
│   ├── wrappers.py          # Model wrapper classes (predict interface)
│   └── archs/               # Vendored PyTorch architecture definitions
│       ├── rrdbnet_arch.py          #   Real-ESRGAN (RRDBNet)
│       ├── nafnet_arch.py           #   NAFNet
│       ├── codeformer_arch.py       #   CodeFormer
│       ├── vqgan_arch.py            #   VQ-GAN autoencoder (used by CodeFormer)
│       ├── old_photo_detect_arch.py #   Old Photo Restore scratch detection (UNet)
│       ├── old_photo_global_arch.py #   Old Photo Restore VAE + mapping network
│       └── old_photo_face_arch.py   #   Old Photo Restore face enhancement (SPADE)
├── utils/
│   ├── downloader.py        # Model weight downloader (HF/GitHub/GDrive)
│   ├── image_ops.py         # Image read/validate/resize/encode
│   └── logging.py           # JSON structured logging setup
├── tests/                   # pytest test suite
│   ├── conftest.py          # Shared fixtures
│   ├── test_image_ops.py    # Image operations unit tests
│   ├── test_downloader.py   # Downloader unit tests
│   └── test_endpoints.py    # API endpoint integration tests
├── requirements.txt         # Shared deps (no torch)
├── requirements-cpu.txt     # CPU torch (used by Docker)
├── requirements-gpu.txt     # CUDA torch (used for local dev)
├── requirements-dev.txt     # Dev deps (pytest, httpx, ruff)
├── ruff.toml                # Linter/formatter config
├── pyproject.toml           # pytest config
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

All six models run real AI inference using vendored PyTorch architectures in `models/archs/` (except LaMa which uses a TorchScript JIT model). Model architectures (block counts, widths, scale factors) are inferred from checkpoint keys at load time — no hard-coded config per variant. This means dropping in a different weight file for the same model family will just work.

Dependencies:
- **DDColor**: installed via pip (`ddcolor` package, uses bundled `basicsr==1.3.4.6`)
- **NAFNet, Real-ESRGAN, CodeFormer, Old Photo Restore**: architectures vendored locally in `models/archs/` to avoid requiring `basicsr>=1.4.2` (which would conflict with DDColor)
- **CodeFormer** additionally uses `facexlib` for face detection and alignment
- **Old Photo Restore** additionally uses `dlib` for face detection and landmark extraction

## Notes

- Images larger than 2048px on either side are automatically resized before processing.
- Uploads larger than 32 MB are rejected with HTTP 413.
- Model weights are stored in `/app/weights` (Docker) and cached across restarts via a named volume.
- Google Drive downloads (NAFNet weights) are handled via `drive.usercontent.google.com` to bypass virus-scan interstitials.
- All log output is JSON-structured for easy integration with log aggregation systems.
- Prometheus metrics are available at `/metrics`.

## Documentation

For deeper reading, see the [`docs/`](docs/) folder:

- **[API Reference](docs/api-reference.md)** — Full endpoint documentation with request/response examples
- **[Architecture](docs/architecture.md)** — System design, module breakdown, processing pipelines
- **[Deployment](docs/deployment.md)** — Docker setup, local dev, environment variables, troubleshooting
- **[Models](docs/models.md)** — Per-model deep dives, variants, papers, weight sources
- **[Development](docs/development.md)** — Contributing guide, adding new models, code conventions
