# Architecture Overview

## System Diagram

```
Client
  │
  │  POST /v1/colorize, /v1/restore, /v1/face-restore, /v1/upscale, /v1/pipeline
  │  (multipart/form-data with image file)
  ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                      │
│                                                         │
│  RequestLoggingMiddleware  → JSON structured logs        │
│  Prometheus Instrumentator → /metrics                   │
│                                                         │
│  1. check_file_size(bytes) → reject if > 32 MB (413)   │
│  2. read_image(file_bytes) → numpy BGR array            │
│  3. validate_and_resize(image) → max 2048px             │
│  4. model_registry[category] → get wrapper              │
│  5. wrapper.predict(image, **kw) → numpy BGR array      │
│  6. encode_image(result, format) → bytes (PNG/JPG/…)    │
│  7. Response(content, media_type) → HTTP response       │
│                                                         │
│  /v1 router + legacy 307 redirects                      │
└─────────────────────────────────────────────────────────┘
         │                                ▲
         ▼                                │
┌─────────────────┐            ┌──────────────────────┐
│  Model Wrappers │            │  utils/image_ops.py  │
│  (wrappers.py)  │            │  read / validate /   │
│                 │            │  resize / encode     │
│  DDColorWrapper │            └──────────────────────┘
│  NAFNetWrapper  │
│  CodeFormer…    │            ┌──────────────────────┐
│  RealESRGAN…    │            │  utils/logging.py    │
└────────┬────────┘            │  JSON structured     │
         │                     │  logging setup       │
         ▼                     └──────────────────────┘
┌─────────────────┐
│  models/archs/  │
│  (PyTorch       │
│   nn.Modules)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Weight files   │
│  /app/weights/  │
│  (downloaded by │
│   downloader.py)│
└─────────────────┘
```

## Module Breakdown

### `main.py`

The FastAPI application. Responsibilities:

- **Lifespan management**: The `lifespan` async context manager runs on startup/shutdown. On startup it detects the device (CUDA/CPU), iterates `MODEL_CONFIG`, downloads weights via `ensure_model_exists()`, and instantiates wrapper objects into the global `model_registry` dict. On shutdown it clears the registry and empties the CUDA cache.
- **Model config**: `MODEL_CONFIG` maps each category (`colorize`, `restore`, `face`, `upscale`) to its environment variable name, default variant, and wrapper class.
- **API versioning**: All image processing endpoints live on a `/v1` router. Legacy un-prefixed paths return 307 redirects to their `/v1` equivalents.
- **File size validation**: `check_file_size()` rejects uploads exceeding 32 MB with HTTP 413.
- **Request logging**: `RequestLoggingMiddleware` logs every request with method, path, status code, duration, and client IP as structured JSON.
- **Prometheus metrics**: `prometheus-fastapi-instrumentator` auto-instruments all routes and exposes `GET /metrics`. `/health` and `/metrics` are excluded from instrumentation.
- **Endpoints**: Five `POST` endpoints on the `/v1` router, each following the identical pattern: size check → decode → validate → predict → encode → respond. One `GET /health` and one `GET /metrics` at root.
- **Error handling**: `FileTooLargeError` → 413, `ValueError` → 400, model missing → 503, other exceptions → 500 with `logger.exception()`.

### `utils/logging.py`

Configures structured JSON logging using `python-json-logger`. The `setup_logging()` function replaces the default `logging.basicConfig()` with a `JsonFormatter` that outputs one JSON object per line to stdout with fields: `timestamp`, `level`, `logger`, `message`, plus any extras.

### `models/wrappers.py`

Four wrapper classes, each with the same interface:

```python
class XxxWrapper:
    def __init__(self, model_path: str, device: str, variant: str = "default"):
        # Load checkpoint, build model, move to device

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # BGR uint8 in, BGR uint8 out
```

| Wrapper | Model | Key behavior |
|---|---|---|
| `DDColorWrapper` | DDColor | Uses the `ddcolor` pip package. Maps variant names to model sizes (`tiny`/`large`). Uses `ColorizationPipeline` for inference. |
| `NAFNetWrapper` | NAFNet | Infers architecture (encoder/decoder block counts, middle blocks, width) from checkpoint key names. No hard-coded config per variant. |
| `RealESRGANWrapper` | Real-ESRGAN (RRDBNet) | Infers architecture (num_feat, num_block, num_grow_ch, scale) from checkpoint. Supports tiled processing for large images. |
| `CodeFormerWrapper` | CodeFormer | Fixed architecture params (dim_embd=512, codebook_size=1024, etc). Uses `facexlib.FaceRestoreHelper` for face detection, alignment, and paste-back. |

### `models/archs/`

Vendored PyTorch `nn.Module` definitions:

| File | Class | Used by |
|---|---|---|
| `rrdbnet_arch.py` | `RRDBNet` | `RealESRGANWrapper` |
| `nafnet_arch.py` | `NAFNet` | `NAFNetWrapper` |
| `codeformer_arch.py` | `CodeFormer` | `CodeFormerWrapper` |
| `vqgan_arch.py` | `VQAutoEncoder` | Used internally by CodeFormer architecture |

These are vendored (copied into the repo) rather than installed via pip to avoid the `basicsr` version conflict (see [Dependency Strategy](#dependency-strategy)).

### `utils/downloader.py`

Handles model weight downloads:

- `MODEL_URLS` dict maps `category → variant → URL`
- `FILENAME_OVERRIDES` provides explicit filenames for URLs that don't have meaningful filenames (Google Drive)
- Downloads use `.part` file handling for atomicity — partial downloads don't leave corrupt files
- Google Drive URLs are resolved via `drive.usercontent.google.com` to bypass the virus-scan interstitial page
- Downloaded files are validated: if the first bytes are `<` (HTML), the download is rejected as corrupt

### `utils/image_ops.py`

Three functions for image I/O:

| Function | Input | Output | Notes |
|---|---|---|---|
| `read_image(bytes)` | Raw file bytes | numpy BGR uint8 | Uses `cv2.imdecode`. Raises `ValueError` on failure. |
| `validate_and_resize(img)` | numpy array | numpy array | Resizes if either dimension > 2048px (preserves aspect ratio). |
| `encode_image(img, fmt)` | numpy array + format string | bytes | JPEG/WebP quality: 95. Raises `ValueError` for unsupported formats. |

---

## Request Flow

```
POST /v1/colorize (file + query params)
  │
  ├─ RequestLoggingMiddleware: start timer
  ├─ Prometheus Instrumentator: track request
  │
  ├─ check_file_size(file_bytes) → 413 if > 32 MB
  ├─ read_image(file_bytes) → 400 if invalid
  ├─ validate_and_resize(image) → resize if > 2048px
  ├─ get_model("colorize") → 503 if not loaded
  ├─ model.predict(image, **kwargs) → 500 if exception
  ├─ encode_image(result, format) → 400 if format unsupported
  └─ Response(content, media_type)
```

---

## Model Loading Lifecycle

```
Server start
  │
  ├─ setup_logging() → configure JSON structured logging
  ├─ detect_device()
  │    └─ Check FORCE_CPU env → check torch.cuda.is_available()
  │
  ├─ For each model in MODEL_CONFIG:
  │    ├─ Read variant from environment variable (or use default)
  │    ├─ ensure_model_exists(category, variant)
  │    │    ├─ Check if weight file exists on disk
  │    │    ├─ Validate it's not a corrupt HTML download
  │    │    └─ Download from URL if missing (with .part atomicity)
  │    ├─ Instantiate wrapper class(model_path, device, variant)
  │    │    ├─ torch.load() the checkpoint
  │    │    ├─ Build nn.Module with inferred architecture
  │    │    ├─ load_state_dict() + eval() + to(device)
  │    │    └─ (CodeFormer: also init FaceRestoreHelper)
  │    └─ Store in model_registry[category]
  │    └─ On failure: log error, skip (endpoint returns 503)
  │
  └─ Log "Startup complete — N/4 models loaded"

Server shutdown
  │
  ├─ model_registry.clear()
  └─ torch.cuda.empty_cache() (if CUDA)
```

---

## Dynamic Architecture Inference

NAFNet and Real-ESRGAN wrappers infer their model architecture from checkpoint keys rather than hard-coding configurations per variant. This means any compatible weight file will work without code changes.

### NAFNet Example

The checkpoint contains keys like `encoders.0.0.conv1.weight`, `encoders.0.1.conv1.weight`, etc. The wrapper:

1. Counts encoder blocks per stage via regex: `encoders\.(\d+)\.(\d+)\.`
2. Counts decoder blocks per stage: `decoders\.(\d+)\.(\d+)\.`
3. Counts middle blocks: `middle_blks\.(\d+)\.`
4. Reads width from `intro.weight` shape

### Real-ESRGAN Example

The checkpoint contains keys like `body.0.rdb1.conv1.weight`, `conv_first.weight`, etc. The wrapper:

1. Reads `num_feat` from `conv_first.weight` shape[0]
2. Reads `num_in_ch` from `conv_first.weight` shape[1]
3. Reads `num_out_ch` from `conv_last.weight` shape[0]
4. Reads `num_grow_ch` from `body.0.rdb1.conv1.weight` shape[0]
5. Counts `num_block` from max body index
6. Infers scale from input channels: 3→4x, 12→2x, 48→1x

---

## Dependency Strategy

The project has a `basicsr` version conflict:

- **DDColor** (pip package) bundles `basicsr==1.3.4.6`
- **NAFNet**, **Real-ESRGAN**, and **CodeFormer** originally depend on `basicsr>=1.4.2`

Solution: The PyTorch `nn.Module` architecture definitions for NAFNet, RRDBNet, CodeFormer, and VQ-GAN are vendored directly in `models/archs/`. This eliminates the `basicsr>=1.4.2` dependency entirely, allowing DDColor to use its bundled version without conflicts.

DDColor is installed with `--no-deps` from git to prevent it from pulling in conflicting dependencies via pip.

---

## Image Processing Pipeline

All endpoints follow the same pipeline:

### 1. Size Check

`check_file_size(file_bytes)` → Rejects uploads exceeding 32 MB with HTTP 413.

### 2. Decode

`read_image(file_bytes)` → Uses `cv2.imdecode` on raw bytes. Returns BGR uint8 numpy array or raises `ValueError`.

### 3. Validate & Resize

`validate_and_resize(image, max_dim=2048)` → If either dimension exceeds 2048px, scales down proportionally using `cv2.INTER_AREA` interpolation.

### 4. Predict

`wrapper.predict(image, **kwargs)` → Model-specific inference. All wrappers:
- Accept BGR uint8 numpy array
- Convert to RGB float32 tensor for model input
- Convert model output back to BGR uint8 numpy array

### 5. Encode

`encode_image(result, output_format)` → Uses `cv2.imencode`. JPEG and WebP use quality 95.

---

## Tiling Strategy (Real-ESRGAN)

For large images, processing the entire image at once can exceed GPU memory. The `RealESRGANWrapper._tile_process()` method splits the image into tiles:

1. Divide image into grid of `tile_size × tile_size` tiles
2. For each tile, add `tile_pad` (10px) overlap padding from neighboring regions
3. Run the model on each padded tile
4. Strip the padding from the output tile (scaled by the upscale factor)
5. Place the clean tile into the output image at the correct position

This allows processing arbitrarily large images with bounded VRAM usage. The default `tile_size` for upscale is 512; for restore it's 0 (no tiling, since NAFNet uses less memory).

---

## Face Detection & Alignment Pipeline (CodeFormer)

The `CodeFormerWrapper` uses `facexlib.FaceRestoreHelper` for a multi-stage pipeline:

1. **Detection**: RetinaFace (`retinaface_resnet50`) detects all faces in the image and returns bounding boxes + 5 landmarks (eyes, nose, mouth corners)
2. **Alignment**: Each face is affine-warped to a canonical 512×512 crop based on the landmarks
3. **Restoration**: Each aligned face is normalized to [-1, 1], passed through the CodeFormer transformer, and denormalized back to uint8
4. **Color preservation**: Luminance is taken from the restored face, chrominance from the original crop (YCrCb space) to prevent color artifacts on B&W or sepia photos
5. **Paste-back**: Restored faces are inverse-affine-transformed and blended back into the (optionally upscaled) original image

If no faces are detected, the image is returned with a simple bicubic upscale.

---

## Observability

### JSON Structured Logging

All log output is JSON-formatted (one object per line) via `python-json-logger`. Each log entry includes `timestamp`, `level`, `logger`, and `message` fields, plus any extras. Request logs from the middleware additionally include `method`, `path`, `status_code`, `duration_ms`, and `client_ip`.

### Prometheus Metrics

The `/metrics` endpoint exposes standard Prometheus metrics via `prometheus-fastapi-instrumentator`:

- `http_request_duration_seconds` — histogram of request latency
- `http_requests_total` — counter of total requests
- `http_requests_inprogress` — gauge of in-flight requests

The `/health` and `/metrics` endpoints are excluded from instrumentation.
