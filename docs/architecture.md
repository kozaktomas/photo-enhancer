# Architecture Overview

## System Diagram

```
Client
  │
  │  POST /colorize, /restore, /face-restore, /upscale
  │  (multipart/form-data with image file)
  ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                      │
│                                                         │
│  1. read_image(file_bytes)         → numpy BGR array    │
│  2. validate_and_resize(image)     → max 2048px         │
│  3. model_registry[category]       → get wrapper        │
│  4. wrapper.predict(image, **kw)   → numpy BGR array    │
│  5. encode_image(result, format)   → bytes (PNG/JPG/…)  │
│  6. Response(content, media_type)  → HTTP response      │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │                                ▲
         ▼                                │
┌─────────────────┐            ┌──────────────────────┐
│  Model Wrappers │            │  utils/image_ops.py  │
│  (wrappers.py)  │            │  read / validate /   │
│                 │            │  resize / encode     │
│  DDColorWrapper │            └──────────────────────┘
│  NAFNetWrapper  │
│  CodeFormer…    │
│  RealESRGAN…    │
└────────┬────────┘
         │
         ▼
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
- **Endpoints**: Four `POST` endpoints, each following the identical pattern: decode → validate → predict → encode → respond. One `GET /health` endpoint.
- **Error handling**: `ValueError` → 400, model missing → 503, other exceptions → 500 with logging.

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

## Model Loading Lifecycle

```
Server start
  │
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

### 1. Decode

`read_image(file_bytes)` → Uses `cv2.imdecode` on raw bytes. Returns BGR uint8 numpy array or raises `ValueError`.

### 2. Validate & Resize

`validate_and_resize(image, max_dim=2048)` → If either dimension exceeds 2048px, scales down proportionally using `cv2.INTER_AREA` interpolation.

### 3. Predict

`wrapper.predict(image, **kwargs)` → Model-specific inference. All wrappers:
- Accept BGR uint8 numpy array
- Convert to RGB float32 tensor for model input
- Convert model output back to BGR uint8 numpy array

### 4. Encode

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
