# Development & Contributing Guide

## Project Structure

```
photo-enhancer/
├── main.py                  # FastAPI app, v1 router, endpoints, middleware, model lifecycle
├── models/
│   ├── __init__.py
│   ├── wrappers.py          # Model wrapper classes (DDColor, NAFNet, CodeFormer, RealESRGAN, OldPhotoRestore)
│   └── archs/               # Vendored PyTorch architecture definitions
│       ├── __init__.py
│       ├── rrdbnet_arch.py   #   RRDBNet (Real-ESRGAN)
│       ├── nafnet_arch.py    #   NAFNet
│       ├── codeformer_arch.py#   CodeFormer transformer
│       ├── vqgan_arch.py     #   VQ-GAN autoencoder (used by CodeFormer)
│       ├── old_photo_detect_arch.py  # UNet scratch detection
│       ├── old_photo_global_arch.py  # VAE + mapping network
│       └── old_photo_face_arch.py    # SPADE face enhancement
├── utils/
│   ├── __init__.py
│   ├── downloader.py        # Model weight download & caching
│   ├── image_ops.py         # Image read/validate/resize/encode
│   └── logging.py           # JSON structured logging setup
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures (mock models, test client, sample images)
│   ├── test_image_ops.py    # Unit tests for image operations
│   ├── test_downloader.py   # Unit tests for downloader
│   └── test_endpoints.py    # Integration tests for API endpoints
├── requirements.txt         # Python dependencies (no torch)
├── requirements-cpu.txt     # CPU torch (used by Docker)
├── requirements-gpu.txt     # CUDA torch (used for local dev)
├── requirements-dev.txt     # Dev dependencies (pytest, httpx, ruff)
├── ruff.toml                # Linter/formatter configuration
├── pyproject.toml           # pytest configuration
├── Dockerfile
├── docker-compose.yml
├── CLAUDE.md                # Claude Code instructions
└── docs/                    # Extended documentation
    ├── api-reference.md
    ├── architecture.md
    ├── deployment.md
    ├── models.md
    └── development.md
```

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_image_ops.py -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=. --cov-report=term-missing
```

All model inference is mocked in tests — models are too large for CI. The `TestClient` uses a swapped no-op lifespan that pre-populates `model_registry` with `MagicMock` objects returning valid 64x64 BGR images.

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, configured in `ruff.toml`:

```bash
# Check for lint issues
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Check formatting
ruff format --check .

# Format all files
ruff format .
```

## How to Add a New Model

Follow these 5 steps to add a new image processing model.

### Step 1: Add download entry (`utils/downloader.py`)

For **single-file models**, add the weight URLs to `MODEL_URLS`:

```python
MODEL_URLS: dict[str, dict[str, str]] = {
    # ... existing entries ...
    "new_category": {
        "variant_a": "https://example.com/model_a.pth",
        "variant_b": "https://example.com/model_b.pth",
    },
}
```

If the download URL doesn't have a meaningful filename (e.g., Google Drive), add an override:

```python
FILENAME_OVERRIDES: dict[str, dict[str, str]] = {
    # ... existing entries ...
    "new_category": {
        "variant_a": "ModelName-variant_a.pth",
    },
}
```

For **multi-file models** (multiple weight files per variant), add to `MODEL_URLS_MULTI` instead:

```python
MODEL_URLS_MULTI: dict[str, dict[str, dict[str, str]]] = {
    # ... existing entries ...
    "new_category": {
        "variant_a": {
            "network_a.pth": "https://example.com/network_a.pth",
            "network_b.pth": "https://example.com/network_b.pth",
        },
    },
}
```

Multi-file models use `ensure_model_files_exist()` which returns a directory path instead of a file path.

### Step 2: Add architecture module (`models/archs/`)

If the model's PyTorch architecture isn't available via pip (or would cause dependency conflicts), vendor the `nn.Module` definition:

```python
# models/archs/new_model_arch.py
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Forward pass
        return x
```

If the architecture is available from an installable package without conflicts, you can import it directly in the wrapper instead.

### Step 3: Create wrapper class (`models/wrappers.py`)

Add a wrapper class following the existing pattern:

```python
class NewModelWrapper:
    """Wrapper for NewModel — real inference."""

    def __init__(self, model_path: str, device: str, variant: str = "variant_a") -> None:
        from models.archs.new_model_arch import NewModel

        self.device = device

        # Load checkpoint
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "params" in state_dict:
            state_dict = state_dict["params"]

        # Prefer inferring architecture from checkpoint keys
        model = NewModel(...)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        self.model = model

        logger.info("NewModelWrapper loaded — %s on %s", model_path, device)

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # BGR uint8 -> RGB float32 [0, 1]
        img = image[:, :, ::-1].astype(np.float32) / 255.0
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_t)

        # Back to BGR uint8
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        return output[:, :, ::-1].copy()
```

Key conventions:
- `__init__` takes `(model_path, device, variant)`
- `predict` takes BGR uint8 numpy array, returns BGR uint8 numpy array
- Use `torch.no_grad()` for inference
- Infer architecture params from checkpoint keys when possible

### Step 4: Add model config (`main.py`)

Add an entry to `MODEL_CONFIG`:

```python
MODEL_CONFIG = {
    # ... existing entries ...
    "new_category": {
        "env_var": "MODEL_NEW",
        "default": "variant_a",
        "wrapper": NewModelWrapper,
    },
}
```

For multi-file models, add `"multi_file": True` to the config entry. This tells the lifespan to use `ensure_model_files_exist()` instead of `ensure_model_exists()`, and the wrapper receives a directory path rather than a file path.

Don't forget to import the wrapper:

```python
from models.wrappers import (
    # ... existing imports ...
    NewModelWrapper,
)
```

### Step 5: Add endpoint (`main.py`)

Add a POST endpoint to the `v1_router`:

```python
@v1_router.post("/new-endpoint")
def new_endpoint(
    file: UploadFile = File(...),
    some_param: float = Query(0.5, ge=0.0, le=1.0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    model = get_model("new_category")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "New model not loaded"},
        )

    try:
        result = model.predict(image, some_param=some_param)
        cuda_clear()
        output = encode_image(result, output_format)
        return Response(
            content=output,
            media_type=MEDIA_TYPES.get(output_format, "image/png"),
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception:
        cuda_clear()
        logger.exception("New endpoint error")
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )
```

Don't forget to add the legacy redirect path and update the `_LEGACY_PATHS` list if desired.

---

## Code Conventions

### Wrapper Pattern

All model wrappers share the same interface:
- **Constructor**: `__init__(self, model_path, device, variant)` — loads weights, builds model, moves to device
- **Predict**: `predict(self, image, **kwargs)` — BGR uint8 numpy in, BGR uint8 numpy out
- Use `**kwargs` for model-specific parameters (e.g., `render_factor`, `fidelity`, `tile_size`)

### Checkpoint Inference

Prefer inferring architecture parameters from checkpoint keys over hard-coding per variant. This makes the wrapper work with any compatible weight file without code changes.

### BGR Arrays

All image I/O uses BGR uint8 numpy arrays (OpenCV convention). Models expect RGB float32 tensors, so wrappers handle the conversion.

### Error Handling

Endpoints use a consistent pattern:
- `FileTooLargeError` from size check → 413
- `ValueError` from image ops → 400
- Model missing from registry → 503
- Any other exception during predict → 500 with `logger.exception()` logged

---

## Testing

### curl Testing

```bash
# Health
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Colorize
curl -X POST http://localhost:8000/v1/colorize -F "file=@photo.jpg" -o output.png

# Restore
curl -X POST "http://localhost:8000/v1/restore?tile_size=256" -F "file=@photo.jpg" -o output.png

# Face restore
curl -X POST "http://localhost:8000/v1/face-restore?fidelity=0.7&upscale=2" -F "file=@photo.jpg" -o output.png

# Upscale
curl -X POST "http://localhost:8000/v1/upscale?scale=4&tile_size=512" -F "file=@photo.jpg" -o output.png

# Old photo restore
curl -X POST http://localhost:8000/v1/old-photo-restore -F "file=@old_photo.jpg" -o output.png

# Pipeline
curl -X POST "http://localhost:8000/v1/pipeline?width=4000" -F "file=@photo.jpg" -o output.png

# Pipeline with old photo restore
curl -X POST "http://localhost:8000/v1/pipeline?old_photo_restore=true&colorize=true" -F "file=@old_photo.jpg" -o output.png
```

### Checking Specific Errors

```bash
# Test invalid image
curl -X POST http://localhost:8000/v1/colorize -F "file=@README.md" -v

# Test file too large (returns 413)
dd if=/dev/zero bs=1M count=33 | curl -X POST http://localhost:8000/v1/colorize -F "file=@-" -v

# Test with model not loaded (if you set an invalid variant)
# Should return 503
```

---

## Dependency Management

### requirements.txt

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
opencv-python-headless>=4.8.0
requests>=2.31.0
numpy>=1.24.0
facexlib>=0.3.0
python-json-logger>=3.0
prometheus-fastapi-instrumentator>=7.0
dlib>=19.24.0
```

Key notes:
- **PyTorch CUDA 12.1**: The `--extra-index-url` line in `requirements-gpu.txt` pulls PyTorch with CUDA 12.1 support.
- **opencv-python-headless**: The headless variant avoids pulling in GUI dependencies.
- **DDColor is NOT in requirements.txt**: It's installed separately from git with `--no-deps` to avoid the `basicsr` conflict.
- **python-json-logger**: Provides JSON-formatted structured logging.
- **prometheus-fastapi-instrumentator**: Auto-instruments routes and exposes `/metrics`.
- **dlib**: Used by `OldPhotoRestoreWrapper` for face detection and 68-point landmark extraction. Requires `cmake` at build time.

### Adding a New Dependency

1. Add to `requirements.txt`
2. Check for conflicts with existing packages (especially `basicsr`)
3. If the package conflicts, consider vendoring the needed code in `models/archs/` instead
