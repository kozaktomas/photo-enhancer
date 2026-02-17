# Development & Contributing Guide

## Project Structure

```
photo-en/
├── main.py                  # FastAPI app, endpoints, model lifecycle
├── models/
│   ├── __init__.py
│   ├── wrappers.py          # Model wrapper classes (DDColor, NAFNet, CodeFormer, RealESRGAN)
│   └── archs/               # Vendored PyTorch architecture definitions
│       ├── __init__.py
│       ├── rrdbnet_arch.py   #   RRDBNet (Real-ESRGAN)
│       ├── nafnet_arch.py    #   NAFNet
│       ├── codeformer_arch.py#   CodeFormer transformer
│       └── vqgan_arch.py     #   VQ-GAN autoencoder (used by CodeFormer)
├── utils/
│   ├── __init__.py
│   ├── downloader.py        # Model weight download & caching
│   └── image_ops.py         # Image read/validate/resize/encode
├── requirements.txt         # Python dependencies (PyTorch CUDA 12.1)
├── Dockerfile               # Container build (python:3.11-slim)
├── docker-compose.yml       # Compose with GPU reservations + weight volume
├── CLAUDE.md                # Claude Code instructions
└── README.md                # Project documentation
```

## How to Add a New Model

Follow these 5 steps to add a new image processing model.

### Step 1: Add download entry (`utils/downloader.py`)

Add the model's weight URLs to `MODEL_URLS`:

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
        # rather than hard-coding per variant
        num_blocks = 1 + max(
            int(m.group(1)) for k in state_dict
            for m in [re.match(r"blocks\.(\d+)\.", k)] if m
        )

        model = NewModel(num_blocks=num_blocks)
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

Don't forget to import the wrapper:

```python
from models.wrappers import (
    # ... existing imports ...
    NewModelWrapper,
)
```

### Step 5: Add endpoint (`main.py`)

Add a POST endpoint following the existing pattern:

```python
@app.post("/new-endpoint")
def new_endpoint(
    file: UploadFile = File(...),
    some_param: float = Query(0.5, ge=0.0, le=1.0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    try:
        image = read_image(file.file.read())
        image = validate_and_resize(image)
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
        output = encode_image(result, output_format)
        return Response(
            content=output,
            media_type=MEDIA_TYPES.get(output_format, "image/png"),
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception:
        logger.error("New endpoint error:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )
```

---

## Code Conventions

### Wrapper Pattern

All model wrappers share the same interface:
- **Constructor**: `__init__(self, model_path, device, variant)` — loads weights, builds model, moves to device
- **Predict**: `predict(self, image, **kwargs)` — BGR uint8 numpy in, BGR uint8 numpy out
- Use `**kwargs` for model-specific parameters (e.g., `render_factor`, `fidelity`, `tile_size`)

### Checkpoint Inference

Prefer inferring architecture parameters from checkpoint keys over hard-coding per variant. This makes the wrapper work with any compatible weight file without code changes. Examples:

- Count blocks: `max(int(m.group(1)) for k in state_dict for m in [re.match(pattern, k)] if m)`
- Read dimensions: `state_dict["layer.weight"].shape[0]`

### BGR Arrays

All image I/O uses BGR uint8 numpy arrays (OpenCV convention). Models expect RGB float32 tensors, so wrappers handle the conversion:

```python
# Input: BGR uint8 → RGB float32 [0,1]
img = image[:, :, ::-1].astype(np.float32) / 255.0
img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

# Output: tensor → RGB float32 → BGR uint8
output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
output = output[:, :, ::-1].copy()  # RGB → BGR
```

### Error Handling

Endpoints use a consistent pattern:
- `ValueError` from image ops → 400
- Model missing from registry → 503
- Any other exception during predict → 500 with `traceback.format_exc()` logged

---

## Testing

### curl Testing

```bash
# Health
curl http://localhost:8000/health

# Colorize
curl -X POST http://localhost:8000/colorize -F "file=@photo.jpg" -o output.png

# Restore
curl -X POST "http://localhost:8000/restore?tile_size=256" -F "file=@photo.jpg" -o output.png

# Face restore
curl -X POST "http://localhost:8000/face-restore?fidelity=0.7&upscale=2" -F "file=@photo.jpg" -o output.png

# Upscale
curl -X POST "http://localhost:8000/upscale?scale=4&tile_size=512" -F "file=@photo.jpg" -o output.png
```

### Checking Specific Errors

```bash
# Test invalid image
curl -X POST http://localhost:8000/colorize -F "file=@README.md" -v

# Test with model not loaded (if you set an invalid variant)
# Should return 503
```

---

## Dependency Management

### requirements.txt

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.1.0
torchvision>=0.16.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
opencv-python-headless>=4.8.0
requests>=2.31.0
numpy>=1.24.0
facexlib>=0.3.0
```

Key notes:
- **PyTorch CUDA 12.1**: The `--extra-index-url` line pulls PyTorch with CUDA 12.1 support. For CPU-only, you can remove this line.
- **opencv-python-headless**: The headless variant avoids pulling in GUI dependencies (not needed for a server).
- **DDColor is NOT in requirements.txt**: It's installed separately from git with `--no-deps` (see Dockerfile line 20) to avoid the `basicsr` conflict.
- **facexlib**: Provides RetinaFace face detection used by CodeFormer. Downloads its own detection model on first use.

### Adding a New Dependency

1. Add to `requirements.txt`
2. Check for conflicts with existing packages (especially `basicsr`)
3. If the package conflicts, consider vendoring the needed code in `models/archs/` instead
