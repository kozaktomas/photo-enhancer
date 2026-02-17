# API Reference

Full reference for the Photo AI Processing Service REST API.

Base URL: `http://localhost:8000`

## Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check and system info |
| `POST` | `/colorize` | Colorize a grayscale photo |
| `POST` | `/restore` | Remove noise or blur |
| `POST` | `/face-restore` | Enhance and restore faces |
| `POST` | `/upscale` | Super-resolution upscaling |

Interactive docs are available at [`/docs`](http://localhost:8000/docs) (Swagger UI) and [`/redoc`](http://localhost:8000/redoc) (ReDoc).

---

## GET /health

Returns server status, device info, and loaded models.

**Request:**

```bash
curl http://localhost:8000/health
```

**Response (200):**

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

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"healthy"` if the server is running |
| `device` | string | `"cuda"` or `"cpu"` |
| `loaded_models` | string[] | List of successfully loaded model categories |
| `cuda_info` | object \| null | GPU info (null when running on CPU) |
| `cuda_info.gpu_name` | string | GPU device name |
| `cuda_info.vram_total_gb` | float | Total VRAM in GB |
| `cuda_info.vram_used_gb` | float | Currently allocated VRAM in GB |

---

## POST /colorize

Colorize a grayscale or B&W photo using the DDColor model.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | — | Input image (JPEG, PNG, WebP, etc.) |
| `render_factor` | query | int | `35` | 1-100 | Controls colorization intensity. Higher values produce more vivid colors. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage
curl -X POST http://localhost:8000/colorize \
  -F "file=@photo.jpg" \
  -o colorized.png

# With parameters
curl -X POST "http://localhost:8000/colorize?render_factor=50&output_format=webp" \
  -F "file=@photo.jpg" \
  -o colorized.webp
```

**Python (requests):**

```python
import requests

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/colorize",
        files={"file": f},
        params={"render_factor": 50, "output_format": "webp"},
    )

resp.raise_for_status()
with open("colorized.webp", "wb") as f:
    f.write(resp.content)
```

### Response

- **200**: Processed image binary (Content-Type matches `output_format`)
- **400**: Invalid image or parameter
- **500**: Processing error
- **503**: Colorization model not loaded

---

## POST /restore

Remove noise or blur from a photo using the NAFNet model. The behavior (denoising vs deblurring) depends on which variant is loaded via the `MODEL_RESTORE` environment variable.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | — | Input image |
| `tile_size` | query | int | `0` | >= 0 | Tile size for processing. `0` = no tiling (process whole image at once). Use tiling to reduce VRAM on large images. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage
curl -X POST http://localhost:8000/restore \
  -F "file=@noisy_photo.jpg" \
  -o restored.png

# With tiling for large images
curl -X POST "http://localhost:8000/restore?tile_size=256&output_format=jpg" \
  -F "file=@noisy_photo.jpg" \
  -o restored.jpg
```

**Python (requests):**

```python
import requests

with open("noisy_photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/restore",
        files={"file": f},
        params={"tile_size": 256},
    )

resp.raise_for_status()
with open("restored.png", "wb") as f:
    f.write(resp.content)
```

### Response

- **200**: Processed image binary
- **400**: Invalid image or parameter
- **500**: Processing error
- **503**: Restoration model not loaded

---

## POST /face-restore

Enhance and restore faces in a photo using the CodeFormer model. Detects all faces in the image, restores each one, and pastes them back. If no faces are detected, returns a bicubic upscale of the original.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | — | Input image |
| `fidelity` | query | float | `0.5` | 0.0-1.0 | Balance between quality and fidelity. Lower values produce higher quality but less faithful results. Higher values stay closer to the input. |
| `upscale` | query | int | `2` | 1-4 | Upscale factor for the output image |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage
curl -X POST http://localhost:8000/face-restore \
  -F "file=@portrait.jpg" \
  -o face_restored.png

# With parameters
curl -X POST "http://localhost:8000/face-restore?fidelity=0.7&upscale=2&output_format=png" \
  -F "file=@portrait.jpg" \
  -o face_restored.png
```

**Python (requests):**

```python
import requests

with open("portrait.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/face-restore",
        files={"file": f},
        params={"fidelity": 0.7, "upscale": 2},
    )

resp.raise_for_status()
with open("face_restored.png", "wb") as f:
    f.write(resp.content)
```

### Response

- **200**: Processed image binary
- **400**: Invalid image or parameter
- **500**: Processing error
- **503**: Face restoration model not loaded

---

## POST /upscale

Upscale an image using the Real-ESRGAN super-resolution model. The model variant determines the native scale factor and style (realistic vs anime).

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | — | Input image |
| `scale` | query | int | `4` | 1-8 | Desired upscale factor |
| `tile_size` | query | int | `512` | >= 0 | Tile size for processing. `0` = no tiling. Use tiling to reduce VRAM on large images. Default is 512 to prevent OOM on typical inputs. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic 4x upscale
curl -X POST http://localhost:8000/upscale \
  -F "file=@small_photo.jpg" \
  -o upscaled.png

# 2x upscale with tiling and WebP output
curl -X POST "http://localhost:8000/upscale?scale=2&tile_size=256&output_format=webp" \
  -F "file=@small_photo.jpg" \
  -o upscaled.webp
```

**Python (requests):**

```python
import requests

with open("small_photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/upscale",
        files={"file": f},
        params={"scale": 4, "tile_size": 512, "output_format": "webp"},
    )

resp.raise_for_status()
with open("upscaled.webp", "wb") as f:
    f.write(resp.content)
```

### Response

- **200**: Processed image binary
- **400**: Invalid image or parameter
- **500**: Processing error
- **503**: Upscale model not loaded

---

## HTTP Status Codes

| Code | Meaning | Description |
|---|---|---|
| `200` | OK | Image processed successfully. Response body is the image binary. |
| `400` | Bad Request | Invalid input: unreadable image, unsupported format, or parameter out of range. |
| `500` | Internal Server Error | Unexpected failure during model inference. Check server logs for the traceback. |
| `503` | Service Unavailable | The requested model failed to load during startup. Check server startup logs. |

### Error Response Format

All error responses return JSON with a `detail` field:

```json
{"detail": "Could not decode image — invalid or unsupported format"}
```

Common error messages:

| Status | Detail | Cause |
|---|---|---|
| 400 | `Could not decode image — invalid or unsupported format` | File is not a valid image |
| 400 | `Unsupported output format: 'bmp'. Supported: png, jpg, webp` | Invalid `output_format` value |
| 503 | `Colorization model not loaded` | DDColor failed to load at startup |
| 503 | `Restoration model not loaded` | NAFNet failed to load at startup |
| 503 | `Face restoration model not loaded` | CodeFormer failed to load at startup |
| 503 | `Upscale model not loaded` | Real-ESRGAN failed to load at startup |
| 500 | `Internal processing error` | Unexpected exception (see server logs) |

## Output Formats

All image endpoints support these output formats via the `output_format` query parameter:

| Format | Content-Type | Notes |
|---|---|---|
| `png` | `image/png` | Default. Lossless compression. |
| `jpg` / `jpeg` | `image/jpeg` | Lossy, quality 95. |
| `webp` | `image/webp` | Lossy, quality 95. Smaller file size than JPEG at similar quality. |
