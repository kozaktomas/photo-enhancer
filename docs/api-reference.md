# API Reference

Full reference for the Photo AI Processing Service REST API.

Base URL: `http://localhost:8000`

## Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check and system info |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/v1/colorize` | Colorize a grayscale photo |
| `POST` | `/v1/restore` | Remove noise or blur |
| `POST` | `/v1/face-restore` | Enhance and restore faces |
| `POST` | `/v1/upscale` | Super-resolution upscaling |
| `POST` | `/v1/old-photo-restore` | Restore old/damaged photos (scratch repair, global restoration, face enhancement) |
| `POST` | `/v1/inpaint` | Fill polygon regions using LaMa inpainting |
| `POST` | `/v1/pipeline` | Run multiple enhancement steps in one request |

Interactive docs are available at [`/docs`](http://localhost:8000/docs) (Swagger UI) and [`/redoc`](http://localhost:8000/redoc) (ReDoc).

> **Legacy paths:** The old un-prefixed paths (`/colorize`, `/restore`, etc.) return a 307 redirect to their `/v1/` equivalents. They are hidden from the API docs.

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
  "loaded_models": ["colorize", "restore", "face", "upscale", "old_photo_restore", "inpaint"],
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

## GET /metrics

Returns Prometheus metrics in text exposition format. Includes standard HTTP metrics (request duration, count, in-progress) plus custom model gauges.

**Request:**

```bash
curl http://localhost:8000/metrics
```

---

## POST /v1/colorize

Colorize a grayscale or B&W photo using the DDColor model.

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image (JPEG, PNG, WebP, etc.) |
| `render_factor` | query | int | `35` | 1-100 | Controls colorization intensity. Higher values produce more vivid colors. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage
curl -X POST http://localhost:8000/v1/colorize \
  -F "file=@photo.jpg" \
  -o colorized.png

# With parameters
curl -X POST "http://localhost:8000/v1/colorize?render_factor=50&output_format=webp" \
  -F "file=@photo.jpg" \
  -o colorized.webp
```

**Python (requests):**

```python
import requests

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/colorize",
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
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: Colorization model not loaded

---

## POST /v1/restore

Remove noise or blur from a photo using the NAFNet model. The behavior (denoising vs deblurring) depends on which variant is loaded via the `MODEL_RESTORE` environment variable.

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image |
| `tile_size` | query | int | `0` | >= 0 | Tile size for processing. `0` = no tiling (process whole image at once). Use tiling to reduce VRAM on large images. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage
curl -X POST http://localhost:8000/v1/restore \
  -F "file=@noisy_photo.jpg" \
  -o restored.png

# With tiling for large images
curl -X POST "http://localhost:8000/v1/restore?tile_size=256&output_format=jpg" \
  -F "file=@noisy_photo.jpg" \
  -o restored.jpg
```

**Python (requests):**

```python
import requests

with open("noisy_photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/restore",
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
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: Restoration model not loaded

---

## POST /v1/face-restore

Enhance and restore faces in a photo using the CodeFormer model. Detects all faces in the image, restores each one, and pastes them back. If no faces are detected, returns a bicubic upscale of the original.

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image |
| `fidelity` | query | float | `0.5` | 0.0-1.0 | Balance between quality and fidelity. Lower values produce higher quality but less faithful results. Higher values stay closer to the input. |
| `upscale` | query | int | `2` | 1-4 | Upscale factor for the output image |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage
curl -X POST http://localhost:8000/v1/face-restore \
  -F "file=@portrait.jpg" \
  -o face_restored.png

# With parameters
curl -X POST "http://localhost:8000/v1/face-restore?fidelity=0.7&upscale=2&output_format=png" \
  -F "file=@portrait.jpg" \
  -o face_restored.png
```

**Python (requests):**

```python
import requests

with open("portrait.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/face-restore",
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
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: Face restoration model not loaded

---

## POST /v1/upscale

Upscale an image using the Real-ESRGAN super-resolution model. The model variant determines the native scale factor and style (realistic vs anime).

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image |
| `scale` | query | int | `4` | 1-8 | Desired upscale factor |
| `tile_size` | query | int | `512` | >= 0 | Tile size for processing. `0` = no tiling. Use tiling to reduce VRAM on large images. Default is 512 to prevent OOM on typical inputs. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic 4x upscale
curl -X POST http://localhost:8000/v1/upscale \
  -F "file=@small_photo.jpg" \
  -o upscaled.png

# 2x upscale with tiling and WebP output
curl -X POST "http://localhost:8000/v1/upscale?scale=2&tile_size=256&output_format=webp" \
  -F "file=@small_photo.jpg" \
  -o upscaled.webp
```

**Python (requests):**

```python
import requests

with open("small_photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/upscale",
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
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: Upscale model not loaded

---

## POST /v1/old-photo-restore

Restore an old or damaged photo. Automatically detects and repairs scratches, performs global quality restoration via a VAE pipeline, and optionally enhances detected faces using a SPADE generator. Based on Microsoft's "Bringing Old Photos Back to Life" (CVPR 2020).

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image (JPEG, PNG, WebP, etc.) |
| `with_scratch` | query | bool | `true` | — | Enable automatic scratch detection and repair |
| `with_face` | query | bool | `true` | — | Enable face enhancement via SPADE generator |
| `scratch_threshold` | query | float | `0.4` | 0.0-1.0 | Sensitivity for scratch detection. Lower values detect more scratches. |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage — full pipeline (scratch detection + global restore + face enhance)
curl -X POST http://localhost:8000/v1/old-photo-restore \
  -F "file=@old_photo.jpg" \
  -o restored.png

# Without scratch detection (for photos without physical damage)
curl -X POST "http://localhost:8000/v1/old-photo-restore?with_scratch=false" \
  -F "file=@old_photo.jpg" \
  -o restored.png

# Adjust scratch sensitivity
curl -X POST "http://localhost:8000/v1/old-photo-restore?scratch_threshold=0.6&output_format=webp" \
  -F "file=@old_photo.jpg" \
  -o restored.webp
```

**Python (requests):**

```python
import requests

with open("old_photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/old-photo-restore",
        files={"file": f},
        params={"with_scratch": True, "scratch_threshold": 0.4},
    )

resp.raise_for_status()
with open("restored.png", "wb") as f:
    f.write(resp.content)
```

### Response

- **200**: Processed image binary (Content-Type matches `output_format`)
- **400**: Invalid image or parameter
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: Old photo restoration model not loaded

---

## POST /v1/inpaint

Fill a polygon-shaped region of an image using the LaMa (Large Mask Inpainting) model. Instead of uploading a separate mask file, you define the inpaint region as a polygon via the `points` query parameter.

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image (JPEG, PNG, WebP, etc.) |
| `points` | query | string (JSON) | required | At least 3 `[x,y]` pairs | JSON array of `[x,y]` points defining the polygon to inpaint, e.g. `[[10,20],[30,40],[50,60]]` |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
# Basic usage — inpaint a rectangular region (-g disables curl's URL globbing for brackets)
curl -g -X POST "http://localhost:8000/v1/inpaint?points=[[100,100],[400,100],[400,400],[100,400]]" \
  -F "file=@photo.jpg" \
  -o inpainted.png

# With WebP output
curl -g -X POST "http://localhost:8000/v1/inpaint?points=[[100,100],[400,100],[400,400],[100,400]]&output_format=webp" \
  -F "file=@photo.jpg" \
  -o inpainted.webp
```

**Python (requests):**

```python
import json
import requests

points = [[100, 100], [400, 100], [400, 400], [100, 400]]

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/inpaint",
        files={"file": f},
        params={"points": json.dumps(points), "output_format": "png"},
    )

resp.raise_for_status()
with open("inpainted.png", "wb") as f:
    f.write(resp.content)
```

### Response

- **200**: Processed image binary (Content-Type matches `output_format`)
- **400**: Invalid image, invalid points JSON, fewer than 3 points, or parameter error
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: Inpainting model not loaded

---

## POST /v1/pipeline

Run multiple enhancement steps in a single request. Pipeline order: old_photo_restore -> colorize -> restore -> upscale -> resize -> face restore.

**File size limit:** 32 MB maximum.

### Parameters

| Parameter | In | Type | Default | Constraints | Description |
|---|---|---|---|---|---|
| `file` | body (form) | file | required | max 32 MB | Input image |
| `old_photo_restore` | query | bool | `false` | — | Enable old photo restoration step (runs first) |
| `colorize` | query | bool | `true` | — | Enable colorization step |
| `restore` | query | bool | `true` | — | Enable restoration step |
| `face_restore` | query | bool | `true` | — | Enable face restoration step |
| `upscale` | query | bool | `true` | — | Enable upscale step |
| `width` | query | int | `2400` | >= 1 | Target output width |
| `height` | query | int | `2400` | >= 1 | Target output height |
| `with_scratch` | query | bool | `true` | — | Enable scratch detection in old photo restore |
| `scratch_threshold` | query | float | `0.4` | 0.0-1.0 | Scratch detection sensitivity |
| `render_factor` | query | int | `35` | 1-100 | Colorization intensity |
| `restore_tile_size` | query | int | `0` | >= 0 | Tile size for restoration |
| `fidelity` | query | float | `0.7` | 0.0-1.0 | Face restoration fidelity |
| `upscale_tile_size` | query | int | `512` | >= 0 | Tile size for upscaling |
| `output_format` | query | string | `png` | `png`, `jpg`, `jpeg`, `webp` | Output image format |

### Examples

**curl:**

```bash
curl -X POST "http://localhost:8000/v1/pipeline?width=4000&output_format=png" \
  -F "file=@photo.jpg" \
  -o pipeline_output.png
```

### Response

- **200**: Processed image binary
- **400**: Invalid image, invalid parameter, or no pipeline steps enabled
- **413**: File too large (exceeds 32 MB)
- **500**: Processing error
- **503**: One or more required models not loaded

---

## HTTP Status Codes

| Code | Meaning | Description |
|---|---|---|
| `200` | OK | Image processed successfully. Response body is the image binary. |
| `400` | Bad Request | Invalid input: unreadable image, unsupported format, or parameter out of range. |
| `413` | Payload Too Large | Uploaded file exceeds the 32 MB size limit. |
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
| 400 | `Invalid points JSON: ...` | `points` param is not valid JSON |
| 400 | `points must be a JSON array of at least 3 [x,y] pairs` | Fewer than 3 points provided |
| 413 | `File too large (35.2 MB). Maximum allowed size is 32 MB.` | Upload exceeds 32 MB |
| 503 | `Colorization model not loaded` | DDColor failed to load at startup |
| 503 | `Restoration model not loaded` | NAFNet failed to load at startup |
| 503 | `Face restoration model not loaded` | CodeFormer failed to load at startup |
| 503 | `Upscale model not loaded` | Real-ESRGAN failed to load at startup |
| 503 | `Old photo restoration model not loaded` | Old Photo Restore failed to load at startup |
| 503 | `Inpainting model not loaded` | LaMa failed to load at startup |
| 500 | `Internal processing error` | Unexpected exception (see server logs) |

## Output Formats

All image endpoints support these output formats via the `output_format` query parameter:

| Format | Content-Type | Notes |
|---|---|---|
| `png` | `image/png` | Default. Lossless compression. |
| `jpg` / `jpeg` | `image/jpeg` | Lossy, quality 95. |
| `webp` | `image/webp` | Lossy, quality 95. Smaller file size than JPEG at similar quality. |
