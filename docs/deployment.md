# Deployment Guide

## Docker Deployment (Recommended)

### Prerequisites

- Docker and Docker Compose
- For GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured

### Quick Start

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Model weights are downloaded automatically on first startup (this may take several minutes).

### GPU Setup

The `docker-compose.yml` reserves all NVIDIA GPUs by default:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Verify GPU access inside the container:

```bash
docker compose exec photo-ai python -c "import torch; print(torch.cuda.is_available())"
```

### CPU-Only Mode

Set `FORCE_CPU=true` in `docker-compose.yml`:

```yaml
environment:
  - FORCE_CPU=true
```

Or remove the entire `deploy.resources.reservations` block if you don't have an NVIDIA GPU.

### Weight Caching

Model weights are persisted in a named Docker volume (`model-weights`) mounted at `/app/weights`. This means:

- Weights survive container rebuilds (`docker compose up --build`)
- First startup downloads all model weights (~1 GB total)
- Subsequent startups load weights from the volume instantly

To force re-download of weights:

```bash
docker compose down -v  # removes the volume
docker compose up --build
```

### Selecting Model Variants

Configure variants via environment variables in `docker-compose.yml`:

```yaml
environment:
  - MODEL_COLORIZE=artistic      # modelscope (default), paper_tiny, artistic
  - MODEL_RESTORE=deblur         # denoise (default), deblur
  - MODEL_FACE=v0.1              # v0.1 (default)
  - MODEL_UPSCALE=x2plus         # x4plus (default), x4anime, x2plus
```

### Changing the Port

Edit the ports mapping in `docker-compose.yml`:

```yaml
ports:
  - "3000:8000"  # host:container
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Git (required for DDColor install)
- CUDA toolkit (optional, for GPU inference)

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (includes PyTorch with CUDA 12.1 support)
pip install -r requirements.txt

# Install DDColor from git (--no-deps to avoid basicsr conflict)
pip install --no-deps git+https://github.com/piddnad/DDColor.git
```

> **Note:** The DDColor install from git is handled automatically by the Dockerfile but must be done manually for local development. The `--no-deps` flag is important to avoid pulling in `basicsr>=1.4.2` which conflicts with the vendored architectures.

### Running

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Model weights will be downloaded to `/app/weights/` by default. On local development, you may want to override this path or create the directory.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_COLORIZE` | `modelscope` | DDColor variant: `modelscope`, `paper_tiny`, `artistic` |
| `MODEL_RESTORE` | `denoise` | NAFNet variant: `denoise`, `deblur` |
| `MODEL_FACE` | `v0.1` | CodeFormer variant: `v0.1` |
| `MODEL_UPSCALE` | `x4plus` | Real-ESRGAN variant: `x4plus`, `x4anime`, `x2plus` |
| `FORCE_CPU` | `false` | Force CPU inference even if CUDA is available |
| `TORCH_HOME` | `/app/weights/.torch` | PyTorch cache directory (set in Dockerfile) |
| `HF_HOME` | `/app/weights/.huggingface` | HuggingFace cache directory (set in Dockerfile) |

---

## Production Considerations

### Workers

The default command runs a single uvicorn worker:

```
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

A single worker is intentional — each worker loads all models into GPU memory. Multiple workers would multiply VRAM usage. If you need concurrency, use an async-aware setup or queue-based architecture instead of multiple workers.

### Reverse Proxy

Place behind nginx or similar for:
- TLS termination
- Request size limits (large image uploads)
- Rate limiting
- Static file serving

Example nginx config snippet:

```nginx
location /api/ {
    proxy_pass http://localhost:8000/;
    client_max_body_size 50M;
    proxy_read_timeout 120s;
}
```

### Resource Limits

- **VRAM**: Models consume roughly 2-4 GB VRAM total depending on variants loaded. The upscale model on large images is the most VRAM-intensive; use `tile_size` to limit per-tile memory.
- **RAM**: Expect ~2-4 GB system RAM for model weights and image buffers.
- **Disk**: Model weights total ~1 GB. The persistent volume needs at least 2 GB to account for download artifacts.

---

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory` or HTTP 500 during processing.

**Solutions:**
- Use the `tile_size` parameter on `/restore` and `/upscale` endpoints to process images in smaller chunks
- Resize input images (the API auto-resizes anything above 2048px, but smaller images use less VRAM)
- Use `FORCE_CPU=true` for processing that exceeds GPU memory
- Choose lighter model variants (e.g., `paper_tiny` instead of `modelscope` for colorization)

### Model Download Failures

**Symptom:** Model returns 503, startup logs show download errors.

**Solutions:**
- Check internet connectivity from the container
- For Google Drive URLs (NAFNet weights): the downloader uses `drive.usercontent.google.com` to bypass the virus-scan interstitial, but Google may still rate-limit. Retry after a few minutes.
- Check that the download didn't save an HTML error page: the downloader validates file headers, but you can manually inspect files in the weights volume
- Remove the weights volume and retry: `docker compose down -v && docker compose up --build`

### Model Returns 503

**Symptom:** Endpoint returns `{"detail": "<Model> model not loaded"}`.

**Cause:** The model failed to load during startup. Check the server logs for the full traceback:

```bash
docker compose logs photo-ai | grep "Failed to load"
```

Common causes:
- Download failed or was interrupted (corrupt `.part` file)
- Invalid variant name in environment variable
- CUDA/driver version mismatch

### Startup Takes Too Long

First startup downloads all model weights (~1 GB). Subsequent startups should be fast if the volume is preserved. If startup is slow on restarts, check that the `model-weights` volume still exists:

```bash
docker volume ls | grep model-weights
```
