"""Photo AI Processing Service — FastAPI application.

REST API for AI-powered photo enhancement: colorization, restoration,
face restoration, and upscaling. Models load at startup and are served
via versioned POST endpoints under ``/v1``.
"""

import logging
import os
import time
import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pretrained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weight enum.*", category=UserWarning)

import cv2  # noqa: E402
import torch  # noqa: E402
from fastapi import APIRouter, FastAPI, File, Query, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, RedirectResponse, Response  # noqa: E402
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402

from models.wrappers import (  # noqa: E402
    CodeFormerWrapper,
    DDColorWrapper,
    NAFNetWrapper,
    OldPhotoRestoreWrapper,
    RealESRGANWrapper,
)
from utils.downloader import ensure_model_exists, ensure_model_files_exist  # noqa: E402
from utils.image_ops import encode_image, read_image, validate_and_resize  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

model_registry: dict[str, object] = {}
device: str = "cpu"

MAX_FILE_SIZE = 32 * 1024 * 1024  # 32 MB

MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def detect_device() -> str:
    """Detect the best available compute device and set the global ``device`` variable.

    Checks ``FORCE_CPU`` env var first, then CUDA, then MPS, falling back to CPU.

    Returns:
        The device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
    """
    global device
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() in ("true", "1", "yes")
    if force_cpu:
        device = "cpu"
        logger.info("Device: CPU (forced via FORCE_CPU)")
        return device

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Device: CUDA — %s (%.1f GB VRAM)", gpu_name, vram)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Device: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        logger.info("Device: CPU (no GPU available)")
    return device


MODEL_CONFIG = {
    "colorize": {
        "env_var": "MODEL_COLORIZE",
        "default": "modelscope",
        "wrapper": DDColorWrapper,
    },
    "restore": {
        "env_var": "MODEL_RESTORE",
        "default": "denoise",
        "wrapper": NAFNetWrapper,
    },
    "face": {
        "env_var": "MODEL_FACE",
        "default": "v0.1",
        "wrapper": CodeFormerWrapper,
    },
    "upscale": {
        "env_var": "MODEL_UPSCALE",
        "default": "x4plus",
        "wrapper": RealESRGANWrapper,
    },
    "old_photo_restore": {
        "env_var": "MODEL_OLD_PHOTO",
        "default": "v1",
        "wrapper": OldPhotoRestoreWrapper,
        "multi_file": True,
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    On startup: detects the compute device, downloads model weights (if needed),
    instantiates wrapper objects, and populates ``model_registry``.
    On shutdown: clears the registry and frees CUDA memory.
    """
    detect_device()

    for category, cfg in MODEL_CONFIG.items():
        variant = os.environ.get(cfg["env_var"], cfg["default"])
        try:
            if cfg.get("multi_file"):
                model_path = ensure_model_files_exist(category, variant)
            else:
                model_path = ensure_model_exists(category, variant)
            wrapper = cfg["wrapper"](model_path, device, variant=variant)
            model_registry[category] = wrapper
            logger.info("Loaded model: %s/%s", category, variant)
        except Exception:
            logger.exception("Failed to load model %s/%s", category, variant)

    logger.info("Startup complete — %d/%d models loaded", len(model_registry), len(MODEL_CONFIG))
    yield

    model_registry.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete — models unloaded")


app = FastAPI(
    title="Photo AI Processing Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 1),
                "client_ip": request.client.host if request.client else None,
            },
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

instrumentator = Instrumentator(
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/health"],
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model(name: str):
    """Look up a model wrapper by category name.

    Args:
        name: Model category key (e.g. ``"colorize"``, ``"restore"``).

    Returns:
        The model wrapper instance, or ``None`` if the model is not loaded.
    """
    if name not in model_registry:
        return None
    return model_registry[name]


def cuda_clear():
    """Free CUDA cache memory if the current device is CUDA."""
    if device == "cuda":
        torch.cuda.empty_cache()


def check_file_size(file_bytes: bytes) -> None:
    """Reject uploads that exceed the maximum allowed file size.

    Args:
        file_bytes: Raw uploaded file content.

    Raises:
        JSONResponse: 413 status if the file exceeds ``MAX_FILE_SIZE``.
    """
    if len(file_bytes) > MAX_FILE_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise FileTooLargeError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed size is {max_mb:.0f} MB."
        )


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the size limit."""


# ---------------------------------------------------------------------------
# v1 API router
# ---------------------------------------------------------------------------

v1_router = APIRouter(prefix="/v1", tags=["v1"])


@v1_router.post("/colorize")
def colorize(
    file: UploadFile = File(...),
    render_factor: int = Query(35, ge=1, le=100),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    """Colorize a grayscale or black-and-white photo using the DDColor model.

    Args:
        file: Input image file (JPEG, PNG, WebP, etc.).
        render_factor: Controls colorization intensity (1-100).
        output_format: Desired output image format.

    Returns:
        The colorized image as binary content.
    """
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    model = get_model("colorize")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Colorization model not loaded"},
        )

    try:
        result = model.predict(image, render_factor=render_factor)
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
        logger.exception("Colorize error")
        return JSONResponse(status_code=500, content={"detail": "Internal processing error"})


@v1_router.post("/restore")
def restore(
    file: UploadFile = File(...),
    tile_size: int = Query(0, ge=0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    """Remove noise or blur from a photo using the NAFNet model.

    Args:
        file: Input image file.
        tile_size: Tile size for processing (0 = no tiling).
        output_format: Desired output image format.

    Returns:
        The restored image as binary content.
    """
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    model = get_model("restore")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Restoration model not loaded"},
        )

    try:
        result = model.predict(image, tile_size=tile_size)
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
        logger.exception("Restore error")
        return JSONResponse(status_code=500, content={"detail": "Internal processing error"})


@v1_router.post("/face-restore")
def face_restore(
    file: UploadFile = File(...),
    fidelity: float = Query(0.5, ge=0.0, le=1.0),
    upscale: int = Query(2, ge=1, le=4),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    """Enhance and restore faces in a photo using the CodeFormer model.

    Detects all faces, restores each one, and pastes them back. Returns a
    bicubic upscale if no faces are detected.

    Args:
        file: Input image file.
        fidelity: Quality vs fidelity balance (0.0-1.0).
        upscale: Output upscale factor (1-4).
        output_format: Desired output image format.

    Returns:
        The face-restored image as binary content.
    """
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    model = get_model("face")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Face restoration model not loaded"},
        )

    try:
        result = model.predict(image, fidelity=fidelity, upscale=upscale)
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
        logger.exception("Face restore error")
        return JSONResponse(status_code=500, content={"detail": "Internal processing error"})


@v1_router.post("/upscale")
def upscale(
    file: UploadFile = File(...),
    scale: int = Query(4, ge=1, le=8),
    tile_size: int = Query(512, ge=0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    """Upscale an image using the Real-ESRGAN super-resolution model.

    Args:
        file: Input image file.
        scale: Desired upscale factor (1-8).
        tile_size: Tile size for processing (0 = no tiling).
        output_format: Desired output image format.

    Returns:
        The upscaled image as binary content.
    """
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    model = get_model("upscale")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Upscale model not loaded"},
        )

    try:
        result = model.predict(image, scale=scale, tile_size=tile_size)
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
        logger.exception("Upscale error")
        return JSONResponse(status_code=500, content={"detail": "Internal processing error"})


@v1_router.post("/old-photo-restore")
def old_photo_restore(
    file: UploadFile = File(...),
    with_scratch: bool = Query(True),
    with_face: bool = Query(True),
    scratch_threshold: float = Query(0.4, ge=0.0, le=1.0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    """Restore an old or damaged photo.

    Detects and repairs scratches, restores global quality via VAE, and
    optionally enhances detected faces using a SPADE generator.

    Args:
        file: Input image file.
        with_scratch: Enable automatic scratch detection and repair.
        with_face: Enable face enhancement.
        scratch_threshold: Sensitivity for scratch detection (0.0-1.0).
        output_format: Desired output image format.

    Returns:
        The restored image as binary content.
    """
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    model = get_model("old_photo_restore")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Old photo restoration model not loaded"},
        )

    try:
        result = model.predict(
            image,
            with_scratch=with_scratch,
            with_face=with_face,
            scratch_threshold=scratch_threshold,
        )
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
        logger.exception("Old photo restore error")
        return JSONResponse(status_code=500, content={"detail": "Internal processing error"})


@v1_router.post("/pipeline")
def pipeline(
    file: UploadFile = File(...),
    old_photo_restore: bool = Query(False),
    restore: bool = Query(True),
    face_restore: bool = Query(True),
    colorize: bool = Query(True),
    upscale: bool = Query(True),
    width: int = Query(2400, ge=1),
    height: int = Query(2400, ge=1),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
    # old photo restore params
    with_scratch: bool = Query(True),
    scratch_threshold: float = Query(0.4, ge=0.0, le=1.0),
    # colorize params
    render_factor: int = Query(35, ge=1, le=100),
    # restore params
    restore_tile_size: int = Query(0, ge=0),
    # face restore params
    fidelity: float = Query(0.7, ge=0.0, le=1.0),
    # upscale params
    upscale_tile_size: int = Query(512, ge=0),
):
    """Run multiple enhancement steps in a single request.

    Pipeline order: old_photo_restore -> colorize -> restore -> upscale -> resize -> face restore.

    Args:
        file: Input image file.
        old_photo_restore: Enable old photo restoration step (off by default).
        restore: Enable restoration step.
        face_restore: Enable face restoration step.
        colorize: Enable colorization step.
        upscale: Enable upscale step.
        width: Target output width.
        height: Target output height.
        output_format: Desired output image format.
        with_scratch: Enable scratch detection in old photo restore.
        scratch_threshold: Scratch detection sensitivity (0.0-1.0).
        render_factor: Colorization intensity (1-100).
        restore_tile_size: Tile size for restoration (0 = no tiling).
        fidelity: Face restoration fidelity (0.0-1.0).
        upscale_tile_size: Tile size for upscaling (0 = no tiling).

    Returns:
        The processed image as binary content.
    """
    try:
        file_bytes = file.file.read()
        check_file_size(file_bytes)
        image = read_image(file_bytes)
        image = validate_and_resize(image)
    except FileTooLargeError as e:
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    steps = []
    if old_photo_restore:
        steps.append(("old_photo_restore", "Old photo restoration"))
    if colorize:
        steps.append(("colorize", "Colorization"))
    if restore:
        steps.append(("restore", "Restoration"))
    if upscale:
        steps.append(("upscale", "Upscale"))
    if face_restore:
        steps.append(("face", "Face restoration"))

    if not steps:
        return JSONResponse(
            status_code=400,
            content={"detail": "At least one pipeline step must be enabled"},
        )

    missing = [label for key, label in steps if key not in model_registry]
    if missing:
        return JSONResponse(
            status_code=503,
            content={"detail": f"{', '.join(missing)} model(s) not loaded"},
        )

    try:
        if old_photo_restore:
            image = model_registry["old_photo_restore"].predict(
                image,
                with_scratch=with_scratch,
                scratch_threshold=scratch_threshold,
            )
            cuda_clear()
        if colorize:
            image = model_registry["colorize"].predict(image, render_factor=render_factor)
            cuda_clear()
        if restore:
            image = model_registry["restore"].predict(image, tile_size=restore_tile_size)
            cuda_clear()

        h, w = image.shape[:2]
        target = max(width, height)
        if upscale and max(w, h) < target:
            image = model_registry["upscale"].predict(image, tile_size=upscale_tile_size)
            cuda_clear()
            h, w = image.shape[:2]

        scale = min(width / w, height / h)
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
            image = cv2.resize(image, (new_w, new_h), interpolation=interp)

        if face_restore:
            image = model_registry["face"].predict(image, fidelity=fidelity, upscale=1)
            cuda_clear()

        output = encode_image(image, output_format)
        return Response(
            content=output,
            media_type=MEDIA_TYPES.get(output_format, "image/png"),
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception:
        cuda_clear()
        logger.exception("Pipeline error")
        return JSONResponse(status_code=500, content={"detail": "Internal processing error"})


app.include_router(v1_router)


# ---------------------------------------------------------------------------
# Legacy redirects (307 preserves POST method and body)
# ---------------------------------------------------------------------------

_LEGACY_PATHS = [
    "/colorize",
    "/restore",
    "/face-restore",
    "/upscale",
    "/old-photo-restore",
    "/pipeline",
]


for _path in _LEGACY_PATHS:

    def _make_redirect(p: str):
        async def _redirect(request: Request):
            url = request.url.replace(path=f"/v1{p}")
            return RedirectResponse(url=str(url), status_code=307)

        return _redirect

    app.add_api_route(
        _path,
        _make_redirect(_path),
        methods=["POST"],
        include_in_schema=False,
    )


# ---------------------------------------------------------------------------
# Infrastructure endpoints (root, not versioned)
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Return server health status, device info, and loaded models.

    Returns:
        JSON with status, device, loaded model list, and optional CUDA info.
    """
    cuda_info = None
    if torch.cuda.is_available():
        cuda_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
        }

    return {
        "status": "healthy",
        "device": device,
        "loaded_models": list(model_registry.keys()),
        "cuda_info": cuda_info,
    }
