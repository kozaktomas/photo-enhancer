import logging
import os
import traceback
import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings(
    "ignore", message=".*weights_only=False.*", category=FutureWarning
)
warnings.filterwarnings("ignore", message=".*pretrained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weight enum.*", category=UserWarning)

import cv2  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, File, Query, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from models.wrappers import (  # noqa: E402
    CodeFormerWrapper,
    DDColorWrapper,
    NAFNetWrapper,
    RealESRGANWrapper,
)
from utils.downloader import ensure_model_exists  # noqa: E402
from utils.image_ops import encode_image, read_image, validate_and_resize  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

model_registry: dict[str, object] = {}
device: str = "cpu"

MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def detect_device() -> str:
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
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    detect_device()

    for category, cfg in MODEL_CONFIG.items():
        variant = os.environ.get(cfg["env_var"], cfg["default"])
        try:
            model_path = ensure_model_exists(category, variant)
            wrapper = cfg["wrapper"](model_path, device, variant=variant)
            model_registry[category] = wrapper
            logger.info("Loaded model: %s/%s", category, variant)
        except Exception:
            logger.error(
                "Failed to load model %s/%s:\n%s",
                category,
                variant,
                traceback.format_exc(),
            )

    logger.info(
        "Startup complete — %d/%d models loaded", len(model_registry), len(MODEL_CONFIG)
    )
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


def get_model(name: str):
    if name not in model_registry:
        return None
    return model_registry[name]


def cuda_clear():
    if device == "cuda":
        torch.cuda.empty_cache()


@app.post("/colorize")
def colorize(
    file: UploadFile = File(...),
    render_factor: int = Query(35, ge=1, le=100),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    try:
        image = read_image(file.file.read())
        image = validate_and_resize(image)
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
        logger.error("Colorize error:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )


@app.post("/restore")
def restore(
    file: UploadFile = File(...),
    tile_size: int = Query(0, ge=0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    try:
        image = read_image(file.file.read())
        image = validate_and_resize(image)
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
        logger.error("Restore error:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )


@app.post("/face-restore")
def face_restore(
    file: UploadFile = File(...),
    fidelity: float = Query(0.5, ge=0.0, le=1.0),
    upscale: int = Query(2, ge=1, le=4),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    try:
        image = read_image(file.file.read())
        image = validate_and_resize(image)
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
        logger.error("Face restore error:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )


@app.post("/upscale")
def upscale(
    file: UploadFile = File(...),
    scale: int = Query(4, ge=1, le=8),
    tile_size: int = Query(512, ge=0),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
):
    try:
        image = read_image(file.file.read())
        image = validate_and_resize(image)
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
        logger.error("Upscale error:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )


@app.post("/pipeline")
def pipeline(
    file: UploadFile = File(...),
    restore: bool = Query(True),
    face_restore: bool = Query(True),
    colorize: bool = Query(True),
    upscale: bool = Query(True),
    width: int = Query(2400, ge=1),
    height: int = Query(2400, ge=1),
    output_format: str = Query("png", pattern="^(png|jpg|jpeg|webp)$"),
    # colorize params
    render_factor: int = Query(35, ge=1, le=100),
    # restore params
    restore_tile_size: int = Query(0, ge=0),
    # face restore params
    fidelity: float = Query(0.7, ge=0.0, le=1.0),
    # upscale params
    upscale_tile_size: int = Query(512, ge=0),
):
    try:
        image = read_image(file.file.read())
        image = validate_and_resize(image)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    steps = []
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
        if colorize:
            image = model_registry["colorize"].predict(
                image, render_factor=render_factor
            )
            cuda_clear()
        if restore:
            image = model_registry["restore"].predict(
                image, tile_size=restore_tile_size
            )
            cuda_clear()

        h, w = image.shape[:2]
        target = max(width, height)
        if upscale and max(w, h) < target:
            image = model_registry["upscale"].predict(
                image, tile_size=upscale_tile_size
            )
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
        logger.error("Pipeline error:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"detail": "Internal processing error"}
        )


@app.get("/health")
def health():
    cuda_info = None
    if torch.cuda.is_available():
        cuda_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            ),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
        }

    return {
        "status": "healthy",
        "device": device,
        "loaded_models": list(model_registry.keys()),
        "cuda_info": cuda_info,
    }
