import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image(file_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — invalid or unsupported format")
    return image


def validate_and_resize(image: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= max_dim and w <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.warning("Image too large (%dx%d), resizing to %dx%d", w, h, new_w, new_h)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image(image: np.ndarray, output_format: str = "png") -> bytes:
    fmt = output_format.lower().strip(".")
    ext_map = {
        "png": ".png",
        "jpg": ".jpg",
        "jpeg": ".jpg",
        "webp": ".webp",
    }
    if fmt not in ext_map:
        raise ValueError(
            f"Unsupported output format: '{output_format}'. Supported: png, jpg, webp"
        )

    params = []
    if fmt in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]

    success, buf = cv2.imencode(ext_map[fmt], image, params)
    if not success:
        raise ValueError(f"Failed to encode image as {fmt}")
    return buf.tobytes()
