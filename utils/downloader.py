import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

MODEL_URLS: dict[str, dict[str, str]] = {
    "colorize": {
        "paper_tiny": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth",
        "modelscope": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
        "artistic": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_artistic.pth",
    },
    "restore": {
        "denoise": "https://drive.google.com/uc?export=download&id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X",
        "deblur": "https://drive.google.com/uc?export=download&id=1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj",
    },
    "face": {
        "v0.1": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    },
    "upscale": {
        "x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "x4anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
}

FILENAME_OVERRIDES: dict[str, dict[str, str]] = {
    "restore": {
        "denoise": "NAFNet-SIDD-width64.pth",
        "deblur": "NAFNet-GoPro-width64.pth",
    },
}


def _resolve_google_drive_url(url: str) -> str:
    """Convert a Google Drive share/uc URL to the usercontent direct-download URL."""
    import re

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        file_id = match.group(1)
        return (
            f"https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t"
        )
    return url


def _download_file(
    url: str, dest: Path, part: Path, category: str, variant: str
) -> None:
    """Download a file, with Google Drive large-file handling."""
    session = requests.Session()

    if "drive.google.com" in url:
        url = _resolve_google_drive_url(url)

    resp = session.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(part, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                logger.info("  %s/%s — %d%%", category, variant, pct)
    resp.close()

    # Sanity check: reject HTML responses saved as model files
    with open(part, "rb") as f:
        header = f.read(16)
    if header.startswith(b"<") or header.startswith(b"<!"):
        part.unlink()
        raise RuntimeError(
            f"Downloaded file for {category}/{variant} appears to be HTML, "
            "not a model checkpoint. The download URL may be invalid."
        )


def ensure_model_exists(
    category: str,
    variant: str,
    weights_dir: str = os.environ.get("WEIGHTS_DIR", "/app/weights"),
) -> str:
    if category not in MODEL_URLS:
        raise ValueError(f"Unknown model category: {category}")
    variants = MODEL_URLS[category]
    if variant not in variants:
        raise ValueError(
            f"Unknown variant '{variant}' for category '{category}'. "
            f"Available: {list(variants.keys())}"
        )

    url = variants[variant]

    filename = (
        FILENAME_OVERRIDES.get(category, {}).get(variant)
        or url.split("/")[-1].split("?")[0]
    )

    dest_dir = Path(weights_dir) / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        # Validate existing file isn't a corrupt HTML download
        with open(dest_path, "rb") as f:
            header = f.read(16)
        if header.startswith(b"<") or header.startswith(b"<!"):
            logger.warning(
                "Existing file %s is HTML (corrupt download), re-downloading",
                dest_path,
            )
            dest_path.unlink()
        else:
            logger.info("Model already exists: %s", dest_path)
            return str(dest_path)

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    logger.info("Downloading %s/%s from %s ...", category, variant, url)

    try:
        _download_file(url, dest_path, part_path, category, variant)
        os.rename(part_path, dest_path)
        logger.info("Saved model to %s", dest_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    return str(dest_path)
