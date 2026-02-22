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

MODEL_URLS_MULTI: dict[str, dict[str, dict[str, str]]] = {
    "old_photo_restore": {
        "v1": {
            "scratch_detection.pt": "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Global/checkpoints/detection/FT_Epoch_latest.pt",
            "vae_a_encoder.pth": "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Global/checkpoints/restoration/VAE_A_quality/latest_net_G.pth",
            "vae_b_decoder.pth": "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Global/checkpoints/restoration/VAE_B_scratch/latest_net_G.pth",
            "mapping_net.pth": "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Global/checkpoints/restoration/mapping_scratch/latest_net_mapping_net.pth",
            "face_enhance_gen.pth": "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Face_Enhancement/checkpoints/Setting_9_epoch_100/latest_net_G.pth",
            "shape_predictor_68_face_landmarks.dat": "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Face_Detection/shape_predictor_68_face_landmarks.dat",
        },
    },
}

FILENAME_OVERRIDES: dict[str, dict[str, str]] = {
    "restore": {
        "denoise": "NAFNet-SIDD-width64.pth",
        "deblur": "NAFNet-GoPro-width64.pth",
    },
}


def _resolve_google_drive_url(url: str) -> str:
    """Convert a Google Drive share/uc URL to a direct-download URL.

    Extracts the file ID from the URL and constructs a
    ``drive.usercontent.google.com`` URL that bypasses the virus-scan
    interstitial page.

    Args:
        url: Original Google Drive URL containing an ``id`` query parameter.

    Returns:
        Direct-download URL, or the original URL if no file ID is found.
    """
    import re

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        file_id = match.group(1)
        return (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        )
    return url


def _download_file(url: str, dest: Path, part: Path, category: str, variant: str) -> None:
    """Download a model weight file to ``part``, validating it is not HTML.

    Google Drive URLs are resolved to direct-download URLs before fetching.
    After download, the first bytes are checked to reject HTML error pages
    that may be served instead of the real model file.

    Args:
        url: Source download URL.
        dest: Final destination path (not written by this function).
        part: Temporary ``.part`` path to stream into.
        category: Model category name (for logging).
        variant: Model variant name (for logging).

    Raises:
        requests.HTTPError: If the HTTP response status is not OK.
        RuntimeError: If the downloaded file appears to be HTML.
    """
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
    """Ensure the weight file for the given model category/variant exists on disk.

    If the file is already present and valid, returns immediately. Otherwise
    downloads it from the URL registered in ``MODEL_URLS``.

    Args:
        category: Model category (e.g. ``"colorize"``, ``"restore"``).
        variant: Variant name within the category.
        weights_dir: Root directory for weight storage.

    Returns:
        Absolute path to the weight file.

    Raises:
        ValueError: If the category or variant is unknown.
        RuntimeError: If the download produces a corrupt (HTML) file.
    """
    if category not in MODEL_URLS:
        raise ValueError(f"Unknown model category: {category}")
    variants = MODEL_URLS[category]
    if variant not in variants:
        raise ValueError(
            f"Unknown variant '{variant}' for category '{category}'. "
            f"Available: {list(variants.keys())}"
        )

    url = variants[variant]

    filename = FILENAME_OVERRIDES.get(category, {}).get(variant) or url.split("/")[-1].split("?")[0]

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


def ensure_model_files_exist(
    category: str,
    variant: str,
    weights_dir: str = os.environ.get("WEIGHTS_DIR", "/app/weights"),
) -> str:
    """Ensure all weight files for a multi-file model exist on disk.

    Downloads any missing files from the URLs registered in ``MODEL_URLS_MULTI``.

    Args:
        category: Model category (e.g. ``"old_photo_restore"``).
        variant: Variant name within the category.
        weights_dir: Root directory for weight storage.

    Returns:
        Absolute path to the directory containing the weight files.

    Raises:
        ValueError: If the category or variant is unknown.
        RuntimeError: If a download produces a corrupt (HTML) file.
    """
    if category not in MODEL_URLS_MULTI:
        raise ValueError(f"Unknown multi-file model category: {category}")
    variants = MODEL_URLS_MULTI[category]
    if variant not in variants:
        raise ValueError(
            f"Unknown variant '{variant}' for category '{category}'. "
            f"Available: {list(variants.keys())}"
        )

    file_urls = variants[variant]
    dest_dir = Path(weights_dir) / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in file_urls.items():
        dest_path = dest_dir / filename
        if dest_path.exists():
            with open(dest_path, "rb") as f:
                header = f.read(16)
            if header.startswith(b"<") or header.startswith(b"<!"):
                logger.warning(
                    "Existing file %s is HTML (corrupt download), re-downloading",
                    dest_path,
                )
                dest_path.unlink()
            else:
                logger.info("Model file already exists: %s", dest_path)
                continue

        part_path = dest_path.with_suffix(dest_path.suffix + ".part")
        logger.info("Downloading %s/%s/%s from %s ...", category, variant, filename, url)

        try:
            _download_file(url, dest_path, part_path, category, f"{variant}/{filename}")
            os.rename(part_path, dest_path)
            logger.info("Saved model file to %s", dest_path)
        except Exception:
            if part_path.exists():
                part_path.unlink()
            raise

    return str(dest_dir)
