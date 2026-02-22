"""Unit tests for utils/image_ops.py."""

import cv2
import numpy as np
import pytest

from utils.image_ops import encode_image, read_image, validate_and_resize


class TestReadImage:
    def test_valid_jpeg(self, sample_image_bytes):
        img = read_image(sample_image_bytes)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR

    def test_valid_png(self, sample_png_bytes):
        img = read_image(sample_png_bytes)
        assert isinstance(img, np.ndarray)
        assert img.shape == (32, 32, 3)

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="Could not decode image"):
            read_image(b"not an image at all")

    def test_empty_bytes_raises(self):
        with pytest.raises((ValueError, cv2.error)):
            read_image(b"")

    def test_truncated_jpeg_raises(self, sample_image_bytes):
        with pytest.raises(ValueError, match="Could not decode image"):
            read_image(sample_image_bytes[:10])


class TestValidateAndResize:
    def test_small_image_unchanged(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = validate_and_resize(img)
        assert result.shape == (100, 200, 3)
        assert result is img  # same object, no copy

    def test_exact_max_dim_unchanged(self):
        img = np.zeros((2048, 1024, 3), dtype=np.uint8)
        result = validate_and_resize(img)
        assert result is img

    def test_oversized_width_resized(self):
        img = np.zeros((1000, 4096, 3), dtype=np.uint8)
        result = validate_and_resize(img)
        assert result.shape[1] == 2048
        assert result.shape[0] == 500  # proportional

    def test_oversized_height_resized(self):
        img = np.zeros((4096, 1000, 3), dtype=np.uint8)
        result = validate_and_resize(img)
        assert result.shape[0] == 2048
        assert result.shape[1] == 500

    def test_both_oversized(self):
        img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        result = validate_and_resize(img)
        # 4000 is the larger dim, scale = 2048/4000
        assert max(result.shape[:2]) <= 2048

    def test_custom_max_dim(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = validate_and_resize(img, max_dim=256)
        assert max(result.shape[:2]) <= 256


class TestEncodeImage:
    def test_encode_png(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = encode_image(img, "png")
        assert data[:4] == b"\x89PNG"

    def test_encode_jpg(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = encode_image(img, "jpg")
        assert data[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_encode_jpeg_alias(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = encode_image(img, "jpeg")
        assert data[:2] == b"\xff\xd8"

    def test_encode_webp(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = encode_image(img, "webp")
        assert data[:4] == b"RIFF"

    def test_unsupported_format_raises(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported output format"):
            encode_image(img, "bmp")

    def test_format_case_insensitive(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = encode_image(img, "PNG")
        assert data[:4] == b"\x89PNG"

    def test_format_strips_dot(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = encode_image(img, ".png")
        assert data[:4] == b"\x89PNG"
