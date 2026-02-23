"""Shared test fixtures for photo-enhancer tests."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _make_mock_model(output_shape=(64, 64, 3)):
    """Create a MagicMock model whose predict() returns a valid BGR image."""
    mock = MagicMock()
    mock.predict.return_value = np.random.randint(0, 255, output_shape, dtype=np.uint8)
    return mock


@pytest.fixture()
def mock_registry():
    """Dict of mock model wrappers for all four model categories."""
    return {
        "colorize": _make_mock_model(),
        "restore": _make_mock_model(),
        "face": _make_mock_model(),
        "upscale": _make_mock_model(),
        "old_photo_restore": _make_mock_model(),
        "inpaint": _make_mock_model(),
    }


@pytest.fixture()
def client(mock_registry):
    """FastAPI TestClient with mocked models and no-op lifespan."""
    import main

    @asynccontextmanager
    async def _noop_lifespan(app):
        main.model_registry.clear()
        main.model_registry.update(mock_registry)
        yield
        main.model_registry.clear()

    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    with TestClient(main.app) as tc:
        yield tc
    main.app.router.lifespan_context = original_lifespan


@pytest.fixture()
def sample_image_bytes():
    """A small valid JPEG image as bytes."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (128, 64, 200)  # some color
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture()
def sample_png_bytes():
    """A small valid PNG image as bytes."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = (50, 100, 150)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture()
def oversized_file_bytes():
    """Bytes exceeding the 32 MB file size limit."""
    return b"\x00" * (33 * 1024 * 1024)
