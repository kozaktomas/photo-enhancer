"""Unit tests for utils/downloader.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.downloader import (
    _resolve_google_drive_url,
    ensure_model_exists,
    ensure_model_files_exist,
)


class TestResolveGoogleDriveUrl:
    def test_extracts_file_id(self):
        url = "https://drive.google.com/uc?export=download&id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X"
        result = _resolve_google_drive_url(url)
        assert "drive.usercontent.google.com" in result
        assert "14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X" in result
        assert "confirm=t" in result

    def test_no_id_returns_original(self):
        url = "https://example.com/model.pth"
        assert _resolve_google_drive_url(url) == url

    def test_id_with_ampersand_prefix(self):
        url = "https://drive.google.com/uc?export=download&id=ABC123_-def"
        result = _resolve_google_drive_url(url)
        assert "ABC123_-def" in result


class TestEnsureModelExists:
    def test_unknown_category_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown model category"):
            ensure_model_exists("nonexistent", "v1", weights_dir=str(tmp_path))

    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown variant"):
            ensure_model_exists("colorize", "nonexistent_variant", weights_dir=str(tmp_path))

    def test_cached_file_returns_immediately(self, tmp_path):
        """If the weight file already exists with valid content, skip download."""
        weight_dir = tmp_path / "colorize"
        weight_dir.mkdir()
        weight_file = weight_dir / "ddcolor_modelscope.pth"
        # Write some binary content (not HTML)
        weight_file.write_bytes(b"\x80\x02" + b"\x00" * 100)

        result = ensure_model_exists("colorize", "modelscope", weights_dir=str(tmp_path))
        assert result == str(weight_file)

    def test_corrupt_html_triggers_redownload(self, tmp_path):
        """If existing file starts with '<', it's treated as corrupt."""
        weight_dir = tmp_path / "colorize"
        weight_dir.mkdir()
        weight_file = weight_dir / "ddcolor_modelscope.pth"
        weight_file.write_bytes(b"<!DOCTYPE html>...")

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "100"}
        mock_resp.iter_content.return_value = [b"\x80\x02" + b"\x00" * 98]
        mock_resp.raise_for_status = MagicMock()

        with patch("utils.downloader.requests.Session") as mock_session_cls:
            mock_session_cls.return_value.get.return_value = mock_resp
            result = ensure_model_exists("colorize", "modelscope", weights_dir=str(tmp_path))

        assert Path(result).exists()
        # Should not start with HTML after re-download
        with open(result, "rb") as f:
            assert not f.read(2).startswith(b"<")

    @patch("utils.downloader.requests.Session")
    def test_successful_download(self, mock_session_cls, tmp_path):
        """Download should save file and return path."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "50"}
        mock_resp.iter_content.return_value = [b"\x80\x02" + b"\x00" * 48]
        mock_resp.raise_for_status = MagicMock()
        mock_session_cls.return_value.get.return_value = mock_resp

        result = ensure_model_exists("face", "v0.1", weights_dir=str(tmp_path))
        assert Path(result).exists()
        assert "codeformer.pth" in result


class TestEnsureModelFilesExist:
    def test_unknown_category_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown multi-file model category"):
            ensure_model_files_exist("nonexistent", "v1", weights_dir=str(tmp_path))

    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown variant"):
            ensure_model_files_exist(
                "old_photo_restore", "nonexistent_variant", weights_dir=str(tmp_path)
            )

    def test_cached_files_return_immediately(self, tmp_path):
        """If all weight files already exist, skip download and return dir."""
        weight_dir = tmp_path / "old_photo_restore"
        weight_dir.mkdir()
        filenames = [
            "scratch_detection.pt",
            "vae_a_encoder.pth",
            "vae_b_decoder.pth",
            "mapping_net.pth",
            "face_enhance_gen.pth",
            "shape_predictor_68_face_landmarks.dat",
        ]
        for fn in filenames:
            (weight_dir / fn).write_bytes(b"\x80\x02" + b"\x00" * 100)

        result = ensure_model_files_exist("old_photo_restore", "v1", weights_dir=str(tmp_path))
        assert result == str(weight_dir)
