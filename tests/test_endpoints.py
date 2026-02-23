"""Integration tests for API endpoints with mocked model registry."""


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "loaded_models" in data

    def test_health_lists_loaded_models(self, client):
        data = client.get("/health").json()
        assert set(data["loaded_models"]) == {
            "colorize",
            "restore",
            "face",
            "upscale",
            "old_photo_restore",
            "inpaint",
        }


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_request" in resp.text or "HELP" in resp.text


class TestColorizeEndpoint:
    def test_success(self, client, sample_image_bytes):
        resp = client.post("/v1/colorize", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_webp_output(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/colorize?output_format=webp",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_invalid_image_returns_400(self, client):
        resp = client.post("/v1/colorize", files={"file": ("bad.jpg", b"not an image")})
        assert resp.status_code == 400
        assert (
            "decode" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()
        )

    def test_oversized_file_returns_413(self, client, oversized_file_bytes):
        resp = client.post("/v1/colorize", files={"file": ("big.jpg", oversized_file_bytes)})
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()


class TestRestoreEndpoint:
    def test_success(self, client, sample_image_bytes):
        resp = client.post("/v1/restore", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 200

    def test_invalid_image_returns_400(self, client):
        resp = client.post("/v1/restore", files={"file": ("bad.jpg", b"nope")})
        assert resp.status_code == 400


class TestFaceRestoreEndpoint:
    def test_success(self, client, sample_image_bytes):
        resp = client.post("/v1/face-restore", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 200

    def test_invalid_image_returns_400(self, client):
        resp = client.post("/v1/face-restore", files={"file": ("bad.jpg", b"x")})
        assert resp.status_code == 400


class TestUpscaleEndpoint:
    def test_success(self, client, sample_image_bytes):
        resp = client.post("/v1/upscale", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 200

    def test_jpg_output(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/upscale?output_format=jpg",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"


class TestOldPhotoRestoreEndpoint:
    def test_success(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/old-photo-restore",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_params(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/old-photo-restore?with_scratch=false&with_face=false&scratch_threshold=0.6",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200

    def test_webp_output(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/old-photo-restore?output_format=webp",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_invalid_image_returns_400(self, client):
        resp = client.post(
            "/v1/old-photo-restore",
            files={"file": ("bad.jpg", b"not an image")},
        )
        assert resp.status_code == 400

    def test_oversized_file_returns_413(self, client, oversized_file_bytes):
        resp = client.post(
            "/v1/old-photo-restore",
            files={"file": ("big.jpg", oversized_file_bytes)},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_503_when_model_missing(self, client, sample_image_bytes):
        import main

        main.model_registry.pop("old_photo_restore", None)
        resp = client.post(
            "/v1/old-photo-restore",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()

    def test_500_on_predict_exception(self, client, sample_image_bytes):
        import main

        main.model_registry["old_photo_restore"].predict.side_effect = RuntimeError("boom")
        resp = client.post(
            "/v1/old-photo-restore",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal processing error"


class TestInpaintEndpoint:
    POINTS = "[[10,10],[50,10],[50,50],[10,50]]"

    def test_success(self, client, sample_image_bytes):
        resp = client.post(
            f"/v1/inpaint?points={self.POINTS}",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_webp_output(self, client, sample_image_bytes):
        resp = client.post(
            f"/v1/inpaint?points={self.POINTS}&output_format=webp",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_invalid_image_returns_400(self, client):
        resp = client.post(
            f"/v1/inpaint?points={self.POINTS}",
            files={"file": ("bad.jpg", b"not an image")},
        )
        assert resp.status_code == 400

    def test_invalid_points_returns_400(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/inpaint?points=not-json",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()

    def test_insufficient_points_returns_400(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/inpaint?points=[[0,0],[1,1]]",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 400
        assert "at least 3" in resp.json()["detail"].lower()

    def test_oversized_file_returns_413(self, client, oversized_file_bytes):
        resp = client.post(
            f"/v1/inpaint?points={self.POINTS}",
            files={"file": ("big.jpg", oversized_file_bytes)},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_503_when_model_missing(self, client, sample_image_bytes):
        import main

        main.model_registry.pop("inpaint", None)
        resp = client.post(
            f"/v1/inpaint?points={self.POINTS}",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()

    def test_500_on_predict_exception(self, client, sample_image_bytes):
        import main

        main.model_registry["inpaint"].predict.side_effect = RuntimeError("boom")
        resp = client.post(
            f"/v1/inpaint?points={self.POINTS}",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal processing error"


class TestPipelineEndpoint:
    def test_success_all_steps(self, client, sample_image_bytes):
        resp = client.post("/v1/pipeline", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 200

    def test_no_steps_returns_400(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/pipeline?colorize=false&restore=false&face_restore=false&upscale=false",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 400
        assert "at least one" in resp.json()["detail"].lower()

    def test_old_photo_restore_in_pipeline(self, client, sample_image_bytes):
        resp = client.post(
            "/v1/pipeline?old_photo_restore=true&colorize=false&restore=false"
            "&face_restore=false&upscale=false",
            files={"file": ("test.jpg", sample_image_bytes)},
        )
        assert resp.status_code == 200

    def test_oversized_file_returns_413(self, client, oversized_file_bytes):
        resp = client.post("/v1/pipeline", files={"file": ("big.jpg", oversized_file_bytes)})
        assert resp.status_code == 413


class TestModelNotLoaded:
    def test_503_when_model_missing(self, client, sample_image_bytes):
        """If a model is removed from the registry, endpoint returns 503."""
        import main

        main.model_registry.pop("colorize", None)
        resp = client.post("/v1/colorize", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()


class TestModelPredictError:
    def test_500_on_predict_exception(self, client, sample_image_bytes):
        """If model.predict() raises, endpoint returns 500."""
        import main

        main.model_registry["colorize"].predict.side_effect = RuntimeError("boom")
        resp = client.post("/v1/colorize", files={"file": ("test.jpg", sample_image_bytes)})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal processing error"


class TestLegacyRedirects:
    def test_colorize_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/colorize",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/colorize" in resp.headers["location"]

    def test_restore_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/restore",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/restore" in resp.headers["location"]

    def test_face_restore_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/face-restore",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/face-restore" in resp.headers["location"]

    def test_upscale_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/upscale",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/upscale" in resp.headers["location"]

    def test_old_photo_restore_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/old-photo-restore",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/old-photo-restore" in resp.headers["location"]

    def test_inpaint_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/inpaint?points=[[0,0],[10,0],[10,10]]",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/inpaint" in resp.headers["location"]

    def test_pipeline_redirect(self, client, sample_image_bytes):
        resp = client.post(
            "/pipeline",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=False,
        )
        assert resp.status_code == 307
        assert "/v1/pipeline" in resp.headers["location"]

    def test_legacy_redirect_follows_to_success(self, client, sample_image_bytes):
        """Following the redirect should reach the v1 endpoint and succeed."""
        resp = client.post(
            "/colorize",
            files={"file": ("test.jpg", sample_image_bytes)},
            follow_redirects=True,
        )
        assert resp.status_code == 200
