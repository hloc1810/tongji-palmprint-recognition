"""Tests for FastAPI backend server and real model inference endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.server import app

WS = Path(__file__).resolve().parents[1]


def test_server_root_serves_html():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        text = response.text
        assert "modelSelect" in text
        assert "sampleGrid" in text
        assert "top5Wrap" in text
        assert "const SAMPLES = [" in text


def test_server_status_and_metadata():
    with TestClient(app) as client:
        status_res = client.get("/api/status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["status"] == "online"
        assert status_data["samples_count"] >= 6
        for m in ("crc", "svm", "knn"):
            assert m in status_data["models"]
            assert status_data["models"][m]["loaded"] is True

        models_res = client.get("/api/models")
        assert models_res.status_code == 200
        models_data = models_res.json()
        assert len(models_data) == 3
        ids = [item["id"] for item in models_data]
        assert "crc" in ids and "svm" in ids and "knn" in ids


def test_server_samples_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/samples")
        assert res.status_code == 200
        samples = res.json()
        assert len(samples) >= 6
        for s in samples:
            assert "data_url" in s and s["data_url"].startswith("data:image/png;base64,")
            assert "palm_id" in s and isinstance(s["palm_id"], int)
            assert "filename" in s


def test_server_recognize_real_image():
    sample_path = WS / "data/raw/session2/00001.bmp"
    assert sample_path.exists(), "Sample test image must exist"
    img_bytes = sample_path.read_bytes()

    with TestClient(app) as client:
        for model in ("crc", "svm", "knn"):
            res = client.post(
                "/api/recognize",
                files={"file": ("00001.bmp", img_bytes, "image/bmp")},
                data={"model": model},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["palm_id"] == 1
            assert data["predicted_palm_id"] == 1
            assert data["true_palm_id"] == 1
            assert data["is_correct"] is True
            assert isinstance(data["score"], float)
            assert isinstance(data["latency_ms"], float)
            assert len(data["top5"]) == 5
            assert data["top5"][0]["palm_id"] == 1


def test_server_recognize_empty_file():
    with TestClient(app) as client:
        res = client.post(
            "/api/recognize",
            files={"file": ("empty.png", b"", "image/png")},
            data={"model": "crc"},
        )
        assert res.status_code == 400
