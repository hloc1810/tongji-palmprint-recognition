"""Test FastAPI inference using real classifiers and synthetic ROI fixtures."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.server as server
from app.server import app
from palmprint.data.dataset import generate_synthetic_roi, read_gray
from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor
from palmprint.models.classical import CollaborativeRidge, LinearSVM, NearestTemplate

WS = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_workspace(tmp_path_factory):
    """An isolated, explicitly synthetic dataset; never a Tongji benchmark."""
    workspace = tmp_path_factory.mktemp("server_synthetic")
    static = workspace / "app/static"
    static.mkdir(parents=True)
    shutil.copy2(WS / "app/static/index.html", static / "index.html")
    generate_synthetic_roi(workspace / "data/raw", palms=6, samples=2, seed=17)
    config = CompCodeConfig()
    extractor = CompCodeExtractor(config)
    names = [f"{(palm - 1) * 10 + 1:05d}.png" for palm in range(1, 7)]
    images = [read_gray(workspace / "data/raw/session2" / name) for name in names]
    features = np.stack([extractor.transform(image) for image in images])
    labels = np.arange(1, 7, dtype=np.int32)
    artifacts = workspace / "artifacts"
    artifacts.mkdir()
    for name, model in (
        ("crc", CollaborativeRidge(lam=0.1)),
        ("svm", LinearSVM(C=1.0)),
        ("knn", NearestTemplate()),
    ):
        model.fit(features, labels)
        joblib.dump(
            {"model": model, "feature_config": asdict(config)}, artifacts / f"{name}.joblib"
        )
    return workspace, names


@pytest.fixture(autouse=True)
def isolated_server(monkeypatch, synthetic_workspace):
    workspace, names = synthetic_workspace
    monkeypatch.setattr(server, "WS", workspace)
    monkeypatch.setattr(
        server,
        "SAMPLE_DEFINITIONS",
        [(i, name, i, f"Synthetic palm {i}") for i, name in enumerate(names, 1)],
    )
    for name, value in (
        ("MODELS", {}),
        ("EXTRACTOR", None),
        ("FEATURE_CONFIG", {}),
        ("SAMPLES", []),
        ("SAMPLE_IMAGES", {}),
    ):
        monkeypatch.setattr(server, name, value)


@pytest.mark.parametrize("path", ["/", "/Tongji_Palmprint_UI.html"])
def test_server_root_serves_html(path):
    with TestClient(app) as client:
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        text = response.text
        assert "modelSelect" in text
        assert "sampleGrid" in text
        assert "top5Wrap" in text
        assert "SAMPLES" in text
        # Confirm no hardcoded base64 in previewImg and dynamic API sample endpoints present
        assert 'id="previewImg"' in text
        assert "/api/sample-image/" in text
        preview_snippet = text[text.find('id="previewImg"') : text.find('id="previewImg"') + 200]
        assert ";base64," not in preview_snippet


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


def test_server_samples_and_image_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/samples")
        assert res.status_code == 200
        samples = res.json()
        assert len(samples) >= 6
        for s in samples:
            assert "image_url" in s and s["image_url"].startswith("/api/sample-image/")
            assert "palm_id" in s and isinstance(s["palm_id"], int)
            assert "filename" in s

            # Fetch the actual sample image via dynamic image endpoint
            img_res = client.get(s["image_url"])
            assert img_res.status_code == 200
            assert img_res.headers["content-type"] == "image/png"
            assert len(img_res.content) > 1000


def test_server_recognize_synthetic_image(synthetic_workspace):
    workspace, names = synthetic_workspace
    sample_path = workspace / "data/raw/session2" / names[0]
    img_bytes = sample_path.read_bytes()

    with TestClient(app) as client:
        for model in ("crc", "svm", "knn"):
            res = client.post(
                "/api/recognize",
                files={"file": ("00001.png", img_bytes, "image/png")},
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
