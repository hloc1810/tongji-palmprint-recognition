"""Contract tests for the shared Tongji Official ROI input path."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from palmprint.data.dataset import build_manifest, decode_gray, read_gray


def _write(path, pixels):
    ok, buffer = cv2.imencode(path.suffix, pixels)
    assert ok
    buffer.tofile(str(path))
    return path


def test_grayscale_uint8_preserves_values(tmp_path):
    gray = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
    image = _write(tmp_path / "roi.png", gray)
    assert np.array_equal(decode_gray(image), gray)
    assert np.array_equal(read_gray(image), gray)


def test_color_roi_has_fixed_opencv_conversion(tmp_path):
    bgr = np.empty((128, 128, 3), dtype=np.uint8)
    bgr[:, :] = [35, 120, 220]
    image = _write(tmp_path / "roi.png", bgr)
    expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(read_gray(image), expected)


def test_alpha_image_uses_the_same_gray_conversion(tmp_path):
    bgra = np.zeros((128, 128, 4), dtype=np.uint8)
    bgra[:, :, :] = [35, 120, 220, 255]
    image = _write(tmp_path / "roi.png", bgra)
    assert np.array_equal(read_gray(image), cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY))
    bgra[0, 0, 3] = 0
    _write(image, bgra)
    assert np.array_equal(read_gray(image), cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY))


def test_reject_unexpected_bit_depth_and_original_hand_image(tmp_path):
    high_bit = _write(tmp_path / "high.tif", np.full((128, 128), 1024, np.uint16))
    with pytest.raises(ValueError, match="8-bit"):
        read_gray(high_bit)
    full_hand = _write(tmp_path / "original.png", np.full((600, 800), 140, np.uint8))
    with pytest.raises(ValueError, match="Official ROI"):
        read_gray(full_hand)


def test_manifest_rejects_noncanonical_dtype(tmp_path):
    for session in ("session1", "session2"):
        folder = tmp_path / session
        folder.mkdir()
        _write(folder / "00001.tif", np.full((128, 128), 8192, np.uint16))
    with pytest.raises(ValueError, match="8-bit"):
        build_manifest(tmp_path, strict=False, folds=1)


def test_cnn_reads_identical_grayscale_as_classical(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from palmprint.models.resnet import PalmROIDataset, preprocessing
    from palmprint.utils import hash_file

    bgr = np.empty((128, 128, 3), dtype=np.uint8)
    bgr[:] = (2, 100, 221)
    file = _write(tmp_path / "00001.png", bgr)
    frame = pd.DataFrame(
        [
            {
                "path": file.name,
                "palm_id": 1,
                "sample_uid": "session1/00001",
                "sha256": hash_file(file),
            }
        ]
    )
    loader = PalmROIDataset(frame, tmp_path, train=False, size=224)
    x, y = loader[0]
    expected = preprocessing(False, 224)(Image.fromarray(read_gray(file)))
    assert np.allclose(x.numpy(), expected.numpy())
    assert y.item() == 0
    assert x.shape == (3, 224, 224)
    assert np.array_equal(loader[0][0].numpy(), x.numpy())  # no evaluation augmentation

    bgr[0, 0] = (90, 90, 90)
    _write(file, bgr)
    with pytest.raises(RuntimeError, match="changed after dataset audit"):
        PalmROIDataset(frame, tmp_path, train=False, size=224)


def test_cnn_rejects_full_hand_and_uint16(tmp_path):
    pytest.importorskip("torch")
    from palmprint.models.resnet import PalmROIDataset

    for name, pixels in [
        ("full.png", np.zeros((600, 800), dtype=np.uint8)),
        ("bit16.tif", np.full((128, 128), 256, dtype=np.uint16)),
    ]:
        image = _write(tmp_path / name, pixels)
        frame = pd.DataFrame([{"path": image.name, "palm_id": 1}])
        with pytest.raises(ValueError):
            PalmROIDataset(frame, tmp_path, train=False, size=224)[0]
