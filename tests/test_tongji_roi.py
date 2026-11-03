"""Geometry and failure-policy tests; synthetic hands are not research results."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from palmprint.data.tongji_roi import (
    ROIExtractionError,
    common_tangent,
    contour_cross_section,
    extract_roi,
)


def synthetic_hand():
    outline = np.array(
        [
            [180, 590],
            [180, 300],
            [35, 220],
            [30, 140],
            [80, 130],
            [200, 240],
            [230, 200],
            [230, 0],
            [280, 0],
            [290, 120],
            [310, 130],
            [330, 120],
            [340, 0],
            [390, 0],
            [400, 110],
            [420, 120],
            [440, 110],
            [460, 0],
            [510, 0],
            [510, 140],
            [530, 150],
            [550, 140],
            [590, 0],
            [640, 0],
            [610, 300],
            [600, 590],
        ],
        np.int32,
    )
    mask = np.zeros((600, 800), np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    y, x = np.mgrid[:600, :800]
    texture = np.uint8(80 + 0.08 * x + 0.05 * y + 12 * np.sin((x + y) / 20))
    return np.where(mask > 0, texture, 0).astype(np.uint8)


def test_tangent_uses_support_points_instead_of_raw_valley_centers():
    x = np.linspace(-2, 2, 801)
    a = np.column_stack((x, 10 - x**2))
    b = np.column_stack((x + 20, 14 - x**2))
    p, q = common_tangent([a, b])
    slope = (q[1] - p[1]) / (q[0] - p[0])
    assert slope == pytest.approx(0.2, abs=1e-6)
    assert p[0] == pytest.approx(-0.1, abs=0.005)
    for patch in (a, b):
        assert np.max(patch[:, 1] - slope * patch[:, 0]) == pytest.approx(
            p[1] - slope * p[0], abs=1e-6
        )


def test_oblique_contour_intersections_are_subpixel():
    rectangle = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)
    axis = np.array([1.0, 0.2])
    axis /= np.linalg.norm(axis)
    left, right = contour_cross_section(rectangle, np.array([50.0, 40.0]), axis)
    np.testing.assert_allclose(left, [0, 30])
    np.testing.assert_allclose(right, [100, 50])


def test_clipped_fingers_geometry_and_c3_nearest_edge():
    image = synthetic_hand()
    result = extract_roi(image)
    assert result.roi.shape == (128, 128)
    assert result.roi.dtype == np.uint8
    marks = result.landmarks
    np.testing.assert_allclose(np.array(marks["gaps"])[:, 0], [310, 420, 530], atol=10)
    corners = np.array(marks["corners"])
    o = np.array(marks["O"])
    u = (np.array(marks["X2"]) - marks["X1"]) / marks["gap_distance"]
    v = np.array([-u[1], u[0]])
    near_edge = (corners[0] + corners[1]) / 2
    assert (near_edge - o) @ v == pytest.approx(0.1 * marks["palm_width"])
    assert (near_edge - o) @ u == pytest.approx(0, abs=1e-8)
    np.testing.assert_allclose(
        np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1), 0.65 * marks["palm_width"]
    )


@pytest.mark.parametrize("degrees,scale", [(8, 1.0), (-8, 1.0), (0, 0.85)])
def test_small_similarity_transforms_preserve_roi_texture(degrees, scale):
    image = synthetic_hand()
    reference = extract_roi(image)
    transform = cv2.getRotationMatrix2D((400, 300), degrees, scale)
    changed = cv2.warpAffine(image, transform, (800, 600))
    actual = extract_roi(changed)
    expected_center = transform @ np.r_[np.mean(reference.landmarks["corners"], axis=0), 1]
    np.testing.assert_allclose(
        np.mean(actual.landmarks["corners"], axis=0), expected_center, atol=8
    )
    assert np.abs(reference.roi.astype(float) - actual.roi).mean() < 7


def test_color_conversion_and_source_unchanged():
    image = synthetic_hand()
    original = image.copy()
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    np.testing.assert_array_equal(extract_roi(image).roi, extract_roi(color).roi)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((600, 800), np.uint8),
        np.full((600, 800), 150, np.uint8),
        np.ones((600, 800), np.uint16),
        np.zeros((600, 800, 2), np.uint8),
    ],
)
def test_invalid_images_fail_without_center_crop(image):
    with pytest.raises(ROIExtractionError):
        extract_roi(image)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "roi_comparison_runner",
        Path(__file__).resolve().parents[1] / "scripts/run_roi_comparison.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_probes_and_missing_gallery_classes_stay_in_denominator():
    runner = load_runner()
    frame = pd.DataFrame(
        [
            {"sample_uid": "session1/a", "session": "session1", "palm_id": 1},
            {"sample_uid": "session1/b", "session": "session1", "palm_id": 2},
            {"sample_uid": "session2/a", "session": "session2", "palm_id": 1},
            {"sample_uid": "session2/b", "session": "session2", "palm_id": 2},
            {"sample_uid": "session2/c", "session": "session2", "palm_id": 1},
        ]
    )
    features = {
        "session1/a": np.array([1.0, 0]),
        "session2/a": np.array([1.0, 0]),
        "session2/b": np.array([0.0, 1]),
    }
    result, predictions = runner.evaluate(features, frame)
    assert result["probe_denominator"] == 3
    assert result["gallery_classes"] == 1
    assert result["rank1"] == pytest.approx(1 / 3)
    assert predictions.predicted_palm_id.tolist() == [1, 1, -1]
    result, predictions = runner.evaluate({}, frame)
    assert result["rank1"] == 0
    assert len(predictions) == 3


def test_pairing_requires_all_preselected_images_and_detects_duplicate_stems(tmp_path):
    runner = load_runner()
    for root in ("original", "official"):
        for session in ("session1", "session2"):
            folder = tmp_path / root / session
            folder.mkdir(parents=True)
            for number in range(1, 11):
                (folder / f"{number:05d}.png").write_bytes(b"index fixture only")
    paired = runner.select_samples(tmp_path / "original", tmp_path / "official", [1])
    assert len(paired) == 20
    assert paired.sample_uid.nunique() == 20
    (tmp_path / "original/session2/00010.png").unlink()
    with pytest.raises(ValueError, match="missing sample"):
        runner.select_samples(tmp_path / "original", tmp_path / "official", [1])
    (tmp_path / "official/session1/00001.bmp").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="Duplicate stem"):
        runner.image_index(tmp_path / "official", "session1")
