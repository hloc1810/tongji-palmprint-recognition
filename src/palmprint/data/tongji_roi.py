"""Tongji 2017 Section 4 geometry, independently implemented for dark-background hands.

The paper specifies the common tangent and scale geometry, but not segmentation
thresholds or a numerical gap detector. Those engineering choices are explicit in
ROIConfig. This module does not claim parity with the authors' implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ROIConfig:
    c1: float = 0.60
    c2: float = 0.65
    c3: float = 0.10
    size: int = 128
    gaussian_kernel: int = 5
    gaussian_sigma: float = 1.5
    threshold: int = 25
    curvature_step_fraction: float = 0.0375
    curvature_minimum: float = 0.55
    curvature_peak_spacing_fraction: float = 0.075
    tangent_patch_fraction: float = 0.18
    minimum_coverage: float = 0.98

    def __post_init__(self):
        if self.size < 2 or not (0 < self.c1 and 0 < self.c2 and 0 <= self.c3):
            raise ValueError("Invalid ROI geometry")
        if self.gaussian_kernel < 3 or self.gaussian_kernel % 2 != 1:
            raise ValueError("Gaussian kernel must be odd and >=3")
        if not (0 < self.threshold < 255 and 0 < self.minimum_coverage <= 1):
            raise ValueError("Invalid mask parameters")


class ROIExtractionError(ValueError):
    """Expected image/geometry failure; never silently substitute a center crop."""


@dataclass
class ROIResult:
    roi: np.ndarray
    mask: np.ndarray
    landmarks: dict


def segment_hand(gray: np.ndarray, cfg: ROIConfig) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (cfg.gaussian_kernel, cfg.gaussian_kernel), cfg.gaussian_sigma)
    _, binary = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ROIExtractionError("empty_hand_mask")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.10 * gray.size:
        raise ROIExtractionError("hand_mask_too_small")
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask, contour.reshape(-1, 2).astype(np.float64)


def detect_gap_patches(mask: np.ndarray, contour: np.ndarray, cfg: ROIConfig):
    """Find three upper, approximately aligned concave webs on the hand contour.

    Finite-scale turning angles work with fingertips clipped by image borders,
    where a single convex-hull defect can span several gaps. The triple score
    favors deep, upper, evenly spaced webs in Tongji's upright acquisition pose.
    """
    h, w = mask.shape
    k = max(3, int(cfg.curvature_step_fraction * w))
    incoming = contour - np.roll(contour, k, axis=0)
    outgoing = np.roll(contour, -k, axis=0) - contour
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    # findContours external contours have negative oriented area; normalize defensively.
    orientation = -np.sign(cv2.contourArea(contour.astype(np.float32), oriented=True))
    angle = np.arctan2(orientation * cross, (incoming * outgoing).sum(axis=1))
    peaks, _ = find_peaks(
        angle,
        height=cfg.curvature_minimum,
        prominence=0.3,
        distance=max(1, int(cfg.curvature_peak_spacing_fraction * w)),
    )
    points = contour[peaks]
    peaks = peaks[
        (points[:, 1] > 0.02 * h)
        & (points[:, 1] < 0.60 * h)
        & (points[:, 0] > 0.06 * w)
        & (points[:, 0] < 0.94 * w)
    ]
    if len(peaks) < 3:
        raise ROIExtractionError(f"fewer_than_three_finger_gaps:{len(peaks)}")
    candidates = []
    for triple in combinations(peaks.tolist(), 3):
        indices = sorted(triple, key=lambda i: contour[i, 0])
        p = contour[indices]
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ratio = lengths.min() / max(lengths.max(), 1)
        if ratio < 0.45 or np.min(np.diff(p[:, 0])) < 0.055 * w:
            continue
        chord = p[2] - p[0]
        d = np.linalg.norm(chord)
        deviation = abs(chord[0] * (p[1, 1] - p[0, 1]) - chord[1] * (p[1, 0] - p[0, 0])) / max(
            d * d, 1
        )
        if deviation > 0.25 or abs(chord[1]) > 0.8 * abs(chord[0]):
            continue
        score = float(angle[indices].sum() - 2 * p[:, 1].mean() / h - (1 - ratio) - 2 * deviation)
        candidates.append((score, indices))
    if not candidates:
        raise ROIExtractionError("no_plausible_finger_gap_triple")
    gap_points = contour[max(candidates, key=lambda item: item[0])[1]]
    first, last = gap_points[[0, 2]]
    separation = float(np.linalg.norm(last - first))
    if not 0.12 * w < separation < 0.75 * w:
        raise ROIExtractionError("implausible_gap_separation")
    radius = cfg.tangent_patch_fraction * separation
    patches = []
    for point in (first, last):
        patch = contour[np.linalg.norm(contour - point, axis=1) <= radius]
        if len(patch) < 3:
            raise ROIExtractionError("insufficient_gap_boundary")
        patches.append(patch)
    return gap_points, patches


def common_tangent(patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Lower common supporting tangent of the two U-shaped gap boundaries."""
    a, b = patches

    def support(p, slope):
        return p[:, 1] - slope * p[:, 0]

    def difference(slope):
        return float(support(b, slope).max() - support(a, slope).max())

    if difference(-1.5) * difference(1.5) > 0:
        raise ROIExtractionError("no_common_gap_tangent")
    slope = brentq(difference, -1.5, 1.5, xtol=1e-10)
    return a[np.argmax(support(a, slope))], b[np.argmax(support(b, slope))]


def contour_cross_section(contour, origin, axis):
    """Intersect a continuous scan line with polygon edges; keep the palm interval."""
    normal = np.array([-axis[1], axis[0]])
    relative = contour - origin
    v = relative @ normal
    u = relative @ axis
    v2, u2 = np.roll(v, -1), np.roll(u, -1)
    crossing = ((v <= 0) & (v2 > 0)) | ((v2 <= 0) & (v > 0))
    locations = np.sort(u[crossing] - v[crossing] * (u2 - u)[crossing] / (v2 - v)[crossing])
    left = locations[locations <= 0]
    right = locations[locations > 0]
    if not len(left) or not len(right):
        raise ROIExtractionError("palm_cross_section_missing")
    return origin + left[-1] * axis, origin + right[0] * axis


def extract_roi(image: np.ndarray, cfg: ROIConfig | None = None) -> ROIResult:
    cfg = cfg or ROIConfig()
    if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ROIExtractionError("expected_uint8_hand_image")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ROIExtractionError("expected_gray_BGR_or_BGRA")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask, contour = segment_hand(gray, cfg)
    gaps, patches = detect_gap_patches(mask, contour, cfg)
    x1, x2 = common_tangent(patches)
    d = float(np.linalg.norm(x2 - x1))
    u = (x2 - x1) / d
    v = np.array([-u[1], u[0]])  # Into the palm for upright Tongji images.
    origin = (x1 + x2) / 2
    scan_center = origin + cfg.c1 * d * v
    o1, o2 = contour_cross_section(contour, scan_center, u)
    width = float(np.linalg.norm(o2 - o1))
    side = cfg.c2 * width
    # c3 is the distance from X-axis to the NEAREST square edge, not its center.
    top_center = origin + cfg.c3 * width * v
    corners = np.stack(
        [
            top_center - side / 2 * u,
            top_center + side / 2 * u,
            top_center + side / 2 * u + side * v,
            top_center - side / 2 * u + side * v,
        ]
    )
    h, w = gray.shape
    if side < 0.15 * min(h, w):
        raise ROIExtractionError("roi_too_small")
    if (corners < 0).any() or (corners[:, 0] > w - 1).any() or (corners[:, 1] > h - 1).any():
        raise ROIExtractionError("roi_outside_image")
    n = cfg.size - 1
    transform = cv2.getAffineTransform(
        corners[[0, 1, 3]].astype(np.float32),
        np.array([[0, 0], [n, 0], [0, n]], np.float32),
    )
    aligned_mask = cv2.warpAffine(mask, transform, (cfg.size, cfg.size), flags=cv2.INTER_NEAREST)
    coverage = float((aligned_mask > 0).mean())
    if coverage < cfg.minimum_coverage:
        raise ROIExtractionError(f"roi_background_fraction:{1 - coverage:.4f}")
    # The Gaussian is for mask detection only; sample original grayscale texture.
    roi = cv2.warpAffine(gray, transform, (cfg.size, cfg.size), flags=cv2.INTER_LINEAR)
    landmarks = {
        "gaps": gaps.tolist(),
        "X1": x1.tolist(),
        "X2": x2.tolist(),
        "O": origin.tolist(),
        "O1": o1.tolist(),
        "O2": o2.tolist(),
        "corners": corners.tolist(),
        "gap_distance": d,
        "palm_width": width,
        "side": side,
        "coverage": coverage,
        "source_to_roi": transform.tolist(),
        "config": asdict(cfg),
    }
    return ROIResult(roi, mask, landmarks)


def draw_overlay(image: np.ndarray, landmarks: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    def point(name):
        return tuple(np.rint(landmarks[name]).astype(int))

    cv2.polylines(canvas, [np.rint(landmarks["corners"]).astype(np.int32)], True, (0, 255, 0), 2)
    cv2.line(canvas, point("X1"), point("X2"), (0, 255, 255), 2)
    cv2.line(canvas, point("O1"), point("O2"), (255, 200, 0), 2)
    for name in ("X1", "X2", "O", "O1", "O2"):
        cv2.circle(canvas, point(name), 4, (0, 0, 255), -1)
        cv2.putText(canvas, name, point(name), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas
