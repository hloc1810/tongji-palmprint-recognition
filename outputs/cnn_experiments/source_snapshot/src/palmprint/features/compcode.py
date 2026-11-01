"""Competitive orientation code and fixed spatial histograms.

The implementation is paper-inspired. Numerical parity with the official MATLAB
CR_CompCode release is NOT claimed without comparing intermediate outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property

import cv2
import numpy as np


@dataclass(frozen=True)
class CompCodeConfig:
    image_size: int = 128
    num_directions: int = 6
    sigma_x: float = 4.85
    sigma_y: float = 9.31
    frequency: float = 0.071
    kernel_size: int = 41
    block_pixels: int = 14
    zero_dc: bool = True
    response: str = "min_real"
    border_mode: str = "reflect101"
    histogram_norm: str = "per_block_l1_then_global_l2"

    def __post_init__(self) -> None:
        if self.image_size < 16 or self.num_directions != 6 or self.block_pixels < 1:
            raise ValueError("Invalid ROI size, block size, or direction count")
        if self.kernel_size < 3 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd and >=3")
        if self.sigma_x <= 0 or self.sigma_y <= 0 or self.frequency <= 0:
            raise ValueError("Gabor parameters must be positive")
        if self.response != "min_real" or self.border_mode != "reflect101":
            raise ValueError("Only min_real and reflect101 are implemented")
        if self.histogram_norm != "per_block_l1_then_global_l2":
            raise ValueError("Unexpected histogram normalization")


class CompCodeExtractor:
    def __init__(self, config: CompCodeConfig):
        self.config = config

    @cached_property
    def kernels(self) -> tuple[np.ndarray, ...]:
        cfg = self.config
        r = cfg.kernel_size // 2
        y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float64)
        filters = []
        for direction in range(cfg.num_directions):
            theta = np.pi * direction / cfg.num_directions
            xr = x * np.cos(theta) + y * np.sin(theta)
            yr = -x * np.sin(theta) + y * np.cos(theta)
            envelope = np.exp(-0.5 * ((xr / cfg.sigma_x)**2 + (yr / cfg.sigma_y)**2))
            kernel = envelope * np.cos(2 * np.pi * cfg.frequency * xr)
            if cfg.zero_dc:
                # Finite support DC correction. Paper-inspired, not MATLAB bit parity.
                kernel -= kernel.mean()
            magnitude = np.linalg.norm(kernel)
            if magnitude < 1e-12:
                raise ValueError("Degenerate Gabor kernel")
            filters.append(np.asarray(kernel / magnitude, dtype=np.float32))
        return tuple(filters)

    def orientation_map(self, gray: np.ndarray) -> np.ndarray:
        if gray.shape != (self.config.image_size, self.config.image_size) or gray.ndim != 2:
            raise ValueError(f"Expected grayscale ROI {self.config.image_size}x{self.config.image_size}")
        if not np.isfinite(gray).all():
            raise ValueError("Non-finite ROI")
        image = gray.astype(np.float32, copy=False) / 255.0
        responses = [cv2.filter2D(image, cv2.CV_32F, k,
                                  borderType=cv2.BORDER_REFLECT_101)
                     for k in self.kernels]
        return np.argmin(np.stack(responses, axis=0), axis=0).astype(np.uint8)

    def histogram(self, codes: np.ndarray, *, global_histogram: bool = False) -> np.ndarray:
        cfg = self.config
        if codes.shape != (cfg.image_size, cfg.image_size):
            raise ValueError("Invalid orientation-map shape")
        if codes.min() < 0 or codes.max() >= cfg.num_directions:
            raise ValueError("CompCode contains invalid direction ID")
        if global_histogram:
            segments = [codes]
        else:
            # Partial edge blocks are included, never silently discarded.
            segments = [codes[y:min(y+cfg.block_pixels, cfg.image_size),
                              x:min(x+cfg.block_pixels, cfg.image_size)]
                        for y in range(0, cfg.image_size, cfg.block_pixels)
                        for x in range(0, cfg.image_size, cfg.block_pixels)]
        parts = []
        for block in segments:
            hist = np.bincount(block.ravel().astype(int), minlength=cfg.num_directions).astype(np.float32)
            hist /= max(float(hist.sum()), 1.0)
            parts.append(hist)
        z = np.concatenate(parts)
        norm = np.linalg.norm(z)
        if norm > 1e-12:
            z /= norm
        return z.astype(np.float32)

    def transform(self, gray: np.ndarray, *, global_histogram: bool = False) -> np.ndarray:
        return self.histogram(self.orientation_map(gray), global_histogram=global_histogram)

    @property
    def feature_dim(self) -> int:
        cfg = self.config
        blocks = (cfg.image_size + cfg.block_pixels - 1) // cfg.block_pixels
        return cfg.num_directions * blocks * blocks

    def metadata(self) -> dict:
        return asdict(self.config) | {"feature_dim": self.feature_dim,
                                      "source_parity": "not_verified"}
