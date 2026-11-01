import numpy as np
import pytest

from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor


def test_orientation_and_hist_shape():
    ext = CompCodeExtractor(CompCodeConfig())
    img = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    codes = ext.orientation_map(img)
    assert codes.shape == (128, 128)
    assert set(np.unique(codes)).issubset(set(range(6)))
    h1 = ext.transform(img)
    h2 = ext.transform(img)
    assert len(h1) == 600  # 10 x 10 blocks of 14px, each with 6 bins
    assert np.allclose(h1, h2)
    assert np.isclose(np.linalg.norm(h1), 1.0)
    assert len(ext.transform(img, global_histogram=True)) == 6


def test_invalid_feature_config():
    with pytest.raises(ValueError):
        CompCodeConfig(kernel_size=20)
    with pytest.raises(ValueError):
        CompCodeConfig(block_pixels=0)
