"""Audit real ROI inputs without fitting models or evaluating session2 recognition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from palmprint.data.dataset import read_gray, resolve_path
from palmprint.data.preparation import load_manifest, prepare
from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor
from palmprint.utils import atomic_json, env_info, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    ws, root = args.workspace.resolve(), args.data.resolve()
    summary = prepare(ws, root, strict=True, folds=5, seed=42)
    manifest, info = load_manifest(ws)
    config = CompCodeConfig(**read_yaml(ws / "configs/compcode.yaml"))
    extractor = CompCodeExtractor(config)
    # Fixed representative IDs. Never choose samples by recognition accuracy.
    samples = manifest[
        (manifest.session == "session1") & manifest.palm_id.isin([1, 150, 300, 450, 600])
    ]
    samples = samples.groupby("palm_id", sort=True).head(1)
    sanity = []
    for row in samples.itertuples(index=False):
        gray = read_gray(resolve_path(root, row.path))
        codes = extractor.orientation_map(gray)
        feature = extractor.histogram(codes)
        if (
            feature.shape != (600,)
            or not np.isfinite(feature).all()
            or not np.isclose(np.linalg.norm(feature), 1.0)
        ):
            raise RuntimeError(f"Real ROI feature sanity failed: {row.sample_uid}")
        sanity.append(
            {
                "sample_uid": row.sample_uid,
                "shape": list(gray.shape),
                "dtype": str(gray.dtype),
                "orientation_codes": np.unique(codes).tolist(),
                "feature_dimension": len(feature),
                "feature_l2_norm": float(np.linalg.norm(feature)),
            }
        )
    result = {
        "audit": summary,
        "manifest_sha256": info["manifest_sha256"],
        "environment": env_info(),
        "session1_feature_sanity": sanity,
        "scope": "Input integrity and feature sanity only; no recognition accuracy measured",
        "source_authenticity": "Layout/content checks do not authenticate download provenance",
        "matlab_parity": "not_verified",
    }
    dest = ws / "outputs/research_correctness/roi_audit.json"
    atomic_json(dest, result)
    print(f"Strict audit passed: {summary['n_images']} images, {summary['n_palms']} Palm IDs")
    print(f"Real session1 feature sanity: {len(sanity)} samples; evidence: {dest}")


if __name__ == "__main__":
    main()
