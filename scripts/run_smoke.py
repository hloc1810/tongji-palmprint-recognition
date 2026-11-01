"""Fully runnable small synthetic end-to-end test, NOT scientific Tongji performance."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from palmprint.data.dataset import generate_synthetic_roi
from palmprint.data.preparation import prepare
from palmprint.reporting.summary import generate_report
from palmprint.training.classical import evaluate_final, final_train, lock_protocol
from palmprint.training.selection import cross_validate


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="palmprint_smoke_") as d:
        root = Path(d)
        ws = root / "workspace"
        shutil.copytree(source / "configs", ws / "configs")
        fake = root / "synthetic_roi"
        generate_synthetic_roi(fake, palms=6, samples=4)
        summary = prepare(ws, fake, strict=False, folds=2)
        assert summary["n_images"] == 48
        for name in ("knn", "svm", "crc"):
            choice = cross_validate(ws, name, ws / "configs/compcode.yaml")
            print("CV", name, choice["best_parameter"], choice["mean_cv_rank1"])
            final_train(ws, name, ws / "configs/compcode.yaml")
        lock_protocol(ws)
        for name in ("knn", "svm", "crc"):
            results = evaluate_final(ws, name, ws / "configs/compcode.yaml")
            assert results["n"] == 24
            assert results["rank1"] <= results["rank5"]
            print("SYNTHETIC TEST", name, results["rank1"])
        print(generate_report(ws))
        print("Smoke test completed. All numbers above are SYNTHETIC, NOT Tongji results.")


if __name__ == "__main__":
    main()
