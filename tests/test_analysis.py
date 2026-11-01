import shutil
from pathlib import Path

import pandas as pd

from palmprint.data.dataset import generate_synthetic_roi
from palmprint.data.preparation import prepare
from palmprint.evaluation.analysis import benchmark_cpu, data_eda, error_analysis, robustness
from palmprint.training.classical import evaluate_final, final_train, lock_protocol
from palmprint.training.selection import cross_validate


def test_secondary_analysis_on_synthetic(tmp_path):
    source = Path(__file__).resolve().parents[1]
    ws = tmp_path / "work"
    shutil.copytree(source / "configs", ws / "configs")
    root = tmp_path / "synthetic"
    generate_synthetic_roi(root, palms=3, samples=4)
    prepare(ws, root, strict=False, folds=2)
    eda = data_eda(ws)
    assert eda["n_palms"] == 3
    cross_validate(ws, "knn", ws / "configs/compcode.yaml")
    final_train(ws, "knn", ws / "configs/compcode.yaml")
    for name in ("svm", "crc"):
        cross_validate(ws, name, ws / "configs/compcode.yaml")
        final_train(ws, name, ws / "configs/compcode.yaml")
    lock_protocol(ws)
    evaluate_final(ws, "knn", ws / "configs/compcode.yaml")
    assert error_analysis(ws, "knn")["model"] == "knn"
    assert benchmark_cpu(ws, "knn", limit=10)["n_queries"] == 10
    rr = robustness(ws, "knn", limit=2)
    assert len(rr["results"]) == 5
    assert (ws / "reports/figures/eda_brightness.png").exists()
    assert pd.read_csv(ws / "outputs/runs/knn/robustness.csv").shape[0] == 5
