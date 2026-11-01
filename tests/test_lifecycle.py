import shutil

import pandas as pd
import pytest

from palmprint.data.dataset import generate_synthetic_roi
from palmprint.data.preparation import prepare
from palmprint.features.extraction import load_features
from palmprint.training.classical import evaluate_final, final_train, lock_protocol, predict_one
from palmprint.training.selection import cross_validate


def test_end_to_end_and_locked_test(tmp_path):
    source = __import__("pathlib").Path(__file__).resolve().parents[1]
    ws = tmp_path / "work"
    shutil.copytree(source / "configs", ws / "configs")
    root = tmp_path / "fake"
    generate_synthetic_roi(root, palms=3, samples=4)
    prepare(ws, root, strict=False, folds=2)
    for name in ("knn", "svm", "crc"):
        cross_validate(ws, name, ws / "configs/compcode.yaml")
        final_train(ws, name, ws / "configs/compcode.yaml")
    lock_protocol(ws)
    for name in ("knn", "svm", "crc"):
        m = evaluate_final(ws, name, ws / "configs/compcode.yaml")
        assert m["n"] == 12
        assert m["rank5"] >= m["rank1"]
        with pytest.raises(RuntimeError, match="already evaluated"):
            evaluate_final(ws, name, ws / "configs/compcode.yaml")
        out = ws / "outputs/runs" / name / "predictions.csv"
        pred = pd.read_csv(out)
        assert len(pred) == 12
        result = predict_one(ws, name, root / "session2/00001.png")
        assert result["predicted_palm_id"] in (1, 2, 3)
    df, X, _ = load_features(ws, ws / "configs/compcode.yaml")
    assert X.shape == (12, 600)
    df2, X2, _ = load_features(ws, ws / "configs/compcode.yaml", sessions=("session2",))
    assert X2.shape == (12, 600)
