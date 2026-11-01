"""CNN training regression tests on fake images only: no Tongji data required."""

import shutil
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize("device", ["directml:0", "cpu"])
def test_cnn_full_train_save_evaluate_and_freeze(tmp_path, device):
    pytest.importorskip("torch")
    if device.startswith("directml"):
        directml = pytest.importorskip("torch_directml")
        if directml.device_count() == 0:
            pytest.skip("No DirectML GPU available")
    from palmprint.data.dataset import generate_synthetic_roi
    from palmprint.data.preparation import prepare
    from palmprint.evaluation.cnn import evaluate_one, predict_one_cnn
    from palmprint.training.cnn import train_one

    root = tmp_path / "synthetic"
    ws = tmp_path / "workspace"
    source_configs = Path(__file__).resolve().parents[1] / "configs"
    shutil.copytree(source_configs, ws / "configs")
    generate_synthetic_roi(root, palms=3, samples=4)
    prepare(ws, root, strict=False, folds=2)
    meta = train_one(
        ws,
        seed=3,
        weights="none",
        head_epochs=1,
        finetune_epochs=1,
        batch_size=6,
        size=32,
        patience=1,
        amp=False,
        accumulate=2,
        grad_clip=0.5,
        device=device,
    )
    assert meta["n_classes"] == 3
    assert meta["head_epochs_final"] == 1
    assert meta["training_config"]["effective_batch_size"] == 12
    assert meta["training_config"]["amp_used"] is False
    assert meta["dev_val_best_loss"] > 0
    hist = pd.read_csv(ws / "outputs/runs/resnet18_seed3/dev_history.csv")
    assert {"train_rank1", "dev_val_rank1", "dev_val_loss", "learning_rates"} <= set(hist.columns)
    assert len(hist) == 2
    assert (ws / "outputs/runs/resnet18_seed3/best_dev.pth").exists()
    with pytest.raises(RuntimeError, match="already frozen"):
        train_one(ws, seed=3, weights="none", head_epochs=1, finetune_epochs=0, size=32)
    result = evaluate_one(ws, seed=3, batch_size=6, device=device)
    assert result["n"] == 12
    from palmprint.reporting.summary import generate_report

    report = generate_report(ws)
    assert report["method"].tolist() == ["ResNet18 ROI classification"]
    assert report.data_scope.tolist() == ["NON_STRICT_fixture_or_subset"]
    pred = predict_one_cnn(ws, 3, root / "session2" / "00001.png")
    assert pred["predicted_palm_id"] in (1, 2, 3)
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_one(ws, seed=3, batch_size=6)


def test_cnn_cli_read_yaml_and_override(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from palmprint import cli_cnn as cnn

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "resnet18.yaml").write_text(
        "batch_size: 7\nhead_epochs: 2\nimage_size: 96\namp: true\n"
    )
    calls = []

    def fake_train(ws, **kwargs):
        calls.append((ws, kwargs))
        return {"ok": True}

    monkeypatch.setattr(cnn, "train_one", fake_train)
    cnn.main(
        ["--workspace", str(tmp_path), "train", "--seed", "91", "--batch-size", "3", "--no-amp"]
    )
    assert calls[0][0] == tmp_path
    assert calls[0][1]["batch_size"] == 3
    assert calls[0][1]["head_epochs"] == 2
    assert calls[0][1]["size"] == 96
    assert calls[0][1]["amp"] is False
    assert '"ok": true' in capsys.readouterr().out.lower()


def test_train_epoch_amp_fallback_and_invalid_settings():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from palmprint.training.cnn import train_epoch, validate_epoch

    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    X = torch.rand(5, 1, 2, 2)
    y = torch.tensor([0, 1, 2, 0, 1])
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    result = train_epoch(
        net,
        loader,
        optimizer,
        torch.device("cpu"),
        accumulate=2,
        label_smoothing=0.05,
        grad_clip=1.0,
        amp=True,
    )
    assert result["samples"] == 5
    assert 0 <= result["train_rank1"] <= 1
    assert result["train_loss"] > 0
    val = validate_epoch(net, loader, torch.device("cpu"), amp=True)
    assert val["val_samples"] == 5
    assert val["dev_val_loss"] > 0
    with pytest.raises(ValueError, match="accumulation"):
        train_epoch(net, loader, optimizer, torch.device("cpu"), accumulate=0)


def test_cnn_dataset_picklable_for_windows_spawn(tmp_path):
    pytest.importorskip("torch")
    import pickle

    from palmprint.models.resnet import make_dataset

    rows = pd.DataFrame({"path": ["session1/00001.png"], "palm_id": [1]})
    dataset = make_dataset(rows, tmp_path, train=False, size=64)
    assert len(pickle.loads(pickle.dumps(dataset))) == 1


def test_explicit_cpu_device_and_unavailable_backend():
    torch = pytest.importorskip("torch")
    from palmprint.training.cnn import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="CNN device"):
        resolve_device("meta")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")
