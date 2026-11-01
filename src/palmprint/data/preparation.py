"""Dataset manifest, split preparation and data audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils import atomic_json, hash_file
from .dataset import build_manifest


def repo_defaults() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_dirs(ws: Path) -> None:
    for d in (
        "data/metadata",
        "data/splits",
        "outputs/cache",
        "outputs/cv",
        "outputs/runs",
        "artifacts",
        "reports/figures",
        "reports/tables",
    ):
        (ws / d).mkdir(parents=True, exist_ok=True)


def prepare(
    ws: Path, data_root: Path, *, strict: bool = True, folds: int = 5, seed: int = 42
) -> dict:
    workspace_dirs(ws)
    if (ws / "artifacts/protocol_lock.json").exists():
        raise RuntimeError("Protocol locked; cannot modify manifest")
    df = build_manifest(data_root, strict=strict, folds=folds, seed=seed)
    df.to_csv(ws / "data/metadata/manifest.csv", index=False)
    df.loc[df.session == "session1", ["sample_uid", "palm_id", "fold"]].to_csv(
        ws / "data/splits/fold_assignments.csv", index=False
    )
    mapping = {
        "data_root": str(data_root.resolve()),
        "strict": strict,
        "folds": folds,
        "seed": seed,
        "manifest_sha256": hash_file(ws / "data/metadata/manifest.csv"),
    }
    atomic_json(ws / "data/metadata/local_dataset.json", mapping)
    dupes = df.groupby("sha256").size()
    summary = {
        "n_images": len(df),
        "n_palms": int(df.palm_id.nunique()),
        "session1": int((df.session == "session1").sum()),
        "session2": int((df.session == "session2").sum()),
        "same_content_groups": int((dupes > 1).sum()),
        "strict": strict,
        "folds": folds,
        "seed": seed,
        "quality_by_session": {
            session: {
                name: {"mean": float(group[name].mean()), "std": float(group[name].std())}
                for name in ("brightness", "contrast", "blur_proxy")
            }
            for session, group in df.groupby("session")
        },
    }
    atomic_json(ws / "data/metadata/audit.json", summary)
    return summary


def load_manifest(ws: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = ws / "data/metadata/manifest.csv"
    info = json.loads((ws / "data/metadata/local_dataset.json").read_text())
    if hash_file(manifest_path) != info["manifest_sha256"]:
        raise RuntimeError("Manifest changed after prepare; prepare again and invalidate runs")
    df = pd.read_csv(manifest_path, dtype={"stem": "string"})
    return df, info
