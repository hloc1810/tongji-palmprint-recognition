"""Read-only verification of existing embedding runs before report aggregation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from palmprint.reporting.evidence import verify_run_evidence

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs/embedding_benchmark"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = ROOT / "data/metadata/manifest.csv"
manifest = pd.read_csv(manifest_path)
probes = manifest[manifest.session == "session2"].reset_index(drop=True)
gallery_frame = manifest[manifest.session == "session1"].reset_index(drop=True)
assert len(probes) == len(gallery_frame) == 6000
records = []
protected = list((ROOT / "src").rglob("*.py"))
protected += list((ROOT / "outputs/runs").rglob("metrics.json"))
protected += list((ROOT / "outputs/runs").rglob("predictions.csv"))
protected += list((ROOT / "outputs/runs").rglob("all_class_scores.npz"))
protected += list((ROOT / "artifacts").glob("*"))
protected = [p for p in protected if p.is_file()]
for seed in (17, 29, 43):
    artifact = ROOT / f"artifacts/resnet18_seed{seed}.pth"
    gallery = ROOT / f"artifacts/resnet18_seed{seed}_gallery512.npz"
    with np.load(gallery, allow_pickle=False) as data:
        assert data["features"].shape == (6000, 512)
        np.testing.assert_allclose(np.linalg.norm(data["features"], axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(data["sample_uid"], gallery_frame.sample_uid.astype(str))
        np.testing.assert_array_equal(data["palm_id"], gallery_frame.palm_id)
        assert str(data["checkpoint_sha256"].item()) == sha(artifact)
        assert str(data["manifest_sha256"].item()) == sha(manifest_path)
    baseline = json.loads((ROOT / f"outputs/runs/resnet18_seed{seed}/metrics.json").read_text())
    for strategy in ("nearest", "centroid"):
        run = ROOT / f"outputs/runs/resnet18_seed{seed}_embedding_{strategy}"
        metric = json.loads((run / "metrics.json").read_text())
        assert metric["artifact_sha256"] == sha(artifact) == baseline["artifact_sha256"]
        assert metric["gallery_sha256"] == sha(gallery)
        assert metric["gallery_session"] == "session1" and metric["probe_session"] == "session2"
        assert metric["seed"] == seed and metric["gallery_strategy"] == strategy
        result = verify_run_evidence(run, expected_probes=probes,
                                     manifest_sha256=sha(manifest_path))
        records.append({**result, "metrics_sha256": sha(run / "metrics.json"),
                        "artifact_sha256": sha(artifact), "gallery_sha256": sha(gallery)})
        print(json.dumps(result), flush=True)
out = {"runs": records,
       "protected_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                            for p in protected}}
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
