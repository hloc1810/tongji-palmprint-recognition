"""Read-only checks after unexpected cross-session degradation; never train or retune."""
from pathlib import Path
import json

import numpy as np
import torch
from palmprint.data.preparation import load_manifest
from palmprint.models.resnet import make_resnet
from palmprint.training.cnn import resolve_device, predict_loader, _loader
from palmprint.utils import atomic_json, hash_file


def main():
    ws = Path.cwd()
    device = resolve_device("directml:0")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    frame, dataset = load_manifest(ws)
    root = Path(dataset["data_root"])
    train = frame[frame.session == "session1"].reset_index(drop=True)
    sample = frame.groupby(["session", "palm_id"], sort=True).head(1).reset_index(drop=True)
    rows = []
    for seed in (17, 29, 43):
        path = ws / f"artifacts/resnet18_seed{seed}.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        gpu = make_resnet(600, "none").to(device)
        gpu.load_state_dict(checkpoint["state_dict"])
        train_logits, train_y = predict_loader(gpu, _loader(train, root, 32, False, 224), device)
        sample_gpu, sample_y = predict_loader(gpu, _loader(sample, root, 32, False, 224), device)
        cpu = make_resnet(600, "none")
        cpu.load_state_dict(checkpoint["state_dict"])
        sample_cpu, cpu_y = predict_loader(cpu, _loader(sample, root, 32, False, 224), torch.device("cpu"))
        assert np.array_equal(sample_y, cpu_y)
        result = {"seed": seed, "artifact_sha256": hash_file(path),
                  "session1_postfit_n": len(train_y),
                  "session1_postfit_rank1_no_augmentation": float((train_logits.argmax(1)+1 == train_y).mean()),
                  "cpu_gpu_parity_n": len(sample_y), "parity_sample_policy": "first ROI of each Palm ID in each session",
                  "max_absolute_logit_difference": float(np.max(np.abs(sample_gpu-sample_cpu))),
                  "mean_absolute_logit_difference": float(np.mean(np.abs(sample_gpu-sample_cpu))),
                  "top1_agreement_cpu_gpu": float((sample_gpu.argmax(1) == sample_cpu.argmax(1)).mean()),
                  "cpu_gpu_allclose_atol1e3_rtol1e3": bool(np.allclose(sample_gpu, sample_cpu, atol=1e-3, rtol=1e-3)),
                  "scope": "posthoc implementation diagnostic only, no fitting or configuration selection"}
        print(result, flush=True)
        rows.append(result)
        atomic_json(ws / "outputs/cnn_experiments/checkpoint_diagnostics.json", rows)


if __name__ == "__main__":
    main()
