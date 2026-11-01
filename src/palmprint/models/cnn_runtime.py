"""Shared CNN devices, data loaders and inference, without a training dependency."""

from __future__ import annotations

import numpy as np

from .resnet import make_dataset, torch_modules


def resolve_device(name: str = "auto"):
    """Select CUDA/CPU by default, or explicitly opt into DirectML.

    Register DirectML before the process's first autograd backward when using
    torch-directml 0.2.5. DirectML keeps inference in float32.
    """
    torch, *_ = torch_modules()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "directml" or name.startswith("directml:"):
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError(
                "DirectML requires a separate environment with torch-directml"
            ) from exc
        index = int(name.split(":", 1)[1]) if ":" in name else 0
        if not 0 <= index < torch_directml.device_count():
            raise ValueError("DirectML device index is unavailable")
        return torch_directml.device(index)
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("CNN device must be auto, cpu, cuda[:N], or directml[:N]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def inference_context(torch, device):
    """DirectML BatchNorm requires no_grad rather than inference tensors."""
    return torch.no_grad() if device.type == "privateuseone" else torch.inference_mode()


def make_loader(frame, root, batch, training, size, *, seed: int = 0, num_workers: int = 0):
    """Keep dataset order for evaluation and seed shuffling for training."""
    torch, _, DataLoader, *_ = torch_modules()
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        make_dataset(frame, root, training, size),
        batch_size=batch,
        shuffle=training,
        num_workers=num_workers,
        generator=generator,
    )


def predict_loader(model, loader, device):
    """Return logits and one-based Palm IDs in evaluation loader order."""
    torch, *_ = torch_modules()
    model.eval()
    logits, truth = [], []
    with inference_context(torch, device):
        for images, labels in loader:
            logits.append(model(images.to(device)).detach().cpu().numpy())
            truth.append(labels.numpy() + 1)
    return np.concatenate(logits), np.concatenate(truth)
