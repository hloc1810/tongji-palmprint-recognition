"""ResNet18 training loops and reproducible two-stage fine-tuning."""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preparation import load_manifest
from ..models.resnet import (
    make_dataset,
    make_resnet,
    set_seed,
    set_trainable,
    torch_modules,
)
from ..utils import atomic_json, hash_file


def resolve_device(name: str = "auto"):
    """Resolve an explicit CNN backend without changing the default CUDA/CPU policy.

    With torch-directml 0.2.5, register the DirectML device before the process's
    first autograd backward; the experiment runner initializes it at startup.
    """
    torch, *_ = torch_modules()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "directml" or name.startswith("directml:"):
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("DirectML requires a separate environment with torch-directml") from exc
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


def _autocast(torch, device, enabled: bool):
    # DirectML uses float32 and does not register a PyTorch autocast module.
    return (torch.autocast(device_type="cuda", dtype=torch.float16)
            if enabled and device.type == "cuda" else nullcontext())


def _cpu_state_dict(model):
    """Independent, portable checkpoint tensors, including DirectML opaque tensors."""
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def _inference_context(torch, device):
    # DirectML 0.2.5 BatchNorm cannot use inference tensors' version counters.
    return torch.no_grad() if device.type == "privateuseone" else torch.inference_mode()

def _scaler(torch, device, use_amp: bool):
    """Use current AMP API, falling back for supported older PyTorch releases."""
    enabled = use_amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, optimizer, device, *, label_smoothing: float = 0.0,
                grad_clip: float = 1.0, accumulate: int = 1, amp: bool = False) -> dict:
    """One train epoch with optional CUDA AMP and exact last-window accumulation."""
    torch, nn, *_ = torch_modules()
    if accumulate < 1 or grad_clip < 0 or not 0 <= label_smoothing < 1:
        raise ValueError("Invalid accumulation, clip or label smoothing")
    model.train()
    # Keep ImageNet BN running means/variances stable when learning from few ROIs/class.
    # layer4 affine BN weights remain trainable when layer4 is unfrozen.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = _scaler(torch, device, amp)
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    for index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        start = (index // accumulate) * accumulate
        window_size = min(accumulate, n_batches - start)
        with _autocast(torch, device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss / window_size).backward()
        total_loss += float(loss.detach().item()) * len(labels)
        correct += int((logits.detach().argmax(1) == labels).sum().item())
        total += len(labels)
        if (index + 1) % accumulate == 0 or index + 1 == n_batches:
            if grad_clip > 0:
                scaler.unscale_(optimizer)  # clip unscaled, not AMP-scaled, gradients
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                         grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return {"train_loss": total_loss / max(total, 1),
            "train_rank1": correct / max(total, 1), "samples": total}


def validate_epoch(model, loader, device, *, amp: bool = False) -> dict:
    """Deterministic validation: no training-time augmentations, gradients or sampling."""
    torch, nn, *_ = torch_modules()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with _inference_context(torch, device):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with _autocast(torch, device, amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += len(labels)
    return {"dev_val_loss": total_loss / max(total, 1),
            "dev_val_rank1": correct / max(total, 1), "val_samples": total}


def _optimizer_and_scheduler(torch, net, stage: str, *, head_lr: float,
                             finetune_lr: float, finetune_head_lr: float,
                             weight_decay: float, epochs: int, schedule: str):
    if stage == "head":
        groups = [{"params": list(net.fc.parameters()), "lr": head_lr}]
    else:
        groups = [{"params": list(net.fc.parameters()), "lr": finetune_head_lr},
                  {"params": list(net.layer4.parameters()), "lr": finetune_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    if schedule not in ("none", "cosine"):
        raise ValueError("lr_schedule must be none or cosine")
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
                 if schedule == "cosine" else None)
    return optimizer, scheduler


def predict_loader(model, loader, device):
    torch, *_ = torch_modules()
    model.eval()
    logits, truth = [], []
    with _inference_context(torch, device):
        for images, labels in loader:
            z = model(images.to(device)).detach().cpu().numpy()
            logits.append(z)
            truth.append(labels.numpy() + 1)
    return np.concatenate(logits), np.concatenate(truth)


def _loader(frame, root, batch, training, size, *, seed: int = 0,
            num_workers: int = 0):
    torch, _, DataLoader, *_ = torch_modules()
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(make_dataset(frame, root, training, size),
                      batch_size=batch, shuffle=training, num_workers=num_workers,
                      generator=generator)


def train_one(ws: Path, *, seed: int = 17, weights: str = "imagenet1k_v1",
              head_epochs: int = 8, finetune_epochs: int = 5,
              batch_size: int = 32, size: int = 224, patience: int = 3,
              head_lr: float = 1e-3, finetune_lr: float = 1e-4,
              finetune_head_lr: float = 3e-4, weight_decay: float = 1e-4,
              lr_schedule: str = "cosine", label_smoothing: float = 0.0,
              grad_clip: float = 1.0, accumulate: int = 1,
              amp: bool = True, num_workers: int = 0, device: str = "auto"):
    """Choose head/fine-tune epochs on session1 fold0, refit on all session1.

    Test session2 remains unused; final schedule has no validation dependence.
    """
    torch, nn, *_ = torch_modules()
    if head_epochs < 1 or finetune_epochs < 0 or batch_size < 1 or patience < 1:
        raise ValueError("Invalid epoch/batch/patience setting")
    if size < 32 or num_workers < 0 or accumulate < 1:
        raise ValueError("Invalid image size/workers/accumulation")
    if min(head_lr, finetune_lr, finetune_head_lr) <= 0 or weight_decay < 0:
        raise ValueError("Invalid learning-rate/weight-decay setting")
    if not 0 <= label_smoothing < 1 or grad_clip < 0 or lr_schedule not in ("none", "cosine"):
        raise ValueError("Invalid training regularization/schedule setting")
    if (ws / "artifacts" / f"resnet18_seed{seed}_freeze.json").exists():
        raise RuntimeError("CNN run already frozen; use new workspace or seed")
    set_seed(seed)
    df, info = load_manifest(ws)
    all_train = df[df.session == "session1"].copy()
    val = all_train[all_train.fold == 0].copy()
    dev = all_train[all_train.fold != 0].copy()
    labels = sorted(all_train.palm_id.unique().tolist())
    if labels != list(range(1, len(labels)+1)) or val.palm_id.nunique() != len(labels):
        raise ValueError("CNN labels must be continuous and each class in validation")
    root = Path(info["data_root"])
    device_request = device
    device = resolve_device(device_request)
    use_amp = bool(amp and device.type == "cuda")
    net = make_resnet(len(labels), weights).to(device)
    train_loader = _loader(dev, root, batch_size, True, size, seed=seed, num_workers=num_workers)
    valid_loader = _loader(val, root, batch_size, False, size, seed=seed,
                           num_workers=num_workers)
    logdir = ws / "outputs/runs" / f"resnet18_seed{seed}"
    logdir.mkdir(parents=True, exist_ok=True)
    histories = []
    best_score, best_val_loss = -1.0, float("inf")
    best_state = None
    best_stage, best_epoch = "head", 1
    head_best_state, head_best_epoch = None, 1
    head_score, head_val_loss = -1.0, float("inf")
    budgets = {"head": head_epochs, "layer4": finetune_epochs}
    lrs = {"head": head_lr, "layer4": finetune_lr}
    for stage, budget in budgets.items():
        if budget <= 0:
            continue
        # Second stage starts from BEST head epoch, not last early-stopped epoch.
        if stage == "layer4" and head_best_state is not None:
            net.load_state_dict(head_best_state)
        set_trainable(net, layer4=(stage == "layer4"))
        optimizer, scheduler = _optimizer_and_scheduler(
            torch, net, stage, head_lr=head_lr, finetune_lr=finetune_lr,
            finetune_head_lr=finetune_head_lr, weight_decay=weight_decay,
            epochs=budget, schedule=lr_schedule)
        bad_epochs = 0
        for epoch in range(1, budget + 1):
            lr_before = [g["lr"] for g in optimizer.param_groups]
            start = time.perf_counter()
            metrics = train_epoch(net, train_loader, optimizer, device,
                                  label_smoothing=label_smoothing, grad_clip=grad_clip,
                                  accumulate=accumulate, amp=use_amp)
            metrics.update(validate_epoch(net, valid_loader, device, amp=use_amp))
            row = {"stage": stage, "epoch": epoch, **metrics,
                   "learning_rates": json.dumps(lr_before),
                   "seconds": time.perf_counter()-start}
            histories.append(row)
            pd.DataFrame(histories).to_csv(logdir / "dev_history.csv", index=False)
            print(f"[CNN seed={seed} dev/{stage} {epoch}/{budget}] "
                  f"loss={metrics['train_loss']:.4f} "
                  f"val_loss={metrics['dev_val_loss']:.4f} "
                  f"val_rank1={metrics['dev_val_rank1']:.4f}", flush=True)
            if scheduler is not None:
                scheduler.step()
            score, vl = metrics["dev_val_rank1"], metrics["dev_val_loss"]
            improved = score > best_score + 1e-9 or (
                abs(score - best_score) <= 1e-9 and vl < best_val_loss - 1e-7)
            if improved:
                best_score, best_val_loss = score, vl
                best_state = _cpu_state_dict(net)
                torch.save({"state_dict": best_state, "selected_stage": stage,
                            "selected_epoch": epoch, "seed": seed},
                           logdir / "best_dev.pth")
                best_stage, best_epoch = stage, epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            if stage == "head" and (score > head_score + 1e-9 or (
                    abs(score-head_score) <= 1e-9 and vl < head_val_loss - 1e-7)):
                head_score, head_val_loss = score, vl
                head_best_epoch = epoch
                head_best_state = _cpu_state_dict(net)
            if bad_epochs >= patience:
                break

    # Preserve a dev checkpoint separately from the final full-session1 checkpoint.
    torch.save({"state_dict": best_state, "selected_stage": best_stage,
                "selected_epoch": best_epoch, "seed": seed}, logdir / "best_dev.pth")
    final_head = head_best_epoch
    final_finetune = best_epoch if best_stage == "layer4" else 0
    # Re-initialize model and fit deterministic epoch schedule on ALL session1.
    # No validation or test images influence optimizer steps during this refit.
    set_seed(seed)
    final_net = make_resnet(len(labels), weights).to(device)
    full_loader = _loader(all_train, root, batch_size, True, size, seed=seed,
                          num_workers=num_workers)
    final_histories = []
    for stage, epochs in (("head", final_head), ("layer4", final_finetune)):
        if not epochs:
            continue
        set_trainable(final_net, layer4=(stage == "layer4"))
        optimizer, scheduler = _optimizer_and_scheduler(
            torch, final_net, stage, head_lr=head_lr, finetune_lr=finetune_lr,
            finetune_head_lr=finetune_head_lr, weight_decay=weight_decay,
            epochs=budgets[stage], schedule=lr_schedule)
        for epoch in range(1, epochs + 1):
            lrs = [g["lr"] for g in optimizer.param_groups]
            start = time.perf_counter()
            row = train_epoch(final_net, full_loader, optimizer, device,
                              label_smoothing=label_smoothing, grad_clip=grad_clip,
                              accumulate=accumulate, amp=use_amp)
            final_histories.append({"stage": stage, "epoch": epoch, **row,
                                    "learning_rates": json.dumps(lrs),
                                    "seconds": time.perf_counter()-start})
            pd.DataFrame(final_histories).to_csv(logdir / "final_history.csv", index=False)
            print(f"[CNN seed={seed} final/{stage} {epoch}/{epochs}] "
                  f"loss={row['train_loss']:.4f} train_rank1={row['train_rank1']:.4f}",
                  flush=True)
            if scheduler is not None:
                scheduler.step()
    training_cfg = {"seed": seed, "weights": weights, "image_size": size,
                    "batch_size": batch_size, "effective_batch_size": batch_size*accumulate,
                    "accumulate": accumulate, "head_epochs_budget": head_epochs,
                    "finetune_epochs_budget": finetune_epochs, "patience": patience,
                    "head_lr": head_lr, "finetune_lr": finetune_lr,
                    "finetune_head_lr": finetune_head_lr, "weight_decay": weight_decay,
                    "lr_schedule": lr_schedule, "label_smoothing": label_smoothing,
                    "grad_clip": grad_clip, "amp_requested": amp, "amp_used": use_amp,
                    "num_workers": num_workers, "batchnorm_running_stats": "frozen",
                    "augmentation": "rotation5_brightness0.08_contrast0.08",
                    "criterion": "rank1_desc_then_val_ce_loss_asc",
                    "torch_version": str(torch.__version__), "device": str(device),
                    "device_request": device_request}
    dst = ws / "artifacts" / f"resnet18_seed{seed}.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_net.cpu().state_dict(), "num_classes": len(labels),
                "image_size": size, "weights_initialization": weights,
                "seed": seed, "selected_stage": best_stage,
                "head_epochs_final": final_head,
                "finetune_epochs_final": final_finetune,
                "training_config": training_cfg}, dst)
    hist_path = logdir / "dev_history.csv"
    pd.DataFrame(histories).to_csv(hist_path, index=False)
    pd.DataFrame(final_histories).to_csv(logdir / "final_history.csv", index=False)
    atomic_json(logdir / "training_config.json", training_cfg)
    meta = {"model": "resnet18", "seed": seed, "weights": weights,
            "dev_val_best_rank1": best_score, "dev_val_best_loss": best_val_loss,
            "selected_stage": best_stage, "head_epochs_final": final_head,
            "finetune_epochs_final": final_finetune,
            "artifact_sha256": hash_file(dst),
            "dev_history_sha256": hash_file(hist_path),
            "manifest_sha256": hash_file(ws / "data/metadata/manifest.csv"),
            "n_classes": len(labels), "training_config": training_cfg,
            "test_status": "not_evaluated"}
    atomic_json(ws / "artifacts" / f"resnet18_seed{seed}_freeze.json", meta)
    return meta
