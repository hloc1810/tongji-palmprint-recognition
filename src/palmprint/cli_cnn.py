"""Command-line interface for CNN training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation.cnn import evaluate_one
from .evaluation.embedding import (
    build_gallery,
    evaluate_embeddings,
    predict_one_embedding,
    validate_embeddings,
)
from .training.cnn import train_one
from .utils import read_yaml


def main(argv=None):
    """CLI uses configs/resnet18.yaml; supplied flags override matching config keys."""
    p = argparse.ArgumentParser("palmprint-cnn")
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    sp = p.add_subparsers(dest="command", required=True)
    train = sp.add_parser("train")
    train.add_argument("--config", type=Path, help="Default: <workspace>/configs/resnet18.yaml")
    train.add_argument("--seed", type=int)
    train.add_argument("--weights", choices=["none", "imagenet1k_v1"])
    train.add_argument("--head-epochs", type=int)
    train.add_argument("--finetune-epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--size", type=int)
    train.add_argument("--patience", type=int)
    train.add_argument("--head-lr", type=float)
    train.add_argument("--finetune-lr", type=float)
    train.add_argument("--finetune-head-lr", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--lr-schedule", choices=["none", "cosine"])
    train.add_argument("--label-smoothing", type=float)
    train.add_argument("--grad-clip", type=float)
    train.add_argument("--accumulate", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--device", help="auto, cpu, cuda[:N], or directml[:N]")
    train.add_argument("--no-amp", action="store_true", default=None)
    ev = sp.add_parser("evaluate")
    ev.add_argument("--seed", type=int, default=17)
    ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--device", default="auto")
    build = sp.add_parser("embedding-gallery", help="Build Session1-only 512-D gallery")
    build.add_argument("--seed", type=int, default=17)
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument("--device", default="auto")
    emb = sp.add_parser("embedding-evaluate", help="Frozen Session1 gallery vs Session2 queries")
    emb.add_argument("--seed", type=int, default=17)
    emb.add_argument("--batch-size", type=int, default=32)
    emb.add_argument("--device", default="auto")
    emb.add_argument("--strategy", choices=["nearest", "centroid", "both"], default="both")
    emb.add_argument("--chunk-size", type=int, default=128)
    dev_emb = sp.add_parser(
        "embedding-validate", help="Session1 fold0 diagnostic; best_dev checkpoint"
    )
    dev_emb.add_argument("--seed", type=int, default=17)
    dev_emb.add_argument("--batch-size", type=int, default=32)
    dev_emb.add_argument("--device", default="auto")
    dev_emb.add_argument("--strategy", choices=["nearest", "centroid", "both"], default="both")
    dev_emb.add_argument("--chunk-size", type=int, default=128)
    pred_emb = sp.add_parser("embedding-predict", help="Single Official ROI identification")
    pred_emb.add_argument("--seed", type=int, default=17)
    pred_emb.add_argument("--device", default="cpu")
    pred_emb.add_argument("--strategy", choices=["nearest", "centroid"], default="nearest")
    pred_emb.add_argument("--image", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "train":
        path = args.config or args.workspace / "configs" / "resnet18.yaml"
        if args.config and not path.is_file():
            raise FileNotFoundError(f"CNN training config not found: {path}")
        conf = read_yaml(path) if path.is_file() else {}
        keys = (
            "seed",
            "weights",
            "head_epochs",
            "finetune_epochs",
            "batch_size",
            "patience",
            "head_lr",
            "finetune_lr",
            "finetune_head_lr",
            "weight_decay",
            "lr_schedule",
            "label_smoothing",
            "grad_clip",
            "accumulate",
            "num_workers",
            "device",
        )
        parameters = {key: conf[key] for key in keys if key in conf}
        if "image_size" in conf:
            parameters["size"] = conf["image_size"]
        if "amp" in conf:
            parameters["amp"] = conf["amp"]
        for key in (*keys, "size"):
            val = getattr(args, key, None)
            if val is not None:
                parameters[key] = val
        if args.no_amp:
            parameters["amp"] = False
        result = train_one(args.workspace, **parameters)
    elif args.command == "evaluate":
        result = evaluate_one(
            args.workspace, seed=args.seed, batch_size=args.batch_size, device=args.device
        )
    elif args.command == "embedding-gallery":
        result = build_gallery(
            args.workspace, seed=args.seed, batch_size=args.batch_size, device=args.device
        )
    elif args.command == "embedding-evaluate":
        strategies = ("nearest", "centroid") if args.strategy == "both" else (args.strategy,)
        result = evaluate_embeddings(
            args.workspace,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            strategies=strategies,
            chunk_size=args.chunk_size,
        )
    elif args.command == "embedding-validate":
        strategies = ("nearest", "centroid") if args.strategy == "both" else (args.strategy,)
        result = validate_embeddings(
            args.workspace,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            strategies=strategies,
            chunk_size=args.chunk_size,
        )
    else:
        result = predict_one_embedding(
            args.workspace, args.seed, args.image, device=args.device, strategy=args.strategy
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
