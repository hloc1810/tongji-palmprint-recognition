"""Explicit experiment lifecycle; test cannot influence CV or final training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.dataset import generate_synthetic_roi
from .data.preparation import prepare
from .features.extraction import load_features
from .reporting.summary import generate_report
from .training.classical import evaluate_final, final_train, lock_protocol, predict_one
from .training.selection import cross_validate


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="palmprint", description="Tongji cross-session experiment CLI")
    p.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Repository root with configs/ and results directories",
    )
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "synthetic", help="Generate only a clearly marked synthetic fixture"
    ).add_argument("--output", type=Path, required=True)
    a = commands.add_parser("prepare")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--allow-subset", action="store_true", help="For smoke tests only")
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--seed", type=int, default=42)
    e = commands.add_parser("features")
    e.add_argument("--representation", choices=["compcode", "global", "hog"], default="compcode")
    c = commands.add_parser("cv")
    c.add_argument("--model", choices=["knn", "svm", "crc"], required=True)
    c.add_argument("--representation", choices=["compcode", "global", "hog"], default="compcode")
    for command in ("train", "evaluate"):
        commands.add_parser(command).add_argument(
            "--model", choices=["knn", "svm", "crc"], required=True
        )
    commands.add_parser(
        "lock", help="Freeze all three classical candidates before any test evaluation"
    )
    q = commands.add_parser("predict")
    q.add_argument("--model", choices=["knn", "svm", "crc"], required=True)
    q.add_argument("--image", required=True, type=Path)
    commands.add_parser("report")
    commands.add_parser("eda")
    error = commands.add_parser("errors")
    error.add_argument("--model", choices=["knn", "svm", "crc"], required=True)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--model", choices=["knn", "svm", "crc"], required=True)
    bench.add_argument("--limit", type=int, default=200)
    stress = commands.add_parser("robustness")
    stress.add_argument("--model", choices=["knn", "svm", "crc"], required=True)
    stress.add_argument("--limit", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    ws = args.workspace.resolve()
    if args.command == "synthetic":
        generate_synthetic_roi(args.output)
        result = {"synthetic": True, "root": str(args.output), "research_valid": False}
    else:
        cfg = ws / "configs/compcode.yaml"
        if args.command == "prepare":
            result = prepare(
                ws, args.data, strict=not args.allow_subset, folds=args.folds, seed=args.seed
            )
        elif args.command == "features":
            df, X, signature = load_features(ws, cfg, args.representation, sessions=("session1",))
            result = {
                "n": len(df),
                "feature_dimension": int(X.shape[1]),
                "representation": args.representation,
                "signature": signature,
            }
        elif args.command == "cv":
            result = cross_validate(ws, args.model, cfg, rep=args.representation)
        elif args.command == "train":
            result = final_train(ws, args.model, cfg)
        elif args.command == "evaluate":
            result = evaluate_final(ws, args.model, cfg)
        elif args.command == "lock":
            result = lock_protocol(ws)
        elif args.command == "predict":
            result = predict_one(ws, args.model, args.image)
        elif args.command == "report":
            result = generate_report(ws).to_dict(orient="records")
        elif args.command in ("eda", "errors", "benchmark", "robustness"):
            from .evaluation.analysis import benchmark_cpu, data_eda, error_analysis, robustness

            if args.command == "eda":
                result = data_eda(ws)
            elif args.command == "errors":
                result = error_analysis(ws, args.model)
            elif args.command == "benchmark":
                result = benchmark_cpu(ws, args.model, limit=args.limit)
            else:
                result = robustness(ws, args.model, limit=args.limit or None)
        else:
            raise RuntimeError("Unknown command")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
