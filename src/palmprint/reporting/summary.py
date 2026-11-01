"""Tables and charts derived from immutable experiment evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.preparation import load_manifest
from ..evaluation.metrics import bootstrap_delta
from .evidence import verify_run_evidence


def generate_report(ws: Path) -> pd.DataFrame:
    """Publication-supporting tables/figures, computed only from genuine test runs.

    Never synthesize scientific values from the smoke test; that fixture is identified
    by the dataset manifest and must not be presented as Tongji research evidence.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..utils import hash_file

    runs = sorted((ws / "outputs/runs").glob("*/metrics.json"))
    if not runs:
        raise FileNotFoundError("No final test runs; train/lock/evaluate first")
    manifest, dataset = load_manifest(ws)
    probes = manifest[manifest.session == "session2"].reset_index(drop=True)
    strict = bool(dataset["strict"])
    scope = "strict_tongji_layout" if strict else "NON_STRICT_fixture_or_subset"
    title_prefix = "" if strict else "NON-STRICT fixture/subset: "
    # Validate every run before exporting a comparison. A consistent score matrix
    # alone is insufficient: its UIDs and labels must also match the audit manifest.
    verified_runs = {}
    for path in runs:
        verified_runs[path.parent.name] = verify_run_evidence(
            path.parent, expected_probes=probes, manifest_sha256=dataset["manifest_sha256"]
        )
        metric = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        # Embedding variants are evaluation-only views of the SAME frozen CNN.
        # They do not have independent training checkpoints or freeze.json files.
        cnn_embedding = "_embedding_" in name and name.startswith("resnet18_seed")
        artifact_name = name.split("_embedding_", 1)[0] if cnn_embedding else name
        frozen_path = ws / "artifacts" / f"{artifact_name}_freeze.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        suffix = ".joblib" if name in ("knn", "svm", "crc") else ".pth"
        artifact_path = ws / "artifacts" / f"{artifact_name}{suffix}"
        if (
            metric.get("artifact_sha256") != frozen["artifact_sha256"]
            or hash_file(artifact_path) != frozen["artifact_sha256"]
            or metric["manifest_sha256"] != frozen["manifest_sha256"]
        ):
            raise RuntimeError(f"Run provenance differs from frozen artifact: {name}")
        if cnn_embedding:
            gallery = ws / "artifacts" / f"{artifact_name}_gallery512.npz"
            if metric.get("gallery_sha256") != hash_file(gallery):
                raise RuntimeError(f"Embedding gallery hash mismatch: {name}")
        if metric.get("score_kind") != metric["model_definition"]["score_kind"]:
            raise RuntimeError(f"Score kind differs from model definition: {path.parent.name}")
        if path.parent.name in ("knn", "svm", "crc"):
            if metric.get("model_definition") != frozen["model_definition"]:
                raise RuntimeError(f"Model definition differs from frozen fit: {path.parent.name}")
    figdir = ws / "reports/figures"
    tables = ws / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    records, frames, audit = [], {}, []
    for path in runs:
        run_dir = path.parent
        verified = verified_runs[run_dir.name]
        m = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "model": run_dir.name,
                "rank1": m["rank1"],
                "rank5": m["rank5"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
                "rank1_ci_lower": m["rank1_ci95_palm_bootstrap"][0],
                "rank1_ci_upper": m["rank1_ci95_palm_bootstrap"][1],
                "matching_ms_per_probe_batch_amortized": m.get(
                    "matching_ms_per_probe_batch_amortized",
                    m.get("cosine_matching_ms_per_image_batch_amortized"),
                ),
                "training_fit_seconds": m.get("training_fit_seconds"),
                "n": m["n"],
                "score_kind": m["score_kind"],
                "method": m["model_definition"]["method"],
                "data_scope": scope,
                "manifest_sha256": dataset["manifest_sha256"],
                "artifact_sha256": m["artifact_sha256"],
                "note": "Batch-amortized score computation; excludes feature extraction for embedding runs",
            }
        )
        pred = pd.read_csv(run_dir / "predictions.csv")
        frames[run_dir.name] = pred.set_index("sample_uid")
        audit.append(
            {
                **verified,
                "scores_sha256": hash_file(run_dir / "all_class_scores.npz"),
                "predictions_sha256": hash_file(run_dir / "predictions.csv"),
            }
        )
        per = pd.read_csv(run_dir / "per_class_metrics.csv")
        per.sort_values(["recall", "palm_id"]).head(min(25, len(per))).to_csv(
            tables / f"hardest_palms_{run_dir.name}.csv", index=False
        )
        # Frequent actual confusion pairs, including predicted labels outside top true classes.
        pairs = (
            pred.loc[pred.correct == 0]
            .groupby(["true_palm_id", "pred_palm_id"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["count", "true_palm_id", "pred_palm_id"], ascending=[False, True, True])
        )
        pairs.head(30).to_csv(tables / f"top_confusions_{run_dir.name}.csv", index=False)
        cm = np.load(run_dir / "confusion_matrix.npy", allow_pickle=False)
        if cm.shape != (len(per), len(per)) or not np.array_equal(
            cm.sum(axis=1), per.support.to_numpy()
        ):
            raise RuntimeError(f"Invalid confusion matrix counts: {run_dir.name}")
        fig_cm, ax_cm = plt.subplots(figsize=(9, 8))
        ax_cm.imshow(cm, interpolation="nearest", aspect="auto")
        ax_cm.set(
            xlabel="Predicted Palm ID (ordered class index)",
            ylabel="True Palm ID (ordered class index)",
            title=f"{title_prefix}Full confusion matrix — {run_dir.name}",
        )
        fig_cm.tight_layout()
        fig_cm.savefig(figdir / f"confusion_full_{run_dir.name}.png", dpi=170)
        plt.close(fig_cm)
        # Focus view selects the most common directed confusion pairs so both
        # their true AND predicted IDs remain visible (not just error-prone rows).
        ids = []
        for row in pairs.itertuples(index=False):
            for ident in (int(row.true_palm_id), int(row.pred_palm_id)):
                if ident not in ids:
                    ids.append(ident)
            if len(ids) >= 20:
                break
        if not ids:
            ids = per.sort_values("recall").palm_id.head(min(20, len(per))).tolist()
        ids = ids[:20]
        ix = [int(np.searchsorted(per.palm_id.to_numpy(), ident)) for ident in ids]
        if any(per.palm_id.iloc[index] != ident for index, ident in zip(ix, ids)):
            raise RuntimeError("Palm class order corrupted")
        fig_small, ax_small = plt.subplots(figsize=(8, 7))
        ax_small.imshow(cm[np.ix_(ix, ix)], interpolation="nearest", aspect="auto")
        ax_small.set(
            xticks=range(len(ids)),
            yticks=range(len(ids)),
            xticklabels=[str(v) for v in ids],
            yticklabels=[str(v) for v in ids],
            xlabel="Predicted Palm ID",
            ylabel="True Palm ID",
            title=f"{title_prefix}Most-confused class pairs — {run_dir.name}",
        )
        ax_small.tick_params(axis="x", labelrotation=90)
        fig_small.tight_layout()
        fig_small.savefig(figdir / f"confusion_focus_{run_dir.name}.png", dpi=170)
        plt.close(fig_small)
        fig, ax = plt.subplots(figsize=(9, 4.7))
        curve = pd.read_csv(run_dir / "cmc_full.csv")
        ax.plot(curve.rank_k.to_numpy(), curve.identification_rate.to_numpy())
        ax.set(
            xlabel="Rank k (distinct Palm IDs)",
            ylabel="Identification rate",
            title=f"{title_prefix}Full CMC — {run_dir.name}",
            ylim=(0, 1.02),
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figdir / f"cmc_full_{run_dir.name}.png", dpi=170)
        plt.close(fig)
    pd.DataFrame(audit).to_csv(tables / "evidence_verification.csv", index=False)
    df = pd.DataFrame(records).sort_values("model").reset_index(drop=True)
    df.to_csv(tables / "model_comparison.csv", index=False)
    paired = []
    keys = sorted(frames)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            xa, xb = frames[a], frames[b]
            if not xa.index.is_unique or not xb.index.is_unique or set(xa.index) != set(xb.index):
                raise RuntimeError(
                    f"Cannot perform paired comparison on unequal samples: {a} vs {b}"
                )
            xb = xb.loc[xa.index]
            if not np.array_equal(xa.true_palm_id.values, xb.true_palm_id.values):
                raise RuntimeError(f"Different ground-truth labels: {a} vs {b}")
            y = xa.true_palm_id.values
            ci = bootstrap_delta(y, xa.pred_palm_id.values, xb.pred_palm_id.values)
            paired.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "delta_rank1_a_minus_b": float(
                        np.mean(xa.pred_palm_id.values == y) - np.mean(xb.pred_palm_id.values == y)
                    ),
                    "ci95_lower": ci[0],
                    "ci95_upper": ci[1],
                    "n_paired_probes": len(y),
                    "bootstrap_cluster": "Palm ID",
                }
            )
    pd.DataFrame(
        paired,
        columns=[
            "model_a",
            "model_b",
            "delta_rank1_a_minus_b",
            "ci95_lower",
            "ci95_upper",
            "n_paired_probes",
            "bootstrap_cluster",
        ],
    ).to_csv(tables / "paired_rank1_differences.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ix = np.arange(len(df))
    for offset, key in ((-0.24, "rank1"), (0, "rank5"), (0.24, "macro_f1")):
        ax.bar(ix + offset, df[key], 0.23, label=key)
    ax.set(
        xticks=ix,
        xticklabels=df.model,
        ylim=(0, 1.02),
        ylabel="Rate",
        title=f"{title_prefix}Cross-session comparison (same probes)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "model_comparison.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    lower = (df.rank1 - df.rank1_ci_lower).clip(lower=0)
    upper = (df.rank1_ci_upper - df.rank1).clip(lower=0)
    ax.errorbar(ix, df.rank1, yerr=np.vstack((lower, upper)), fmt="o", capsize=4)
    ax.set(
        xticks=ix,
        xticklabels=df.model,
        ylim=(0, 1.02),
        ylabel="Rank-1",
        title=f"{title_prefix}Rank-1 with 95% Palm-ID bootstrap intervals",
    )
    fig.tight_layout()
    fig.savefig(figdir / "rank1_confidence_intervals.png", dpi=170)
    plt.close(fig)
    cv_rows = []
    for sel_path in sorted((ws / "outputs/cv").glob("*_selection.json")):
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        cv_rows.append(
            {
                "model": sel["model"],
                "representation": sel["representation"],
                "parameter": sel["best_parameter"],
                "cv_mean_rank1": sel["mean_cv_rank1"],
                "cv_fold_std_rank1": sel["std_cv_rank1"],
                "folds": sel["n_folds"],
                "cv_csv_sha256": sel["cv_csv_sha256"],
            }
        )
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(tables / "cv_selection_summary.csv", index=False)
    return df
