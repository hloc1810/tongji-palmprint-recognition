"""Repository documentation checks independent of model execution."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_internal_markdown_links_exist() -> None:
    documents = [ROOT / "README.md", ROOT / "data/README.md"]
    for folder in ("docs", "reports", "experiments"):
        documents.extend((ROOT / folder).rglob("*.md"))
    for document in documents:
        for destination in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if destination.startswith(("http:", "https:", "mailto:", "#")):
                continue
            relative_path = destination.split("#", 1)[0].split("?", 1)[0]
            assert (document.parent / relative_path).exists(), (
                f"Broken documentation link: {document.relative_to(ROOT)} -> {destination}"
            )


def test_document_paths_are_revision_neutral() -> None:
    for document in (ROOT / "docs").glob("*.md"):
        assert not re.search(r"(?:_v\d+(?:_\d+)?|_20\d{2}|_fixed|_clean)$", document.stem, re.I), (
            f"Revision-dependent documentation filename: {document.name}"
        )


def test_modules_grouped_by_responsibility() -> None:
    package = ROOT / "src/palmprint"
    for name in ("data", "features", "models", "training", "evaluation", "reporting"):
        assert (package / name / "__init__.py").is_file()
    for old in ("data.py", "features.py", "models.py", "experiment.py", "cnn.py", "reporting.py"):
        assert not (package / old).exists()
    assert (package / "training/selection.py").is_file()
    assert (package / "reporting/evidence.py").is_file()
