"""Documentation references must track the packaged repository."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_relative_markdown_links_resolve():
    missing = []
    docs_to_check = [ROOT / "README.md", ROOT / "data/README.md"]
    if (ROOT / "docs").is_dir():
        docs_to_check.extend((ROOT / "docs").glob("*.md"))
    if (ROOT / "reports").is_dir():
        docs_to_check.extend((ROOT / "reports").rglob("*.md"))
    for doc in docs_to_check:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                continue
            relative = target.split("#")[0]
            if relative and not (doc.parent / relative).exists():
                missing.append(f"{doc.relative_to(ROOT)}: {target}")
    assert missing == []


def test_documented_entrypoints_exist():
    assert (ROOT / "src/palmprint/cli.py").is_file()
    assert (ROOT / "src/palmprint/cli_cnn.py").is_file()
    assert (ROOT / "configs/resnet18.yaml").is_file()
    assert (ROOT / "app/main.py").is_file()
