"""Documentation references must track the packaged repository."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_relative_markdown_links_resolve():
    missing = []
    for doc in [
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "reports").rglob("*.md"),
        ROOT / "data/README.md",
    ]:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
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
    assert (ROOT / "reports/report/REPORT_TEMPLATE.md").is_file()
    assert (ROOT / "reports/slides/SLIDE_OUTLINE.md").is_file()
