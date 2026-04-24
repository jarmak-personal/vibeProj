"""Documentation example linting."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BLOCK_RE = re.compile(r"```(?:python|py)\n(?P<body>.*?)```", re.DOTALL)
LAT_LON_TRANSFORM_RE = re.compile(
    r"\bt\.transform(?:_buffers)?\(\s*lat(?:_array)?\s*,\s*lon(?:_array)?\b"
)


def _markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def test_docs_do_not_show_default_transform_lat_lon_order():
    """Default Transformer examples should use always_xy=(lon, lat) order."""
    failures: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in PYTHON_BLOCK_RE.finditer(text):
            block = match.group("body")
            if "always_xy=False" in block:
                continue
            if LAT_LON_TRANSFORM_RE.search(block):
                line = text[: match.start("body")].count("\n") + 1
                failures.append(f"{path.relative_to(ROOT)}:{line}")

    assert not failures, "Misleading default lat/lon transform examples:\n" + "\n".join(failures)
