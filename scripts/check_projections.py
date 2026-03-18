#!/usr/bin/env python3
"""Projection registry consistency checks.

Ensures the multi-step projection registration process (CLAUDE.md) is complete.
Every projection must be registered, imported, and have fused kernel support.

Uses a ratchet baseline: fails only when NEW violations are introduced.

Usage:
    uv run python scripts/check_projections.py --all
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ratchet baseline: known violation count. Only decrease this, never increase.
_VIOLATION_BASELINE = 0

# Paths
_PROJECTIONS_DIR = REPO_ROOT / "src" / "vibeproj" / "projections"
_INIT_PY = _PROJECTIONS_DIR / "__init__.py"
_FUSED_KERNELS = REPO_ROOT / "src" / "vibeproj" / "fused_kernels.py"

# Files to skip when scanning projection modules
_SKIP_STEMS = {"__init__", "base"}


@dataclass(frozen=True)
class LintError:
    code: str
    path: Path
    line: int
    message: str

    def render(self, repo_root: Path) -> str:
        relative = self.path.relative_to(repo_root)
        return f"{relative}:{self.line}: {self.code} {self.message}"


def _parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _iter_projection_modules() -> list[Path]:
    """Return all projection .py files (excluding __init__.py, base.py)."""
    return sorted(
        p
        for p in _PROJECTIONS_DIR.glob("*.py")
        if p.stem not in _SKIP_STEMS and "__pycache__" not in p.parts
    )


def _extract_register_calls(tree: ast.AST) -> list[str]:
    """Extract projection names from register("name", ...) calls in an AST."""
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match: register("name", ...)
        func = node.func
        if isinstance(func, ast.Name) and func.id == "register":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.append(node.args[0].value)
    return names


def _extract_init_imports(tree: ast.AST) -> set[str]:
    """Extract module names imported in projections/__init__.py."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("vibeproj.projections."):
                # from vibeproj.projections.sinusoidal import ...
                mod = node.module.split(".")[-1]
                modules.add(mod)
            elif node.module and "." not in node.module:
                # from sinusoidal import ...  (relative)
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vibeproj.projections."):
                    mod = alias.name.split(".")[-1]
                    modules.add(mod)
    # Also handle: from vibeproj.projections import sinusoidal, ...
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "vibeproj.projections":
            for alias in node.names:
                modules.add(alias.name)
    return modules


def _extract_block(source: str, varname: str) -> str:
    """Extract the { ... } block after a top-level assignment like `_SUPPORTED = {`."""
    pattern = rf"^{re.escape(varname)}\s*=\s*\{{"
    m = re.search(pattern, source, re.MULTILINE)
    if not m:
        return ""
    start = m.start()
    depth = 0
    for i in range(m.end() - 1, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    return ""


def _extract_supported_set(source: str) -> set[str]:
    """Extract projection names from _SUPPORTED = { ("name", "forward"), ... }."""
    block = _extract_block(source, "_SUPPORTED")
    names = set()
    for m in re.finditer(r'\(\s*"(\w+)"\s*,\s*"(?:forward|inverse)"\s*\)', block):
        names.add(m.group(1))
    return names


def _extract_source_map_names(source: str) -> set[str]:
    """Extract projection names from _SOURCE_MAP = { ("name", "forward"): ... }."""
    block = _extract_block(source, "_SOURCE_MAP")
    names = set()
    for m in re.finditer(r'\(\s*"(\w+)"\s*,\s*"(?:forward|inverse)"\s*\)\s*:', block):
        names.add(m.group(1))
    return names


# ===================================================================
# PROJ001: Every projection module must call register()
# ===================================================================


def check_register_calls(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    for path in _iter_projection_modules():
        tree = _parse_module(path)
        names = _extract_register_calls(tree)
        if not names:
            errors.append(
                LintError(
                    code="PROJ001",
                    path=path,
                    line=1,
                    message=f"Projection module '{path.stem}' does not call register(). "
                    f'Add: register("<name>", <ClassName>()) at module bottom.',
                )
            )
    return errors


# ===================================================================
# PROJ002: Every projection module must be imported in __init__.py
# ===================================================================


def check_init_imports(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    init_tree = _parse_module(_INIT_PY)
    imported = _extract_init_imports(init_tree)

    for path in _iter_projection_modules():
        if path.stem not in imported:
            errors.append(
                LintError(
                    code="PROJ002",
                    path=_INIT_PY,
                    line=1,
                    message=f"Projection module '{path.stem}' is not imported in "
                    f"projections/__init__.py. Add it to the import block.",
                )
            )
    return errors


# ===================================================================
# PROJ003: Every registered projection must have _SUPPORTED entries
# ===================================================================


def check_supported_entries(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    fused_source = _FUSED_KERNELS.read_text(encoding="utf-8")
    supported = _extract_supported_set(fused_source)

    # Collect all registered projection names
    registered = set()
    for path in _iter_projection_modules():
        tree = _parse_module(path)
        registered.update(_extract_register_calls(tree))

    for name in sorted(registered):
        if name not in supported:
            errors.append(
                LintError(
                    code="PROJ003",
                    path=_FUSED_KERNELS,
                    line=1,
                    message=f"Projection '{name}' is registered but has no entries in "
                    f"_SUPPORTED. Add ('{name}', 'forward') and ('{name}', 'inverse').",
                )
            )
    return errors


# ===================================================================
# PROJ004: Every _SUPPORTED entry must have a _SOURCE_MAP entry
# ===================================================================


def check_source_map_entries(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    fused_source = _FUSED_KERNELS.read_text(encoding="utf-8")
    supported = _extract_supported_set(fused_source)
    source_map = _extract_source_map_names(fused_source)

    missing = supported - source_map
    for name in sorted(missing):
        errors.append(
            LintError(
                code="PROJ004",
                path=_FUSED_KERNELS,
                line=1,
                message=f"Projection '{name}' is in _SUPPORTED but has no _SOURCE_MAP "
                f"entry. Add kernel source templates and register them.",
            )
        )
    return errors


# ===================================================================
# Aggregation
# ===================================================================


def run_checks(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    errors.extend(check_register_calls(repo_root))
    errors.extend(check_init_imports(repo_root))
    errors.extend(check_supported_entries(repo_root))
    errors.extend(check_source_map_entries(repo_root))
    return sorted(errors, key=lambda e: (str(e.path), e.line, e.code))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check projection registry consistency")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all to run all checks")

    errors = run_checks(REPO_ROOT)
    count = len(errors)

    if count > _VIOLATION_BASELINE:
        for error in errors:
            print(error.render(REPO_ROOT))
        print(
            f"\ncheck_projections: FAIL — {count} violations found, "
            f"baseline is {_VIOLATION_BASELINE}.",
            file=sys.stderr,
        )
        return 1

    if count < _VIOLATION_BASELINE:
        print(
            f"check_projections: passed ({count} known violations, "
            f"baseline {_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _VIOLATION_BASELINE to {count}."
        )
    else:
        print(f"check_projections: passed ({count} violations, baseline holds).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
