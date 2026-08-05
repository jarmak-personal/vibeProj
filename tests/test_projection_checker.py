"""Regression tests for projection registry checker discovery."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_NAME = "vibeproj_check_projections_test"
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_projections.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_projections = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = check_projections
_SPEC.loader.exec_module(check_projections)


def test_private_projection_helpers_are_not_treated_as_projection_modules(tmp_path, monkeypatch):
    public_module = tmp_path / "public_projection.py"
    private_helper = tmp_path / "_equal_area.py"
    public_module.write_text("class PublicProjection:\n    pass\n", encoding="utf-8")
    private_helper.write_text("def qsfn(value):\n    return value\n", encoding="utf-8")
    monkeypatch.setattr(check_projections, "_PROJECTIONS_DIR", tmp_path)

    assert check_projections._iter_projection_modules() == [public_module]
    errors = check_projections.check_register_calls(tmp_path)
    assert [(error.code, error.path) for error in errors] == [("PROJ001", public_module)]
