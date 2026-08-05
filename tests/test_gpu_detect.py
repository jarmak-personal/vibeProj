"""Tests for GPU-dependent internal strategy selection."""

import pytest

from vibeproj.fused_kernels import _resolve_helmert_trig_mode, _resolve_tmerc_forward_mode
from vibeproj.gpu_detect import _supports_fixed_int64_trig


@pytest.mark.parametrize(
    ("major", "minor", "ratio", "expected"),
    [
        (8, 9, 64, True),  # Ada consumer
        (8, 9, 16, True),  # Conservative lower ratio boundary
        (8, 9, 2, False),  # Hypothetical strong-fp64 Ada
        (8, 6, 64, False),  # Ampere consumer is not validated yet
        (9, 0, 2, False),  # Hopper datacenter
        (10, 0, 64, False),  # Unvalidated future architecture
    ],
)
def test_fixed_int64_trig_support_gate(major, minor, ratio, expected):
    assert _supports_fixed_int64_trig(major, minor, ratio) is expected


def test_explicit_helmert_trig_modes():
    assert _resolve_helmert_trig_mode("fp64") == "fp64"
    assert _resolve_helmert_trig_mode("int64") == "int64"


def test_invalid_helmert_trig_mode():
    with pytest.raises(ValueError, match="Invalid Helmert trig mode"):
        _resolve_helmert_trig_mode("fastish")


def test_explicit_tmerc_forward_modes():
    assert _resolve_tmerc_forward_mode("fp64") == "fp64"
    assert _resolve_tmerc_forward_mode("int64") == "int64"


def test_invalid_tmerc_forward_mode():
    with pytest.raises(ValueError, match="Invalid TM forward mode"):
        _resolve_tmerc_forward_mode("fastish")
