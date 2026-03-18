"""Projection registry — maps projection names to implementation classes."""

from __future__ import annotations

from vibeproj.projections.base import Projection

# Registry: projection_name -> Projection instance
PROJECTION_REGISTRY: dict[str, Projection] = {}


def register(name: str, projection: Projection):
    """Register a projection implementation."""
    PROJECTION_REGISTRY[name] = projection


def get_projection(name: str) -> Projection:
    """Look up a projection by name."""
    if name not in PROJECTION_REGISTRY:
        from vibeproj.exceptions import UnsupportedProjectionError

        supported = sorted(PROJECTION_REGISTRY.keys())
        raise UnsupportedProjectionError(f"Unknown projection '{name}'. Supported: {supported}")
    return PROJECTION_REGISTRY[name]


# Import all projection modules to trigger registration
from vibeproj.projections import (  # noqa: E402, F401
    albers_equal_area,
    azimuthal_equidistant,
    cylindrical_equal_area,
    eckert_iv,
    eckert_vi,
    equal_earth,
    geostationary,
    gnomonic,
    krovak,
    lambert_azimuthal_equal_area,
    lambert_conformal_conic,
    mercator,
    mollweide,
    natural_earth,
    oblique_mercator,
    oblique_stereographic,
    orthographic,
    plate_carree,
    robinson,
    sinusoidal,
    stereographic,
    transverse_mercator,
    winkel_tripel,
)
