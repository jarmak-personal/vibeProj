"""Ellipsoid definitions for coordinate projections.

Ported from cuProj (NVIDIA/RAPIDS) ellipsoid.hpp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ellipsoid:
    """Reference ellipsoid parameters.

    Constructed from semi-major axis and inverse flattening.
    All derived parameters are computed at construction time.
    """

    a: float  # semi-major axis (meters)
    b: float  # semi-minor axis (meters)
    f: float  # flattening
    e: float  # first eccentricity
    es: float  # first eccentricity squared
    n: float  # third flattening

    @classmethod
    def from_af(cls, a: float, inverse_flattening: float) -> Ellipsoid:
        """Create ellipsoid from semi-major axis and inverse flattening."""
        f = 1.0 / inverse_flattening
        b = a * (1.0 - f)
        es = 2 * f - f * f
        e = math.sqrt(es)
        alpha = math.asin(e)
        n = math.pow(math.tan(alpha / 2), 2)
        return cls(a=a, b=b, f=f, e=e, es=es, n=n)


# Standard ellipsoids
WGS84 = Ellipsoid.from_af(6378137.0, 298.257223563)
GRS80 = Ellipsoid.from_af(6378137.0, 298.257222101)
# Web Mercator sphere — f=0 so from_af() can't be used (1/inf → NaN in derived params)
SPHERE = Ellipsoid(
    a=6378137.0,
    b=6378137.0,
    f=0.0,
    e=0.0,
    es=0.0,
    n=0.0,
)
