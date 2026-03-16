"""Equidistant Cylindrical / Plate Carree projection.

The trivial projection: x = lon, y = lat (in radians, scaled by the ellipsoid).
Used as the default for lat/lon raster data (climate models, global datasets).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class PlateCarree(Projection):
    name = "eqc"

    def setup(self, params: ProjectionParams) -> dict:
        lat_ts = math.radians(params.lat_1) if params.lat_1 != 0 else 0.0
        return {
            "a": params.ellipsoid.a,
            "cos_lat_ts": math.cos(lat_ts),
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        cos_lat_ts = computed["cos_lat_ts"]
        x = lam * cos_lat_ts
        y = phi
        return x, y

    def inverse(self, x, y, params, computed, xp):
        cos_lat_ts = computed["cos_lat_ts"]
        lam = x / cos_lat_ts
        phi = y
        return lam, phi


register("eqc", PlateCarree())
