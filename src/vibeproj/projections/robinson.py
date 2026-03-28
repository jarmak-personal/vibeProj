"""Robinson projection.

Compromise pseudocylindrical projection for world maps. Used by Rand McNally
and formerly by National Geographic. Table-based with interpolation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# Robinson lookup table at 5° intervals from 0° to 90°
# (X_factor, Y_factor) — normalized to X=1 at equator, Y=1 at equator
_TABLE_X = [
    1.0000,
    0.9986,
    0.9954,
    0.9900,
    0.9822,
    0.9730,
    0.9600,
    0.9427,
    0.9216,
    0.8962,
    0.8679,
    0.8350,
    0.7986,
    0.7597,
    0.7186,
    0.6732,
    0.6213,
    0.5722,
    0.5322,
]
_TABLE_Y = [
    0.0000,
    0.0620,
    0.1240,
    0.1860,
    0.2480,
    0.3100,
    0.3720,
    0.4340,
    0.4958,
    0.5571,
    0.6176,
    0.6769,
    0.7346,
    0.7903,
    0.8435,
    0.8936,
    0.9394,
    0.9761,
    1.0000,
]
_FXC = 0.8487  # scale factor for x
_FYC = 1.3523  # scale factor for y
_C1 = 11.459155902616464  # 1/5° in radians = 180/(5*pi) ... actually 5° intervals
_RC1 = 0.08726646259971647  # 5° in radians


class Robinson(Projection):
    """Compromise pseudocylindrical projection with table-based interpolation."""

    name = "robin"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        abs_phi = xp.abs(phi)
        phi_deg = abs_phi * (180.0 / math.pi)

        is_scalar = not hasattr(phi_deg, "__len__")
        if is_scalar:
            phi_deg = xp.asarray([phi_deg], dtype=float)

        idx = xp.clip(phi_deg / 5.0, 0, 17).astype(int)
        frac = phi_deg / 5.0 - idx

        x_arr = xp.array(_TABLE_X, dtype=float)
        y_arr = xp.array(_TABLE_Y, dtype=float)
        idx_safe = xp.clip(idx, 0, 17)
        idx_next = xp.clip(idx + 1, 0, 18)
        X = x_arr[idx_safe] + frac * (x_arr[idx_next] - x_arr[idx_safe])
        Y = y_arr[idx_safe] + frac * (y_arr[idx_next] - y_arr[idx_safe])

        if is_scalar:
            X = float(X[0])
            Y = float(Y[0])

        x = _FXC * X * lam
        y = _FYC * Y * xp.sign(phi)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        abs_y = xp.abs(y) / _FYC

        is_scalar = not hasattr(abs_y, "__len__")
        if is_scalar:
            abs_y = xp.asarray([abs_y], dtype=float)

        y_arr = xp.array(_TABLE_Y, dtype=abs_y.dtype if hasattr(abs_y, "dtype") else float)
        idx = xp.searchsorted(y_arr, abs_y, side="right") - 1
        idx = xp.clip(idx, 0, 17)
        idx_next = xp.clip(idx + 1, 0, 18)
        y0 = y_arr[idx]
        y1 = y_arr[idx_next]
        frac = (abs_y - y0) / xp.maximum(y1 - y0, EPS_DENOM)
        phi_deg = (idx + frac) * 5.0
        x_arr = xp.array(_TABLE_X, dtype=abs_y.dtype if hasattr(abs_y, "dtype") else float)
        X = x_arr[idx] + frac * (x_arr[idx_next] - x_arr[idx])

        if is_scalar:
            phi_deg = float(phi_deg[0])
            X = float(X[0])

        phi = phi_deg * (math.pi / 180.0) * xp.sign(y)
        lam = x / (_FXC * xp.maximum(X, EPS_DENOM))
        return lam, phi


register("robin", Robinson())
