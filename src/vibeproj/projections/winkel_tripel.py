"""Winkel Tripel projection.

Compromise projection used by National Geographic since 1998.
Arithmetic mean of Plate Carrée and Aitoff projections.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class WinkelTripel(Projection):
    name = "wintri"

    def setup(self, params: ProjectionParams) -> dict:
        # Standard parallel (default: acos(2/pi) ≈ 50.46°)
        lat1 = math.radians(params.lat_1) if params.lat_1 != 0 else math.acos(2.0 / math.pi)
        return {
            "a": params.ellipsoid.a,
            "cos_phi1": math.cos(lat1),
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        cos_phi1 = computed["cos_phi1"]
        cos_phi = xp.cos(phi)
        alpha = xp.arccos(xp.clip(cos_phi * xp.cos(lam / 2), -1.0, 1.0))
        sinc_alpha = xp.where(xp.abs(alpha) < 1e-10, 1.0, xp.sin(alpha) / alpha)
        # Aitoff component
        x_aitoff = 2 * cos_phi * xp.sin(lam / 2) / sinc_alpha
        y_aitoff = xp.sin(phi) / sinc_alpha
        # Plate Carrée component
        x_eqc = lam * cos_phi1
        y_eqc = phi
        # Average
        x = (x_aitoff + x_eqc) / 2
        y = (y_aitoff + y_eqc) / 2
        return x, y

    def inverse(self, x, y, params, computed, xp):
        cos_phi1 = computed["cos_phi1"]
        # Newton iteration for the inverse
        lam = x * 2
        phi = y
        for _ in range(20):
            cos_phi = xp.cos(phi)
            sin_phi = xp.sin(phi)
            cos_half_lam = xp.cos(lam / 2)
            sin_half_lam = xp.sin(lam / 2)
            alpha = xp.arccos(xp.clip(cos_phi * cos_half_lam, -1.0, 1.0))
            sinc_alpha = xp.where(xp.abs(alpha) < 1e-10, 1.0, xp.sin(alpha) / alpha)
            fx = (2 * cos_phi * sin_half_lam / sinc_alpha + lam * cos_phi1) / 2 - x
            fy = (sin_phi / sinc_alpha + phi) / 2 - y
            # Approximate Jacobian (use finite differences conceptually, but simplified)
            dlam = -fx * 0.5
            dphi = -fy * 0.5
            lam = lam + dlam
            phi = phi + dphi
            if hasattr(dlam, "__len__"):
                if xp.all((xp.abs(dlam) < 1e-10) & (xp.abs(dphi) < 1e-10)):
                    break
            elif abs(float(dlam)) < 1e-10 and abs(float(dphi)) < 1e-10:
                break
        return lam, phi


register("wintri", WinkelTripel())
