"""Winkel Tripel projection.

Compromise projection used by National Geographic since 1998.
Arithmetic mean of Plate Carrée and Aitoff projections.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_ANGLE, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class WinkelTripel(Projection):
    """Compromise projection averaging Plate Carree and Aitoff."""

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
        sinc_alpha = xp.where(xp.abs(alpha) < EPS_ANGLE, 1.0, xp.sin(alpha) / alpha)
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
        eps = EPS_ANGLE
        # Initial guess
        lam = x * 2.0
        phi = y * 1.0

        for _ in range(10):
            cp = xp.cos(phi)
            sp = xp.sin(phi)
            ch = xp.cos(lam * 0.5)
            sh = xp.sin(lam * 0.5)

            D = cp * ch
            alpha = xp.arccos(xp.clip(D, -1.0, 1.0))
            sa = xp.sin(alpha)

            # sinc_alpha and its reciprocal, with guard for alpha ~ 0
            small = xp.abs(alpha) < eps
            sa_safe = xp.where(small, 1.0, sa)
            rsinc = xp.where(small, 1.0, alpha / sa_safe)  # 1 / sinc(alpha)

            # G = (sa - alpha * D) / sa^3, derivative factor
            sa3 = sa_safe * sa_safe * sa_safe
            G = xp.where(small, 0.0, (sa_safe - alpha * D) / sa3)

            # Forward residuals
            f1 = (2.0 * cp * sh * rsinc + lam * cos_phi1) * 0.5 - x
            f2 = (sp * rsinc + phi) * 0.5 - y

            # Analytical Jacobian entries (2x2 per point)
            J11 = (cp * ch * rsinc + cp * cp * sh * sh * G + cos_phi1) * 0.5
            J12 = sp * sh * (D * G - rsinc)
            J21 = sp * cp * sh * G * 0.25
            J22 = (cp * rsinc + sp * sp * ch * G + 1.0) * 0.5

            # Solve via Cramer's rule: J * [dlam, dphi] = -[f1, f2]
            det = J11 * J22 - J12 * J21
            det_safe = xp.where(xp.abs(det) < 1e-30, 1e-30, det)

            dlam = (J22 * f1 - J12 * f2) / det_safe
            dphi = (J11 * f2 - J21 * f1) / det_safe

            lam = lam - dlam
            phi = phi - dphi

            # Convergence check
            if hasattr(dlam, "__len__"):
                if xp.all((xp.abs(dlam) < eps) & (xp.abs(dphi) < eps)):
                    break
            elif abs(float(dlam)) < eps and abs(float(dphi)) < eps:
                break

        return lam, phi


register("wintri", WinkelTripel())
