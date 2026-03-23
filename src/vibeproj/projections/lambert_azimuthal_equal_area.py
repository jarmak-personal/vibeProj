"""Lambert Azimuthal Equal Area (LAEA) projection.

Equal-area azimuthal projection. The EU standard for statistical mapping (EPSG:3035).
Also used for MODIS polar products.

Math from PROJ laea.c and Snyder, "Map Projections: A Working Manual".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

_EPS10 = 1e-10
_HALF_PI = math.pi / 2.0


def _qsfn_scalar(sin_phi, e):
    """Scalar q-function."""
    if e < _EPS10:
        return 2.0 * sin_phi
    e_sin = e * sin_phi
    return (1.0 - e * e) * (
        sin_phi / (1.0 - e_sin * e_sin)
        - (1.0 / (2.0 * e)) * math.log((1.0 - e_sin) / (1.0 + e_sin))
    )


def _qsfn_array(sin_phi, e, xp):
    """Vectorized q-function."""
    if e < _EPS10:
        return 2.0 * sin_phi
    e_sin = e * sin_phi
    return (1.0 - e * e) * (
        sin_phi / (1.0 - e_sin * e_sin) - (1.0 / (2.0 * e)) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
    )


def _authlat(q, qp, apa, xp):
    """Convert from geodetic to authalic latitude."""
    beta = xp.arcsin(xp.clip(q / qp, -1.0, 1.0))
    return beta


class LambertAzimuthalEqualArea(Projection):
    name = "laea"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        ec = e.e
        es = e.es

        phi0 = math.radians(params.lat_0)
        sin_phi0 = math.sin(phi0)
        cos_phi0 = math.cos(phi0)

        qp = _qsfn_scalar(1.0, ec)  # q at the pole
        q0 = _qsfn_scalar(sin_phi0, ec)

        # Authalic latitude of origin
        beta0 = math.asin(q0 / qp) if abs(qp) > _EPS10 else phi0

        # Determine mode
        if abs(abs(phi0) - _HALF_PI) < _EPS10:
            mode = "north_pole" if phi0 > 0 else "south_pole"
        elif abs(phi0) < _EPS10:
            mode = "equatorial"
        else:
            mode = "oblique"

        # Normalized (without a) — pipeline multiplies by a
        Rq = math.sqrt(qp / 2.0)

        sin_beta0 = math.sin(beta0)
        cos_beta0 = math.cos(beta0)

        # D: the a's cancel: (a * cos_phi0/sqrt(...)) / (a * Rq * cos_beta0)
        D = (
            cos_phi0 / math.sqrt(1.0 - es * sin_phi0 * sin_phi0) / (Rq * cos_beta0)
            if abs(cos_beta0) > _EPS10
            else 1.0
        )

        return {
            "mode": mode,
            "Rq": Rq,
            "D": D,
            "qp": qp,
            "sin_beta0": sin_beta0,
            "cos_beta0": cos_beta0,
            "e": ec,
            "es": es,
            "a": e.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        mode = computed["mode"]
        Rq = computed["Rq"]
        D = computed["D"]
        qp = computed["qp"]
        e = computed["e"]
        sin_beta0 = computed["sin_beta0"]
        cos_beta0 = computed["cos_beta0"]

        q = _qsfn_array(xp.sin(phi), e, xp)
        beta = xp.arcsin(xp.clip(q / qp, -1.0, 1.0))
        sin_beta = xp.sin(beta)
        cos_beta = xp.cos(beta)

        if mode == "oblique":
            sin_lam = xp.sin(lam)
            cos_lam = xp.cos(lam)
            b = 1.0 + sin_beta0 * sin_beta + cos_beta0 * cos_beta * cos_lam
            b = Rq * xp.sqrt(2.0 / xp.maximum(b, 1e-30))
            x = b * D * cos_beta * sin_lam
            y = (b / D) * (cos_beta0 * sin_beta - sin_beta0 * cos_beta * cos_lam)
        elif mode == "equatorial":
            sin_lam = xp.sin(lam)
            cos_lam = xp.cos(lam)
            b = 1.0 + cos_beta * cos_lam
            b = Rq * xp.sqrt(2.0 / xp.maximum(b, 1e-30))
            x = b * D * cos_beta * sin_lam
            y = (b / D) * sin_beta
        elif mode == "north_pole":
            q_diff = qp - q
            rho = Rq * xp.sqrt(xp.maximum(q_diff, 0.0))
            x = rho * xp.sin(lam)
            y = -rho * xp.cos(lam)
        else:  # south_pole
            q_diff = qp + q
            rho = Rq * xp.sqrt(xp.maximum(q_diff, 0.0))
            x = rho * xp.sin(lam)
            y = rho * xp.cos(lam)

        return x, y

    def inverse(self, x, y, params, computed, xp):
        mode = computed["mode"]
        Rq = computed["Rq"]
        D = computed["D"]
        qp = computed["qp"]
        e = computed["e"]
        es = computed["es"]
        sin_beta0 = computed["sin_beta0"]
        cos_beta0 = computed["cos_beta0"]

        x_adj = x / D
        y_adj = y * D

        rho = xp.sqrt(x_adj * x_adj + y_adj * y_adj)

        if mode == "oblique" or mode == "equatorial":
            ce = 2.0 * xp.arcsin(xp.clip(rho / (2.0 * Rq), -1.0, 1.0))
            sin_ce = xp.sin(ce)
            cos_ce = xp.cos(ce)

            if mode == "oblique":
                sin_beta = cos_ce * sin_beta0 + y_adj * sin_ce * cos_beta0 / xp.maximum(rho, 1e-30)
                lam = xp.arctan2(
                    x_adj * sin_ce,
                    rho * cos_beta0 * cos_ce - y_adj * sin_beta0 * sin_ce,
                )
            else:
                sin_beta = y_adj * sin_ce / xp.maximum(rho, 1e-30)
                lam = xp.arctan2(x_adj * sin_ce, rho * cos_ce)
        elif mode == "north_pole":
            sin_beta = 1.0 - (rho * rho) / (Rq * Rq)
            lam = xp.arctan2(x, -y)
        else:  # south_pole
            sin_beta = (rho * rho) / (Rq * Rq) - 1.0
            lam = xp.arctan2(x, y)

        q = qp * sin_beta
        # Iterative phi from q
        phi = xp.arcsin(xp.clip(q / 2.0, -1.0, 1.0))
        for _ in range(15):
            sin_phi = xp.sin(phi)
            e_sin = e * sin_phi
            one_minus = 1.0 - e_sin * e_sin
            cos_phi = xp.cos(phi)
            cos_phi = xp.where(xp.abs(cos_phi) < 1e-30, 1e-30, cos_phi)
            dphi = (one_minus * one_minus / (2.0 * cos_phi)) * (
                q / (1.0 - es)
                - sin_phi / one_minus
                + (1.0 / (2.0 * e)) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
            )
            phi = phi + dphi
            if xp is np:
                if hasattr(dphi, "__len__"):
                    if xp.all(xp.abs(dphi) < 1e-14):
                        break
                elif abs(float(dphi)) < 1e-14:
                    break

        return lam, phi


register("laea", LambertAzimuthalEqualArea())
