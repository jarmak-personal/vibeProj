"""Lambert Azimuthal Equal Area (LAEA) projection.

Equal-area azimuthal projection. The EU standard for statistical mapping (EPSG:3035).
Also used for MODIS polar products.

Math from PROJ laea.c and Snyder, "Map Projections: A Working Manual".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections._equal_area import (
    authalic_q,
    authalic_q_scalar,
    geodetic_latitude_from_authalic_q,
)
from vibeproj.projections.base import EPS_ANGLE, EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams
_HALF_PI = math.pi / 2.0


def _antipode_scale(denominator, radius, xp):
    """Return a safe LAEA scale; callers apply the antipode sentinel atomically."""
    safe_denominator = xp.where(denominator > EPS_ANGLE, denominator, xp.nan)
    scale = radius * xp.sqrt(2.0 / safe_denominator)
    return xp.where(denominator <= EPS_ANGLE, 0.0, scale)


class LambertAzimuthalEqualArea(Projection):
    """Equal-area azimuthal projection for continental and polar mapping."""

    name = "laea"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        ec = e.e
        es = e.es
        if not math.isfinite(e.a) or e.a <= 0.0:
            raise ValueError("LAEA semi-major axis must be finite and positive")
        if not math.isfinite(ec) or not math.isfinite(es) or not (0.0 <= es < 1.0):
            raise ValueError("LAEA eccentricity must be finite with 0 <= e^2 < 1")

        phi0 = math.radians(params.lat_0)
        if not math.isfinite(phi0) or abs(phi0) > _HALF_PI:
            raise ValueError("LAEA latitude of origin must be finite and within [-90, 90]")
        sin_phi0 = math.sin(phi0)
        cos_phi0 = math.cos(phi0)

        qp = authalic_q_scalar(1.0, ec)  # q at the pole
        q0 = authalic_q_scalar(sin_phi0, ec)

        # Authalic latitude of origin
        beta0 = math.asin(max(-1.0, min(1.0, q0 / qp)))

        # Determine mode
        if abs(abs(phi0) - _HALF_PI) < EPS_ANGLE:
            mode = "north_pole" if phi0 > 0 else "south_pole"
        elif abs(phi0) < EPS_ANGLE:
            mode = "equatorial"
        else:
            mode = "oblique"

        # Normalized (without a) — pipeline multiplies by a
        Rq = math.sqrt(qp / 2.0)

        sin_beta0 = math.sin(beta0)
        cos_beta0 = math.cos(beta0)

        # D: the a's cancel: (a * cos_phi0/sqrt(...)) / (a * Rq * cos_beta0)
        if mode in ("north_pole", "south_pole"):
            D = 1.0
        else:
            D = cos_phi0 / math.sqrt(1.0 - es * sin_phi0 * sin_phi0) / (Rq * cos_beta0)
        if not all(math.isfinite(value) for value in (qp, beta0, Rq, D)):
            raise ValueError("LAEA setup produced non-finite authalic parameters")

        return {
            "mode": mode,
            "Rq": Rq,
            "D": D,
            "qp": qp,
            "sin_beta0": sin_beta0,
            "cos_beta0": cos_beta0,
            "phi0": phi0,
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
        numpy_valid = getattr(xp, "__name__", None) == "numpy" and bool(
            xp.all(xp.isfinite(lam)) and xp.all(xp.abs(phi) <= _HALF_PI)
        )
        finite_lam = lam if numpy_valid else xp.where(xp.isfinite(lam), lam, 0.0)
        finite_phi = phi if numpy_valid else xp.where(xp.isfinite(phi), phi, 0.0)

        q = authalic_q(xp.sin(finite_phi), e, xp)
        beta = xp.arcsin(xp.clip(q / qp, -1.0, 1.0))
        sin_beta = xp.sin(beta)
        cos_beta = xp.cos(beta)

        if mode == "oblique":
            sin_lam = xp.sin(finite_lam)
            cos_lam = xp.cos(finite_lam)
            b = 1.0 + sin_beta0 * sin_beta + cos_beta0 * cos_beta * cos_lam
            if numpy_valid and bool(xp.all(b > EPS_ANGLE)):
                b = Rq * xp.sqrt(2.0 / b)
                return (
                    b * D * cos_beta * sin_lam,
                    (b / D) * (cos_beta0 * sin_beta - sin_beta0 * cos_beta * cos_lam),
                )
            antipode = b <= EPS_ANGLE
            b = _antipode_scale(b, Rq, xp)
            x = b * D * cos_beta * sin_lam
            y = (b / D) * (cos_beta0 * sin_beta - sin_beta0 * cos_beta * cos_lam)
            x = xp.where(antipode, xp.inf, x)
            y = xp.where(antipode, xp.inf, y)
        elif mode == "equatorial":
            sin_lam = xp.sin(finite_lam)
            cos_lam = xp.cos(finite_lam)
            b = 1.0 + cos_beta * cos_lam
            if numpy_valid and bool(xp.all(b > EPS_ANGLE)):
                b = Rq * xp.sqrt(2.0 / b)
                return b * D * cos_beta * sin_lam, (b / D) * sin_beta
            antipode = b <= EPS_ANGLE
            b = _antipode_scale(b, Rq, xp)
            x = b * D * cos_beta * sin_lam
            y = (b / D) * sin_beta
            x = xp.where(antipode, xp.inf, x)
            y = xp.where(antipode, xp.inf, y)
        elif mode == "north_pole":
            q_diff = qp - q
            rho = xp.sqrt(xp.maximum(q_diff, 0.0))
            x = rho * xp.sin(finite_lam)
            y = -rho * xp.cos(finite_lam)
            if numpy_valid and bool(xp.all(xp.abs(phi + _HALF_PI) > EPS_ANGLE)):
                return x, y
            antipode = xp.abs(phi + _HALF_PI) <= EPS_ANGLE
            x = xp.where(antipode, xp.inf, x)
            y = xp.where(antipode, xp.inf, y)
        else:  # south_pole
            q_diff = qp + q
            rho = xp.sqrt(xp.maximum(q_diff, 0.0))
            x = rho * xp.sin(finite_lam)
            y = rho * xp.cos(finite_lam)
            if numpy_valid and bool(xp.all(xp.abs(phi - _HALF_PI) > EPS_ANGLE)):
                return x, y
            antipode = xp.abs(phi - _HALF_PI) <= EPS_ANGLE
            x = xp.where(antipode, xp.inf, x)
            y = xp.where(antipode, xp.inf, y)

        latitude_outside = xp.isfinite(phi) & (xp.abs(phi) > _HALF_PI)
        has_nan = xp.isnan(lam) | xp.isnan(phi)
        infinite = xp.isinf(lam) | xp.isinf(phi)
        x = xp.where(has_nan, xp.nan, xp.where(infinite | latitude_outside, xp.inf, x))
        y = xp.where(has_nan, xp.nan, xp.where(infinite | latitude_outside, xp.inf, y))

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
        phi0 = computed["phi0"]

        if mode == "oblique" or mode == "equatorial":
            x_adj = x / D
            y_adj = y * D
            rho = xp.sqrt(x_adj * x_adj + y_adj * y_adj)
            outside = rho > 2.0 * Rq
            ce = 2.0 * xp.arcsin(xp.clip(rho / (2.0 * Rq), -1.0, 1.0))
            sin_ce = xp.sin(ce)
            cos_ce = xp.cos(ce)

            if mode == "oblique":
                sin_beta = cos_ce * sin_beta0 + y_adj * sin_ce * cos_beta0 / xp.maximum(
                    rho, EPS_DENOM
                )
                lam = xp.arctan2(
                    x_adj * sin_ce,
                    rho * cos_beta0 * cos_ce - y_adj * sin_beta0 * sin_ce,
                )
            else:
                sin_beta = y_adj * sin_ce / xp.maximum(rho, EPS_DENOM)
                lam = xp.arctan2(x_adj * sin_ce, rho * cos_ce)
        elif mode == "north_pole":
            rho = xp.sqrt(x * x + y * y)
            outside = rho * rho > 2.0 * qp
            sin_beta = 1.0 - (rho * rho) / qp
            lam = xp.arctan2(x, -y)
        else:  # south_pole
            rho = xp.sqrt(x * x + y * y)
            outside = rho * rho > 2.0 * qp
            sin_beta = (rho * rho) / qp - 1.0
            lam = xp.arctan2(x, y)

        sin_beta = xp.clip(sin_beta, -1.0, 1.0)
        phi = geodetic_latitude_from_authalic_q(qp * sin_beta, qp, e, es, xp)

        center = rho == 0.0
        lam = xp.where(center, 0.0, lam)
        phi = xp.where(center, phi0, phi)
        has_nan = xp.isnan(x) | xp.isnan(y)
        lam = xp.where(has_nan, xp.nan, xp.where(outside, xp.inf, lam))
        phi = xp.where(has_nan, xp.nan, xp.where(outside, xp.inf, phi))

        return lam, phi


register("laea", LambertAzimuthalEqualArea())
