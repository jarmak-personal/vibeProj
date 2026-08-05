"""Equal Earth projection (2018).

Modern equal-area pseudocylindrical projection with pleasing aesthetics.
Polynomial formulas — closed-form forward and inverse.
EPSG: 8857, 8858, 8859.

On the ellipsoid the input geodetic latitude is first converted to authalic
latitude (β) so that equal-area properties are preserved. The polynomial
formulas then operate on β. Inverse recovers geodetic latitude from β via
the iterative q-inversion used by LAEA/CEA.
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
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# Polynomial coefficients (Šavrič, Patterson & Jenny 2018)
_A1 = 1.340264
_A2 = -0.081106
_A3 = 0.000893
_A4 = 0.003796

# Scale factor for x: 2√3/3 from the parametric equations
_M = 2.0 * math.sqrt(3.0) / 3.0


class EqualEarth(Projection):
    """Equal-area pseudocylindrical projection (Savric, Patterson & Jenny, 2018)."""

    name = "eqearth"

    def setup(self, params: ProjectionParams) -> dict:
        ec = params.ellipsoid.e
        es = params.ellipsoid.es
        qp = authalic_q_scalar(1.0, ec)  # q at the pole
        rqda = math.sqrt(qp / 2.0)  # R_A / a (authalic sphere radius / semi-major)
        return {
            "a": params.ellipsoid.a,
            "e": ec,
            "es": es,
            "qp": qp,
            "rqda": rqda,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        ec = computed["e"]
        qp = computed["qp"]
        rqda = computed["rqda"]
        numpy_valid = getattr(xp, "__name__", None) == "numpy" and bool(
            xp.all(xp.isfinite(lam)) and xp.all(xp.isfinite(phi))
        )
        finite_lam = lam if numpy_valid else xp.where(xp.isfinite(lam), lam, 0.0)
        finite_phi = phi if numpy_valid else xp.where(xp.isfinite(phi), phi, 0.0)

        # Geodetic → authalic latitude
        q = authalic_q(xp.sin(finite_phi), ec, xp)
        beta = xp.arcsin(xp.clip(q / qp, -1.0, 1.0))

        # Equal Earth polynomial on authalic latitude
        theta = xp.arcsin(math.sqrt(3.0) / 2.0 * xp.sin(beta))
        t2 = theta * theta
        t6 = t2 * t2 * t2
        d = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
        x = rqda * _M * finite_lam * xp.cos(theta) / d
        y = rqda * theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))
        if numpy_valid:
            return x, y
        phi_nonfinite = ~xp.isfinite(phi)
        x = xp.where(phi_nonfinite | xp.isnan(lam), xp.nan, xp.where(xp.isinf(lam), lam, x))
        y = xp.where(phi_nonfinite, xp.nan, y)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        ec = computed["e"]
        es = computed["es"]
        qp = computed["qp"]
        rqda = computed["rqda"]
        numpy_valid = getattr(xp, "__name__", None) == "numpy" and bool(
            xp.all(xp.isfinite(x)) and xp.all(xp.isfinite(y))
        )
        finite_x = x if numpy_valid else xp.where(xp.isfinite(x), x, 0.0)
        finite_y = y if numpy_valid else xp.where(xp.isfinite(y), y, 0.0)

        # Remove rqda scaling
        y_s = finite_y / rqda
        x_s = finite_x / rqda

        # Newton iteration to recover theta from y_s
        theta = y_s
        for _ in range(12):
            t2 = theta * theta
            t6 = t2 * t2 * t2
            fy = theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2)) - y_s
            fpy = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
            theta = theta - fy / fpy
        t2 = theta * theta
        t6 = t2 * t2 * t2
        d = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
        lam = x_s * d / (_M * xp.cos(theta))

        # Recover authalic latitude β from theta
        sin_beta = xp.clip(xp.sin(theta) * 2.0 / math.sqrt(3.0), -1.0, 1.0)

        # Authalic → geodetic latitude via iterative q-inversion
        q = qp * sin_beta
        phi = geodetic_latitude_from_authalic_q(q, qp, ec, es, xp)

        if numpy_valid:
            return lam, phi
        y_nonfinite = ~xp.isfinite(y)
        lam = xp.where(y_nonfinite | xp.isnan(x), xp.nan, xp.where(xp.isinf(x), x, lam))
        phi = xp.where(y_nonfinite, xp.nan, phi)

        return lam, phi


register("eqearth", EqualEarth())
