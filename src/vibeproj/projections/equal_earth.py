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
from vibeproj.projections.base import EPS_ANGLE, EPS_CONV, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# Polynomial coefficients (Šavrič, Patterson & Jenny 2018)
_A1 = 1.340264
_A2 = -0.081106
_A3 = 0.000893
_A4 = 0.003796

# Scale factor for x: 2√3/3 from the parametric equations
_M = 2.0 * math.sqrt(3.0) / 3.0


def _qsfn_scalar(sin_phi, e):
    """Scalar q-function for authalic latitude."""
    if e < EPS_ANGLE:
        return 2.0 * sin_phi
    e_sin = e * sin_phi
    return (1.0 - e * e) * (
        sin_phi / (1.0 - e_sin * e_sin) - (0.5 / e) * math.log((1.0 - e_sin) / (1.0 + e_sin))
    )


def _qsfn_array(sin_phi, e, xp):
    """Vectorised q-function."""
    if e < EPS_ANGLE:
        return 2.0 * sin_phi
    e_sin = e * sin_phi
    return (1.0 - e * e) * (
        sin_phi / (1.0 - e_sin * e_sin) - (0.5 / e) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
    )


class EqualEarth(Projection):
    name = "eqearth"

    def setup(self, params: ProjectionParams) -> dict:
        ec = params.ellipsoid.e
        es = params.ellipsoid.es
        qp = _qsfn_scalar(1.0, ec)  # q at the pole
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

        # Geodetic → authalic latitude
        q = _qsfn_array(xp.sin(phi), ec, xp)
        beta = xp.arcsin(xp.clip(q / qp, -1.0, 1.0))

        # Equal Earth polynomial on authalic latitude
        theta = xp.arcsin(math.sqrt(3.0) / 2.0 * xp.sin(beta))
        t2 = theta * theta
        t6 = t2 * t2 * t2
        d = _A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)
        x = rqda * _M * lam * xp.cos(theta) / d
        y = rqda * theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))
        return x, y

    def inverse(self, x, y, params, computed, xp):
        ec = computed["e"]
        es = computed["es"]
        qp = computed["qp"]
        rqda = computed["rqda"]

        # Remove rqda scaling
        y_s = y / rqda
        x_s = x / rqda

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
        phi = xp.arcsin(xp.clip(q / 2.0, -1.0, 1.0))
        for _ in range(15):
            sin_phi = xp.sin(phi)
            e_sin = ec * sin_phi
            one_minus = 1.0 - e_sin * e_sin
            dphi = (one_minus * one_minus / (2.0 * xp.cos(phi))) * (
                q / (1.0 - es)
                - sin_phi / one_minus
                + (0.5 / ec) * xp.log((1.0 - e_sin) / (1.0 + e_sin))
            )
            phi = phi + dphi
            if hasattr(dphi, "__len__"):
                if xp.all(xp.abs(dphi) < EPS_CONV):
                    break
            elif abs(float(dphi)) < EPS_CONV:
                break

        return lam, phi


register("eqearth", EqualEarth())
