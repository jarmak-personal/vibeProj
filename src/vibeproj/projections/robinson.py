"""Robinson projection.

Compromise pseudocylindrical projection for world maps. Used by Rand McNally
and formerly by National Geographic. Uses cubic polynomial interpolation on
the Robinson coefficient table, matching PROJ's approach.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vibeproj.projections import register
from vibeproj.projections.base import EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# ---------------------------------------------------------------------------
# Robinson cubic polynomial coefficient tables (from PROJ PJ_robin.c)
#
# Each interval spans 5 degrees of latitude.  The polynomial variable *z*
# is the offset within the interval **in degrees** (0 <= z <= 5).
#
#   value = c0 + z * (c1 + z * (c2 + z * c3))
#
# 19 entries: indices 0-17 are the primary intervals (0-90 deg).
# Index 18 is used by the inverse to bracket the last interval.
# ---------------------------------------------------------------------------

_COEFS_X = [
    # (c0, c1, c2, c3)
    (1.0, 2.2199e-17, -7.15515e-05, 3.1103e-06),
    (0.9986, -0.000482243, -2.4897e-05, -1.3309e-06),
    (0.9954, -0.00083103, -4.48605e-05, -9.86701e-07),
    (0.99, -0.00135364, -5.9661e-05, 3.6777e-06),
    (0.9822, -0.00167442, -4.49547e-06, -5.72411e-06),
    (0.973, -0.00214868, -9.03571e-05, 1.8736e-08),
    (0.96, -0.00305085, -9.00761e-05, 1.64917e-06),
    (0.9427, -0.00382792, -6.53386e-05, -2.6154e-06),
    (0.9216, -0.00467746, -0.00010457, 4.81243e-06),
    (0.8962, -0.00536223, -3.23831e-05, -5.43432e-06),
    (0.8679, -0.00609363, -0.000113898, 3.32484e-06),
    (0.835, -0.00698325, -6.40253e-05, 9.34959e-07),
    (0.7986, -0.00755338, -5.00009e-05, 9.35324e-07),
    (0.7597, -0.00798324, -3.5971e-05, -2.27626e-06),
    (0.7186, -0.00851367, -7.01149e-05, -8.6303e-06),
    (0.6732, -0.00986209, -0.000199569, 1.91974e-05),
    (0.6213, -0.010418, 8.83923e-05, 6.24051e-06),
    (0.5722, -0.00906601, 0.000182, 6.24051e-06),
    (0.5322, -0.00677797, 0.000275608, 6.24051e-06),
]

_COEFS_Y = [
    (-5.20417e-18, 0.0124, 1.21431e-18, -8.45284e-11),
    (0.062, 0.0124, -1.26793e-09, 4.22642e-10),
    (0.124, 0.0124, 5.07171e-09, -1.60604e-09),
    (0.186, 0.0123999, -1.90189e-08, 6.00152e-09),
    (0.248, 0.0124002, 7.10039e-08, -2.24e-08),
    (0.31, 0.0123992, -2.64997e-07, 8.35986e-08),
    (0.372, 0.0124029, 9.88983e-07, -3.11994e-07),
    (0.434, 0.0123893, -3.69093e-06, -4.35621e-07),
    (0.4958, 0.0123198, -1.02252e-05, -3.45523e-07),
    (0.5571, 0.0121916, -1.54081e-05, -5.82288e-07),
    (0.6176, 0.0119938, -2.41424e-05, -5.25327e-07),
    (0.6769, 0.011713, -3.20223e-05, -5.16405e-07),
    (0.7346, 0.0113541, -3.97684e-05, -6.09052e-07),
    (0.7903, 0.0109107, -4.89042e-05, -1.04739e-06),
    (0.8435, 0.0103431, -6.4615e-05, -1.40374e-09),
    (0.8936, 0.00969686, -6.4636e-05, -8.547e-06),
    (0.9394, 0.00840947, -0.000192841, -4.2106e-06),
    (0.9761, 0.00616527, -0.000256, -4.2106e-06),
    (1.0, 0.00328947, -0.000319159, -4.2106e-06),
]

# Flat arrays for vectorised indexing (used in forward/inverse)
# Pre-built numpy arrays — avoids per-call allocation in forward/inverse.
_XC0 = np.array([c[0] for c in _COEFS_X[:18]])
_XC1 = np.array([c[1] for c in _COEFS_X[:18]])
_XC2 = np.array([c[2] for c in _COEFS_X[:18]])
_XC3 = np.array([c[3] for c in _COEFS_X[:18]])

_YC0 = np.array([c[0] for c in _COEFS_Y])  # 19 entries (used by inverse search)
_YC1 = np.array([c[1] for c in _COEFS_Y])
_YC2 = np.array([c[2] for c in _COEFS_Y])
_YC3 = np.array([c[3] for c in _COEFS_Y])

_FXC = 0.8487  # scale factor for x
_FYC = 1.3523  # scale factor for y
_C1 = 11.45915590261646417544  # 180/(5*pi), converts radians to 5-degree intervals
_NEWTON_EPS = 1e-12
_NEWTON_ITERS = 4  # cubic converges quadratically from linear initial guess


class Robinson(Projection):
    """Compromise pseudocylindrical projection with cubic polynomial interpolation."""

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

        is_scalar = not hasattr(phi, "__len__")
        if is_scalar:
            abs_phi = xp.asarray([abs_phi], dtype=float)

        # Interval index and offset in degrees (z in [0, 5])
        idx = xp.clip((abs_phi * _C1).astype(int), 0, 17)
        z = abs_phi * (180.0 / math.pi) - 5.0 * idx

        # Coefficient arrays (zero-copy view for NumPy, single transfer for CuPy)
        xc0 = xp.asarray(_XC0)
        xc1 = xp.asarray(_XC1)
        xc2 = xp.asarray(_XC2)
        xc3 = xp.asarray(_XC3)
        yc0 = xp.asarray(_YC0[:18])
        yc1 = xp.asarray(_YC1[:18])
        yc2 = xp.asarray(_YC2[:18])
        yc3 = xp.asarray(_YC3[:18])

        X = xc0[idx] + z * (xc1[idx] + z * (xc2[idx] + z * xc3[idx]))
        Y = yc0[idx] + z * (yc1[idx] + z * (yc2[idx] + z * yc3[idx]))

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

        # Coefficient arrays (zero-copy view for NumPy, single transfer for CuPy)
        yc0_arr = xp.asarray(_YC0)  # 19 entries
        yc1_arr = xp.asarray(_YC1)
        yc2_arr = xp.asarray(_YC2)
        yc3_arr = xp.asarray(_YC3)
        xc0_arr = xp.asarray(_XC0)
        xc1_arr = xp.asarray(_XC1)
        xc2_arr = xp.asarray(_XC2)
        xc3_arr = xp.asarray(_XC3)

        # Find interval from YC0 values (c0 coefficients at each knot)
        idx = xp.searchsorted(yc0_arr, abs_y, side="right") - 1
        idx = xp.clip(idx, 0, 17)

        # Gather coefficients for the found interval
        c0 = yc0_arr[idx]
        c1 = yc1_arr[idx]
        c2 = yc2_arr[idx]
        c3 = yc3_arr[idx]

        # Linear initial guess for z (in degrees, 0..5)
        c0_next = yc0_arr[xp.clip(idx + 1, 0, 18)]
        z = 5.0 * (abs_y - c0) / xp.maximum(c0_next - c0, EPS_DENOM)
        z = xp.clip(z, 0.0, 5.0)

        # Newton-Raphson iteration on V(z) - abs_y = 0
        # V(z)  = (c0 - abs_y) + z*(c1 + z*(c2 + z*c3))
        # V'(z) = c1 + z*(2*c2 + z*3*c3)
        c0_shifted = c0 - abs_y
        for _ in range(_NEWTON_ITERS):
            val = c0_shifted + z * (c1 + z * (c2 + z * c3))
            deriv = c1 + z * (2.0 * c2 + z * 3.0 * c3)
            deriv = xp.where(xp.abs(deriv) < EPS_DENOM, EPS_DENOM, deriv)
            dz = val / deriv
            z = z - dz
            if xp.all(xp.abs(dz) < _NEWTON_EPS):
                break

        phi_deg = 5.0 * idx + z
        X = xc0_arr[idx] + z * (xc1_arr[idx] + z * (xc2_arr[idx] + z * xc3_arr[idx]))

        if is_scalar:
            phi_deg = float(phi_deg[0])
            X = float(X[0])

        phi = phi_deg * (math.pi / 180.0) * xp.sign(y)
        lam = x / (_FXC * xp.maximum(X, EPS_DENOM))
        return lam, phi


register("robin", Robinson())
