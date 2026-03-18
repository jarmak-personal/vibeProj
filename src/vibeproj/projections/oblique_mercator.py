"""Hotine Oblique Mercator projection.

Conformal oblique cylindrical projection used for national grids where the area
of interest lies along a great circle at an oblique angle (e.g. Malaysia RSO,
Alaska zone 1). Supports EPSG variants A and B.

Math follows Snyder "Map Projections: A Working Manual" pp. 274-278
and the EPSG Guidance Note 7-2, section 2.4.3.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

_EPS = 1e-10
_HALF_PI = math.pi / 2.0


def _tsfn(phi, sin_phi, e):
    """Conformal latitude function t(φ).

    t = tan(π/4 - φ/2) · ((1+e·sinφ)/(1-e·sinφ))^(e/2)
    """
    e_sin = e * sin_phi
    return math.tan(_HALF_PI / 2.0 - phi / 2.0) * ((1.0 + e_sin) / (1.0 - e_sin)) ** (e / 2.0)


def _tsfn_array(phi, sin_phi, e, xp):
    """Vectorised conformal latitude function."""
    e_sin = e * sin_phi
    return xp.tan(math.pi / 4.0 - phi / 2.0) * ((1.0 + e_sin) / (1.0 - e_sin)) ** (e / 2.0)


def _phi_from_t(t, e, xp):
    """Iterative inverse of the conformal latitude function."""
    phi = _HALF_PI - 2.0 * xp.arctan(t)
    for _ in range(15):
        sin_phi = xp.sin(phi)
        e_sin = e * sin_phi
        phi_new = _HALF_PI - 2.0 * xp.arctan(t * ((1.0 - e_sin) / (1.0 + e_sin)) ** (e / 2.0))
        dphi = phi_new - phi
        phi = phi_new
        if hasattr(dphi, "__len__"):
            if xp.all(xp.abs(dphi) < 1e-14):
                break
        elif abs(float(dphi)) < 1e-14:
            break
    return phi


class ObliqueMercator(Projection):
    name = "omerc"

    def setup(self, params: ProjectionParams) -> dict:
        ell = params.ellipsoid
        a = ell.a
        e = ell.e
        es = ell.es

        phi_c = math.radians(params.lat_0)  # latitude of projection centre
        lam_c = math.radians(params.lon_0)  # longitude of projection centre
        alpha_c = math.radians(params.extra.get("alpha_c", 0.0))  # azimuth at centre
        gamma_c = math.radians(params.extra.get("gamma_c", 0.0))  # rectified grid angle
        k_0 = params.k_0
        no_uoff = params.extra.get("no_uoff", False)  # True for variant B

        sin_phi_c = math.sin(phi_c)
        cos_phi_c = math.cos(phi_c)
        one_es = 1.0 - es

        B = math.sqrt(1.0 + es * cos_phi_c**4 / one_es)
        A_norm = B * k_0 * math.sqrt(one_es) / (1.0 - es * sin_phi_c**2)

        t_c = _tsfn(phi_c, sin_phi_c, e)

        D = B * math.sqrt(one_es) / (cos_phi_c * math.sqrt(1.0 - es * sin_phi_c**2))
        D_sq = D * D
        if D_sq < 1.0:
            D_sq = 1.0
            D = 1.0

        sgn = 1.0 if phi_c >= 0.0 else -1.0
        F = D + math.sqrt(D_sq - 1.0) * sgn
        H = F * t_c**B  # Snyder's E
        G = (F - 1.0 / F) / 2.0

        gamma_0 = math.asin(math.sin(alpha_c) / D)
        lam_0 = lam_c - math.asin(G * math.tan(gamma_0)) / B

        # u at the projection centre (normalized by a)
        if abs(math.cos(alpha_c)) < _EPS:
            u_c = sgn * A_norm * (lam_c - lam_0)
        else:
            u_c = sgn * (A_norm / B) * math.atan2(math.sqrt(D_sq - 1.0), math.cos(alpha_c))

        if no_uoff:
            u_c = 0.0

        return {
            "a": a,
            "e": e,
            "B": B,
            "A_norm": A_norm,
            "H": H,
            "sin_gamma0": math.sin(gamma_0),
            "cos_gamma0": math.cos(gamma_0),
            "sin_gamma_c": math.sin(gamma_c),
            "cos_gamma_c": math.cos(gamma_c),
            "u_c": u_c,
            "lam0": lam_0,  # adjusted central meridian (NOT lon_0)
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        e = computed["e"]
        B = computed["B"]
        A_norm = computed["A_norm"]
        H = computed["H"]
        sin_g0 = computed["sin_gamma0"]
        cos_g0 = computed["cos_gamma0"]
        sin_gc = computed["sin_gamma_c"]
        cos_gc = computed["cos_gamma_c"]
        u_c = computed["u_c"]

        # Conformal latitude function (vectorised)
        sin_phi = xp.sin(phi)
        t = _tsfn_array(phi, sin_phi, e, xp)

        Q = H / xp.power(t, B)
        S = (Q - 1.0 / Q) / 2.0
        T = (Q + 1.0 / Q) / 2.0
        V = xp.sin(B * lam)
        U = (-V * cos_g0 + S * sin_g0) / T
        # Clamp U to avoid log singularity
        U = xp.clip(U, -0.9999999999, 0.9999999999)

        v = A_norm / (2.0 * B) * xp.log((1.0 - U) / (1.0 + U))
        u = (A_norm / B) * xp.arctan2(S * cos_g0 + V * sin_g0, xp.cos(B * lam))
        u = u - u_c

        # Apply rectified grid rotation
        x = v * cos_gc + u * sin_gc
        y = u * cos_gc - v * sin_gc
        return x, y

    def inverse(self, x, y, params, computed, xp):
        e = computed["e"]
        B = computed["B"]
        A_norm = computed["A_norm"]
        H = computed["H"]
        sin_g0 = computed["sin_gamma0"]
        cos_g0 = computed["cos_gamma0"]
        sin_gc = computed["sin_gamma_c"]
        cos_gc = computed["cos_gamma_c"]
        u_c = computed["u_c"]

        # Undo rectified grid rotation
        v = x * cos_gc - y * sin_gc
        u = x * sin_gc + y * cos_gc + u_c

        # Oblique Mercator inverse (PROJ formulation)
        Qp = xp.exp(-B * v / A_norm)
        Sp = (Qp - 1.0 / Qp) / 2.0
        Tp = (Qp + 1.0 / Qp) / 2.0
        Vp = xp.sin(B * u / A_norm)
        Up = (Vp * cos_g0 + Sp * sin_g0) / Tp
        Up = xp.clip(Up, -0.9999999999, 0.9999999999)

        # Recover t from U'
        t = xp.power(H / xp.sqrt((1.0 + Up) / (1.0 - Up)), 1.0 / B)

        # Iterative conformal latitude inversion
        phi = _phi_from_t(t, e, xp)

        # Recover lambda
        lam = -xp.arctan2(Sp * cos_g0 - Vp * sin_g0, xp.cos(B * u / A_norm)) / B

        return lam, phi


register("omerc", ObliqueMercator())
