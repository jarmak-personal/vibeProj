"""Lambert Conformal Conic projection (1SP and 2SP).

Conformal conic projection used for aeronautical charts, US State Plane (E-W zones),
and national mapping (France EPSG:2154, Canada, etc.).

Math from PROJ lcc.c and Snyder, "Map Projections: A Working Manual" (USGS PP 1395).
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


def _tsfn(phi, sin_phi, e):
    """Compute Snyder's t-function: tan(pi/4 - phi/2) / ((1 - e*sin(phi))/(1 + e*sin(phi)))^(e/2)."""
    sin_phi *= e
    return math.tan(0.5 * (_HALF_PI - phi)) / ((1.0 - sin_phi) / (1.0 + sin_phi)) ** (0.5 * e)


def _tsfn_array(phi, sin_phi, e, xp):
    """Vectorized tsfn."""
    esp = e * sin_phi
    return xp.tan(0.5 * (_HALF_PI - phi)) / ((1.0 - esp) / (1.0 + esp)) ** (0.5 * e)


def _msfn(sin_phi, cos_phi, es):
    """Compute m-function: cos(phi) / sqrt(1 - es * sin^2(phi))."""
    return cos_phi / math.sqrt(1.0 - es * sin_phi * sin_phi)


def _phi2(ts, e, xp):
    """Inverse of tsfn: determine phi from ts (vectorized, iterative)."""
    half_e = 0.5 * e
    phi = _HALF_PI - 2.0 * xp.arctan(ts)
    for _ in range(15):
        dphi = (
            _HALF_PI
            - 2.0 * xp.arctan(ts * ((1.0 - e * xp.sin(phi)) / (1.0 + e * xp.sin(phi))) ** half_e)
            - phi
        )
        phi = phi + dphi
        if xp is np:
            if hasattr(dphi, "__len__"):
                if xp.all(xp.abs(dphi) < 1e-14):
                    break
            elif abs(float(dphi)) < 1e-14:
                break
    return phi


class LambertConformalConic(Projection):
    name = "lcc"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        es = e.es
        ec = e.e

        phi0 = math.radians(params.lat_0)
        phi1 = math.radians(params.lat_1) if params.lat_1 != 0 else phi0
        phi2 = math.radians(params.lat_2) if params.lat_2 != 0 else phi1

        sin_phi1 = math.sin(phi1)
        cos_phi1 = math.cos(phi1)
        m1 = _msfn(sin_phi1, cos_phi1, es)
        t1 = _tsfn(phi1, sin_phi1, ec)

        if abs(phi1 - phi2) < _EPS10:
            # 1SP case
            n = sin_phi1
        else:
            sin_phi2 = math.sin(phi2)
            cos_phi2 = math.cos(phi2)
            m2 = _msfn(sin_phi2, cos_phi2, es)
            t2 = _tsfn(phi2, sin_phi2, ec)
            n = math.log(m1 / m2) / math.log(t1 / t2)

        F = m1 / (n * t1**n)
        t0 = _tsfn(phi0, math.sin(phi0), ec)
        # rho0 normalized (without a) — pipeline multiplies by a
        rho0 = F * t0**n * params.k_0

        return {
            "n": n,
            "F": F,
            "rho0": rho0,
            "e": ec,
            "es": es,
            "a": e.a,
            "k0": params.k_0,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        n = computed["n"]
        F = computed["F"]
        rho0 = computed["rho0"]
        e = computed["e"]
        k0 = computed["k0"]

        sin_phi = xp.sin(phi)
        ts = _tsfn_array(phi, sin_phi, e, xp)
        rho = F * ts**n * k0

        theta = n * lam
        x = rho * xp.sin(theta)
        y = rho0 - rho * xp.cos(theta)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        n = computed["n"]
        F = computed["F"]
        rho0 = computed["rho0"]
        e = computed["e"]
        k0 = computed["k0"]

        rho = xp.sqrt(x * x + (rho0 - y) ** 2)
        if n < 0:
            rho = -rho

        theta = xp.arctan2(x, rho0 - y)
        if n < 0:
            theta = -theta

        lam = theta / n
        ts = (rho / (F * k0)) ** (1.0 / n)
        phi = _phi2(ts, e, xp)
        return lam, phi


register("lcc", LambertConformalConic())
