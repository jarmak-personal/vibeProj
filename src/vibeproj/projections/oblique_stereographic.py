"""Oblique Stereographic projection (double projection method).

Conformal projection used for the Netherlands national grid (EPSG:28992).
Uses conformal sphere as intermediate: ellipsoid → conformal sphere → stereographic.

Math from PROJ sterea.c and "Map Projections" by Snyder.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import EPS_CONV, EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams


class ObliqueStereographic(Projection):
    """Conformal projection via double projection through a conformal sphere."""

    name = "sterea"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        ec = e.e
        es = e.es
        phi0 = math.radians(params.lat_0)

        sin_phi0 = math.sin(phi0)
        cos_phi0 = math.cos(phi0)

        # Conformal sphere parameters
        R = math.sqrt(1 - es) / (1 - es * sin_phi0 * sin_phi0)
        n = math.sqrt(1 + es * cos_phi0**4 / (1 - es))
        S1 = (1 + sin_phi0) / (1 - sin_phi0)
        S2_e = ((1 - ec * sin_phi0) / (1 + ec * sin_phi0)) ** ec
        w1 = (S1 * S2_e) ** n
        sin_chi0 = (w1 - 1) / (w1 + 1)
        cos_chi0 = math.sqrt(1 - sin_chi0 * sin_chi0)
        c = (n + sin_phi0) * (1 - sin_chi0) / ((n - sin_phi0) * (1 + sin_chi0))
        # c should be close to 1
        w2 = c * w1

        chi0 = math.asin((w2 - 1) / (w2 + 1))
        cos_chi0 = math.cos(chi0)
        sin_chi0 = math.sin(chi0)

        return {
            "a": e.a,
            "e": ec,
            "n": n,
            "c": c,
            "R": R,
            "sin_chi0": sin_chi0,
            "cos_chi0": cos_chi0,
            "k0": params.k_0,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        ec = computed["e"]
        n = computed["n"]
        c = computed["c"]
        R = computed["R"]
        sin_chi0 = computed["sin_chi0"]
        cos_chi0 = computed["cos_chi0"]
        k0 = computed["k0"]

        # Geodetic → conformal sphere
        sin_phi = xp.sin(phi)
        S = ((1 + sin_phi) / (1 - sin_phi)) * ((1 - ec * sin_phi) / (1 + ec * sin_phi)) ** ec
        w = c * S**n
        chi = xp.arcsin((w - 1) / (w + 1))
        lam_s = n * lam

        # Oblique stereographic on conformal sphere (radius R)
        sin_chi = xp.sin(chi)
        cos_chi = xp.cos(chi)
        cos_lam_s = xp.cos(lam_s)
        sin_lam_s = xp.sin(lam_s)

        k_denom = 1 + sin_chi0 * sin_chi + cos_chi0 * cos_chi * cos_lam_s
        x = 2 * R * k0 * cos_chi * sin_lam_s / k_denom
        y = 2 * R * k0 * (cos_chi0 * sin_chi - sin_chi0 * cos_chi * cos_lam_s) / k_denom
        return x, y

    def inverse(self, x, y, params, computed, xp):
        ec = computed["e"]
        n = computed["n"]
        c = computed["c"]
        R = computed["R"]
        sin_chi0 = computed["sin_chi0"]
        cos_chi0 = computed["cos_chi0"]
        k0 = computed["k0"]

        x_s = x / (2 * R * k0)
        y_s = y / (2 * R * k0)
        rho = xp.sqrt(x_s * x_s + y_s * y_s)

        ce = 2 * xp.arctan(rho)
        sin_ce = xp.sin(ce)
        cos_ce = xp.cos(ce)

        sin_chi = cos_ce * sin_chi0 + y_s * sin_ce * cos_chi0 / xp.maximum(rho, EPS_DENOM)
        sin_chi = xp.clip(sin_chi, -1.0, 1.0)

        lam_s = xp.arctan2(x_s * sin_ce, rho * cos_chi0 * cos_ce - y_s * sin_chi0 * sin_ce)
        lam = lam_s / n

        # Conformal sphere → geodetic (iterative)
        psi = 0.5 * (xp.log((1 + sin_chi) / xp.maximum(1 - sin_chi, EPS_DENOM)) - math.log(c)) / n
        phi = 2 * xp.arctan(xp.exp(psi)) - math.pi / 2
        for _ in range(15):
            sin_phi = xp.sin(phi)
            e_sin = ec * sin_phi
            psi_calc = xp.log(
                xp.tan(math.pi / 4 + phi / 2) * ((1 - e_sin) / (1 + e_sin)) ** (ec / 2)
            )
            dphi = (
                (psi - psi_calc) * xp.cos(phi) * (1 - ec * ec * sin_phi * sin_phi) / (1 - ec * ec)
            )
            phi = phi + dphi
            if hasattr(dphi, "__len__"):
                if xp.all(xp.abs(dphi) < EPS_CONV):
                    break
            elif abs(float(dphi)) < EPS_CONV:
                break

        return lam, phi


register("sterea", ObliqueStereographic())
