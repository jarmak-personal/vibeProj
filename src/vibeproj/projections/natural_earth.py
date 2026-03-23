"""Natural Earth projection.

Compromise pseudocylindrical projection for physical world maps.
Polynomial formulas with closed-form forward and inverse.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

# Polynomial coefficients from Tom Patterson / Bojan Šavrič
_A0 = 0.8707
_A1 = -0.131979
_A2 = -0.013791
_A3 = 0.003971
_A4 = -0.001529
_B0 = 1.007226
_B1 = 0.015085
_B2 = -0.044475
_B3 = 0.028874
_B4 = -0.005916


class NaturalEarth(Projection):
    name = "natearth"

    def setup(self, params: ProjectionParams) -> dict:
        return {
            "a": params.ellipsoid.a,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        p2 = phi * phi
        p4 = p2 * p2
        x = lam * (_A0 + p2 * (_A1 + p2 * (_A2 + p4 * (_A3 + p2 * _A4))))
        y = phi * (_B0 + p2 * (_B1 + p4 * (_B2 + p2 * (_B3 + p2 * _B4))))
        return x, y

    def inverse(self, x, y, params, computed, xp):
        # Newton iteration to recover phi from y
        phi = y
        for _ in range(15):
            p2 = phi * phi
            p4 = p2 * p2
            fy = phi * (_B0 + p2 * (_B1 + p4 * (_B2 + p2 * (_B3 + p2 * _B4)))) - y
            fpy = _B0 + p2 * (3 * _B1 + p4 * (7 * _B2 + p2 * (9 * _B3 + 11 * p2 * _B4)))
            dphi = -fy / fpy
            phi = phi + dphi
            if xp is np:
                if hasattr(dphi, "__len__"):
                    if xp.all(xp.abs(dphi) < 1e-14):
                        break
                elif abs(float(dphi)) < 1e-14:
                    break
        p2 = phi * phi
        p4 = p2 * p2
        lam = x / (_A0 + p2 * (_A1 + p2 * (_A2 + p4 * (_A3 + p2 * _A4))))
        return lam, phi


register("natearth", NaturalEarth())
