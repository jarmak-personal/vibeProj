"""Transverse Mercator projection — ported from cuProj / PROJ.

Implements the Poder/Engsager exact Transverse Mercator using 6th-order
series expansion. This is the projection used by UTM and many national grids.

Original C++ code: Copyright (c) 2023 NVIDIA CORPORATION (Apache 2.0)
Based on PROJ tmerc.cpp by Knud Poder and Karsten Engsager (MIT).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

ETMERC_ORDER = 6


def _gatg(p, B, cos_2B, sin_2B, xp):
    """Gaussian <-> Geographic trigonometric series (vectorized).

    p: list of scalar coefficients (length ETMERC_ORDER)
    B, cos_2B, sin_2B: arrays of same shape
    """
    two_cos_2B = 2.0 * cos_2B
    h2 = 0.0
    h1 = p[-1]
    for i in range(len(p) - 2, -1, -1):
        h = -h2 + two_cos_2B * h1 + p[i]
        h2 = h1
        h1 = h
    return B + h * sin_2B


def _clenshaw_complex(a, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, xp):
    """Complex Clenshaw summation (vectorized).

    a: list of scalar coefficients
    sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i: arrays
    Returns (R, I) tuple of arrays.
    """
    r = 2.0 * cos_arg_r * cosh_arg_i
    i = -2.0 * sin_arg_r * sinh_arg_i

    hr = a[-1]
    hi = 0.0
    hr1 = 0.0
    hi1 = 0.0
    for k in range(len(a) - 2, -1, -1):
        hr2, hi2 = hr1, hi1
        hr1, hi1 = hr, hi
        hr = -hr2 + r * hr1 - i * hi1 + a[k]
        hi = -hi2 + i * hr1 + r * hi1

    r = sin_arg_r * cosh_arg_i
    i = cos_arg_r * sinh_arg_i
    R = r * hr - i * hi
    Im = r * hi + i * hr
    return R, Im


def _clenshaw_real(a, arg_r, xp):
    """Real Clenshaw summation (vectorized).

    a: list of scalar coefficients
    arg_r: array
    """
    cos_arg_r = xp.cos(arg_r)
    r = 2.0 * cos_arg_r

    hr = a[-1]
    hr1 = 0.0
    for k in range(len(a) - 2, -1, -1):
        hr2 = hr1
        hr1 = hr
        hr = -hr2 + r * hr1 + a[k]
    return xp.sin(arg_r) * hr


class TransverseMercator(Projection):
    """Transverse Mercator using Poder/Engsager 6th-order series expansion."""

    name = "tmerc"

    def setup(self, params: ProjectionParams) -> dict:
        """Compute 6th-order series coefficients from ellipsoid parameters."""
        e = params.ellipsoid
        n = e.n

        # Determine central meridian and scale for UTM
        k0 = params.k_0
        lam0 = math.radians(params.lon_0)
        phi0 = math.radians(params.lat_0)
        x0 = params.x_0
        y0 = params.y_0

        if params.utm_zone > 0:
            zone = params.utm_zone
            lam0 = math.radians((zone - 0.5) * 6.0 - 180.0)
            k0 = 0.9996
            phi0 = 0.0
            x0 = 500000.0
            y0 = 10000000.0 if params.south else 0.0

        np_ = n  # running power of n

        # Coefficients: Gaussian -> Geodetic (cgb) and Geodetic -> Gaussian (cbg)
        cgb = [0.0] * ETMERC_ORDER
        cbg = [0.0] * ETMERC_ORDER

        cgb[0] = n * (
            2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0)))))
        )
        cbg[0] = n * (
            -2
            + n
            * (2 / 3.0 + n * (4 / 3.0 + n * (-82 / 45.0 + n * (32 / 45.0 + n * (4642 / 4725.0)))))
        )
        np_ *= n
        cgb[1] = np_ * (
            7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0))))
        )
        cbg[1] = np_ * (
            5 / 3.0 + n * (-16 / 15.0 + n * (-13 / 9.0 + n * (904 / 315.0 + n * (-1522 / 945.0))))
        )
        np_ *= n
        cgb[2] = np_ * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))))
        cbg[2] = np_ * (-26 / 15.0 + n * (34 / 21.0 + n * (8 / 5.0 + n * (-12686 / 2835.0))))
        np_ *= n
        cgb[3] = np_ * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0)))
        cbg[3] = np_ * (1237 / 630.0 + n * (-12 / 5.0 + n * (-24832 / 14175.0)))
        np_ *= n
        cgb[4] = np_ * (4174 / 315.0 + n * (-144838 / 6237.0))
        cbg[4] = np_ * (-734 / 315.0 + n * (109598 / 31185.0))
        np_ *= n
        cgb[5] = np_ * (601676 / 22275.0)
        cbg[5] = np_ * (444337 / 155925.0)

        # Normalized meridian quadrant
        np2 = n * n
        Qn = k0 / (1 + n) * (1 + np2 * (1 / 4.0 + np2 * (1 / 64.0 + np2 / 256.0)))

        # Coefficients: ellipsoidal N,E -> spherical N,E (utg) and reverse (gtu)
        utg = [0.0] * ETMERC_ORDER
        gtu = [0.0] * ETMERC_ORDER
        np_ = n

        utg[0] = n * (
            -0.5
            + n
            * (
                2 / 3.0
                + n * (-37 / 96.0 + n * (1 / 360.0 + n * (81 / 512.0 + n * (-96199 / 604800.0))))
            )
        )
        gtu[0] = n * (
            0.5
            + n
            * (
                -2 / 3.0
                + n * (5 / 16.0 + n * (41 / 180.0 + n * (-127 / 288.0 + n * (7891 / 37800.0))))
            )
        )

        np_ = n * n
        utg[1] = np_ * (
            -1 / 48.0
            + n * (-1 / 15.0 + n * (437 / 1440.0 + n * (-46 / 105.0 + n * (1118711 / 3870720.0))))
        )
        gtu[1] = np_ * (
            13 / 48.0
            + n * (-3 / 5.0 + n * (557 / 1440.0 + n * (281 / 630.0 + n * (-1983433 / 1935360.0))))
        )

        np_ *= n
        utg[2] = np_ * (-17 / 480.0 + n * (37 / 840.0 + n * (209 / 4480.0 + n * (-5569 / 90720.0))))
        gtu[2] = np_ * (
            61 / 240.0 + n * (-103 / 140.0 + n * (15061 / 26880.0 + n * (167603 / 181440.0)))
        )

        np_ *= n
        utg[3] = np_ * (-4397 / 161280.0 + n * (11 / 504.0 + n * (830251 / 7257600.0)))
        gtu[3] = np_ * (49561 / 161280.0 + n * (-179 / 168.0 + n * (6601661 / 7257600.0)))

        np_ *= n
        utg[4] = np_ * (-4583 / 161280.0 + n * (108847 / 3991680.0))
        gtu[4] = np_ * (34729 / 80640.0 + n * (-3418889 / 1995840.0))

        np_ *= n
        utg[5] = np_ * (-20648693 / 638668800.0)
        gtu[5] = np_ * (212378941 / 319334400.0)

        # Gaussian latitude of origin
        Z = _gatg_scalar(cbg, phi0)
        # Origin northing offset
        Zb = -Qn * (Z + _clenshaw_real_scalar(gtu, 2 * Z))

        return {
            "cgb": cgb,
            "cbg": cbg,
            "utg": utg,
            "gtu": gtu,
            "Qn": Qn,
            "Zb": Zb,
            "lam0": lam0,
            "k0": k0,
            "x0": x0,
            "y0": y0,
            "a": e.a,
        }

    def forward(self, lam, phi, params, computed, xp):
        """Geographic -> Transverse Mercator.

        lam: longitude relative to central meridian (radians)
        phi: latitude (radians)
        """
        cbg = computed["cbg"]
        gtu = computed["gtu"]
        Qn = computed["Qn"]
        Zb = computed["Zb"]

        # Geodetic lat -> Gaussian lat
        Cn = _gatg(cbg, phi, xp.cos(2 * phi), xp.sin(2 * phi), xp)

        # Gaussian lat/lon -> complex spherical lat
        sin_Cn = xp.sin(Cn)
        cos_Cn = xp.cos(Cn)
        sin_Ce = xp.sin(lam)
        cos_Ce = xp.cos(lam)

        cos_Cn_cos_Ce = cos_Cn * cos_Ce
        Cn = xp.arctan2(sin_Cn, cos_Cn_cos_Ce)

        inv_denom = 1.0 / xp.hypot(sin_Cn, cos_Cn_cos_Ce)
        tan_Ce = sin_Ce * cos_Cn * inv_denom

        # Complex spherical -> ellipsoidal normalized
        Ce = xp.arcsinh(tan_Ce)

        # Optimized trig/hyperbolic for Clenshaw
        two_inv = 2.0 * inv_denom
        two_inv_sq = two_inv * inv_denom
        tmp_r = cos_Cn_cos_Ce * two_inv_sq
        sin_arg_r = sin_Cn * tmp_r
        cos_arg_r = cos_Cn_cos_Ce * tmp_r - 1.0

        sinh_arg_i = tan_Ce * two_inv
        cosh_arg_i = two_inv_sq - 1.0

        dCn, dCe = _clenshaw_complex(gtu, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, xp)
        Cn = Cn + dCn
        Ce = Ce + dCe

        y = Qn * Cn + Zb  # northing
        x = Qn * Ce  # easting
        return x, y

    def inverse(self, x, y, params, computed, xp):
        """Transverse Mercator -> Geographic.

        x, y: projection-native coordinates (before offset/scale removal — the pipeline handles that)
        Actually: x, y are already scaled/offset-removed by the pipeline.
        """
        utg = computed["utg"]
        cgb = computed["cgb"]
        Qn = computed["Qn"]
        Zb = computed["Zb"]

        # Normalize
        Cn = (y - Zb) / Qn
        Ce = x / Qn

        # Normalized -> complex spherical
        sin_arg_r = xp.sin(2 * Cn)
        cos_arg_r = xp.cos(2 * Cn)

        exp_2_Ce = xp.exp(2 * Ce)
        half_inv = 0.5 / exp_2_Ce
        sinh_arg_i = 0.5 * exp_2_Ce - half_inv
        cosh_arg_i = 0.5 * exp_2_Ce + half_inv

        dCn, dCe = _clenshaw_complex(utg, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, xp)
        Cn = Cn + dCn
        Ce = Ce + dCe

        # Complex spherical -> Gaussian lat/lon
        sin_Cn = xp.sin(Cn)
        cos_Cn = xp.cos(Cn)

        sinhCe = xp.sinh(Ce)
        Ce = xp.arctan2(sinhCe, cos_Cn)
        modulus_Ce = xp.hypot(sinhCe, cos_Cn)
        Cn = xp.arctan2(sin_Cn, modulus_Ce)

        # Gaussian lat -> Geodetic lat
        tmp = 2.0 * modulus_Ce / (sinhCe * sinhCe + 1.0)
        sin_2_Cn = sin_Cn * tmp
        cos_2_Cn = tmp * modulus_Ce - 1.0

        phi = _gatg(cgb, Cn, cos_2_Cn, sin_2_Cn, xp)
        lam = Ce
        return lam, phi


def _gatg_scalar(p, B):
    """Scalar version of gatg for setup computations."""
    cos_2B = math.cos(2 * B)
    sin_2B = math.sin(2 * B)
    two_cos_2B = 2.0 * cos_2B
    h2 = 0.0
    h1 = p[-1]
    for i in range(len(p) - 2, -1, -1):
        h = -h2 + two_cos_2B * h1 + p[i]
        h2 = h1
        h1 = h
    return B + h * sin_2B


def _clenshaw_real_scalar(a, arg_r):
    """Scalar version of clenshaw_real for setup computations."""
    cos_arg_r = math.cos(arg_r)
    r = 2.0 * cos_arg_r
    hr = a[-1]
    hr1 = 0.0
    for k in range(len(a) - 2, -1, -1):
        hr2 = hr1
        hr1 = hr
        hr = -hr2 + r * hr1 + a[k]
    return math.sin(arg_r) * hr


_tmerc = TransverseMercator()
register("tmerc", _tmerc)
