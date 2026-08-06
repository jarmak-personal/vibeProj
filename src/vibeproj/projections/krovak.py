"""Krovak projection.

Oblique conformal conic projection used for Czech/Slovak national grids.
The projection core is north oriented. CRS axis metadata exposes either regular
Krovak (EPSG:5513, X=Southing/Y=Westing) or Krovak North Orientated
(EPSG:5514, Easting/Northing).

Math follows PROJ krovak.cpp and EPSG Guidance Note 7-2, section 2.4.4.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj._conformal import conformal_to_geodetic_coefficients
from vibeproj.projections import register
from vibeproj.projections.base import EPS_CONV, EPS_DENOM, Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

_HALF_PI = math.pi / 2.0
_STANDARD_BESSEL_A = 6_377_397.155
_STANDARD_BESSEL_E = 0.08169683122252751
_STANDARD_DERIVED_SCALARS = (
    1.0005974983716484,
    1.0034191639671806,
    0.9799247046208296,
    0.20353742648834952,
    9.931008767325844,
    0.5043488898136432,
    0.8634999695099853,
    0.0034133319161083263,
)
_STANDARD_CGB = (
    0.003346491640862642,
    6.53254012663727e-06,
    1.7488220271061576e-08,
    5.3234553990234705e-11,
    1.737731799565109e-13,
    5.947923104361088e-16,
)


def _strategy_semantics(
    params: ProjectionParams,
    *,
    a: float,
    e: float,
    B: float,
    k: float,
    n: float,
    r_0_norm: float,
    tan_half_p: float,
    sin_alpha_c: float,
    cos_alpha_c: float,
    cgb: tuple[float, ...],
    log_k: float,
    lam0: float,
    x0: float,
    y0: float,
) -> str:
    """Classify the Krovak policy domain once from immutable setup scalars."""
    if params.ellipsoid.es == 0.0:
        return "spherical"

    if params.operation_method == "Krovak":
        orientation = "regular"
        expected_easting_sign = expected_northing_sign = -1.0
    elif params.operation_method == "Krovak (North Orientated)":
        orientation = "north_oriented"
        expected_easting_sign = expected_northing_sign = 1.0
    else:
        return "custom"

    scalars = (
        a,
        e,
        B,
        k,
        n,
        r_0_norm,
        tan_half_p,
        sin_alpha_c,
        cos_alpha_c,
        log_k,
        lam0,
        x0,
        y0,
        params.x_unit_to_m,
        params.y_unit_to_m,
        params.easting_axis_sign,
        params.northing_axis_sign,
        *cgb,
    )
    if (
        len(cgb) != 6
        or not all(math.isfinite(value) for value in scalars)
        or params.x_unit_to_m == 0.0
        or params.y_unit_to_m == 0.0
        or B <= 0.0
        or k <= 0.0
        or n <= 0.0
        or r_0_norm <= 0.0
        or tan_half_p <= 0.0
        or params.easting_axis_sign != expected_easting_sign
        or params.northing_axis_sign != expected_northing_sign
    ):
        return "invalid_setup"

    derived_scalars = (
        B,
        k,
        n,
        r_0_norm,
        tan_half_p,
        sin_alpha_c,
        cos_alpha_c,
        log_k,
    )
    standard = (
        math.isclose(a, _STANDARD_BESSEL_A, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(e, _STANDARD_BESSEL_E, rel_tol=0.0, abs_tol=2e-16)
        and all(
            math.isclose(value, expected, rel_tol=2e-15, abs_tol=2e-15)
            for value, expected in zip(derived_scalars, _STANDARD_DERIVED_SCALARS, strict=True)
        )
        and all(
            math.isclose(value, expected, rel_tol=2e-15, abs_tol=2e-15)
            for value, expected in zip(cgb, _STANDARD_CGB, strict=True)
        )
    )
    return f"standard_bessel.{orientation}" if standard else "custom"


class Krovak(Projection):
    """Oblique conformal conic projection for Czech/Slovak national grids."""

    name = "krovak"

    def setup(self, params: ProjectionParams) -> dict:
        ell = params.ellipsoid
        a = ell.a
        e = ell.e
        es = ell.es

        phi_c = math.radians(params.lat_0)  # latitude of projection centre
        lam_0 = math.radians(params.lon_0)  # longitude of origin
        alpha_c = math.radians(params.extra.get("alpha_c", 30.28813975277778))
        phi_p = math.radians(params.extra.get("phi_p", 78.5))
        k_p = params.k_0

        sin_phi_c = math.sin(phi_c)
        cos_phi_c = math.cos(phi_c)
        one_es = 1.0 - es

        # Gaussian conformal sphere parameters (PROJ formulation)
        B = math.sqrt(1.0 + es * cos_phi_c**4 / one_es)
        u0 = math.asin(sin_phi_c / B)

        # Conformal mapping constant k (PROJ formula)
        # g = ((1+e·sinφ_c)/(1-e·sinφ_c))^(B·e/2)
        # k = tan(π/4+u0/2) · tan(π/4+φ_c/2)^(-B) · g
        e_sin_c = e * sin_phi_c
        g = ((1.0 + e_sin_c) / (1.0 - e_sin_c)) ** (B * e / 2.0)
        k = math.tan(math.pi / 4.0 + u0 / 2.0) * math.tan(math.pi / 4.0 + phi_c / 2.0) ** (-B) * g

        # Cone constant and radius at pseudo standard parallel
        n = math.sin(phi_p)
        n0 = math.sqrt(one_es) / (1.0 - es * sin_phi_c**2)  # A/a
        r_0_norm = k_p * n0 / math.tan(phi_p)  # already normalised by a

        # Precompute tan(pi/4 + phi_p/2) for cone formula
        tan_half_p = math.tan(math.pi / 4.0 + phi_p / 2.0)

        sin_alpha_c = math.sin(alpha_c)
        cos_alpha_c = math.cos(alpha_c)
        cgb = conformal_to_geodetic_coefficients(ell.n)
        log_k = math.log(k)
        x0 = -params.x_0
        y0 = -params.y_0
        strategy_semantics = _strategy_semantics(
            params,
            a=a,
            e=e,
            B=B,
            k=k,
            n=n,
            r_0_norm=r_0_norm,
            tan_half_p=tan_half_p,
            sin_alpha_c=sin_alpha_c,
            cos_alpha_c=cos_alpha_c,
            cgb=cgb,
            log_k=log_k,
            lam0=lam_0,
            x0=x0,
            y0=y0,
        )

        return {
            "a": a,
            "e": e,
            "B": B,
            "k": k,
            "n": n,
            "r_0_norm": r_0_norm,
            "tan_half_p": tan_half_p,
            "sin_alpha_c": sin_alpha_c,
            "cos_alpha_c": cos_alpha_c,
            # Immutable setup-time scalars for the guarded fp64 inverse.
            # Native dispatch ignores them and keeps its historical ABI.
            "cgb": cgb,
            "log_k": log_k,
            "_strategy_krovak_semantics": strategy_semantics,
            "lam0": lam_0,
            # Krovak's false-origin convention subtracts these values from
            # the north-oriented core coordinates (opposite the generic EPSG
            # false-easting/northing convention used by other projections).
            # Keep the negation in shared setup so generic forward and inverse
            # apply one symmetric offset contract. This matches PROJ forward
            # coordinates for custom nonzero offsets while deliberately
            # avoiding the asymmetric inverse exposed by legacy PROJ pipelines.
            "x0": x0,
            "y0": y0,
        }

    def forward(self, lam, phi, params, computed, xp):
        finite_input = xp.isfinite(lam) & xp.isfinite(phi)
        nan_input = xp.isnan(lam) | xp.isnan(phi)
        lam = xp.where(finite_input, lam, 0.0)
        phi = xp.where(finite_input, phi, 0.0)
        e = computed["e"]
        B = computed["B"]
        k = computed["k"]
        n = computed["n"]
        r_0_norm = computed["r_0_norm"]
        tan_half_p = computed["tan_half_p"]
        sin_ac = computed["sin_alpha_c"]
        cos_ac = computed["cos_alpha_c"]

        # Step 1: Gaussian conformal sphere (PROJ formulation)
        # gfi = ((1+e·sinφ)/(1-e·sinφ))^(B·e/2)
        # Q = k · tan(π/4+φ/2)^B / gfi
        sin_phi = xp.sin(phi)
        e_sin = e * sin_phi
        gfi = ((1.0 + e_sin) / (1.0 - e_sin)) ** (B * e / 2.0)
        Q = k * xp.tan(math.pi / 4.0 + phi / 2.0) ** B / gfi
        U = 2.0 * xp.arctan(Q) - _HALF_PI

        # V = B * (lam_0 - lam); pipeline gives lam = lambda - lam_0, so V = -B * lam
        V = -B * lam

        # Step 2: Oblique coordinates on Gaussian sphere
        cos_U = xp.cos(U)
        sin_U = xp.sin(U)
        cos_V = xp.cos(V)
        sin_V = xp.sin(V)
        T = xp.arcsin(cos_ac * sin_U + sin_ac * cos_U * cos_V)
        D = xp.arcsin(cos_U * sin_V / xp.cos(T))

        # Step 3: Oblique cone
        theta = n * D
        r_norm = r_0_norm * xp.power(tan_half_p / xp.tan(math.pi / 4.0 + T / 2.0), n)

        # Step 4: Grid coordinates (North Orientated → negate)
        x = -r_norm * xp.sin(theta)
        y = -r_norm * xp.cos(theta)
        invalid_x = xp.where(nan_input, xp.nan, xp.inf * params.easting_axis_sign)
        invalid_y = xp.where(nan_input, xp.nan, xp.inf * params.northing_axis_sign)
        x = xp.where(finite_input, x, invalid_x)
        y = xp.where(finite_input, y, invalid_y)
        return x, y

    def inverse(self, x, y, params, computed, xp):
        finite_input = xp.isfinite(x) & xp.isfinite(y)
        nan_input = xp.isnan(x) | xp.isnan(y)
        x = xp.where(finite_input, x, 0.0)
        y = xp.where(finite_input, y, 0.0)
        e = computed["e"]
        B = computed["B"]
        k = computed["k"]
        n = computed["n"]
        r_0_norm = computed["r_0_norm"]
        tan_half_p = computed["tan_half_p"]
        sin_ac = computed["sin_alpha_c"]
        cos_ac = computed["cos_alpha_c"]

        # Step 1: recover r and theta from grid (undo negation)
        r_norm = xp.sqrt(x * x + y * y)
        theta = xp.arctan2(-x, -y)

        # Step 2: inverse cone → T, D
        D = theta / n
        T = (
            2.0
            * xp.arctan(xp.power(r_0_norm / xp.maximum(r_norm, EPS_DENOM), 1.0 / n) * tan_half_p)
            - _HALF_PI
        )

        # Step 3: inverse oblique → U, V
        cos_T = xp.cos(T)
        sin_T = xp.sin(T)
        cos_D = xp.cos(D)
        sin_D = xp.sin(D)
        U = xp.arcsin(cos_ac * sin_T - sin_ac * cos_T * cos_D)
        V = xp.arcsin(cos_T * sin_D / xp.cos(U))

        # Step 4: inverse Gaussian sphere → phi
        # t = [tan(π/4+U/2)/k]^(1/B)
        t = xp.power(xp.tan(math.pi / 4.0 + U / 2.0) / k, 1.0 / B)

        # Iterative conformal latitude inversion (PROJ formula)
        # φ = 2·atan(t · ((1+e·sinφ)/(1-e·sinφ))^(e/2)) - π/2
        phi = 2.0 * xp.arctan(t) - _HALF_PI
        for _ in range(15):
            sin_phi = xp.sin(phi)
            e_sin = e * sin_phi
            phi_new = 2.0 * xp.arctan(t * ((1.0 + e_sin) / (1.0 - e_sin)) ** (e / 2.0)) - _HALF_PI
            dphi = phi_new - phi
            phi = phi_new
            if hasattr(dphi, "__len__"):
                if xp.all(xp.abs(dphi) < EPS_CONV):
                    break
            elif abs(float(dphi)) < EPS_CONV:
                break

        # Step 5: recover lambda
        lam = -V / B

        invalid = xp.where(nan_input, xp.nan, xp.inf)
        lam = xp.where(finite_input, lam, invalid)
        phi = xp.where(finite_input, phi, invalid)

        return lam, phi


register("krovak", Krovak())
