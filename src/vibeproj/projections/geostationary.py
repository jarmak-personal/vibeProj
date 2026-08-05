"""Geostationary Satellite View projection.

Perspective projection from geostationary orbit. Critical for GOES, Meteosat,
and Himawari satellite imagery. Both sweep-axis conventions are supported.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import register
from vibeproj.projections.base import Projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

GEOS_LIMB_TOLERANCE = 0.0
GEOS_DISCRIMINANT_TOLERANCE = 0.0
GEOS_SCAN_ANGLE_LIMIT = math.pi / 2.0


class Geostationary(Projection):
    """Perspective projection from geostationary orbit."""

    name = "geos"

    def setup(self, params: ProjectionParams) -> dict:
        e = params.ellipsoid
        h = float(params.extra.get("h", 35_785_831.0))
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError(f"Geostationary satellite height must be finite and positive: {h!r}")
        sweep_axis = params.extra.get("sweep_axis")
        if sweep_axis is None:
            sweep_axis = (
                "x" if params.operation_method == "Geostationary Satellite (Sweep X)" else "y"
            )
        if sweep_axis not in ("x", "y"):
            raise ValueError(f"Geostationary sweep axis must be 'x' or 'y', got {sweep_axis!r}")
        # Radius ratio
        r_eq = e.a
        r_pol = e.b
        if not all(math.isfinite(value) and value > 0.0 for value in (r_eq, r_pol)):
            raise ValueError("Geostationary ellipsoid radii must be finite and positive")
        if r_pol > r_eq:
            raise ValueError("Geostationary polar radius cannot exceed its equatorial radius")
        if h / r_eq > 1.0e10:
            raise ValueError("Geostationary satellite height must not exceed 1e10 radii")
        H = h + r_eq  # distance from Earth center to satellite
        if not all(math.isfinite(value) for value in (H, r_eq * r_eq, r_pol * r_pol, H * H)):
            raise ValueError("Geostationary height and ellipsoid radii exceed fp64 range")
        inv_r_eq2 = 1.0 / (r_eq * r_eq)
        inv_r_pol2 = 1.0 / (r_pol * r_pol)
        return {
            "a": e.a,
            "h": h,
            "H": H,
            "r_eq2": r_eq * r_eq,
            "r_pol2": r_pol * r_pol,
            "inv_r_eq2": inv_r_eq2,
            "inv_r_pol2": inv_r_pol2,
            "sweep_axis": sweep_axis,
            "lam0": math.radians(params.lon_0),
            "x0": params.x_0,
            "y0": params.y_0,
        }

    def forward(self, lam, phi, params, computed, xp):
        H = computed["H"]
        h = computed["h"]
        r_eq2 = computed["r_eq2"]
        r_pol2 = computed["r_pol2"]
        a = computed["a"]
        sweep_axis = computed["sweep_axis"]
        numpy_valid = getattr(xp, "__name__", None) == "numpy" and bool(
            xp.all(xp.isfinite(lam)) and xp.all(xp.abs(phi) <= 0.5 * math.pi)
        )
        finite_lam = lam if numpy_valid else xp.where(xp.isfinite(lam), lam, 0.0)
        finite_phi = phi if numpy_valid else xp.where(xp.isfinite(phi), phi, 0.0)

        # Geographic to geocentric latitude
        phi_gc = xp.arctan(r_pol2 / r_eq2 * xp.tan(finite_phi))
        cos_phi_gc = xp.cos(phi_gc)
        sin_phi_gc = xp.sin(phi_gc)

        # Geocentric earth radius (CGMS standard)
        r_pol = math.sqrt(r_pol2)
        r_earth = r_pol / xp.sqrt(1.0 - (r_eq2 - r_pol2) / r_eq2 * cos_phi_gc * cos_phi_gc)

        sin_lam = xp.sin(finite_lam)
        cos_lam = xp.cos(finite_lam)

        point_x = r_earth * cos_phi_gc * cos_lam
        Sx = H - point_x
        Sy = -r_earth * cos_phi_gc * sin_lam
        Sz = r_earth * sin_phi_gc
        sn = xp.sqrt(Sx * Sx + Sy * Sy + Sz * Sz)

        if sweep_axis == "x":
            x = xp.arcsin(xp.clip(-Sy / sn, -1.0, 1.0)) * (h / a)
            y = xp.arctan2(Sz, Sx) * (h / a)
        else:
            x = xp.arctan2(-Sy, Sx) * (h / a)
            y = xp.arcsin(xp.clip(Sz / sn, -1.0, 1.0)) * (h / a)

        # Ellipsoid identity reduces the full CGMS visibility predicate to
        # H * point_x - a², avoiding four temporary 1M-element arrays.
        visibility = H * point_x - r_eq2
        visible = visibility >= -GEOS_LIMB_TOLERANCE
        if numpy_valid and bool(xp.all(visible)):
            return x, y
        latitude_outside = xp.isfinite(phi) & (xp.abs(phi) > GEOS_SCAN_ANGLE_LIMIT)
        has_nan = xp.isnan(lam) | xp.isnan(phi)
        infinite = xp.isinf(lam) | xp.isinf(phi)
        x = xp.where(
            has_nan,
            xp.nan,
            xp.where(infinite | latitude_outside | ~visible, xp.inf, x),
        )
        y = xp.where(
            has_nan,
            xp.nan,
            xp.where(infinite | latitude_outside | ~visible, xp.inf, y),
        )
        return x, y

    def inverse(self, x, y, params, computed, xp):
        H = computed["H"]
        h = computed["h"]
        r_eq2 = computed["r_eq2"]
        r_pol2 = computed["r_pol2"]
        a = computed["a"]
        sweep_axis = computed["sweep_axis"]

        # Recover scanning angles (pipeline passes x_norm = x_physical / a)
        x_angle = x * a / h
        y_angle = y * a / h
        numpy_valid = getattr(xp, "__name__", None) == "numpy" and bool(
            xp.all(xp.isfinite(x)) and xp.all(xp.isfinite(y))
        )
        finite_x_angle = x_angle if numpy_valid else xp.where(xp.isfinite(x_angle), x_angle, 0.0)
        finite_y_angle = y_angle if numpy_valid else xp.where(xp.isfinite(y_angle), y_angle, 0.0)

        sin_x = xp.sin(finite_x_angle)
        cos_x = xp.cos(finite_x_angle)
        sin_y = xp.sin(finite_y_angle)
        cos_y = xp.cos(finite_y_angle)

        if sweep_axis == "x":
            a_coeff = (
                cos_x * cos_x * (cos_y * cos_y + sin_y * sin_y * r_eq2 / r_pol2) + sin_x * sin_x
            )
            b_coeff = -2 * H * cos_x * cos_y
        else:
            a_coeff = cos_y * cos_y + sin_y * sin_y * r_eq2 / r_pol2
            b_coeff = -2 * H * cos_y * cos_x
        c_coeff = H * H - a * a

        discrim = b_coeff * b_coeff - 4 * a_coeff * c_coeff
        r_s = (-b_coeff - xp.sqrt(xp.maximum(discrim, 0.0))) / (2 * a_coeff)

        if sweep_axis == "x":
            P_x = H - r_s * cos_x * cos_y
            P_y = r_s * sin_x
            P_z = r_s * cos_x * sin_y
        else:
            P_x = H - r_s * cos_y * cos_x
            P_y = r_s * cos_y * sin_x
            P_z = r_s * sin_y

        lam = xp.arctan2(P_y, P_x)
        # Geocentric → geodetic latitude (CGMS standard: r_eq²/r_pol² factor)
        phi = xp.arctan(P_z * r_eq2 / (xp.sqrt(P_x**2 + P_y**2) * r_pol2))

        if (
            numpy_valid
            and bool(xp.all(xp.abs(x_angle) < GEOS_SCAN_ANGLE_LIMIT))
            and bool(xp.all(xp.abs(y_angle) < GEOS_SCAN_ANGLE_LIMIT))
            and bool(xp.all(discrim >= -GEOS_DISCRIMINANT_TOLERANCE))
        ):
            return lam, phi
        outside_scan_domain = (xp.abs(x_angle) >= GEOS_SCAN_ANGLE_LIMIT) | (
            xp.abs(y_angle) >= GEOS_SCAN_ANGLE_LIMIT
        )
        on_earth = discrim >= -GEOS_DISCRIMINANT_TOLERANCE
        has_nan = xp.isnan(x) | xp.isnan(y)
        infinite = xp.isinf(x) | xp.isinf(y)
        lam = xp.where(
            has_nan,
            xp.nan,
            xp.where(infinite | outside_scan_domain | ~on_earth, xp.inf, lam),
        )
        phi = xp.where(
            has_nan,
            xp.nan,
            xp.where(infinite | outside_scan_domain | ~on_earth, xp.inf, phi),
        )

        return lam, phi


register("geos", Geostationary())
