"""Helmert 7-parameter datum transformation.

Transforms geodetic coordinates between two ellipsoids via geocentric (ECEF)
intermediate using the Position Vector convention (EPSG method 9606).

Pipeline: geodetic(src) -> ECEF -> Helmert rotate/translate/scale -> ECEF -> geodetic(dst)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeproj.ellipsoid import Ellipsoid

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True, slots=True)
class HelmertParams:
    """Helmert 7-parameter datum transformation (Position Vector convention).

    Parameters are always stored in Position Vector convention.
    Coordinate Frame convention rotations are negated at extraction time.
    """

    tx: float  # X-axis translation (meters)
    ty: float  # Y-axis translation (meters)
    tz: float  # Z-axis translation (meters)
    rx: float  # X-axis rotation (radians)
    ry: float  # Y-axis rotation (radians)
    rz: float  # Z-axis rotation (radians)
    ds: float  # scale factor (1.0 + ppm * 1e-6)
    src_ellipsoid: Ellipsoid
    dst_ellipsoid: Ellipsoid

    def inverted(self) -> HelmertParams:
        """Return the inverse transformation (swap src/dst, negate params)."""
        return HelmertParams(
            tx=-self.tx,
            ty=-self.ty,
            tz=-self.tz,
            rx=-self.rx,
            ry=-self.ry,
            rz=-self.rz,
            ds=1.0 / self.ds if self.ds != 0.0 else 1.0,
            src_ellipsoid=self.dst_ellipsoid,
            dst_ellipsoid=self.src_ellipsoid,
        )


def geodetic_to_ecef(lat_rad, lon_rad, a, es, xp):
    """Convert geodetic (lat, lon in radians) to ECEF (X, Y, Z) in meters.

    Parameters
    ----------
    lat_rad, lon_rad : array_like
        Geodetic coordinates in radians.
    a : float
        Semi-major axis.
    es : float
        First eccentricity squared.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    X, Y, Z : arrays
        ECEF coordinates in meters.
    """
    sin_lat = xp.sin(lat_rad)
    cos_lat = xp.cos(lat_rad)
    sin_lon = xp.sin(lon_rad)
    cos_lon = xp.cos(lon_rad)

    N = a / xp.sqrt(1.0 - es * sin_lat * sin_lat)
    X = N * cos_lat * cos_lon
    Y = N * cos_lat * sin_lon
    Z = N * (1.0 - es) * sin_lat
    return X, Y, Z


def ecef_to_geodetic(X, Y, Z, a, es, xp):
    """Convert ECEF (X, Y, Z) to geodetic (lat, lon in radians).

    Uses iterative Bowring method (converges in ~3 iterations for sub-mm).

    Parameters
    ----------
    X, Y, Z : array_like
        ECEF coordinates in meters.
    a : float
        Semi-major axis.
    es : float
        First eccentricity squared.
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    lat_rad, lon_rad : arrays
        Geodetic coordinates in radians.
    """
    p = xp.sqrt(X * X + Y * Y)
    lon = xp.arctan2(Y, X)

    # Initial latitude estimate
    lat = xp.arctan2(Z, p * (1.0 - es))

    for _ in range(10):
        sin_lat = xp.sin(lat)
        N = a / xp.sqrt(1.0 - es * sin_lat * sin_lat)
        lat = xp.arctan2(Z + es * N * sin_lat, p)

    return lat, lon


def apply_helmert(lat_deg, lon_deg, params: HelmertParams, xp):
    """Apply Helmert 7-parameter datum shift.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Geodetic coordinates in degrees on the source ellipsoid.
    params : HelmertParams
        Transformation parameters (Position Vector convention).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    lat_deg_out, lon_deg_out : arrays
        Geodetic coordinates in degrees on the destination ellipsoid.
    """
    lat_rad = lat_deg * DEG_TO_RAD
    lon_rad = lon_deg * DEG_TO_RAD

    src = params.src_ellipsoid
    dst = params.dst_ellipsoid

    # Geodetic -> ECEF on source ellipsoid
    X, Y, Z = geodetic_to_ecef(lat_rad, lon_rad, src.a, src.es, xp)

    # Helmert: X' = ds * R * X + T  (Position Vector convention)
    tx, ty, tz = params.tx, params.ty, params.tz
    rx, ry, rz = params.rx, params.ry, params.rz
    ds = params.ds

    X2 = ds * (X - rz * Y + ry * Z) + tx
    Y2 = ds * (rz * X + Y - rx * Z) + ty
    Z2 = ds * (-ry * X + rx * Y + Z) + tz

    # ECEF -> geodetic on destination ellipsoid
    lat_out, lon_out = ecef_to_geodetic(X2, Y2, Z2, dst.a, dst.es, xp)

    return lat_out * RAD_TO_DEG, lon_out * RAD_TO_DEG
