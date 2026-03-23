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

    # Time-dependent (15-parameter) rates — default 0 preserves 7-param behavior
    dtx: float = 0.0  # X translation rate (m/yr)
    dty: float = 0.0  # Y translation rate (m/yr)
    dtz: float = 0.0  # Z translation rate (m/yr)
    drx: float = 0.0  # X rotation rate (rad/yr)
    dry: float = 0.0  # Y rotation rate (rad/yr)
    drz: float = 0.0  # Z rotation rate (rad/yr)
    dds: float = 0.0  # scale rate (ppm/yr * 1e-6)
    t_epoch: float = 0.0  # reference epoch (decimal year)

    @property
    def has_rates(self) -> bool:
        """True if time-dependent rate parameters are present."""
        return (
            self.dtx != 0.0
            or self.dty != 0.0
            or self.dtz != 0.0
            or self.drx != 0.0
            or self.dry != 0.0
            or self.drz != 0.0
            or self.dds != 0.0
        )

    def at_epoch(self, epoch: float) -> HelmertParams:
        """Compute effective 7-parameter transformation at a given epoch.

        Folds the rate terms into the base parameters:
            param_eff = param + rate * (epoch - t_epoch)

        Returns a standard (non-time-dependent) HelmertParams with rates zeroed.
        """
        dt = epoch - self.t_epoch
        return HelmertParams(
            tx=self.tx + self.dtx * dt,
            ty=self.ty + self.dty * dt,
            tz=self.tz + self.dtz * dt,
            rx=self.rx + self.drx * dt,
            ry=self.ry + self.dry * dt,
            rz=self.rz + self.drz * dt,
            ds=self.ds + self.dds * dt,
            src_ellipsoid=self.src_ellipsoid,
            dst_ellipsoid=self.dst_ellipsoid,
        )

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
            dtx=-self.dtx,
            dty=-self.dty,
            dtz=-self.dtz,
            drx=-self.drx,
            dry=-self.dry,
            drz=-self.drz,
            dds=-self.dds,
            t_epoch=self.t_epoch,
        )


def geodetic_to_ecef(lat_rad, lon_rad, a, es, xp, h=None):
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
    h : array_like, optional
        Ellipsoidal height in meters. When None, height=0 (exact current behavior).

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
    if h is not None:
        X = (N + h) * cos_lat * cos_lon
        Y = (N + h) * cos_lat * sin_lon
        Z = (N * (1.0 - es) + h) * sin_lat
    else:
        X = N * cos_lat * cos_lon
        Y = N * cos_lat * sin_lon
        Z = N * (1.0 - es) * sin_lat
    return X, Y, Z


def ecef_to_geodetic(X, Y, Z, a, es, xp, return_height=False):
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
    return_height : bool, optional
        When True, recover and return ellipsoidal height. Default False.

    Returns
    -------
    lat_rad, lon_rad : arrays
        Geodetic coordinates in radians.
    h : arrays (only when return_height=True)
        Ellipsoidal height in meters.
    """
    p = xp.sqrt(X * X + Y * Y)
    lon = xp.arctan2(Y, X)

    # Initial latitude estimate
    lat = xp.arctan2(Z, p * (1.0 - es))

    for _ in range(10):
        sin_lat = xp.sin(lat)
        N = a / xp.sqrt(1.0 - es * sin_lat * sin_lat)
        lat = xp.arctan2(Z + es * N * sin_lat, p)

    if return_height:
        sin_lat = xp.sin(lat)
        cos_lat = xp.cos(lat)
        N = a / xp.sqrt(1.0 - es * sin_lat * sin_lat)
        # Normal case: h = p / cos(lat) - N
        h = p / cos_lat - N
        # Near-pole guard: |cos(lat)| < 1e-10 → use Z-based formula
        # Always compute both and select via where() — avoids an implicit
        # D→H sync that xp.any() would cause on CuPy arrays.
        near_pole = xp.abs(cos_lat) < 1e-10
        h_pole = xp.abs(Z) / xp.abs(sin_lat) - N * (1.0 - es)
        h = xp.where(near_pole, h_pole, h)
        return lat, lon, h

    return lat, lon


def apply_helmert(lat_deg, lon_deg, params: HelmertParams, xp, h=None):
    """Apply Helmert 7-parameter datum shift.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Geodetic coordinates in degrees on the source ellipsoid.
    params : HelmertParams
        Transformation parameters (Position Vector convention).
    xp : module
        Array module (numpy or cupy).
    h : array_like, optional
        Ellipsoidal height in meters. When provided, height is transformed
        through the ECEF intermediate and recovered on the destination ellipsoid.

    Returns
    -------
    lat_deg_out, lon_deg_out : arrays
        Geodetic coordinates in degrees on the destination ellipsoid.
    h_out : arrays (only when h is not None)
        Transformed ellipsoidal height in meters.
    """
    lat_rad = lat_deg * DEG_TO_RAD
    lon_rad = lon_deg * DEG_TO_RAD

    src = params.src_ellipsoid
    dst = params.dst_ellipsoid

    # Geodetic -> ECEF on source ellipsoid
    X, Y, Z = geodetic_to_ecef(lat_rad, lon_rad, src.a, src.es, xp, h=h)

    # Helmert: X' = ds * R * X + T  (Position Vector convention)
    tx, ty, tz = params.tx, params.ty, params.tz
    rx, ry, rz = params.rx, params.ry, params.rz
    ds = params.ds

    X2 = ds * (X - rz * Y + ry * Z) + tx
    Y2 = ds * (rz * X + Y - rx * Z) + ty
    Z2 = ds * (-ry * X + rx * Y + Z) + tz

    # ECEF -> geodetic on destination ellipsoid
    if h is not None:
        lat_out, lon_out, h_out = ecef_to_geodetic(
            X2, Y2, Z2, dst.a, dst.es, xp, return_height=True
        )
        return lat_out * RAD_TO_DEG, lon_out * RAD_TO_DEG, h_out
    else:
        lat_out, lon_out = ecef_to_geodetic(X2, Y2, Z2, dst.a, dst.es, xp)
        return lat_out * RAD_TO_DEG, lon_out * RAD_TO_DEG
