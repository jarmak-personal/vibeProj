"""Transform pipeline — chains common pre/post operations with the projection core.

The pipeline handles:
1. Axis swap (lat/lon -> lon/lat)
2. Degree/radian conversion
3. Central meridian subtraction
4. Core projection (forward/inverse)
5. Scale by semi-major axis
6. False easting/northing

This matches the cuProj operation pipeline architecture but runs on NumPy/CuPy arrays.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vibeproj.projections import get_projection

if TYPE_CHECKING:
    from vibeproj.crs import ProjectionParams

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Lazy CuPy reference for fused kernel fast-path
_cupy_module = None


def _get_cupy():
    global _cupy_module
    if _cupy_module is None:
        try:
            import cupy

            _cupy_module = cupy
        except ImportError:
            _cupy_module = False
    return _cupy_module if _cupy_module is not False else None


def _try_fused(
    arg1,
    arg2,
    xp,
    *,
    projection_name,
    direction,
    computed,
    src_north_first,
    dst_north_first,
    out_x=None,
    out_y=None,
    precision="auto",
):
    """Attempt fused kernel execution. Returns None if not available."""
    cp = _get_cupy()
    if cp is None or xp is not cp:
        return None
    try:
        from vibeproj.fused_kernels import can_fuse, fused_transform
    except ImportError:
        return None
    if not can_fuse(projection_name, direction):
        return None
    return fused_transform(
        arg1,
        arg2,
        projection_name=projection_name,
        direction=direction,
        computed=computed,
        src_north_first=src_north_first,
        dst_north_first=dst_north_first,
        xp=xp,
        out_x=out_x,
        out_y=out_y,
        precision=precision,
    )


def _wrap_to_pi(angle, xp):
    """Wrap angle to [-pi, pi]."""
    return angle - 2.0 * math.pi * xp.round(angle / (2.0 * math.pi))


class TransformPipeline:
    """Executes a coordinate transformation between two CRS.

    Handles the full pipeline: CRS resolution, parameter setup,
    pre-processing (unit conversion, axis swap), projection math,
    and post-processing (scale, offset).
    """

    def __init__(self, src_params: ProjectionParams, dst_params: ProjectionParams):
        self.src = src_params
        self.dst = dst_params

        # Axis order flags for input/output swap
        self.src_north_first = src_params.north_first
        self.dst_north_first = dst_params.north_first

        # Determine which direction we're going
        if src_params.projection_name == "longlat" and dst_params.projection_name != "longlat":
            # Geographic -> Projected (forward)
            self.mode = "forward"
            self.proj_params = dst_params
            self.projection = get_projection(dst_params.projection_name)
        elif src_params.projection_name != "longlat" and dst_params.projection_name == "longlat":
            # Projected -> Geographic (inverse)
            self.mode = "inverse"
            self.proj_params = src_params
            self.projection = get_projection(src_params.projection_name)
        elif src_params.projection_name != "longlat" and dst_params.projection_name != "longlat":
            # Projected -> Projected: inverse src, then forward dst
            self.mode = "proj_to_proj"
            self.src_projection = get_projection(src_params.projection_name)
            self.dst_projection = get_projection(dst_params.projection_name)
            self.src_computed = self.src_projection.setup(src_params)
            self.dst_computed = self.dst_projection.setup(dst_params)
        else:
            # Geographic -> Geographic (possibly different datums)
            self.mode = "longlat_to_longlat"

        if self.mode in ("forward", "inverse"):
            self.computed = self.projection.setup(self.proj_params)

    def transform(self, x, y, xp, *, out_x=None, out_y=None, precision="auto"):
        """Execute the transform pipeline.

        For forward (geographic -> projected):
            x = latitude (degrees), y = longitude (degrees)  [pyproj convention]
            Returns (easting, northing) in meters.

        For inverse (projected -> geographic):
            x = easting, y = northing
            Returns (latitude, longitude) in degrees.

        out_x, out_y: optional pre-allocated output arrays (avoids allocation).
        precision: "auto", "fp32", or "fp64" — compute precision for GPU kernels.
        """
        if self.mode == "forward":
            return self._forward(x, y, xp, out_x=out_x, out_y=out_y, precision=precision)
        elif self.mode == "inverse":
            return self._inverse(x, y, xp, out_x=out_x, out_y=out_y, precision=precision)
        elif self.mode == "proj_to_proj":
            return self._proj_to_proj(x, y, xp)
        else:
            return x, y  # longlat -> longlat identity

    def _forward(self, arg1, arg2, xp, *, out_x=None, out_y=None, precision="auto"):
        """Geographic -> Projected.

        Input follows source CRS axis order (lat/lon for EPSG:4326).
        Output follows destination CRS axis order.
        """
        # Fast path: fused CUDA kernel (single launch, no intermediate arrays)
        fused = _try_fused(
            arg1,
            arg2,
            xp,
            projection_name=self.projection.name,
            direction="forward",
            computed=self.computed,
            src_north_first=self.src_north_first,
            dst_north_first=self.dst_north_first,
            out_x=out_x,
            out_y=out_y,
            precision=precision,
        )
        if fused is not None:
            return fused

        # Source axis order: geographic CRS is (lat, lon) when north_first
        if self.src_north_first:
            lat, lon = arg1, arg2
        else:
            lon, lat = arg1, arg2

        computed = self.computed
        a = computed.get("a", self.proj_params.ellipsoid.a)
        x0 = computed.get("x0", self.proj_params.x_0)
        y0 = computed.get("y0", self.proj_params.y_0)
        lam0 = computed.get("lam0", math.radians(self.proj_params.lon_0))

        # Convert to radians
        phi = lat * DEG_TO_RAD
        lam = lon * DEG_TO_RAD

        # Subtract central meridian
        lam = _wrap_to_pi(lam - lam0, xp)

        # Core projection: returns (easting, northing) always
        easting, northing = self.projection.forward(lam, phi, self.proj_params, computed, xp)

        # Scale by semi-major axis and add false easting/northing
        easting = easting * a + x0
        northing = northing * a + y0

        # Output in destination CRS axis order
        if self.dst_north_first:
            return northing, easting
        return easting, northing

    def _inverse(self, arg1, arg2, xp, *, out_x=None, out_y=None, precision="auto"):
        """Projected -> Geographic.

        Input follows source CRS axis order.
        Output follows destination CRS axis order (lat/lon for EPSG:4326).
        """
        # Fast path: fused CUDA kernel
        fused = _try_fused(
            arg1,
            arg2,
            xp,
            projection_name=self.projection.name,
            direction="inverse",
            computed=self.computed,
            src_north_first=self.src_north_first,
            dst_north_first=self.dst_north_first,
            out_x=out_x,
            out_y=out_y,
            precision=precision,
        )
        if fused is not None:
            return fused

        # Source is projected: interpret per its axis order
        if self.src_north_first:
            northing, easting = arg1, arg2
        else:
            easting, northing = arg1, arg2

        computed = self.computed
        a = computed.get("a", self.proj_params.ellipsoid.a)
        x0 = computed.get("x0", self.proj_params.x_0)
        y0 = computed.get("y0", self.proj_params.y_0)
        lam0 = computed.get("lam0", math.radians(self.proj_params.lon_0))

        # Remove false easting/northing and scale
        x = (easting - x0) / a
        y = (northing - y0) / a

        # Core inverse projection
        lam, phi = self.projection.inverse(x, y, self.proj_params, computed, xp)

        # Add back central meridian
        lam = _wrap_to_pi(lam + lam0, xp)

        # Convert to degrees
        lat = phi * RAD_TO_DEG
        lon = lam * RAD_TO_DEG

        # Output in destination CRS axis order (geographic)
        if self.dst_north_first:
            return lat, lon
        return lon, lat

    def _proj_to_proj(self, x, y, xp):
        """Projected -> Projected via geographic intermediate.

        Decomposes into two fused kernel calls when available:
        1. Source projected -> geographic (inverse)
        2. Geographic -> destination projected (forward)
        """
        # Build sub-pipelines lazily
        if not hasattr(self, "_p2p_inv"):
            from vibeproj.crs import ProjectionParams

            geo = ProjectionParams(
                projection_name="longlat",
                ellipsoid=self.src.ellipsoid,
                north_first=True,  # intermediate is always (lat, lon)
            )
            self._p2p_inv = TransformPipeline(self.src, geo)
            self._p2p_fwd = TransformPipeline(geo, self.dst)

        # Step 1: source projected -> geographic (may use fused inverse kernel)
        lat, lon = self._p2p_inv.transform(x, y, xp)

        # Step 2: geographic -> destination projected (may use fused forward kernel)
        return self._p2p_fwd.transform(lat, lon, xp)
