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
    from vibeproj.helmert import HelmertParams

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
    stream=None,
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
        stream=stream,
    )


def _wrap_to_pi(angle, xp):
    """Wrap angle to [-pi, pi]."""
    return angle - 2.0 * math.pi * xp.round(angle / (2.0 * math.pi))


def _apply_datum_shift(
    lat,
    lon,
    helmert: HelmertParams,
    xp,
    h=None,
    out_lat=None,
    out_lon=None,
    out_h=None,
    stream=None,
):
    """Apply Helmert datum shift. Tries fused GPU kernel first, falls back to xp.

    Returns 2-tuple (lat, lon) when h is None, 3-tuple (lat, lon, h) when h is provided.
    out_lat, out_lon, out_h: optional pre-allocated output arrays.
    stream: optional CUDA stream forwarded to the fused Helmert kernel.
    """
    cp = _get_cupy()
    if cp is not None and xp is cp:
        try:
            from vibeproj.fused_kernels import fused_helmert_shift

            result = fused_helmert_shift(
                lat,
                lon,
                helmert,
                xp,
                h=h,
                out_lat=out_lat,
                out_lon=out_lon,
                out_h=out_h,
                stream=stream,
            )
            if result is not None:
                return result
        except ImportError:
            pass
    from vibeproj.helmert import apply_helmert

    result = apply_helmert(lat, lon, helmert, xp, h=h)
    # Write into pre-allocated output buffers when provided (xp fallback path)
    if h is not None:
        rlat, rlon, rh = result
        if out_lat is not None:
            out_lat[:] = rlat
            rlat = out_lat
        if out_lon is not None:
            out_lon[:] = rlon
            rlon = out_lon
        if out_h is not None:
            out_h[:] = rh
            rh = out_h
        return rlat, rlon, rh
    else:
        rlat, rlon = result
        if out_lat is not None:
            out_lat[:] = rlat
            rlat = out_lat
        if out_lon is not None:
            out_lon[:] = rlon
            rlon = out_lon
        return rlat, rlon


class TransformPipeline:
    """Executes a coordinate transformation between two CRS.

    Handles the full pipeline: CRS resolution, parameter setup,
    pre-processing (unit conversion, axis swap), projection math,
    and post-processing (scale, offset).
    """

    def __init__(
        self,
        src_params: ProjectionParams,
        dst_params: ProjectionParams,
        *,
        helmert: HelmertParams | None = None,
    ):
        self.src = src_params
        self.dst = dst_params
        self._helmert = helmert

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

    def transform(
        self, x, y, xp, *, z=None, out_x=None, out_y=None, out_z=None, precision="auto", stream=None
    ):
        """Execute the transform pipeline.

        For forward (geographic -> projected):
            x = latitude (degrees), y = longitude (degrees)  [pyproj convention]
            Returns (easting, northing) in meters.

        For inverse (projected -> geographic):
            x = easting, y = northing
            Returns (latitude, longitude) in degrees.

        z: optional ellipsoidal height. Transformed through Helmert when present,
           passed through unchanged for projection-only transforms.
        out_x, out_y: optional pre-allocated output arrays (avoids allocation).
        out_z: optional pre-allocated output height array.
        precision: "auto", "fp32", or "fp64" — compute precision for GPU kernels.
        stream: optional CUDA stream for async kernel execution.

        Returns 2-tuple when z is None, 3-tuple when z is provided.
        """
        if self.mode == "forward":
            return self._forward(
                x,
                y,
                xp,
                z=z,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                precision=precision,
                stream=stream,
            )
        elif self.mode == "inverse":
            return self._inverse(
                x,
                y,
                xp,
                z=z,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                precision=precision,
                stream=stream,
            )
        elif self.mode == "proj_to_proj":
            return self._proj_to_proj(
                x,
                y,
                xp,
                z=z,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                precision=precision,
                stream=stream,
            )
        else:
            # longlat -> longlat: apply datum shift if needed, otherwise identity
            if self._helmert is not None:
                result = _apply_datum_shift(
                    x,
                    y,
                    self._helmert,
                    xp,
                    h=z,
                    out_lat=out_x,
                    out_lon=out_y,
                    out_h=out_z,
                    stream=stream,
                )
                if z is not None:
                    return result  # already a 3-tuple
                return result
            # Identity: write into pre-allocated buffers when provided
            if out_x is not None:
                out_x[:] = x
                x = out_x
            if out_y is not None:
                out_y[:] = y
                y = out_y
            if z is not None:
                if out_z is not None:
                    out_z[:] = z
                    z = out_z
                return x, y, z
            return x, y

    def _forward(
        self,
        arg1,
        arg2,
        xp,
        *,
        z=None,
        out_x=None,
        out_y=None,
        out_z=None,
        precision="auto",
        stream=None,
    ):
        """Geographic -> Projected.

        Input follows source CRS axis order (lat/lon for EPSG:4326).
        Output follows destination CRS axis order.
        z is transformed through Helmert when present, then passed through projection.
        """
        # Fast path: fused CUDA kernel (single launch, no intermediate arrays)
        # Skipped when datum shift is needed (fused kernels don't include Helmert).
        if self._helmert is None:
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
                stream=stream,
            )
            if fused is not None:
                if z is not None:
                    return (*fused, z)
                return fused

        # Source axis order: geographic CRS is (lat, lon) when north_first
        if self.src_north_first:
            lat, lon = arg1, arg2
        else:
            lon, lat = arg1, arg2

        # Datum shift: transform geographic coords (and z) to destination ellipsoid
        z_out = z
        if self._helmert is not None:
            result = _apply_datum_shift(lat, lon, self._helmert, xp, h=z, stream=stream)
            if z is not None:
                lat, lon, z_out = result
            else:
                lat, lon = result

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

        # Core projection: returns (easting, northing) always — 2D only
        easting, northing = self.projection.forward(lam, phi, self.proj_params, computed, xp)

        # Scale by semi-major axis and add false easting/northing
        easting = easting * a + x0
        northing = northing * a + y0

        # Output in destination CRS axis order
        if self.dst_north_first:
            rx, ry = northing, easting
        else:
            rx, ry = easting, northing

        # Write into pre-allocated output buffers when provided (xp fallback path)
        if out_x is not None:
            out_x[:] = rx
            rx = out_x
        if out_y is not None:
            out_y[:] = ry
            ry = out_y

        if z is not None:
            if out_z is not None:
                out_z[:] = z_out
                z_out = out_z
            return rx, ry, z_out
        return rx, ry

    def _inverse(
        self,
        arg1,
        arg2,
        xp,
        *,
        z=None,
        out_x=None,
        out_y=None,
        out_z=None,
        precision="auto",
        stream=None,
    ):
        """Projected -> Geographic.

        Input follows source CRS axis order.
        Output follows destination CRS axis order (lat/lon for EPSG:4326).
        z passes through projection inverse (2D), then is transformed by Helmert.
        """
        # Fast path: fused CUDA kernel
        # Skipped when datum shift is needed (fused kernels don't include Helmert).
        if self._helmert is None:
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
                stream=stream,
            )
            if fused is not None:
                if z is not None:
                    return (*fused, z)
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

        # Core inverse projection — 2D only, z passes through
        lam, phi = self.projection.inverse(x, y, self.proj_params, computed, xp)

        # Add back central meridian
        lam = _wrap_to_pi(lam + lam0, xp)

        # Convert to degrees
        lat = phi * RAD_TO_DEG
        lon = lam * RAD_TO_DEG

        # Datum shift: transform geographic coords (and z) to destination ellipsoid
        z_out = z
        if self._helmert is not None:
            result = _apply_datum_shift(lat, lon, self._helmert, xp, h=z, stream=stream)
            if z is not None:
                lat, lon, z_out = result
            else:
                lat, lon = result

        # Output in destination CRS axis order (geographic)
        if self.dst_north_first:
            rx, ry = lat, lon
        else:
            rx, ry = lon, lat

        # Write into pre-allocated output buffers when provided (xp fallback path)
        if out_x is not None:
            out_x[:] = rx
            rx = out_x
        if out_y is not None:
            out_y[:] = ry
            ry = out_y

        if z is not None:
            if out_z is not None:
                out_z[:] = z_out
                z_out = out_z
            return rx, ry, z_out
        return rx, ry

    def _proj_to_proj(
        self,
        x,
        y,
        xp,
        *,
        z=None,
        out_x=None,
        out_y=None,
        out_z=None,
        precision="auto",
        stream=None,
    ):
        """Projected -> Projected via geographic intermediate.

        Decomposes into two fused kernel calls when available:
        1. Source projected -> geographic (inverse)
        2. Geographic -> destination projected (forward)

        z passes through both projection steps (2D) and is transformed by Helmert.
        out_x, out_y, out_z: optional pre-allocated output arrays.
        precision: compute precision forwarded to sub-pipelines.
        stream: optional CUDA stream for async kernel execution.
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
        # z passes through unchanged (projection is 2D)
        lat, lon = self._p2p_inv.transform(x, y, xp, precision=precision, stream=stream)

        # Step 2: datum shift (if cross-datum) — transforms z when present
        z_out = z
        if self._helmert is not None:
            result = _apply_datum_shift(lat, lon, self._helmert, xp, h=z, stream=stream)
            if z is not None:
                lat, lon, z_out = result
            else:
                lat, lon = result

        # Step 3: geographic -> destination projected (may use fused forward kernel)
        # z passes through unchanged (projection is 2D)
        result = self._p2p_fwd.transform(
            lat,
            lon,
            xp,
            out_x=out_x,
            out_y=out_y,
            precision=precision,
            stream=stream,
        )
        if z is not None:
            if out_z is not None:
                out_z[:] = z_out
                return (*result, out_z)
            return (*result, z_out)
        return result
