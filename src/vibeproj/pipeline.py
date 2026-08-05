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
import threading
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibeproj.projections import get_projection

if TYPE_CHECKING:
    from vibeproj._datum_corrections import DatumCorrectionData
    from vibeproj.crs import ProjectionParams
    from vibeproj.helmert import HelmertParams

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Lazy CuPy reference for fused kernel fast-path
_cupy_module = None


@dataclass(frozen=True, slots=True)
class TransformScratch:
    """Reusable device intermediates owned and synchronized by Transformer."""

    first_x: object
    first_y: object
    second_x: object
    second_y: object

    def pair(self, index: int, size: int):
        if index == 0:
            return self.first_x[:size], self.first_y[:size]
        if index == 1:
            return self.second_x[:size], self.second_y[:size]
        raise ValueError(f"Invalid scratch pair index: {index}")


def _scratch_pair(scratch: TransformScratch | None, index: int, size: int):
    return (None, None) if scratch is None else scratch.pair(index, size)


def _lat_lon_outputs(out_x, out_y, north_first: bool):
    """Map public axis-order buffers to internal (lat, lon) buffers."""
    return (out_x, out_y) if north_first else (out_y, out_x)


def _setup_projection(projection, params):
    """Set up projection math and attach canonical strategy metadata once."""
    from vibeproj.transcendentals import attach_projection_strategy_metadata

    computed = projection.setup(params)
    return attach_projection_strategy_metadata(
        computed,
        operation_method=params.operation_method,
        eccentricity_squared=params.ellipsoid.es,
        latitude_origin_degrees=params.lat_0,
    )


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
    transcendentals="auto",
    execution_context=None,
    stream=None,
):
    """Attempt fused kernel execution. Returns None if not available."""
    cp = _get_cupy()
    if cp is None or xp is not cp:
        return None
    try:
        from vibeproj.fused_kernels import can_fuse, fused_transform
    except ImportError:
        warnings.warn(
            "Fused CUDA kernels unavailable — falling back to element-wise path. "
            "GPU-resident data will be processed without kernel fusion.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not can_fuse(projection_name, direction):
        return None
    from vibeproj.transcendentals import projection_strategy_domain

    domain = projection_strategy_domain(projection_name, direction, computed)
    if execution_context is None:
        # Compatibility for direct private-helper callers. Public entry points
        # construct one immutable plan and pass it through every stage.
        from vibeproj.transcendentals import (
            TranscendentalOperation,
            detect_device_capability,
            resolve_transcendental_strategy,
        )

        operation = (
            TranscendentalOperation.TMERC_FORWARD
            if projection_name == "tmerc" and direction == "forward"
            else TranscendentalOperation.PROJECTION
        )
        array_device = getattr(getattr(arg1, "device", None), "id", None)
        device = detect_device_capability(
            cp, device_id=None if array_device is None else int(array_device)
        )
        transcendental_impl = resolve_transcendental_strategy(
            operation,
            transcendentals,
            device=device,
            domain=domain,
            precision=precision,
            workload_size=int(arg1.size),
        ).implementation_id
    else:
        transcendental_impl = execution_context.projection_implementation(
            projection_name, direction, domain
        )
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
        transcendental_impl=transcendental_impl,
        stream=stream,
    )


def _wrap_to_pi(angle, xp):
    """Wrap angle to [-pi, pi]."""
    finite = xp.isfinite(angle)
    safe_angle = xp.where(finite, angle, 0.0)
    wrapped = safe_angle - 2.0 * math.pi * xp.round(safe_angle / (2.0 * math.pi))
    return xp.where(finite, wrapped, angle)


def _apply_datum_shift(
    lat,
    lon,
    helmert: HelmertParams,
    xp,
    h=None,
    out_lat=None,
    out_lon=None,
    out_h=None,
    transcendentals="auto",
    execution_context=None,
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

            if execution_context is None:
                from vibeproj.transcendentals import (
                    TranscendentalOperation,
                    detect_device_capability,
                    resolve_transcendental_strategy,
                )

                array_device = getattr(getattr(lat, "device", None), "id", None)
                device = detect_device_capability(
                    cp, device_id=None if array_device is None else int(array_device)
                )
                helmert_implementation = resolve_transcendental_strategy(
                    TranscendentalOperation.HELMERT,
                    transcendentals,
                    device=device,
                    precision="auto",
                    workload_size=int(lat.size),
                ).implementation_id
            else:
                helmert_implementation = execution_context.helmert_implementation
            result = fused_helmert_shift(
                lat,
                lon,
                helmert,
                xp,
                h=h,
                out_lat=out_lat,
                out_lon=out_lon,
                out_h=out_h,
                transcendental_impl=helmert_implementation,
                stream=stream,
            )
            if result is not None:
                return result
        except ImportError:
            warnings.warn(
                "Fused Helmert CUDA kernel unavailable — falling back to element-wise path.",
                RuntimeWarning,
                stacklevel=2,
            )
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


def _apply_svd_correction(
    lat,
    lon,
    correction: DatumCorrectionData,
    xp,
    *,
    negate: bool = False,
    out_lat=None,
    out_lon=None,
    stream=None,
):
    """Apply SVD datum correction. Tries fused GPU kernel first, falls back to xp."""
    cp = _get_cupy()
    if cp is not None and xp is cp:
        try:
            from vibeproj.fused_kernels import fused_svd_correction

            result = fused_svd_correction(
                lat,
                lon,
                correction,
                xp,
                negate=negate,
                out_lat=out_lat,
                out_lon=out_lon,
                stream=stream,
            )
            if result is not None:
                return result
        except ImportError:
            pass

    from vibeproj._datum_corrections import apply_svd_correction

    return apply_svd_correction(
        lat,
        lon,
        correction,
        xp,
        negate=negate,
        out_lat=out_lat,
        out_lon=out_lon,
    )


class TransformPipeline:
    """Executes a coordinate transformation between two CRS.

    Handles the full pipeline: CRS resolution, parameter setup,
    pre-processing (unit conversion, axis swap), projection math,
    and post-processing (scale, offset).
    """

    @property
    def needs_scratch(self) -> bool:
        """Whether zero-allocation fused execution needs intermediate buffers."""
        if self.mode == "proj_to_proj":
            return True
        if self.mode in ("forward", "inverse"):
            return self._helmert is not None or self._svd_correction is not None
        return self._helmert is not None and self._svd_correction is not None

    def build_execution_context(
        self,
        *,
        precision: str,
        transcendentals: str,
        device,
        workload_size: int | None,
        _normalized: bool = False,
    ):
        """Resolve every implementation once at the public-call boundary."""
        from vibeproj.transcendentals import (
            NATIVE_LIBDEVICE,
            ExecutionContext,
            ProjectionImplementation,
            TranscendentalOperation,
            normalize_compute_precision,
            normalize_transcendental_policy,
            projection_strategy_domain,
            resolve_transcendental_strategy,
        )

        if _normalized:
            normalized_precision = precision
            normalized_policy = transcendentals
        else:
            normalized_precision = normalize_compute_precision(precision)
            normalized_policy = normalize_transcendental_policy(transcendentals)
        stages: list[tuple[str, str, str, TranscendentalOperation]] = []

        def add_projection(name: str, direction: str, computed: dict) -> None:
            domain = projection_strategy_domain(name, direction, computed)
            if name == "tmerc" and direction == "forward":
                operation = TranscendentalOperation.TMERC_FORWARD
            else:
                operation = TranscendentalOperation.PROJECTION
            stages.append((name, direction, domain, operation))

        if self.mode == "forward":
            add_projection(self.projection.name, "forward", self.computed)
        elif self.mode == "inverse":
            add_projection(self.projection.name, "inverse", self.computed)
        elif self.mode == "proj_to_proj":
            add_projection(self.src_projection.name, "inverse", self.src_computed)
            add_projection(self.dst_projection.name, "forward", self.dst_computed)

        decisions = []
        implementations = []
        for projection, direction, domain, operation in stages:
            decision = resolve_transcendental_strategy(
                operation,
                normalized_policy,
                device=device,
                domain=domain,
                precision=normalized_precision,
                workload_size=workload_size,
                _normalized=True,
            )
            decisions.append(decision)
            implementations.append(
                ProjectionImplementation(
                    projection=projection,
                    direction=direction,
                    domain=domain,
                    implementation_id=decision.implementation_id,
                )
            )

        helmert_implementation = NATIVE_LIBDEVICE
        if self._helmert is not None:
            decision = resolve_transcendental_strategy(
                TranscendentalOperation.HELMERT,
                normalized_policy,
                device=device,
                precision=normalized_precision,
                workload_size=workload_size,
                _normalized=True,
            )
            decisions.append(decision)
            helmert_implementation = decision.implementation_id

        return ExecutionContext(
            precision=normalized_precision,
            transcendentals=normalized_policy,
            device=device,
            workload_size=workload_size,
            projection_implementations=tuple(implementations),
            helmert_implementation=helmert_implementation,
            decisions=tuple(decisions),
        )

    def __init__(
        self,
        src_params: ProjectionParams,
        dst_params: ProjectionParams,
        *,
        helmert: HelmertParams | None = None,
        svd_correction: DatumCorrectionData | None = None,
        svd_negate: bool = False,
    ):
        self.src = src_params
        self.dst = dst_params
        self._helmert = helmert
        self._svd_correction = svd_correction
        self._svd_negate = svd_negate

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
            self.src_computed = _setup_projection(self.src_projection, src_params)
            self.dst_computed = _setup_projection(self.dst_projection, dst_params)
        else:
            # Geographic -> Geographic (possibly different datums)
            self.mode = "longlat_to_longlat"

        if self.mode in ("forward", "inverse"):
            self.computed = _setup_projection(self.projection, self.proj_params)
            self.computed.setdefault("x_unit_to_m", self.proj_params.x_unit_to_m)
            self.computed.setdefault("y_unit_to_m", self.proj_params.y_unit_to_m)
        elif self.mode == "proj_to_proj":
            self.src_computed.setdefault("x_unit_to_m", self.src.x_unit_to_m)
            self.src_computed.setdefault("y_unit_to_m", self.src.y_unit_to_m)
            self.dst_computed.setdefault("x_unit_to_m", self.dst.x_unit_to_m)
            self.dst_computed.setdefault("y_unit_to_m", self.dst.y_unit_to_m)
            self._p2p_inv: TransformPipeline | None = None
            self._p2p_fwd: TransformPipeline | None = None
            self._p2p_lock = threading.RLock()

    def transform(
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
        transcendentals="auto",
        scratch: TransformScratch | None = None,
        execution_context=None,
        stream=None,
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
        if execution_context is None:
            from vibeproj.transcendentals import detect_device_capability

            is_cupy = getattr(xp, "__name__", "").split(".", 1)[0] == "cupy"
            device_id = int(x.device.id) if is_cupy else None
            device = detect_device_capability(xp, device_id=device_id)
            execution_context = self.build_execution_context(
                precision=precision,
                transcendentals=transcendentals,
                device=device,
                workload_size=int(x.size),
            )
        precision = execution_context.precision
        transcendentals = execution_context.transcendentals
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
                transcendentals=transcendentals,
                scratch=scratch,
                execution_context=execution_context,
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
                transcendentals=transcendentals,
                scratch=scratch,
                execution_context=execution_context,
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
                transcendentals=transcendentals,
                scratch=scratch,
                execution_context=execution_context,
                stream=stream,
            )
        else:
            # longlat -> longlat: apply datum shift + SVD if needed, otherwise identity
            if self._helmert is not None or self._svd_correction is not None:
                # Resolve axis order: Helmert and SVD expect (lat, lon)
                if self.src_north_first:
                    lat, lon = x, y
                else:
                    lon, lat = x, y

                z_out = z
                final_lat, final_lon = _lat_lon_outputs(out_x, out_y, self.dst_north_first)
                if self._helmert is not None:
                    if self._svd_correction is None:
                        helmert_out_lat, helmert_out_lon = final_lat, final_lon
                    else:
                        helmert_out_lat, helmert_out_lon = _scratch_pair(scratch, 0, x.size)
                    result = _apply_datum_shift(
                        lat,
                        lon,
                        self._helmert,
                        xp,
                        h=z,
                        out_lat=helmert_out_lat,
                        out_lon=helmert_out_lon,
                        out_h=out_z,
                        transcendentals=transcendentals,
                        execution_context=execution_context,
                        stream=stream,
                    )
                    if z is not None:
                        lat, lon, z_out = result
                    else:
                        lat, lon = result

                if self._svd_correction is not None:
                    lat, lon = _apply_svd_correction(
                        lat,
                        lon,
                        self._svd_correction,
                        xp,
                        negate=self._svd_negate,
                        out_lat=final_lat,
                        out_lon=final_lon,
                        stream=stream,
                    )

                # Restore output axis order
                if self.dst_north_first:
                    rx, ry = lat, lon
                else:
                    rx, ry = lon, lat

                if out_x is not None and rx is not out_x:
                    out_x[:] = rx
                    rx = out_x
                if out_y is not None and ry is not out_y:
                    out_y[:] = ry
                    ry = out_y
                if z is not None:
                    if out_z is not None and z_out is not out_z:
                        out_z[:] = z_out
                        z_out = out_z
                    return rx, ry, z_out
                return rx, ry

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
        transcendentals="auto",
        scratch: TransformScratch | None = None,
        execution_context=None,
        stream=None,
    ):
        """Geographic -> Projected.

        Input follows source CRS axis order (lat/lon for EPSG:4326).
        Output follows destination CRS axis order.
        z is transformed through Helmert when present, then passed through projection.
        """
        # Fast path: fused CUDA kernel (single launch, no intermediate arrays)
        # Skipped when datum shift or SVD correction is needed.
        if self._helmert is None and self._svd_correction is None:
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
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
            if fused is not None:
                if z is not None:
                    z_out = z
                    if out_z is not None and z_out is not out_z:
                        out_z[:] = z_out
                        z_out = out_z
                    return (*fused, z_out)
                return fused

        # Source axis order: geographic CRS is (lat, lon) when north_first
        if self.src_north_first:
            lat, lon = arg1, arg2
        else:
            lon, lat = arg1, arg2

        # Datum shift: transform geographic coords (and z) to destination ellipsoid
        z_out = z
        stage_index = 0
        if self._helmert is not None:
            helmert_out_lat, helmert_out_lon = _scratch_pair(scratch, stage_index, arg1.size)
            result = _apply_datum_shift(
                lat,
                lon,
                self._helmert,
                xp,
                h=z,
                out_lat=helmert_out_lat,
                out_lon=helmert_out_lon,
                out_h=out_z,
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
            stage_index = 1
            if z is not None:
                lat, lon, z_out = result
            else:
                lat, lon = result

        # SVD correction: additive correction on geographic coords (degrees)
        if self._svd_correction is not None:
            svd_out_lat, svd_out_lon = _scratch_pair(scratch, stage_index, arg1.size)
            lat, lon = _apply_svd_correction(
                lat,
                lon,
                self._svd_correction,
                xp,
                negate=self._svd_negate,
                out_lat=svd_out_lat,
                out_lon=svd_out_lon,
                stream=stream,
            )

        # After Helmert/SVD, try fused projection kernel on the shifted coords.
        # Helmert/SVD output is always (lat, lon) = north_first=True.
        if self._helmert is not None or self._svd_correction is not None:
            fused = _try_fused(
                lat,
                lon,
                xp,
                projection_name=self.projection.name,
                direction="forward",
                computed=self.computed,
                src_north_first=True,  # Helmert/SVD always outputs (lat, lon)
                dst_north_first=self.dst_north_first,
                out_x=out_x,
                out_y=out_y,
                precision=precision,
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
            if fused is not None:
                if z is not None:
                    if out_z is not None and z_out is not out_z:
                        out_z[:] = z_out
                        z_out = out_z
                    return (*fused, z_out)
                return fused

        computed = self.computed
        a = computed.get("a", self.proj_params.ellipsoid.a)
        x0 = computed.get("x0", self.proj_params.x_0)
        y0 = computed.get("y0", self.proj_params.y_0)
        x_unit_to_m = computed.get("x_unit_to_m", self.proj_params.x_unit_to_m)
        y_unit_to_m = computed.get("y_unit_to_m", self.proj_params.y_unit_to_m)
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

        # Projected CRS I/O stays in the CRS native linear units.
        easting = easting / x_unit_to_m
        northing = northing / y_unit_to_m

        # Output in destination CRS axis order
        if self.dst_north_first:
            rx, ry = northing, easting
        else:
            rx, ry = easting, northing

        # Write into pre-allocated output buffers when provided (xp fallback path)
        if out_x is not None and rx is not out_x:
            out_x[:] = rx
            rx = out_x
        if out_y is not None and ry is not out_y:
            out_y[:] = ry
            ry = out_y

        if z is not None:
            if out_z is not None and z_out is not out_z:
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
        transcendentals="auto",
        scratch: TransformScratch | None = None,
        execution_context=None,
        stream=None,
    ):
        """Projected -> Geographic.

        Input follows source CRS axis order.
        Output follows destination CRS axis order (lat/lon for EPSG:4326).
        z passes through projection inverse (2D), then is transformed by Helmert.
        """
        final_lat, final_lon = _lat_lon_outputs(out_x, out_y, self.dst_north_first)
        # Fast path: fused CUDA kernel
        if self._helmert is None and self._svd_correction is None:
            # No datum shift or SVD: run fused inverse with final output axis order.
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
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
            if fused is not None:
                if z is not None:
                    z_out = z
                    if out_z is not None and z_out is not out_z:
                        out_z[:] = z_out
                        z_out = out_z
                    return (*fused, z_out)
                return fused
        elif self._helmert is not None or self._svd_correction is not None:
            # With datum shift/SVD: fused inverse -> SVD -> Helmert -> axis reorder.
            # Fused inverse outputs (lat, lon) for SVD/Helmert input.
            inverse_out_lat, inverse_out_lon = _scratch_pair(scratch, 0, arg1.size)
            fused = _try_fused(
                arg1,
                arg2,
                xp,
                projection_name=self.projection.name,
                direction="inverse",
                computed=self.computed,
                src_north_first=self.src_north_first,
                dst_north_first=True,  # SVD/Helmert expects (lat, lon)
                out_x=inverse_out_lat,
                out_y=inverse_out_lon,
                precision=precision,
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
            if fused is not None:
                lat, lon = fused
                z_out = z
                # SVD correction (before Helmert in inverse direction)
                if self._svd_correction is not None:
                    if self._helmert is None:
                        svd_out_lat, svd_out_lon = final_lat, final_lon
                    else:
                        svd_out_lat, svd_out_lon = _scratch_pair(scratch, 1, arg1.size)
                    lat, lon = _apply_svd_correction(
                        lat,
                        lon,
                        self._svd_correction,
                        xp,
                        negate=self._svd_negate,
                        out_lat=svd_out_lat,
                        out_lon=svd_out_lon,
                        stream=stream,
                    )
                if self._helmert is not None:
                    result = _apply_datum_shift(
                        lat,
                        lon,
                        self._helmert,
                        xp,
                        h=z,
                        out_lat=final_lat,
                        out_lon=final_lon,
                        out_h=out_z,
                        transcendentals=transcendentals,
                        execution_context=execution_context,
                        stream=stream,
                    )
                    if z is not None:
                        lat, lon, z_out = result
                    else:
                        lat, lon = result
                if self.dst_north_first:
                    rx, ry = lat, lon
                else:
                    rx, ry = lon, lat
                # Write into pre-allocated output buffers when provided
                if out_x is not None and rx is not out_x:
                    out_x[:] = rx
                    rx = out_x
                if out_y is not None and ry is not out_y:
                    out_y[:] = ry
                    ry = out_y
                if z is not None:
                    if out_z is not None and z_out is not out_z:
                        out_z[:] = z_out
                        z_out = out_z
                    return rx, ry, z_out
                return rx, ry

        # Source is projected: interpret per its axis order
        if self.src_north_first:
            northing, easting = arg1, arg2
        else:
            easting, northing = arg1, arg2

        computed = self.computed
        a = computed.get("a", self.proj_params.ellipsoid.a)
        x0 = computed.get("x0", self.proj_params.x_0)
        y0 = computed.get("y0", self.proj_params.y_0)
        x_unit_to_m = computed.get("x_unit_to_m", self.proj_params.x_unit_to_m)
        y_unit_to_m = computed.get("y_unit_to_m", self.proj_params.y_unit_to_m)
        lam0 = computed.get("lam0", math.radians(self.proj_params.lon_0))

        # Projected CRS inputs are expressed in the CRS native linear units.
        easting = easting * x_unit_to_m
        northing = northing * y_unit_to_m

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

        # SVD correction (before Helmert in inverse direction)
        if self._svd_correction is not None:
            if self._helmert is None:
                svd_out_lat, svd_out_lon = final_lat, final_lon
            else:
                svd_out_lat, svd_out_lon = _scratch_pair(scratch, 0, arg1.size)
            lat, lon = _apply_svd_correction(
                lat,
                lon,
                self._svd_correction,
                xp,
                negate=self._svd_negate,
                out_lat=svd_out_lat,
                out_lon=svd_out_lon,
                stream=stream,
            )

        # Datum shift: transform geographic coords (and z) to destination ellipsoid
        z_out = z
        if self._helmert is not None:
            result = _apply_datum_shift(
                lat,
                lon,
                self._helmert,
                xp,
                h=z,
                out_lat=final_lat,
                out_lon=final_lon,
                out_h=out_z,
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
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
        if out_x is not None and rx is not out_x:
            out_x[:] = rx
            rx = out_x
        if out_y is not None and ry is not out_y:
            out_y[:] = ry
            ry = out_y

        if z is not None:
            if out_z is not None and z_out is not out_z:
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
        transcendentals="auto",
        scratch: TransformScratch | None = None,
        execution_context=None,
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
        if self._p2p_inv is None:
            with self._p2p_lock:
                if self._p2p_inv is None:
                    from vibeproj.crs import ProjectionParams

                    geo = ProjectionParams(
                        projection_name="longlat",
                        ellipsoid=self.src.ellipsoid,
                        north_first=True,  # intermediate is always (lat, lon)
                    )
                    self._p2p_inv = TransformPipeline(self.src, geo)
                    self._p2p_fwd = TransformPipeline(geo, self.dst)
        assert self._p2p_inv is not None and self._p2p_fwd is not None

        # Step 1: source projected -> geographic (may use fused inverse kernel)
        # z passes through unchanged (projection is 2D)
        inverse_out_lat, inverse_out_lon = _scratch_pair(scratch, 0, x.size)
        lat, lon = self._p2p_inv.transform(
            x,
            y,
            xp,
            out_x=inverse_out_lat,
            out_y=inverse_out_lon,
            precision=precision,
            transcendentals=transcendentals,
            execution_context=execution_context,
            stream=stream,
        )

        # Step 2: datum shift (if cross-datum) — transforms z when present
        z_out = z
        if self._helmert is not None:
            helmert_out_lat, helmert_out_lon = _scratch_pair(scratch, 1, x.size)
            result = _apply_datum_shift(
                lat,
                lon,
                self._helmert,
                xp,
                h=z,
                out_lat=helmert_out_lat,
                out_lon=helmert_out_lon,
                out_h=out_z,
                transcendentals=transcendentals,
                execution_context=execution_context,
                stream=stream,
            )
            if z is not None:
                lat, lon, z_out = result
            else:
                lat, lon = result

        # Step 2b: SVD correction (after Helmert, before forward projection)
        if self._svd_correction is not None:
            svd_pair_index = 0 if self._helmert is not None else 1
            svd_out_lat, svd_out_lon = _scratch_pair(scratch, svd_pair_index, x.size)
            lat, lon = _apply_svd_correction(
                lat,
                lon,
                self._svd_correction,
                xp,
                negate=self._svd_negate,
                out_lat=svd_out_lat,
                out_lon=svd_out_lon,
                stream=stream,
            )

        # Step 3: geographic -> destination projected (may use fused forward kernel)
        # z passes through unchanged (projection is 2D)
        result = self._p2p_fwd.transform(
            lat,
            lon,
            xp,
            out_x=out_x,
            out_y=out_y,
            precision=precision,
            transcendentals=transcendentals,
            execution_context=execution_context,
            stream=stream,
        )
        if z is not None:
            if out_z is not None and z_out is not out_z:
                out_z[:] = z_out
                return (*result, out_z)
            return (*result, z_out)
        return result
