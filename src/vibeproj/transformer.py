"""High-level Transformer API — drop-in replacement for pyproj.Transformer.

Usage:
    from vibeproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform(lon, lat)           # always_xy=True (default)
    lon, lat = t.transform(x, y, direction="INVERSE")
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from vibeproj.crs import parse_crs_input, resolve_transform
from vibeproj.pipeline import TransformPipeline
from vibeproj.runtime import get_array_module, to_device


class Transformer:
    """GPU-accelerated coordinate transformer.

    - transform(x, y) where x=lon, y=lat for geographic CRS (always_xy=True default)
    - direction="FORWARD" or "INVERSE"
    - Accepts scalars, lists, numpy arrays, or cupy arrays

    When CuPy is available and inputs are on GPU, transforms run on GPU.
    Otherwise falls back to NumPy on CPU.
    """

    def __init__(self, crs_from, crs_to, *, always_xy=True):
        src_params, dst_params, src_crs, dst_crs = resolve_transform(crs_from, crs_to)

        # Store raw inputs for pickle serialization
        self._crs_from_input = crs_from
        self._crs_to_input = crs_to
        self._always_xy = always_xy

        # Resolve display labels for __repr__
        src_epsg = src_crs.to_epsg()
        dst_epsg = dst_crs.to_epsg()
        self._src_label = f"EPSG:{src_epsg}" if src_epsg else str(crs_from)
        self._dst_label = f"EPSG:{dst_epsg}" if dst_epsg else str(crs_to)

        # Datum shift detection: warn when ellipsoids differ significantly
        src_ell = src_crs.ellipsoid
        dst_ell = dst_crs.ellipsoid
        if (
            src_ell is not None
            and dst_ell is not None
            and abs(src_ell.semi_major_metre - dst_ell.semi_major_metre) > 1.0
        ):
            src_datum = src_crs.datum.name if src_crs.datum else "unknown"
            dst_datum = dst_crs.datum.name if dst_crs.datum else "unknown"
            warnings.warn(
                f"Source and destination CRS use different datums "
                f"({src_datum} \u2192 {dst_datum}). vibeProj performs projection "
                f"math only \u2014 datum shifts (Helmert, NTv2) are not applied. "
                f"Results may differ from pyproj by meters to hundreds of meters.",
                stacklevel=2,
            )

        # always_xy=True forces (x, y) = (lon, lat) / (easting, northing) order,
        # matching shapely/geopandas conventions regardless of CRS native axis order.
        if always_xy:
            src_params = dataclasses.replace(src_params, north_first=False)
            dst_params = dataclasses.replace(dst_params, north_first=False)

        self._pipeline = TransformPipeline(src_params, dst_params)
        self._src_params = src_params
        self._dst_params = dst_params
        # Build the inverse pipeline lazily
        self._inv_pipeline = None

    @staticmethod
    def from_crs(crs_from, crs_to, *, always_xy=True) -> Transformer:
        """Create a Transformer from source and target CRS.

        Parameters
        ----------
        crs_from, crs_to :
            EPSG integer (4326), string ("EPSG:4326"), or tuple (("EPSG", 4326)).
        always_xy : bool, default True
            If True, input/output axis order is always (x, y) — i.e.
            (longitude, latitude) for geographic CRS and (easting, northing)
            for projected CRS. This matches shapely and geopandas conventions.
            If False, uses the CRS native axis order (pyproj default).
        """
        return Transformer(crs_from, crs_to, always_xy=always_xy)

    def __repr__(self) -> str:
        proj = self._dst_params.projection_name
        if proj == "longlat":
            proj = self._src_params.projection_name
        fused = "fused" if self.is_fused else "xp"
        return f"Transformer({self._src_label} \u2192 {self._dst_label}, {proj}, {fused})"

    @property
    def is_fused(self) -> bool:
        """True if fused GPU kernels are available for this transform."""
        from vibeproj.fused_kernels import can_fuse

        pipeline = self._pipeline
        if pipeline.mode == "forward" or pipeline.mode == "inverse":
            return can_fuse(pipeline.projection.name, pipeline.mode)
        elif pipeline.mode == "proj_to_proj":
            return can_fuse(pipeline.src_projection.name, "inverse") and can_fuse(
                pipeline.dst_projection.name, "forward"
            )
        return False

    def compile(self, *, precision="auto"):
        """Pre-compile fused NVRTC kernels for this transformer.

        Call this to front-load kernel compilation latency before the
        first transform. No-op if CuPy is not available.
        """
        try:
            from vibeproj.fused_kernels import compile_kernels
        except ImportError:
            return

        pipeline = self._pipeline
        if pipeline.mode in ("forward", "inverse"):
            compile_kernels([pipeline.projection.name], precision=precision)
        elif pipeline.mode == "proj_to_proj":
            names = [pipeline.src_projection.name, pipeline.dst_projection.name]
            compile_kernels(names, precision=precision)

    def __getstate__(self):
        return {
            "crs_from": self._crs_from_input,
            "crs_to": self._crs_to_input,
            "always_xy": self._always_xy,
        }

    def __setstate__(self, state):
        self.__init__(state["crs_from"], state["crs_to"], always_xy=state["always_xy"])

    def transform(self, x, y, z=None, direction="FORWARD"):
        """Transform coordinates.

        Parameters
        ----------
        x, y : scalar, list, numpy array, or cupy array
            Input coordinates. With always_xy=True (default): x=longitude, y=latitude
            for geographic CRS. With always_xy=False: native CRS axis order.
        z : scalar, list, numpy array, or cupy array, optional
            Vertical coordinate. Passed through unchanged (no vertical datum transform).
        direction : str
            "FORWARD" or "INVERSE".

        Returns
        -------
        tuple of arrays (or scalars if scalar input)
            Transformed (x, y) or (x, y, z) if z was provided.
        """
        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        # Detect scalar input
        is_scalar = isinstance(x, (int, float)) and isinstance(y, (int, float))

        # Determine array module from input
        xp = get_array_module(x)
        if xp is np:
            xp = get_array_module(y)

        if is_scalar:
            x = xp.asarray([x], dtype="f8")
            y = xp.asarray([y], dtype="f8")
        else:
            x = to_device(x, xp)
            y = to_device(y, xp)
            # Ensure float dtype
            if not xp.issubdtype(x.dtype, xp.floating):
                x = x.astype(xp.float64)
            if not xp.issubdtype(y.dtype, xp.floating):
                y = y.astype(xp.float64)

        # Handle z passthrough
        if z is not None and not is_scalar:
            z_out = to_device(z, xp)
            if not xp.issubdtype(z_out.dtype, xp.floating):
                z_out = z_out.astype(xp.float64)
        else:
            z_out = z

        if direction == "FORWARD":
            rx, ry = self._pipeline.transform(x, y, xp)
        else:
            if self._inv_pipeline is None:
                self._inv_pipeline = TransformPipeline(self._dst_params, self._src_params)
            rx, ry = self._inv_pipeline.transform(x, y, xp)

        # Check for non-finite output values
        if rx.size > 0 and (xp.any(~xp.isfinite(rx)) or xp.any(~xp.isfinite(ry))):
            warnings.warn(
                "Transform produced non-finite values (NaN or inf). "
                "Input coordinates may be outside the projection's valid domain.",
                stacklevel=2,
            )

        if is_scalar:
            # Convert back to Python floats
            if hasattr(rx, "get"):
                rx, ry = float(rx.get()[0]), float(ry.get()[0])
            else:
                rx, ry = float(rx[0]), float(ry[0])

        if z is not None:
            return rx, ry, z_out
        return rx, ry

    def transform_buffers(
        self, x, y, z=None, *, direction="FORWARD", out_x=None, out_y=None,
        precision="auto", stream=None,
    ):
        """Zero-overhead transform for device-resident arrays.

        Designed for integration with vibeSpatial's OwnedGeometryArray.
        Skips scalar detection, dtype conversion, and array module inference.

        Parameters
        ----------
        x, y : cupy.ndarray or numpy.ndarray
            Coordinate arrays (fp64 storage per ADR-0002).
        z : cupy.ndarray or numpy.ndarray, optional
            Vertical coordinate array. Passed through unchanged.
        direction : str
            "FORWARD" or "INVERSE".
        out_x, out_y : cupy.ndarray or numpy.ndarray, optional
            Pre-allocated fp64 output arrays. Avoids allocation.
        precision : str
            "fp64" = full double precision.
            "fp32" = fp32 compute with fp64 I/O (ADR-0002 mixed precision).
            "auto" = fp64 (projection math is trig-dominated / SFU-bound).
        stream : cupy.cuda.Stream, optional
            CUDA stream for async kernel execution. Enables overlapping
            projection compute with data transfers in pipeline workloads.

        Returns
        -------
        tuple of arrays
            Transformed (out_x, out_y) or (out_x, out_y, z). Same objects if pre-allocated.
        """
        xp = get_array_module(x)

        if direction == "FORWARD":
            pipeline = self._pipeline
        else:
            if self._inv_pipeline is None:
                self._inv_pipeline = TransformPipeline(self._dst_params, self._src_params)
            pipeline = self._inv_pipeline

        rx, ry = pipeline.transform(
            x, y, xp, out_x=out_x, out_y=out_y, precision=precision, stream=stream,
        )
        if z is not None:
            return rx, ry, z
        return rx, ry
