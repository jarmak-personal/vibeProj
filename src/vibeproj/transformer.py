"""High-level Transformer API — drop-in replacement for pyproj.Transformer.

Usage:
    from vibeproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform(lon, lat)           # always_xy=True (default)
    lon, lat = t.transform(x, y, direction="INVERSE")
"""

from __future__ import annotations

import dataclasses
import threading
import warnings

import numpy as np

from vibeproj.crs import resolve_transform
from vibeproj.pipeline import TransformPipeline
from vibeproj.runtime import get_array_module, to_device


class Transformer:
    """GPU-accelerated coordinate transformer.

    - transform(x, y) where x=lon, y=lat for geographic CRS (always_xy=True default)
    - direction="FORWARD" or "INVERSE"
    - Accepts scalars, lists, numpy arrays, or cupy arrays

    When CuPy is available and inputs are on GPU, transforms run on GPU.
    Otherwise falls back to NumPy on CPU.

    Thread Safety
    -------------
    Transformer instances are safe to share across threads. The internal
    kernel cache uses an RLock to serialize NVRTC compilation on first use;
    subsequent calls are lock-free. Call ``compile()`` at startup to
    front-load compilation if you want deterministic latency.
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

        # Datum shift detection and Helmert extraction
        src_ell = src_crs.ellipsoid
        dst_ell = dst_crs.ellipsoid
        self._cross_datum = (
            src_ell is not None
            and dst_ell is not None
            and abs(src_ell.semi_major_metre - dst_ell.semi_major_metre) > 1.0
        )
        self._helmert = None
        if self._cross_datum:
            from vibeproj.crs import extract_helmert

            self._helmert = extract_helmert(src_crs, dst_crs)
            if self._helmert is None:
                # No Helmert available (grid-only datum shift)
                src_datum = src_crs.datum.name if src_crs.datum else "unknown"
                dst_datum = dst_crs.datum.name if dst_crs.datum else "unknown"
                warnings.warn(
                    f"Source and destination CRS use different datums "
                    f"({src_datum} \u2192 {dst_datum}). No Helmert transformation "
                    f"available \u2014 grid-based shifts (NTv2) are not yet supported. "
                    f"Results may differ from pyproj by meters to hundreds of meters.",
                    stacklevel=2,
                )

        # always_xy=True forces (x, y) = (lon, lat) / (easting, northing) order,
        # matching shapely/geopandas conventions regardless of CRS native axis order.
        if always_xy:
            src_params = dataclasses.replace(src_params, north_first=False)
            dst_params = dataclasses.replace(dst_params, north_first=False)

        self._pipeline = TransformPipeline(src_params, dst_params, helmert=self._helmert)
        self._src_params = src_params
        self._dst_params = dst_params
        # Build the inverse pipeline lazily (protected by lock for thread safety)
        self._inv_pipeline = None
        self._inv_pipeline_lock = threading.Lock()

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

    @property
    def accuracy(self) -> str:
        """Rough accuracy classification for this transform.

        Returns
        -------
        str
            "sub-millimeter" — same datum, projection math only.
            "sub-meter" — nearly identical datums (e.g. WGS84/NAD83).
            "degraded — no datum shift applied" — different datums; results
            may differ from pyproj by meters to hundreds of meters.
        """
        if self._cross_datum and self._helmert is None:
            return "degraded \u2014 no datum shift applied"
        if self._cross_datum and self._helmert is not None:
            return "sub-meter"
        return "sub-millimeter"

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
        if self._helmert is not None:
            from vibeproj.fused_kernels import compile_helmert_kernel

            compile_helmert_kernel()

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
                with self._inv_pipeline_lock:
                    if self._inv_pipeline is None:
                        inv_helmert = self._helmert.inverted() if self._helmert else None
                        self._inv_pipeline = TransformPipeline(
                            self._dst_params, self._src_params, helmert=inv_helmert
                        )
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
        self,
        x,
        y,
        z=None,
        *,
        direction="FORWARD",
        out_x=None,
        out_y=None,
        precision="auto",
        stream=None,
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
                inv_helmert = self._helmert.inverted() if self._helmert else None
                self._inv_pipeline = TransformPipeline(
                    self._dst_params, self._src_params, helmert=inv_helmert
                )
            pipeline = self._inv_pipeline

        rx, ry = pipeline.transform(
            x,
            y,
            xp,
            out_x=out_x,
            out_y=out_y,
            precision=precision,
            stream=stream,
        )
        if z is not None:
            return rx, ry, z
        return rx, ry

    def transform_chunked(
        self,
        x,
        y,
        z=None,
        *,
        direction="FORWARD",
        chunk_size=1_000_000,
    ):
        """Transform large host-resident arrays in GPU-sized chunks.

        Transfers chunks to GPU, transforms via fused kernel, and copies
        results back to the host. Reuses pre-allocated device buffers across
        chunks to minimize allocation overhead.

        Falls back to CPU ``transform()`` when CuPy is not available.

        Parameters
        ----------
        x, y : array-like
            Input coordinate arrays (host memory).
        z : array-like, optional
            Vertical coordinate. Passed through unchanged.
        direction : str
            "FORWARD" or "INVERSE".
        chunk_size : int, default 1_000_000
            Coordinates per GPU chunk. Larger values use more GPU memory
            but reduce per-chunk overhead.

        Returns
        -------
        tuple of numpy.ndarray
            Transformed (x, y) or (x, y, z) on the host.
        """
        try:
            import cupy as cp
        except ImportError:
            return self.transform(x, y, z=z, direction=direction)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.size

        if n == 0:
            result = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
            return (*result, np.asarray(z, dtype=np.float64)) if z is not None else result

        # Resolve pipeline once
        if direction == "FORWARD":
            pipeline = self._pipeline
        else:
            if self._inv_pipeline is None:
                with self._inv_pipeline_lock:
                    if self._inv_pipeline is None:
                        inv_helmert = self._helmert.inverted() if self._helmert else None
                        self._inv_pipeline = TransformPipeline(
                            self._dst_params, self._src_params, helmert=inv_helmert
                        )
            pipeline = self._inv_pipeline

        out_x = np.empty(n, dtype=np.float64)
        out_y = np.empty(n, dtype=np.float64)

        # Pre-allocate device buffers (reused across chunks)
        buf_size = min(chunk_size, n)
        dev_x = cp.empty(buf_size, dtype=cp.float64)
        dev_y = cp.empty(buf_size, dtype=cp.float64)
        dev_ox = cp.empty(buf_size, dtype=cp.float64)
        dev_oy = cp.empty(buf_size, dtype=cp.float64)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            size = end - start

            # H2D: host numpy slice → device buffer
            dev_x[:size].set(x[start:end])
            dev_y[:size].set(y[start:end])

            # Transform on GPU (zero-alloc via pre-allocated output buffers)
            pipeline.transform(
                dev_x[:size],
                dev_y[:size],
                cp,
                out_x=dev_ox[:size],
                out_y=dev_oy[:size],
            )

            # D2H: device → host numpy slice
            out_x[start:end] = cp.asnumpy(dev_ox[:size])
            out_y[start:end] = cp.asnumpy(dev_oy[:size])

        if z is not None:
            return out_x, out_y, np.asarray(z, dtype=np.float64)
        return out_x, out_y
