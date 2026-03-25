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


def _resolve_epoch(user_epoch, src_crs):
    """Resolve the evaluation epoch for time-dependent Helmert.

    Priority: user-provided epoch > source CRS coordinate epoch > None.
    """
    if user_epoch is not None:
        return float(user_epoch)
    try:
        ce = src_crs.coordinate_epoch
        if ce is not None:
            return float(ce)
    except (AttributeError, TypeError):
        pass
    return None


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

    def __init__(self, crs_from, crs_to, *, always_xy=True, datum_shift="accurate", epoch=None):
        if datum_shift not in ("accurate", "fast"):
            raise ValueError(f"datum_shift must be 'accurate' or 'fast', got {datum_shift!r}")

        src_params, dst_params, src_crs, dst_crs = resolve_transform(crs_from, crs_to)

        # Store raw inputs for pickle serialization
        self._crs_from_input = crs_from
        self._crs_to_input = crs_to
        self._always_xy = always_xy
        self._datum_shift = datum_shift
        self._epoch = epoch

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
        self._helmert_has_rates = False
        self._epoch_applied = False
        if self._cross_datum:
            from vibeproj.crs import extract_helmert

            helmert_raw = extract_helmert(src_crs, dst_crs)
            if helmert_raw is not None:
                self._helmert_has_rates = helmert_raw.has_rates
                if datum_shift == "accurate" and helmert_raw.has_rates:
                    eval_epoch = _resolve_epoch(epoch, src_crs)
                    if eval_epoch is not None:
                        self._helmert = helmert_raw.at_epoch(eval_epoch)
                        self._epoch_applied = True
                    else:
                        self._helmert = helmert_raw
                else:
                    self._helmert = helmert_raw
            if self._helmert is None and helmert_raw is None:
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
    def from_crs(
        crs_from, crs_to, *, always_xy=True, datum_shift="accurate", epoch=None
    ) -> Transformer:
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
        datum_shift : str, default "accurate"
            "accurate" — use 15-parameter time-dependent Helmert when available,
            evaluating rate terms at the given *epoch*. Falls back to 7-parameter
            when no rates are present or no epoch can be resolved.
            "fast" — always use the base 7-parameter Helmert (ignores rate terms).
        epoch : float, optional
            Decimal year at which to evaluate the time-dependent Helmert
            (e.g. 2024.0). Only used when *datum_shift="accurate"*.
            If omitted, the source CRS coordinate epoch is used when available.
        """
        return Transformer(
            crs_from, crs_to, always_xy=always_xy, datum_shift=datum_shift, epoch=epoch
        )

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
            "sub-decimeter" — cross-datum with 15-param time-dependent Helmert
            evaluated at a known epoch.
            "sub-meter" — cross-datum with 7-param Helmert.
            "degraded — no datum shift applied" — different datums; results
            may differ from pyproj by meters to hundreds of meters.
        """
        if self._cross_datum and self._helmert is None:
            return "degraded \u2014 no datum shift applied"
        if self._cross_datum and self._helmert is not None:
            if self._epoch_applied:
                return "sub-decimeter"
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
            "datum_shift": self._datum_shift,
            "epoch": self._epoch,
        }

    def __setstate__(self, state):
        self.__init__(
            state["crs_from"],
            state["crs_to"],
            always_xy=state["always_xy"],
            datum_shift=state.get("datum_shift", "accurate"),
            epoch=state.get("epoch"),
        )

    def transform(self, x, y, z=None, direction="FORWARD"):
        """Transform coordinates.

        Parameters
        ----------
        x, y : scalar, list, numpy array, or cupy array
            Input coordinates. With always_xy=True (default): x=longitude, y=latitude
            for geographic CRS. With always_xy=False: native CRS axis order.
        z : scalar, list, numpy array, or cupy array, optional
            Ellipsoidal height in meters. When a Helmert datum shift is active,
            z is transformed through the ECEF intermediate (correctness fix).
            When no datum shift is needed, z is passed through unchanged.
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

        # Prepare z for pipeline: only route through Helmert when active
        z_pipeline = None  # z to pass into pipeline (None = no z transform)
        z_passthrough = z  # z to return as-is when not routing through pipeline
        if z is not None and self._helmert is not None:
            if is_scalar:
                z_pipeline = (
                    xp.asarray([z], dtype="f8")
                    if isinstance(z, (int, float))
                    else xp.asarray([float(z)], dtype="f8")
                )
            else:
                z_pipeline = to_device(z, xp)
                if not xp.issubdtype(z_pipeline.dtype, xp.floating):
                    z_pipeline = z_pipeline.astype(xp.float64)
            z_passthrough = None  # pipeline will return z
        elif z is not None and not is_scalar:
            z_passthrough = to_device(z, xp)
            if not xp.issubdtype(z_passthrough.dtype, xp.floating):
                z_passthrough = z_passthrough.astype(xp.float64)

        if direction == "FORWARD":
            result = self._pipeline.transform(x, y, xp, z=z_pipeline)
        else:
            if self._inv_pipeline is None:
                with self._inv_pipeline_lock:
                    if self._inv_pipeline is None:
                        inv_helmert = self._helmert.inverted() if self._helmert else None
                        self._inv_pipeline = TransformPipeline(
                            self._dst_params, self._src_params, helmert=inv_helmert
                        )
            result = self._inv_pipeline.transform(x, y, xp, z=z_pipeline)

        if z_pipeline is not None:
            rx, ry, z_out = result
        else:
            rx, ry = result
            z_out = z_passthrough

        # Check for non-finite output values.
        # For GPU arrays, skip this check — it forces an implicit D→H sync
        # (xp.any() returns a device scalar whose truthiness triggers .get()).
        # The transform_buffers() zero-copy path already skips this.
        # Only check on CPU (NumPy) where there is no sync cost.
        if xp is np and rx.size > 0 and (xp.any(~xp.isfinite(rx)) or xp.any(~xp.isfinite(ry))):
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
            if z_out is not None and hasattr(z_out, "__len__"):
                if hasattr(z_out, "get"):
                    z_out = float(z_out.get()[0])
                else:
                    z_out = float(z_out[0])

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
        out_z=None,
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
            Ellipsoidal height array. Transformed through Helmert when a datum
            shift is active; passed through unchanged otherwise.
        direction : str
            "FORWARD" or "INVERSE".
        out_x, out_y : cupy.ndarray or numpy.ndarray, optional
            Pre-allocated fp64 output arrays. Avoids allocation.
        out_z : cupy.ndarray or numpy.ndarray, optional
            Pre-allocated fp64 output height array. Only used when z is provided
            and a Helmert datum shift is active.
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
            Transformed (out_x, out_y) or (out_x, out_y, z_out). Same objects if pre-allocated.
        """
        xp = get_array_module(x)

        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        if direction == "FORWARD":
            pipeline = self._pipeline
        else:
            if self._inv_pipeline is None:
                inv_helmert = self._helmert.inverted() if self._helmert else None
                self._inv_pipeline = TransformPipeline(
                    self._dst_params, self._src_params, helmert=inv_helmert
                )
            pipeline = self._inv_pipeline

        # Route z through pipeline only when Helmert is active
        z_pipeline = z if (z is not None and self._helmert is not None) else None

        result = pipeline.transform(
            x,
            y,
            xp,
            z=z_pipeline,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            precision=precision,
            stream=stream,
        )
        if z_pipeline is not None:
            return result  # already (rx, ry, z_out)
        if z is not None:
            rx, ry = result
            return rx, ry, z
        return result

    def _get_pinned_buffers(self, buf_size, *, chunk_z=False):
        """Return pooled pinned-memory staging buffers for 2 stream slots.

        Returns a list of 2 dicts (one per stream slot).  Each dict has:
            "in_x", "in_y", "out_x", "out_y"  (and "in_z", "out_z" when chunk_z)
        All values are NumPy arrays backed by pinned (page-locked) host memory.
        Pinned buffers enable ``cudaMemcpyAsync`` for true overlap of H<->D
        transfers with GPU compute.

        Each stream slot gets its own pinned buffers to avoid data races:
        while stream A is still copying D->H into slot 0's output buffers,
        the CPU can safely write the next chunk into slot 1's input buffers.

        Buffers are cached on the Transformer instance and only grow (never
        shrink). Once z slots are allocated they are kept, avoiding thrash
        when alternating between 2D and 3D workloads.
        """
        import cupy as cp

        need_alloc = (
            not hasattr(self, "_pinned_bufs")
            or self._pinned_buf_size < buf_size
            or (chunk_z and not self._pinned_has_z)
        )
        if need_alloc:
            # Grow-only: never shrink size, never drop z capability
            buf_size = max(buf_size, getattr(self, "_pinned_buf_size", 0))
            chunk_z = chunk_z or getattr(self, "_pinned_has_z", False)
            nbytes = buf_size * np.dtype(np.float64).itemsize
            # 2 slots x (in_x, in_y, out_x, out_y) = 8 buffers
            # 2 slots x (in_x, in_y, in_z, out_x, out_y, out_z) = 12 buffers
            bufs_per_slot = 6 if chunk_z else 4
            n_bufs = 2 * bufs_per_slot
            pinned_mems = [cp.cuda.alloc_pinned_memory(nbytes) for _ in range(n_bufs)]
            arrs = [np.frombuffer(mem, dtype=np.float64, count=buf_size) for mem in pinned_mems]
            slots = []
            for s in range(2):
                base = s * bufs_per_slot
                slot = {
                    "in_x": arrs[base],
                    "in_y": arrs[base + 1],
                    "out_x": arrs[base + 2],
                    "out_y": arrs[base + 3],
                }
                if chunk_z:
                    slot["in_z"] = arrs[base + 4]
                    slot["out_z"] = arrs[base + 5]
                slots.append(slot)
            # Keep references to prevent GC of the underlying pinned memory
            self._pinned_mems = pinned_mems
            self._pinned_bufs = slots
            self._pinned_buf_size = buf_size
            self._pinned_has_z = chunk_z
        return self._pinned_bufs

    def _get_dev_buffers(self, buf_size, *, chunk_z=False):
        """Return pooled device buffer pairs for 2 stream slots.

        Each slot has: "x", "y", "ox", "oy" (and "z", "oz" when chunk_z).
        Cached on the Transformer instance with grow-only semantics.
        """
        import cupy as cp

        need_alloc = (
            not hasattr(self, "_dev_bufs")
            or self._dev_buf_size < buf_size
            or (chunk_z and not self._dev_has_z)
        )
        if need_alloc:
            buf_size = max(buf_size, getattr(self, "_dev_buf_size", 0))
            chunk_z = chunk_z or getattr(self, "_dev_has_z", False)
            slots = []
            for _ in range(2):
                slot = {
                    "x": cp.empty(buf_size, dtype=cp.float64),
                    "y": cp.empty(buf_size, dtype=cp.float64),
                    "ox": cp.empty(buf_size, dtype=cp.float64),
                    "oy": cp.empty(buf_size, dtype=cp.float64),
                }
                if chunk_z:
                    slot["z"] = cp.empty(buf_size, dtype=cp.float64)
                    slot["oz"] = cp.empty(buf_size, dtype=cp.float64)
                slots.append(slot)
            self._dev_bufs = slots
            self._dev_buf_size = buf_size
            self._dev_has_z = chunk_z
        return self._dev_bufs

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

        Uses a double-buffered pipeline with pinned host memory and 2 CUDA
        streams to overlap H<->D transfers with GPU compute.  Each stream
        owns a dedicated set of device buffers so chunk N can execute on
        stream A while chunk N+1 transfers on stream B.

        Falls back to CPU ``transform()`` when CuPy is not available.

        Parameters
        ----------
        x, y : array-like
            Input coordinate arrays (host memory).
        z : array-like, optional
            Ellipsoidal height. Transformed through Helmert when a datum
            shift is active; passed through unchanged otherwise.
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

        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

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

        # Determine if z needs to be chunked through Helmert
        chunk_z = z is not None and self._helmert is not None
        z_arr = np.asarray(z, dtype=np.float64) if z is not None else None

        # Final host output arrays
        out_x = np.empty(n, dtype=np.float64)
        out_y = np.empty(n, dtype=np.float64)
        out_z = np.empty(n, dtype=np.float64) if chunk_z else None

        # --- Double-buffered stream pipeline setup ---
        buf_size = min(chunk_size, n)

        # Pinned host staging buffers — 2 slots (pooled on the Transformer)
        pin_slots = self._get_pinned_buffers(buf_size, chunk_z=chunk_z)

        # 2 non-blocking CUDA streams
        streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]

        # 2 stream slots of device buffers (pooled on the Transformer)
        dev_bufs = self._get_dev_buffers(buf_size, chunk_z=chunk_z)

        # Build list of (start, end) for all chunks
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunks.append((start, end))

        # Track which stream slot holds pending D2H output that has not yet
        # been copied into the final host output arrays.  We defer the sync
        # and host-side copy so the GPU can run ahead.
        pending = [None, None]  # pending[slot] = (start, end, size) or None
        # Keep references to Helmert z output arrays alive until D2H completes.
        # The pipeline may return a freshly-allocated z array (from Helmert) that
        # would be freed when `result` is reassigned, while the D2H memcpyAsync
        # is still in-flight on the stream.
        pending_z_ref = [None, None]

        def _flush_slot(slot_idx):
            """Sync stream and copy pinned output into final host arrays."""
            if pending[slot_idx] is None:
                return
            p_start, p_end, p_size = pending[slot_idx]
            ps = pin_slots[slot_idx]
            streams[slot_idx].synchronize()
            out_x[p_start:p_end] = ps["out_x"][:p_size]
            out_y[p_start:p_end] = ps["out_y"][:p_size]
            if chunk_z:
                out_z[p_start:p_end] = ps["out_z"][:p_size]
            pending[slot_idx] = None
            pending_z_ref[slot_idx] = None

        for chunk_idx, (start, end) in enumerate(chunks):
            size = end - start
            slot = chunk_idx % 2
            stream = streams[slot]
            db = dev_bufs[slot]
            ps = pin_slots[slot]

            # Before reusing this slot's pinned buffers, flush any
            # pending D2H data from the previous iteration on this slot.
            _flush_slot(slot)

            # Stage input: CPU memcpy into pinned buffers (fast, no GPU involved)
            ps["in_x"][:size] = x[start:end]
            ps["in_y"][:size] = y[start:end]
            if chunk_z:
                ps["in_z"][:size] = z_arr[start:end]

            # --- All GPU work for this chunk on its dedicated stream ---
            nbytes = size * 8  # fp64 = 8 bytes per element
            with stream:
                # H2D async: pinned host -> device (stream-ordered).
                # Use memcpyAsync (kind=1 = cudaMemcpyHostToDevice) for
                # truly non-blocking transfers. CuPy's .set() can
                # synchronize the host thread even with pinned memory.
                cp.cuda.runtime.memcpyAsync(
                    db["x"].data.ptr, ps["in_x"].ctypes.data, nbytes, 1, stream.ptr
                )
                cp.cuda.runtime.memcpyAsync(
                    db["y"].data.ptr, ps["in_y"].ctypes.data, nbytes, 1, stream.ptr
                )
                if chunk_z:
                    cp.cuda.runtime.memcpyAsync(
                        db["z"].data.ptr, ps["in_z"].ctypes.data, nbytes, 1, stream.ptr
                    )

                # GPU compute (kernel launch is stream-ordered via the
                # stream= parameter propagated through pipeline.transform
                # into fused_transform / fused_helmert_shift).
                if chunk_z:
                    result = pipeline.transform(
                        db["x"][:size],
                        db["y"][:size],
                        cp,
                        z=db["z"][:size],
                        out_x=db["ox"][:size],
                        out_y=db["oy"][:size],
                        out_z=db["oz"][:size],
                        stream=stream,
                    )
                else:
                    result = pipeline.transform(
                        db["x"][:size],
                        db["y"][:size],
                        cp,
                        out_x=db["ox"][:size],
                        out_y=db["oy"][:size],
                        stream=stream,
                    )

                # D2H async: device -> pinned host (stream-ordered).
                # Use memcpyAsync (kind=2 = cudaMemcpyDeviceToHost) for
                # truly non-blocking transfers. CuPy's .get(out=) blocks
                # the host thread (~0.31ms), serializing the pipeline and
                # defeating double-buffering overlap.
                # The caller must synchronize the stream before reading
                # from the pinned buffer (_flush_slot handles this).
                #
                # For x/y, db["ox"]/db["oy"] are the pre-allocated output
                # buffers that the fused kernel writes into (via out_x/out_y).
                # For z, the pipeline may return a DIFFERENT device array
                # (Helmert output) rather than writing to db["oz"], so we
                # must use result[2] to get the actual z output pointer.
                cp.cuda.runtime.memcpyAsync(
                    ps["out_x"].ctypes.data, db["ox"].data.ptr, nbytes, 2, stream.ptr
                )
                cp.cuda.runtime.memcpyAsync(
                    ps["out_y"].ctypes.data, db["oy"].data.ptr, nbytes, 2, stream.ptr
                )
                if chunk_z:
                    cp.cuda.runtime.memcpyAsync(
                        ps["out_z"].ctypes.data, result[2].data.ptr, nbytes, 2, stream.ptr
                    )

            # Record that this slot has pending output.
            # Keep a reference to the z result to prevent GC while D2H is in-flight.
            pending[slot] = (start, end, size)
            if chunk_z:
                pending_z_ref[slot] = result[2]

        # Flush any remaining pending slots
        _flush_slot(0)
        _flush_slot(1)

        if z is not None:
            if chunk_z:
                return out_x, out_y, out_z
            return out_x, out_y, z_arr
        return out_x, out_y
