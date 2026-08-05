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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from vibeproj.crs import build_datum_operation_plan, resolve_transform
from vibeproj.pipeline import TransformPipeline, TransformScratch
from vibeproj.runtime import get_array_module, to_device

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from vibeproj.crs import CRSInput
    from vibeproj.transcendentals import (
        DeviceCapability,
        StrategyExplanation,
        TranscendentalPolicy,
    )


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


@dataclasses.dataclass(slots=True)
class _ScratchSlot:
    """Per-device/per-stream scratch protected during host-side enqueue."""

    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    capacity: int = 0
    scratch: TransformScratch | None = None
    completion_event: Any = None
    stream: Any = None
    leases: int = 0
    last_used: int = 0


@dataclasses.dataclass(slots=True)
class _ChunkWorkspace:
    """Serialized, persistent double-buffered resources for one CUDA device."""

    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    streams: tuple[Any, Any] | None = None
    pinned_mems: list[Any] = dataclasses.field(default_factory=list)
    pinned_slots: list[dict[str, Any]] | None = None
    pinned_size: int = 0
    pinned_has_z: bool = False
    device_slots: list[dict[str, Any]] | None = None
    device_size: int = 0
    device_has_z: bool = False


_MAX_SCRATCH_SLOTS_PER_DEVICE = 8


class Transformer:
    """GPU-accelerated coordinate transformer.

    - transform(x, y) where x=lon, y=lat for geographic CRS (always_xy=True default)
    - direction="FORWARD" or "INVERSE"
    - Accepts scalars, lists, numpy arrays, or cupy arrays

    When CuPy is available and inputs are on GPU, transforms run on GPU.
    Otherwise falls back to NumPy on CPU.

    Thread Safety
    -------------
    Transformer instances are safe to share across threads. NVRTC compilation
    is serialized on first use. Device-resident calls lease stream-specific
    scratch as needed. ``transform_chunked()`` calls sharing one Transformer
    and CUDA device serialize around that device's persistent staging
    workspace while retaining two-stream overlap within each call.
    """

    def __init__(
        self,
        crs_from: CRSInput,
        crs_to: CRSInput,
        *,
        always_xy: bool = True,
        datum_shift: Literal["accurate", "fast"] = "accurate",
        epoch: float | None = None,
    ) -> None:
        """Create a Transformer from source and target CRS.

        Prefer the :meth:`from_crs` static method, which has the same
        signature and is consistent with the pyproj API.

        Parameters
        ----------
        crs_from, crs_to :
            EPSG integer (4326), string ("EPSG:4326"), or pyproj CRS.
        always_xy : bool, default True
            If True, input/output order is (x, y) = (lon, lat).
        datum_shift : {"accurate", "fast"}, default "accurate"
            "accurate" uses 15-param time-dependent Helmert + SVD corrections
            when available. "fast" uses base 7-param Helmert only.
        epoch : float, optional
            Evaluation epoch for time-dependent Helmert (e.g. 2024.0).
        """
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

        # Datum shift detection and Helmert extraction.  Datum/reference-frame
        # changes are CRS operations, not ellipsoid-size comparisons.
        self._datum_plan = build_datum_operation_plan(src_crs, dst_crs)
        self._cross_datum = self._datum_plan.cross_datum
        self._helmert = None
        self._helmert_has_rates = False
        self._epoch_applied = False
        should_try_helmert = self._cross_datum and (
            self._datum_plan.best_has_helmert
            or (
                self._datum_plan.has_available_helmert
                and not self._datum_plan.uses_authoritative_noop
            )
        )
        if should_try_helmert:
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

        # SVD-compressed datum correction lookup
        self._svd_correction = None
        self._svd_negate = False
        if self._cross_datum:
            from vibeproj._datum_corrections import (
                get_datum_correction,
                is_reverse_direction,
            )

            # Build authority code strings for lookup
            src_epsg_code = src_crs.to_epsg()
            dst_epsg_code = dst_crs.to_epsg()
            if src_epsg_code is not None and dst_epsg_code is not None:
                src_auth = f"EPSG:{src_epsg_code}"
                dst_auth = f"EPSG:{dst_epsg_code}"
                # Check geographic CRS codes too (strip projection)
                src_geo = src_crs.geodetic_crs
                dst_geo = dst_crs.geodetic_crs
                src_geo_epsg = src_geo.to_epsg() if src_geo else None
                dst_geo_epsg = dst_geo.to_epsg() if dst_geo else None

                # Try projected codes first, then geographic codes
                correction = get_datum_correction(src_auth, dst_auth)
                if correction is None and src_geo_epsg and dst_geo_epsg:
                    src_geo_auth = f"EPSG:{src_geo_epsg}"
                    dst_geo_auth = f"EPSG:{dst_geo_epsg}"
                    correction = get_datum_correction(src_geo_auth, dst_geo_auth)
                    if correction is not None:
                        self._svd_negate = is_reverse_direction(src_geo_auth, dst_geo_auth)
                else:
                    if correction is not None:
                        self._svd_negate = is_reverse_direction(src_auth, dst_auth)

                self._svd_correction = correction

        if (
            self._cross_datum
            and self._helmert is None
            and self._svd_correction is None
            and self._datum_plan.warning_level == "unsupported"
        ):
            # No Helmert or SVD correction available (grid-only datum shift)
            src_datum = self._datum_plan.source_datum or "unknown"
            dst_datum = self._datum_plan.target_datum or "unknown"
            grid_msg = ""
            if self._datum_plan.missing_grids:
                grid_msg = f" Missing grids: {', '.join(self._datum_plan.missing_grids)}."
            warnings.warn(
                f"Source and destination CRS use different datums "
                f"({src_datum} \u2192 {dst_datum}). No Helmert transformation "
                f"available \u2014 grid-based shifts (NTv2) are not yet supported. "
                f"Results may differ from pyproj by meters to hundreds of meters."
                f"{grid_msg}",
                RuntimeWarning,
                stacklevel=2,
            )

        # always_xy=True forces (x, y) = (lon, lat) / (easting, northing) order,
        # matching shapely/geopandas conventions regardless of CRS native axis order.
        if always_xy:
            src_params = dataclasses.replace(src_params, north_first=False)
            dst_params = dataclasses.replace(dst_params, north_first=False)

        self._pipeline = TransformPipeline(
            src_params,
            dst_params,
            helmert=self._helmert,
            svd_correction=self._svd_correction,
            svd_negate=self._svd_negate,
        )
        self._src_params = src_params
        self._dst_params = dst_params
        # Build the inverse pipeline lazily (protected by lock for thread safety)
        self._inv_pipeline: TransformPipeline | None = None
        self._inv_pipeline_lock = threading.Lock()
        self._scratch_slots: dict[tuple[int, int], _ScratchSlot] = {}
        self._scratch_slots_lock = threading.RLock()
        self._scratch_retired: list[tuple[TransformScratch, Any]] = []
        self._scratch_clock = 0
        self._chunk_workspaces: dict[int, _ChunkWorkspace] = {}
        self._chunk_workspaces_lock = threading.RLock()
        # Kept as a compatibility view for callers/tests that inspect the
        # historical per-device device-buffer cache.
        self._device_buffer_cache: dict[int, dict[str, Any]] = {}
        self._device_buffer_cache_lock = self._chunk_workspaces_lock

    @staticmethod
    def from_crs(
        crs_from: CRSInput,
        crs_to: CRSInput,
        *,
        always_xy: bool = True,
        datum_shift: Literal["accurate", "fast"] = "accurate",
        epoch: float | None = None,
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

    def explain_strategy(
        self,
        *,
        transcendentals: TranscendentalPolicy = "auto",
        precision: str = "auto",
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        device: DeviceCapability | None = None,
        workload_size: int | None = None,
    ) -> StrategyExplanation:
        """Explain transcendental decisions for one transform direction.

        The result is immutable and contains a decision for every projection
        stage plus Helmert when active.  Passing *device* makes hardware policy
        tests deterministic; otherwise the current device is detected lazily.
        ``precision`` accepts ``"auto"``, ``"fp64"``, ``"fp32"``, or ``"ds"``;
        ``transcendentals`` accepts ``"auto"``, ``"native"``, or
        ``"accelerated"``. Pass ``workload_size`` to preview size-aware
        ``"auto"`` selection. ``None`` represents compilation/planning without
        a concrete array size and selects an otherwise-qualified accelerated
        implementation. This method does not materialize coordinate arrays.
        """
        from vibeproj.transcendentals import (
            StrategyExplanation,
            TranscendentalOperation,
            detect_device_capability,
            normalize_compute_precision,
            normalize_transcendental_policy,
            resolve_transcendental_strategy,
        )

        precision = normalize_compute_precision(precision)
        requested = normalize_transcendental_policy(transcendentals)
        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")
        if device is None:
            device = detect_device_capability()

        if direction == "FORWARD":
            pipeline = self._pipeline
        else:
            inv_helmert = self._helmert.inverted() if self._helmert else None
            pipeline = TransformPipeline(
                self._dst_params,
                self._src_params,
                helmert=inv_helmert,
                svd_correction=self._svd_correction,
                svd_negate=not self._svd_negate,
            )

        contexts: list[tuple[Any, str]] = []

        def add_projection(name: str, stage_direction: str, computed: dict) -> None:
            if name == "tmerc" and stage_direction == "forward":
                domain = "utm" if computed.get("is_utm", False) else "global"
                contexts.append((TranscendentalOperation.TMERC_FORWARD, domain))
            else:
                contexts.append((TranscendentalOperation.PROJECTION, f"{name}.{stage_direction}"))

        helmert_context = (TranscendentalOperation.HELMERT, "global")
        if pipeline.mode == "forward":
            if self._helmert is not None:
                contexts.append(helmert_context)
            add_projection(pipeline.projection.name, "forward", pipeline.computed)
        elif pipeline.mode == "inverse":
            add_projection(pipeline.projection.name, "inverse", pipeline.computed)
            if self._helmert is not None:
                contexts.append(helmert_context)
        elif pipeline.mode == "proj_to_proj":
            add_projection(pipeline.src_projection.name, "inverse", pipeline.src_computed)
            if self._helmert is not None:
                contexts.append(helmert_context)
            add_projection(pipeline.dst_projection.name, "forward", pipeline.dst_computed)
        elif self._helmert is not None:
            contexts.append(helmert_context)

        decisions = tuple(
            resolve_transcendental_strategy(
                operation,
                requested,
                device=device,
                domain=domain,
                precision=precision,
                workload_size=workload_size,
            )
            for operation, domain in contexts
        )
        return StrategyExplanation(
            requested_policy=requested,
            direction=direction,
            device=device,
            workload_size=workload_size,
            decisions=decisions,
        )

    @property
    def accuracy(self) -> str:
        """Rough accuracy classification for this transform.

        Returns
        -------
        str
            "sub-millimeter" — same datum, projection math only.
            "sub-5cm" — cross-datum with SVD-compressed grid correction.
            "sub-decimeter" — cross-datum with 15-param time-dependent Helmert
            evaluated at a known epoch.
            "sub-meter" — cross-datum with 7-param Helmert.
            "datum no-op (... m PROJ accuracy)" — PROJ selected an explicit
            no-op datum operation with meter-level expected accuracy.
            "degraded — no datum shift applied" — different datums; results
            may differ from pyproj by meters to hundreds of meters.
        """
        if self._svd_correction is not None:
            return "sub-5cm"
        if self._cross_datum and self._helmert is None:
            if (
                self._datum_plan.uses_authoritative_noop
                and self._datum_plan.expected_accuracy_m is not None
            ):
                acc = f"{self._datum_plan.expected_accuracy_m:g}"
                return f"datum no-op ({acc} m PROJ accuracy)"
            return "degraded \u2014 no datum shift applied"
        if self._cross_datum and self._helmert is not None:
            if self._epoch_applied:
                return "sub-decimeter"
            return "sub-meter"
        return "sub-millimeter"

    def compile(
        self,
        *,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> None:
        """Pre-compile fused NVRTC kernels for this transformer.

        ``precision`` accepts ``"auto"``, ``"fp64"``, ``"fp32"``, or ``"ds"``.
        ``transcendentals`` accepts ``"auto"``, ``"native"``, or
        ``"accelerated"`` independently. Compilation has no concrete workload
        size, so ``"auto"`` selects an otherwise-qualified implementation
        without applying runtime crossover thresholds. This front-loads kernel
        compilation only; first-use stream scratch and chunk staging workspaces
        are still allocated lazily. No-op if CuPy is unavailable.
        """
        from vibeproj.transcendentals import (
            normalize_compute_precision,
            normalize_transcendental_policy,
        )

        precision = normalize_compute_precision(precision)
        transcendentals = normalize_transcendental_policy(transcendentals)
        try:
            from vibeproj.fused_kernels import compile_kernels
        except ImportError:
            return

        pipeline = self._pipeline
        tmerc_domain = None
        if pipeline.mode == "forward" and pipeline.projection.name == "tmerc":
            tmerc_domain = "utm" if pipeline.computed.get("is_utm", False) else "global"
        elif pipeline.mode == "proj_to_proj" and pipeline.dst_projection.name == "tmerc":
            tmerc_domain = "utm" if pipeline.dst_computed.get("is_utm", False) else "global"
        from vibeproj.transcendentals import (
            NATIVE_LIBDEVICE,
            TranscendentalOperation,
            detect_device_capability,
            resolve_transcendental_strategy,
        )

        device = detect_device_capability()
        tmerc_impl = NATIVE_LIBDEVICE
        if tmerc_domain is not None:
            tmerc_impl = resolve_transcendental_strategy(
                TranscendentalOperation.TMERC_FORWARD,
                transcendentals,
                device=device,
                domain=tmerc_domain,
                precision=precision,
            ).implementation_id
        if pipeline.mode in ("forward", "inverse"):
            compile_kernels(
                [pipeline.projection.name],
                precision=precision,
                transcendental_impl=tmerc_impl,
            )
        elif pipeline.mode == "proj_to_proj":
            names = [pipeline.src_projection.name, pipeline.dst_projection.name]
            compile_kernels(names, precision=precision, transcendental_impl=tmerc_impl)
        if self._helmert is not None:
            from vibeproj.fused_kernels import compile_helmert_kernel

            helmert_impl = resolve_transcendental_strategy(
                TranscendentalOperation.HELMERT,
                transcendentals,
                device=device,
            ).implementation_id
            compile_helmert_kernel(transcendental_impl=helmert_impl)
        if self._svd_correction is not None:
            from vibeproj.fused_kernels import compile_svd_kernel

            compile_svd_kernel()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "crs_from": self._crs_from_input,
            "crs_to": self._crs_to_input,
            "always_xy": self._always_xy,
            "datum_shift": self._datum_shift,
            "epoch": self._epoch,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(  # type: ignore[misc]
            state["crs_from"],
            state["crs_to"],
            always_xy=state["always_xy"],
            datum_shift=state.get("datum_shift", "accurate"),
            epoch=state.get("epoch"),
        )

    def _pipeline_for_direction(self, direction: str) -> TransformPipeline:
        """Return the direction pipeline, constructing the inverse once."""
        if direction == "FORWARD":
            return self._pipeline
        if self._inv_pipeline is None:
            with self._inv_pipeline_lock:
                if self._inv_pipeline is None:
                    inv_helmert = self._helmert.inverted() if self._helmert else None
                    self._inv_pipeline = TransformPipeline(
                        self._dst_params,
                        self._src_params,
                        helmert=inv_helmert,
                        svd_correction=self._svd_correction,
                        svd_negate=not self._svd_negate,
                    )
        return self._inv_pipeline

    @staticmethod
    def _input_device_capability(x, xp):
        """Resolve capability from the input array without changing devices."""
        from vibeproj.transcendentals import detect_device_capability

        is_cupy = getattr(xp, "__name__", "").split(".", 1)[0] == "cupy"
        device_id = int(x.device.id) if is_cupy else None
        return detect_device_capability(xp, device_id=device_id)

    def _build_execution_context(
        self,
        pipeline: TransformPipeline,
        x,
        xp,
        *,
        precision: str,
        transcendentals: str,
        workload_size: int,
    ):
        return pipeline.build_execution_context(
            precision=precision,
            transcendentals=transcendentals,
            device=self._input_device_capability(x, xp),
            workload_size=workload_size,
            _normalized=True,
        )

    @overload
    def transform(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: None = None,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        *,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[Any, Any]: ...

    @overload
    def transform(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        *,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[Any, Any, Any]: ...

    def transform(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | None = None,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        *,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[Any, Any] | tuple[Any, Any, Any]:
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
        precision : {"auto", "fp64", "fp32", "ds"}
            Numeric compute precision for fused GPU kernels.
        transcendentals : {"auto", "native", "accelerated"}
            Transcendental implementation policy, independent of *precision*.
            ``"auto"`` uses the concrete input size and device capability when
            applying qualified acceleration crossover thresholds.

        Returns
        -------
        tuple of arrays (or scalars if scalar input)
            Transformed (x, y) or (x, y, z) if z was provided.
        """
        from vibeproj.transcendentals import (
            normalize_compute_precision,
            normalize_transcendental_policy,
        )

        precision = normalize_compute_precision(precision)
        transcendentals = normalize_transcendental_policy(transcendentals)
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
            if not xp.issubdtype(x.dtype, xp.floating):  # type: ignore[union-attr]
                x = x.astype(xp.float64)  # type: ignore[union-attr]
            if not xp.issubdtype(y.dtype, xp.floating):  # type: ignore[union-attr]
                y = y.astype(xp.float64)  # type: ignore[union-attr]

        # Prepare z for pipeline: only route through Helmert when active
        z_pipeline = None  # z to pass into pipeline (None = no z transform)
        z_passthrough = z  # z to return as-is when not routing through pipeline
        if z is not None and self._helmert is not None:
            if is_scalar:
                z_pipeline = (
                    xp.asarray([z], dtype="f8")
                    if isinstance(z, (int, float))
                    else xp.asarray([float(z)], dtype="f8")  # type: ignore[arg-type]
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

        pipeline = self._pipeline_for_direction(direction)
        execution_context = self._build_execution_context(
            pipeline,
            x,
            xp,
            precision=precision,
            transcendentals=transcendentals,
            workload_size=int(x.size),
        )
        is_cupy = getattr(xp, "__name__", "").split(".", 1)[0] == "cupy"
        if is_cupy:
            with xp.cuda.Device(int(x.device.id)):
                result = pipeline.transform(
                    x,
                    y,
                    xp,
                    z=z_pipeline,
                    precision=precision,
                    transcendentals=transcendentals,
                    execution_context=execution_context,
                )
        else:
            result = pipeline.transform(
                x,
                y,
                xp,
                z=z_pipeline,
                precision=precision,
                transcendentals=transcendentals,
                execution_context=execution_context,
            )

        if z_pipeline is not None:
            rx, ry, z_out = result
        else:
            rx, ry = result
            z_out = z_passthrough

        # Check for non-finite output values.
        # For GPU arrays, skip this check — it forces an implicit D→H sync
        # (xp.any() returns a device scalar whose truthiness triggers .get()).
        # The device-resident transform_buffers() path already skips this.
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

    @overload
    def transform_buffers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: None = None,
        *,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        out_x: ArrayLike | None = None,
        out_y: ArrayLike | None = None,
        out_z: ArrayLike | None = None,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
        stream: Any = None,
    ) -> tuple[Any, Any]: ...

    @overload
    def transform_buffers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        *,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        out_x: ArrayLike | None = None,
        out_y: ArrayLike | None = None,
        out_z: ArrayLike | None = None,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
        stream: Any = None,
    ) -> tuple[Any, Any, Any]: ...

    def transform_buffers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | None = None,
        *,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        out_x: ArrayLike | None = None,
        out_y: ArrayLike | None = None,
        out_z: ArrayLike | None = None,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
        stream: Any = None,
    ) -> tuple[Any, Any] | tuple[Any, Any, Any]:
        """Transform device-resident arrays with optional pre-allocated outputs.

        Designed for integration with vibeSpatial's OwnedGeometryArray.
        It skips scalar detection and dtype conversion. With pre-allocated
        outputs, repeated same-size GPU calls on a warmed, cached stream avoid
        output and correction-scratch allocation. First use, cache growth, and
        kernel compilation can still allocate.

        Parameters
        ----------
        x, y : cupy.ndarray or numpy.ndarray
            Coordinate arrays (fp64 storage per ADR-0002).
        z : cupy.ndarray or numpy.ndarray, optional
            Ellipsoidal height array. Transformed through Helmert when a datum
            shift is active; passed through unchanged by projection/SVD-only
            paths.
        direction : str
            "FORWARD" or "INVERSE".
        out_x, out_y : cupy.ndarray or numpy.ndarray, optional
            Pre-allocated fp64 output arrays. Avoids allocation.
        out_z : cupy.ndarray or numpy.ndarray, optional
            Pre-allocated fp64 output height array. Whenever ``z`` is provided,
            this buffer is honored and returned for transformed or passthrough
            height paths.
        precision : {"auto", "fp64", "fp32", "ds"}
            Numeric compute precision for fused GPU kernels.
        transcendentals : {"auto", "native", "accelerated"}
            Hardware-aware transcendental implementation policy. This is
            independent of numeric compute *precision*. ``"auto"`` uses the
            array size when applying qualified acceleration thresholds.
        stream : cupy.cuda.Stream, optional
            CUDA stream for asynchronous execution. ``None`` uses the current
            stream on the input array's device; the legacy null stream is
            supported. Explicit streams from another device are rejected. The
            caller owns synchronization.

        Returns
        -------
        tuple of arrays
            Transformed ``(out_x, out_y)`` or ``(out_x, out_y, z_out)``.
            Supplied output buffers are returned identically.
        """
        from vibeproj.transcendentals import (
            normalize_compute_precision,
            normalize_transcendental_policy,
        )

        precision = normalize_compute_precision(precision)
        transcendentals = normalize_transcendental_policy(transcendentals)
        xp = get_array_module(x)
        is_cupy = getattr(xp, "__name__", "").split(".", 1)[0] == "cupy"
        if is_cupy:
            stream = self._normalize_cuda_stream(x, xp, stream)

        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        pipeline = self._pipeline_for_direction(direction)
        execution_context = self._build_execution_context(
            pipeline,
            x,
            xp,
            precision=precision,
            transcendentals=transcendentals,
            workload_size=int(x.size),
        )

        with self._transform_scratch_context(x, xp, pipeline, stream) as scratch:
            kwargs = dict(
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
            if stream is not None and is_cupy:
                with xp.cuda.Device(int(x.device.id)), stream:
                    return pipeline.transform(x, y, xp, **kwargs)
            return pipeline.transform(x, y, xp, **kwargs)

    @contextmanager
    def _transform_scratch_context(self, x, xp, pipeline, stream):
        """Lease bounded, event-safe scratch for one device/stream."""
        is_cupy = getattr(xp, "__name__", "").split(".", 1)[0] == "cupy"
        if not is_cupy or not pipeline.needs_scratch:
            yield None
            return

        device_id = int(x.device.id)
        with xp.cuda.Device(device_id):
            active_stream = stream if stream is not None else xp.cuda.get_current_stream()
            key = (device_id, int(active_stream.ptr))
            with self._scratch_slots_lock:
                self._prune_scratch_retired_locked()
                slot = self._scratch_slots.get(key)
                admitted = slot is not None
                if slot is None:
                    slot = _ScratchSlot()
                    device_slots = [
                        (candidate_key, candidate)
                        for candidate_key, candidate in self._scratch_slots.items()
                        if candidate_key[0] == device_id
                    ]
                    if len(device_slots) >= _MAX_SCRATCH_SLOTS_PER_DEVICE:
                        evictable = [item for item in device_slots if item[1].leases == 0]
                        if evictable:
                            evict_key, evicted = min(evictable, key=lambda item: item[1].last_used)
                            del self._scratch_slots[evict_key]
                            if evicted.scratch is not None:
                                if self._event_complete(evicted.completion_event):
                                    evicted.scratch = None
                                else:
                                    self._scratch_retired.append(
                                        (evicted.scratch, evicted.completion_event)
                                    )
                    if sum(k[0] == device_id for k in self._scratch_slots) < (
                        _MAX_SCRATCH_SLOTS_PER_DEVICE
                    ):
                        self._scratch_slots[key] = slot
                        admitted = True
                self._scratch_clock += 1
                slot.last_used = self._scratch_clock
                slot.stream = active_stream
                slot.leases += 1

            try:
                with slot.lock:
                    size = int(x.size)
                    if slot.scratch is None or slot.capacity < size:
                        capacity = max(size, max(1, slot.capacity * 2))
                        if slot.scratch is not None:
                            if not self._event_complete(slot.completion_event):
                                with self._scratch_slots_lock:
                                    self._scratch_retired.append(
                                        (slot.scratch, slot.completion_event)
                                    )
                        slot.scratch = TransformScratch(
                            xp.empty(capacity, dtype=xp.float64),
                            xp.empty(capacity, dtype=xp.float64),
                            xp.empty(capacity, dtype=xp.float64),
                            xp.empty(capacity, dtype=xp.float64),
                        )
                        slot.capacity = capacity
                        slot.completion_event = None
                    try:
                        yield slot.scratch
                    finally:
                        event_type = getattr(xp.cuda, "Event", None)
                        if event_type is not None:
                            event = event_type(disable_timing=True)
                            event.record(active_stream)
                            slot.completion_event = event
            finally:
                with self._scratch_slots_lock:
                    slot.leases -= 1
                    self._scratch_clock += 1
                    slot.last_used = self._scratch_clock
                    if not admitted and slot.scratch is not None:
                        if not self._event_complete(slot.completion_event):
                            self._scratch_retired.append((slot.scratch, slot.completion_event))
                    self._prune_scratch_retired_locked()

    @staticmethod
    def _event_complete(event) -> bool:
        if event is None:
            return True
        try:
            query = getattr(event, "query", None)
            if query is not None:
                return bool(query())
            return bool(event.done)
        except (AttributeError, RuntimeError):
            return False

    def _prune_scratch_retired_locked(self) -> None:
        self._scratch_retired[:] = [
            item for item in self._scratch_retired if not self._event_complete(item[1])
        ]

    @staticmethod
    def _normalize_cuda_stream(x, xp, stream):
        """Bind null/current streams to the input device and reject mismatches."""
        device_id = int(x.device.id)
        with xp.cuda.Device(device_id):
            if stream is None:
                return xp.cuda.get_current_stream()
            stream_device = int(getattr(stream, "device_id", -1))
            if stream_device >= 0 and stream_device != device_id:
                raise ValueError(
                    f"CUDA stream device {stream_device} does not match input device {device_id}."
                )
            return stream

    def _get_chunk_workspace(self, device_id: int) -> _ChunkWorkspace:
        with self._chunk_workspaces_lock:
            workspace = self._chunk_workspaces.get(device_id)
            if workspace is None:
                workspace = _ChunkWorkspace()
                self._chunk_workspaces[device_id] = workspace
            return workspace

    def _get_pinned_buffers(self, buf_size, *, chunk_z=False, workspace=None):
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

        if workspace is None:
            device_id = int(cp.cuda.runtime.getDevice())
            workspace = self._get_chunk_workspace(device_id)
        need_alloc = (
            workspace.pinned_slots is None
            or workspace.pinned_size < buf_size
            or (chunk_z and not workspace.pinned_has_z)
        )
        if need_alloc:
            # Grow-only: never shrink size, never drop z capability
            buf_size = max(buf_size, workspace.pinned_size)
            chunk_z = chunk_z or workspace.pinned_has_z
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
            workspace.pinned_mems = pinned_mems
            workspace.pinned_slots = slots
            workspace.pinned_size = buf_size
            workspace.pinned_has_z = chunk_z
        return workspace.pinned_slots

    def _get_dev_buffers(self, buf_size, *, chunk_z=False, workspace=None, device_id=None):
        """Return pooled device buffer pairs for 2 stream slots.

        Each slot has: "x", "y", "ox", "oy" (and "z", "oz" when chunk_z).
        Cached on the Transformer instance with grow-only semantics.
        """
        import cupy as cp

        if device_id is None:
            device_id = int(cp.cuda.runtime.getDevice())
        if workspace is None:
            workspace = self._get_chunk_workspace(device_id)
        with self._device_buffer_cache_lock:
            need_alloc = (
                workspace.device_slots is None
                or workspace.device_size < buf_size
                or (chunk_z and not workspace.device_has_z)
            )
            if need_alloc:
                capacity = max(buf_size, workspace.device_size)
                has_z = chunk_z or workspace.device_has_z
                with cp.cuda.Device(device_id):
                    slots = []
                    for _ in range(2):
                        slot = {
                            "x": cp.empty(capacity, dtype=cp.float64),
                            "y": cp.empty(capacity, dtype=cp.float64),
                            "ox": cp.empty(capacity, dtype=cp.float64),
                            "oy": cp.empty(capacity, dtype=cp.float64),
                        }
                        if has_z:
                            slot["z"] = cp.empty(capacity, dtype=cp.float64)
                            slot["oz"] = cp.empty(capacity, dtype=cp.float64)
                        slots.append(slot)
                workspace.device_slots = slots
                workspace.device_size = capacity
                workspace.device_has_z = has_z
            self._device_buffer_cache[device_id] = {
                "size": workspace.device_size,
                "has_z": workspace.device_has_z,
                "slots": workspace.device_slots,
            }
            return workspace.device_slots

    def _transform_chunked_gpu(
        self,
        cp,
        workspace: _ChunkWorkspace,
        device_id: int,
        pipeline: TransformPipeline,
        x: np.ndarray,
        y: np.ndarray,
        z_arr: np.ndarray | None,
        *,
        chunk_z: bool,
        chunk_size: int,
        buf_size: int,
        precision: str,
        transcendentals: str,
    ):
        """Execute one serialized call using persistent two-stream resources."""
        from vibeproj.transcendentals import detect_device_capability

        if workspace.streams is None:
            workspace.streams = (
                cp.cuda.Stream(non_blocking=True),
                cp.cuda.Stream(non_blocking=True),
            )
        streams = workspace.streams
        pin_slots = self._get_pinned_buffers(buf_size, chunk_z=chunk_z, workspace=workspace)
        dev_bufs = self._get_dev_buffers(
            buf_size,
            chunk_z=chunk_z,
            workspace=workspace,
            device_id=device_id,
        )
        execution_context = pipeline.build_execution_context(
            precision=precision,
            transcendentals=transcendentals,
            device=detect_device_capability(cp, device_id=device_id),
            workload_size=buf_size,
            _normalized=True,
        )

        n = x.size
        out_x = np.empty(n, dtype=np.float64)
        out_y = np.empty(n, dtype=np.float64)
        out_z = np.empty(n, dtype=np.float64) if chunk_z else None
        pending: list[tuple[int, int, int] | None] = [None, None]
        pending_z_ref = [None, None]

        def flush_slot(slot_index: int) -> None:
            item = pending[slot_index]
            if item is None:
                return
            start, end, size = item
            pin = pin_slots[slot_index]
            streams[slot_index].synchronize()
            out_x[start:end] = pin["out_x"][:size]
            out_y[start:end] = pin["out_y"][:size]
            if chunk_z:
                out_z[start:end] = pin["out_z"][:size]
            pending[slot_index] = None
            pending_z_ref[slot_index] = None

        for chunk_index, start in enumerate(range(0, n, chunk_size)):
            end = min(start + chunk_size, n)
            size = end - start
            slot_index = chunk_index % 2
            stream = streams[slot_index]
            device = dev_bufs[slot_index]
            pin = pin_slots[slot_index]
            flush_slot(slot_index)

            pin["in_x"][:size] = x[start:end]
            pin["in_y"][:size] = y[start:end]
            if chunk_z:
                pin["in_z"][:size] = z_arr[start:end]

            nbytes = size * np.dtype(np.float64).itemsize
            with stream:
                cp.cuda.runtime.memcpyAsync(
                    device["x"].data.ptr, pin["in_x"].ctypes.data, nbytes, 1, stream.ptr
                )
                cp.cuda.runtime.memcpyAsync(
                    device["y"].data.ptr, pin["in_y"].ctypes.data, nbytes, 1, stream.ptr
                )
                if chunk_z:
                    cp.cuda.runtime.memcpyAsync(
                        device["z"].data.ptr,
                        pin["in_z"].ctypes.data,
                        nbytes,
                        1,
                        stream.ptr,
                    )

                with self._transform_scratch_context(
                    device["x"][:size], cp, pipeline, stream
                ) as scratch:
                    result = pipeline.transform(
                        device["x"][:size],
                        device["y"][:size],
                        cp,
                        z=device["z"][:size] if chunk_z else None,
                        out_x=device["ox"][:size],
                        out_y=device["oy"][:size],
                        out_z=device["oz"][:size] if chunk_z else None,
                        precision=precision,
                        transcendentals=transcendentals,
                        scratch=scratch,
                        execution_context=execution_context,
                        stream=stream,
                    )

                cp.cuda.runtime.memcpyAsync(
                    pin["out_x"].ctypes.data, device["ox"].data.ptr, nbytes, 2, stream.ptr
                )
                cp.cuda.runtime.memcpyAsync(
                    pin["out_y"].ctypes.data, device["oy"].data.ptr, nbytes, 2, stream.ptr
                )
                if chunk_z:
                    cp.cuda.runtime.memcpyAsync(
                        pin["out_z"].ctypes.data, result[2].data.ptr, nbytes, 2, stream.ptr
                    )

            pending[slot_index] = (start, end, size)
            if chunk_z:
                pending_z_ref[slot_index] = result[2]

        flush_slot(0)
        flush_slot(1)
        if z_arr is not None:
            return (out_x, out_y, out_z) if chunk_z else (out_x, out_y, z_arr)
        return out_x, out_y

    @overload
    def transform_chunked(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: None = None,
        *,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        chunk_size: int = 1_000_000,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def transform_chunked(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        *,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        chunk_size: int = 1_000_000,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def transform_chunked(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | None = None,
        *,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        chunk_size: int = 1_000_000,
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform large host-resident arrays in GPU-sized chunks.

        Uses a double-buffered pipeline with pinned host memory and two CUDA
        streams to overlap transfers with GPU compute. Each Transformer keeps
        a grow-only workspace per CUDA device containing persistent streams,
        pinned buffers, and device buffers. Calls sharing that Transformer and
        device serialize around the complete workspace lifecycle; the two
        streams still overlap work within a call. Workspace and correction
        scratch allocation occurs on first use or growth and is reused by
        subsequent same-size calls.

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
            but reduce per-chunk overhead. Size-aware ``"auto"`` dispatch uses
            the planned buffer size once and reuses that decision for every
            chunk, including a smaller final chunk.
        precision : {"auto", "fp64", "fp32", "ds"}
            Numeric compute precision for fused GPU kernels.
        transcendentals : {"auto", "native", "accelerated"}
            Hardware-aware transcendental implementation policy, independent
            of *precision*.

        Returns
        -------
        tuple of numpy.ndarray
            Transformed (x, y) or (x, y, z) on the host.
        """
        from vibeproj.transcendentals import (
            normalize_compute_precision,
            normalize_transcendental_policy,
        )

        precision = normalize_compute_precision(precision)
        transcendentals = normalize_transcendental_policy(transcendentals)
        try:
            import cupy as cp
        except ImportError:
            return self.transform(  # type: ignore[arg-type,misc]
                x,
                y,
                z=z,
                direction=direction,
                precision=precision,
                transcendentals=transcendentals,
            )

        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.size

        if n == 0:
            result = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
            return (*result, np.asarray(z, dtype=np.float64)) if z is not None else result

        pipeline = self._pipeline_for_direction(direction)

        # Determine if z needs to be chunked through Helmert
        chunk_z = z is not None and self._helmert is not None
        z_arr = np.asarray(z, dtype=np.float64) if z is not None else None

        buf_size = min(chunk_size, n)
        device_id = int(cp.cuda.runtime.getDevice())
        workspace = self._get_chunk_workspace(device_id)
        with workspace.lock, cp.cuda.Device(device_id):
            return self._transform_chunked_gpu(
                cp,
                workspace,
                device_id,
                pipeline,
                x,
                y,
                z_arr,
                chunk_z=chunk_z,
                chunk_size=chunk_size,
                buf_size=buf_size,
                precision=precision,
                transcendentals=transcendentals,
            )

    def transform_bounds(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        *,
        densify_pts: int = 21,
        direction: Literal["FORWARD", "INVERSE"] = "FORWARD",
        precision: str = "auto",
        transcendentals: TranscendentalPolicy = "auto",
    ) -> tuple[float, float, float, float]:
        """Transform a bounding box, densifying edges to handle projection curvature.

        Densifies the four edges of the input bounding box, transforms all
        points, and returns the min/max envelope of the transformed result.
        This correctly handles non-linear projection distortion that would
        be missed by transforming only the four corners.

        Parameters
        ----------
        left, bottom, right, top : float
            Bounding box coordinates.  With ``always_xy=True`` (default):
            left/right are x (longitude), bottom/top are y (latitude).
        densify_pts : int, default 21
            Number of additional intermediate points per edge (not counting
            corner endpoints).  Matches pyproj/GDAL convention: 0 means
            corners only, 21 (the default) adds 21 points between each pair
            of adjacent corners.  Clamped to a minimum of 0.
        direction : {"FORWARD", "INVERSE"}
            Transform direction.
        precision : {"auto", "fp64", "fp32", "ds"}
            Numeric compute precision forwarded to the densified transform.
        transcendentals : {"auto", "native", "accelerated"}
            Hardware-aware transcendental implementation policy, independent
            of *precision*. ``"auto"`` uses the densified point count.

        Returns
        -------
        tuple of four floats
            ``(left, bottom, right, top)`` of the transformed bounding box.

        Notes
        -----
        When the transformed result crosses the antimeridian (±180°),
        the returned ``left`` will be greater than ``right``
        (e.g. ``left=153, right=-162``), matching the pyproj convention.
        """
        from vibeproj.transcendentals import (
            normalize_compute_precision,
            normalize_transcendental_policy,
        )

        precision = normalize_compute_precision(precision)
        transcendentals = normalize_transcendental_policy(transcendentals)
        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        densify_pts = max(densify_pts, 0)

        # Use CuPy when available so the transform hits the fused GPU path.
        xp = get_array_module()

        # Total sample points per edge including both corner endpoints.
        # pyproj convention: densify_pts is the number of *additional*
        # intermediate points, so total = densify_pts + 2.
        pts_per_edge = densify_pts + 2

        # Densify the four edges.  Corner points are shared between adjacent
        # edges, so we exclude the last point on each edge to avoid
        # duplicates.
        #
        # Edge layout (n = pts_per_edge - 1 points per edge, last excluded):
        #   bottom: x varies left->right, y = bottom
        #   right:  x = right, y varies bottom->top
        #   top:    x varies right->left, y = top
        #   left:   x = left, y varies top->bottom

        n = pts_per_edge - 1  # points per edge excluding the closing corner
        total = 4 * n

        # Pre-allocate two contiguous arrays and fill slices directly to
        # avoid intermediate temporaries and concatenation.
        x_all = xp.empty(total, dtype=np.float64)
        y_all = xp.empty(total, dtype=np.float64)

        x_all[0:n] = xp.linspace(left, right, pts_per_edge)[:-1]
        y_all[0:n] = bottom

        x_all[n : 2 * n] = right
        y_all[n : 2 * n] = xp.linspace(bottom, top, pts_per_edge)[:-1]

        x_all[2 * n : 3 * n] = xp.linspace(right, left, pts_per_edge)[:-1]
        y_all[2 * n : 3 * n] = top

        x_all[3 * n : 4 * n] = left
        y_all[3 * n : 4 * n] = xp.linspace(top, bottom, pts_per_edge)[:-1]

        # Transform all edge points at once
        tx, ty = self.transform(
            x_all,
            y_all,
            direction=direction,
            precision=precision,
            transcendentals=transcendentals,
        )

        # Detect antimeridian crossing before filtering (needs edge structure).
        # Only relevant when the output CRS is geographic (longitudes).
        # Vectorized: reshape to (4 edges, n pts), diff along each edge.
        # abs(NaN) > 180 is False, so non-finite points are safely ignored.
        out_params = self._src_params if direction == "INVERSE" else self._dst_params
        crosses_antimeridian = False
        if out_params.projection_name == "longlat":
            diffs = xp.abs(xp.diff(tx.reshape(4, n), axis=1))
            crosses_antimeridian = bool(xp.any(diffs > 180.0))

        # Filter non-finite values (projections can produce NaN/inf for
        # out-of-domain coordinates)
        finite_mask = xp.isfinite(tx) & xp.isfinite(ty)
        tx = tx[finite_mask]
        ty = ty[finite_mask]

        if tx.size == 0:
            raise ValueError(
                "All transformed coordinates are non-finite. "
                "The input bounding box may be outside the projection's valid domain."
            )

        if crosses_antimeridian:
            # Shift longitudes to [0, 360) so min/max span the crossing correctly.
            # After shifting back, left > right signals the antimeridian crossing
            # (matching pyproj convention).  Note: -180 % 360 == 180, and we use
            # strict > 180 so ±180° stays as +180 (same geographic meridian).
            tx_shifted = tx % 360.0
            left_x = float(tx_shifted.min())
            right_x = float(tx_shifted.max())
            if left_x > 180.0:
                left_x -= 360.0
            if right_x > 180.0:
                right_x -= 360.0
            return left_x, float(ty.min()), right_x, float(ty.max())

        return float(tx.min()), float(ty.min()), float(tx.max()), float(ty.max())
