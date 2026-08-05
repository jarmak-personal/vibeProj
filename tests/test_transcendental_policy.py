"""Public API and host resolver tests for transcendental policies."""

from __future__ import annotations

import dataclasses
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import vibeproj
from vibeproj import DeviceCapability, Transformer
from vibeproj.transcendentals import (
    HELMERT_FIXED_Q62,
    HELMERT_FIXED_Q62_MIN_ELEMENTS,
    NATIVE_LIBDEVICE,
    TMERC_FIXED_Q62,
    TMERC_FIXED_Q62_MIN_ELEMENTS,
    TranscendentalOperation,
    _resolve_transcendental_strategy_cached,
    list_transcendental_strategies,
    resolve_transcendental_strategy,
)

ADA = DeviceCapability(
    backend="cuda",
    compute_capability=(8, 9),
    fp32_to_fp64_ratio=64,
    name="mock Ada",
    device_id=0,
)
HOPPER = DeviceCapability(
    backend="cuda",
    compute_capability=(9, 0),
    fp32_to_fp64_ratio=2,
    name="mock Hopper",
    device_id=0,
)
CPU = DeviceCapability(backend="cpu", name="CPU")


@pytest.mark.parametrize("policy", ["auto", "native", "accelerated"])
def test_cpu_transform_accepts_every_policy(policy):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    expected = transformer.transform([2.0], [49.0], transcendentals="native")
    actual = transformer.transform([2.0], [49.0], transcendentals=policy)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda transformer: transformer.transform(object(), object(), transcendentals="fast"),
        lambda transformer: transformer.transform_buffers(
            object(), object(), transcendentals="fast"
        ),
        lambda transformer: transformer.transform_chunked(
            object(), object(), transcendentals="fast"
        ),
        lambda transformer: transformer.transform_bounds(
            0.0, 0.0, 1.0, 1.0, transcendentals="fast"
        ),
        lambda transformer: transformer.compile(transcendentals="fast"),
        lambda transformer: transformer.explain_strategy(transcendentals="fast"),
    ],
)
def test_invalid_policy_fails_before_input_or_device_work(call):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.raises(ValueError, match="Invalid transcendentals policy.*fast"):
        call(transformer)


def test_invalid_warm_up_policy_fails_early():
    with pytest.raises(ValueError, match="Invalid transcendentals policy"):
        vibeproj.warm_up(transcendentals="fast")


def test_registry_is_immutable_and_contains_stable_ids():
    registry = list_transcendental_strategies()
    assert isinstance(registry, tuple)
    assert {entry.implementation_id for entry in registry} == {
        NATIVE_LIBDEVICE,
        HELMERT_FIXED_Q62,
        TMERC_FIXED_Q62,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        registry[0].family = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("operation", "domain", "threshold", "accelerated_id"),
    [
        (
            TranscendentalOperation.TMERC_FORWARD,
            "utm",
            TMERC_FIXED_Q62_MIN_ELEMENTS,
            TMERC_FIXED_Q62,
        ),
        (
            TranscendentalOperation.HELMERT,
            "global",
            HELMERT_FIXED_Q62_MIN_ELEMENTS,
            HELMERT_FIXED_Q62,
        ),
    ],
)
def test_auto_strategy_uses_exact_workload_crossover(operation, domain, threshold, accelerated_id):
    below = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA,
        domain=domain,
        workload_size=threshold - 1,
    )
    at = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA,
        domain=domain,
        workload_size=threshold,
    )
    explicit = resolve_transcendental_strategy(
        operation,
        "accelerated",
        device=ADA,
        domain=domain,
        workload_size=1,
    )
    unspecified = resolve_transcendental_strategy(
        operation,
        "auto",
        device=ADA,
        domain=domain,
        workload_size=None,
    )

    assert below.implementation_id == NATIVE_LIBDEVICE
    assert below.workload_size == threshold - 1
    assert at.implementation_id == accelerated_id
    assert at.workload_size == threshold
    assert explicit.implementation_id == accelerated_id
    assert unspecified.implementation_id == accelerated_id


@pytest.mark.parametrize(
    ("operation", "domain", "expected"),
    [
        (TranscendentalOperation.HELMERT, "global", HELMERT_FIXED_Q62),
        (TranscendentalOperation.TMERC_FORWARD, "utm", TMERC_FIXED_Q62),
    ],
)
def test_ada_auto_selects_qualified_implementations(operation, domain, expected):
    decision = resolve_transcendental_strategy(operation, device=ADA, domain=domain)
    assert decision.implementation_id == expected
    assert decision.fallback is False


@pytest.mark.parametrize("policy", ["auto", "native"])
def test_hopper_normally_remains_native(policy):
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT, policy, device=HOPPER
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is False
    assert "native" in decision.reason


def test_accelerated_unsupported_device_has_explicit_native_fallback():
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT, "accelerated", device=HOPPER
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert "fell back to native" in decision.reason


def test_accelerated_unsupported_domain_has_explicit_native_fallback():
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="global",
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert "not accuracy-qualified" in decision.reason


@pytest.mark.parametrize("precision", ["fp32", "ds"])
def test_tmerc_acceleration_does_not_change_or_override_compute_precision(precision):
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="utm",
        precision=precision,
    )
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert precision in decision.reason


def test_non_tmerc_accelerated_request_is_observable_native_fallback():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    explanation = transformer.explain_strategy(transcendentals="accelerated", device=ADA)
    assert explanation.requested_policy == "accelerated"
    assert len(explanation.decisions) == 1
    decision = explanation.decisions[0]
    assert decision.operation is TranscendentalOperation.PROJECTION
    assert decision.domain == "webmerc.forward"
    assert decision.implementation_id == NATIVE_LIBDEVICE
    assert decision.fallback is True
    assert "no accuracy-qualified" in decision.reason


def test_utm_explanation_resolves_by_policy_without_array_materialization():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    accelerated = transformer.explain_strategy(transcendentals="accelerated", device=ADA)
    native = transformer.explain_strategy(transcendentals="native", device=ADA)
    assert [decision.implementation_id for decision in accelerated.decisions] == [TMERC_FIXED_Q62]
    assert [decision.implementation_id for decision in native.decisions] == [NATIVE_LIBDEVICE]


def test_proj_to_proj_inverse_explanation_includes_each_projection_stage():
    transformer = Transformer.from_crs("EPSG:32631", "EPSG:3857")
    explanation = transformer.explain_strategy(
        transcendentals="accelerated", direction="INVERSE", device=ADA
    )
    assert [decision.domain for decision in explanation.decisions] == [
        "webmerc.inverse",
        "utm",
    ]
    assert explanation.decisions[0].fallback is True
    assert explanation.decisions[1].implementation_id == TMERC_FIXED_Q62


def test_helmert_explanation_includes_projection_and_datum_stage():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    assert transformer._helmert is not None
    explanation = transformer.explain_strategy(transcendentals="accelerated", device=ADA)
    assert [decision.operation for decision in explanation.decisions] == [
        TranscendentalOperation.HELMERT,
        TranscendentalOperation.TMERC_FORWARD,
    ]
    assert explanation.decisions[0].implementation_id == HELMERT_FIXED_Q62
    assert explanation.decisions[1].implementation_id == NATIVE_LIBDEVICE
    assert explanation.decisions[1].fallback is True


@pytest.mark.parametrize(
    ("precision", "policy", "expected"),
    [
        ("auto", "auto", TMERC_FIXED_Q62),
        ("fp64", "accelerated", TMERC_FIXED_Q62),
        ("fp32", "accelerated", NATIVE_LIBDEVICE),
        ("fp64", "native", NATIVE_LIBDEVICE),
    ],
)
def test_compile_resolves_exact_tmerc_cache_variant(monkeypatch, precision, policy, expected):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    observed = []

    def capture(*args, **kwargs):
        observed.append(kwargs["transcendental_impl"])

    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr("vibeproj.fused_kernels.compile_kernels", capture)
    transformer.compile(precision=precision, transcendentals=policy)
    assert observed == [expected]


def test_compile_resolves_helmert_and_non_utm_fallback_independently(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    projection_impl = []
    helmert_impl = []
    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", lambda: ADA)
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_kernels",
        lambda *args, **kwargs: projection_impl.append(kwargs["transcendental_impl"]),
    )
    monkeypatch.setattr(
        "vibeproj.fused_kernels.compile_helmert_kernel",
        lambda **kwargs: helmert_impl.append(kwargs["transcendental_impl"]),
    )
    transformer.compile(transcendentals="accelerated")
    assert projection_impl == [NATIVE_LIBDEVICE]
    assert helmert_impl == [HELMERT_FIXED_Q62]


def test_transform_propagates_precision_and_policy_independently(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = transformer.transform([2.0], [49.0])
    observed = []
    original = transformer._pipeline.transform

    def capture(*args, **kwargs):
        observed.append((kwargs["precision"], kwargs["transcendentals"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformer._pipeline, "transform", capture)
    transformer.transform([2.0], [49.0], precision="fp32", transcendentals="native")
    assert observed == [("fp32", "native")]

    transformer.transform(x, y, direction="INVERSE", transcendentals="accelerated")
    assert transformer._inv_pipeline is not None


def test_precision_and_policy_reach_projection_dispatch(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    observed = []

    def unavailable_fused(*args, **kwargs):
        observed.append((kwargs["precision"], kwargs["transcendentals"]))
        return None

    monkeypatch.setattr("vibeproj.pipeline._try_fused", unavailable_fused)
    transformer.transform([2.0], [49.0], precision="ds", transcendentals="accelerated")
    assert observed == [("ds", "accelerated")]


def test_inverse_lazy_pipeline_propagates_precision_and_policy(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = transformer.transform([2.0], [49.0])
    transformer.transform(x, y, direction="INVERSE")
    assert transformer._inv_pipeline is not None
    observed = []
    original = transformer._inv_pipeline.transform

    def capture(*args, **kwargs):
        observed.append((kwargs["precision"], kwargs["transcendentals"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformer._inv_pipeline, "transform", capture)
    transformer.transform(
        x,
        y,
        direction="INVERSE",
        precision="fp64",
        transcendentals="native",
    )
    assert observed == [("fp64", "native")]


def test_pickle_roundtrip_has_no_persisted_per_call_policy():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    restored = pickle.loads(pickle.dumps(transformer))
    assert "transcendentals" not in restored.__getstate__()
    explanation = restored.explain_strategy(transcendentals="native", device=CPU)
    assert all(decision.implementation_id == NATIVE_LIBDEVICE for decision in explanation.decisions)


@pytest.mark.parametrize(
    "call",
    [
        lambda transformer: transformer.transform(object(), object(), precision="half"),
        lambda transformer: transformer.transform_buffers(object(), object(), precision="half"),
        lambda transformer: transformer.transform_chunked(object(), object(), precision="half"),
        lambda transformer: transformer.transform_bounds(0.0, 0.0, 1.0, 1.0, precision="half"),
        lambda transformer: transformer.compile(precision="half"),
        lambda transformer: transformer.explain_strategy(precision="half"),
    ],
)
def test_invalid_precision_fails_before_input_or_device_work(call):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    with pytest.raises(ValueError, match="Invalid precision.*half"):
        call(transformer)


def test_invalid_warm_up_precision_fails_early():
    with pytest.raises(ValueError, match="Invalid precision"):
        vibeproj.warm_up(precision="half")


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("EPSG:4326", "EPSG:4326"),
        ("EPSG:4326", "EPSG:4277"),
        ("EPSG:4326", "EPSG:27700"),
        ("EPSG:27700", "EPSG:32631"),
        ("EPSG:4267", "EPSG:4269"),
    ],
)
def test_transform_buffers_returns_every_supplied_output_buffer(source, target):
    transformer = Transformer.from_crs(source, target)
    if source == "EPSG:27700":
        x = np.array([470000.0])
        y = np.array([235000.0])
    else:
        x = np.array([-1.0])
        y = np.array([52.0])
    z = np.array([10.0])
    expected = transformer.transform_buffers(x, y, z)
    out_x = np.empty_like(x)
    out_y = np.empty_like(y)
    out_z = np.empty_like(z)
    actual = transformer.transform_buffers(x, y, z, out_x=out_x, out_y=out_y, out_z=out_z)
    assert actual[0] is out_x
    assert actual[1] is out_y
    assert actual[2] is out_z
    np.testing.assert_allclose(actual, expected)


def test_longlat_helmert_receives_final_axis_mapped_buffers(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4277")
    out_x = np.empty(1)
    out_y = np.empty(1)
    out_z = np.empty(1)
    observed = {}

    def fake_shift(lat, lon, helmert, xp, **kwargs):
        observed.update(kwargs)
        kwargs["out_lat"][:] = lat
        kwargs["out_lon"][:] = lon
        kwargs["out_h"][:] = kwargs["h"]
        return kwargs["out_lat"], kwargs["out_lon"], kwargs["out_h"]

    monkeypatch.setattr("vibeproj.pipeline._apply_datum_shift", fake_shift)
    result = transformer.transform_buffers(
        np.array([-1.0]),
        np.array([52.0]),
        np.array([10.0]),
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
    )
    # always_xy=True means public x=lon/y=lat, while Helmert uses lat/lon.
    assert observed["out_lat"] is out_y
    assert observed["out_lon"] is out_x
    assert observed["out_h"] is out_z
    assert result[0] is out_x
    assert result[1] is out_y
    assert result[2] is out_z


def test_strategy_decisions_are_cached_by_full_context(monkeypatch):
    import vibeproj.transcendentals as transcendental_module

    _resolve_transcendental_strategy_cached.cache_clear()
    original = transcendental_module._implementation
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(transcendental_module, "_implementation", counted)
    first = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="utm",
        precision="fp64",
    )
    second = resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=ADA,
        domain="utm",
        precision="fp64",
    )
    assert second is first
    assert len(calls) == 2  # native + accelerated, built only on the first call

    resolve_transcendental_strategy(
        TranscendentalOperation.TMERC_FORWARD,
        "accelerated",
        device=HOPPER,
        domain="utm",
        precision="fp64",
    )
    assert len(calls) == 4


class _FakeDeviceContext:
    def __init__(self, device_id):
        self.id = device_id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeStream:
    def __init__(self, ptr, device_id=-1):
        self.ptr = ptr
        self.device_id = device_id
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.entered = False
        return False


class _FakeEvent:
    def __init__(self, *, disable_timing=True):
        self.complete = False
        self.stream = None

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.complete


class _FakeCupy:
    __name__ = "cupy"
    float64 = np.float64

    def __init__(self):
        self.device_id = 0
        self.stream = _FakeStream(0)
        self.empty_calls = []
        self.events = []

        def new_event(**kwargs):
            event = _FakeEvent(**kwargs)
            self.events.append(event)
            return event

        self.cuda = SimpleNamespace(
            Device=lambda device_id: _FakeDeviceContext(device_id),
            Event=new_event,
            get_current_stream=lambda: self.stream,
            runtime=SimpleNamespace(getDevice=lambda: self.device_id),
        )

    def empty(self, size, dtype):
        self.empty_calls.append((self.device_id, size, dtype))
        return np.empty(size, dtype=dtype)


def test_scratch_cache_is_grow_only_and_keyed_by_device_and_stream():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()

    def coordinate(size, device_id):
        return SimpleNamespace(size=size, device=SimpleNamespace(id=device_id))

    stream_a = _FakeStream(11)
    with transformer._transform_scratch_context(
        coordinate(8, 0), fake_cp, transformer._pipeline, stream_a
    ) as first:
        assert first is not None
    assert len(fake_cp.empty_calls) == 4
    with transformer._transform_scratch_context(
        coordinate(4, 0), fake_cp, transformer._pipeline, stream_a
    ) as reused:
        assert reused is first
    assert len(fake_cp.empty_calls) == 4

    with transformer._transform_scratch_context(
        coordinate(8, 0), fake_cp, transformer._pipeline, _FakeStream(12)
    ):
        pass
    with transformer._transform_scratch_context(
        coordinate(8, 1), fake_cp, transformer._pipeline, stream_a
    ):
        pass
    assert len(fake_cp.empty_calls) == 12
    assert set(transformer._scratch_slots) == {(0, 11), (0, 12), (1, 11)}


def test_alternating_chunk_streams_allocate_scratch_only_during_warmup():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()
    streams = (_FakeStream(21), _FakeStream(22))
    coordinate = SimpleNamespace(size=8, device=SimpleNamespace(id=0))

    for chunk_index in range(8):
        with transformer._transform_scratch_context(
            coordinate,
            fake_cp,
            transformer._pipeline,
            streams[chunk_index % 2],
        ) as scratch:
            assert scratch is not None
    assert len(fake_cp.empty_calls) == 8  # four arrays for each stream, once


def test_scratch_growth_retires_inflight_generation_until_event_completes():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()
    stream = _FakeStream(31)

    def coordinate(size):
        return SimpleNamespace(size=size, device=SimpleNamespace(id=0))

    with transformer._transform_scratch_context(
        coordinate(8), fake_cp, transformer._pipeline, stream
    ) as first:
        pass
    with transformer._transform_scratch_context(
        coordinate(32), fake_cp, transformer._pipeline, stream
    ):
        pass

    assert any(scratch is first for scratch, _event in transformer._scratch_retired)
    fake_cp.events[0].complete = True
    with transformer._scratch_slots_lock:
        transformer._prune_scratch_retired_locked()
    assert all(scratch is not first for scratch, _event in transformer._scratch_retired)


def test_scratch_cache_is_bounded_across_many_streams_and_prunes_completed_events():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    fake_cp = _FakeCupy()
    coordinate = SimpleNamespace(size=8, device=SimpleNamespace(id=0))

    for pointer in range(100, 120):
        with transformer._transform_scratch_context(
            coordinate, fake_cp, transformer._pipeline, _FakeStream(pointer)
        ):
            pass

    assert len(transformer._scratch_slots) == 8
    assert len(transformer._scratch_retired) == 12
    for event in fake_cp.events:
        event.complete = True
    with transformer._scratch_slots_lock:
        transformer._prune_scratch_retired_locked()
    assert transformer._scratch_retired == []


def test_chunk_device_buffers_are_cached_per_cuda_device(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    fake_cp.device_id = 0
    device_zero = transformer._get_dev_buffers(8)
    assert transformer._get_dev_buffers(4) is device_zero
    fake_cp.device_id = 1
    device_one = transformer._get_dev_buffers(8)
    assert device_one is not device_zero
    assert set(transformer._device_buffer_cache) == {0, 1}


def test_transform_buffers_resolves_one_immutable_context_from_input_device(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    coordinate = SimpleNamespace(size=32, device=SimpleNamespace(id=7))
    observed = []
    detections = []

    monkeypatch.setattr("vibeproj.transformer.get_array_module", lambda value: fake_cp)

    def fake_detect(xp=None, *, device_id=None):
        detections.append((xp, device_id))
        return dataclasses.replace(ADA, device_id=device_id)

    def fake_transform(pipeline_self, x, y, xp, **kwargs):
        observed.append(kwargs["execution_context"])
        return kwargs["out_x"], kwargs["out_y"]

    monkeypatch.setattr("vibeproj.transcendentals.detect_device_capability", fake_detect)
    monkeypatch.setattr("vibeproj.pipeline.TransformPipeline.transform", fake_transform)
    out_x = np.empty(32)
    out_y = np.empty(32)
    transformer.transform_buffers(coordinate, coordinate, out_x=out_x, out_y=out_y)

    assert detections == [(fake_cp, 7)]
    assert len(observed) == 1
    assert observed[0].device.device_id == 7
    assert observed[0].workload_size == 32
    with pytest.raises(dataclasses.FrozenInstanceError):
        observed[0].workload_size = 1


def test_transform_buffers_rejects_stream_from_another_device(monkeypatch):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    coordinate = SimpleNamespace(size=1, device=SimpleNamespace(id=0))
    monkeypatch.setattr("vibeproj.transformer.get_array_module", lambda value: fake_cp)

    with pytest.raises(ValueError, match="stream device 1.*input device 0"):
        transformer.transform_buffers(
            coordinate,
            coordinate,
            out_x=np.empty(1),
            out_y=np.empty(1),
            stream=_FakeStream(123, device_id=1),
        )


@pytest.mark.parametrize(
    ("source", "target", "direction"),
    [
        ("EPSG:4326", "EPSG:4326", "FORWARD"),
        ("EPSG:4326", "EPSG:3857", "FORWARD"),
        ("EPSG:4326", "EPSG:3857", "INVERSE"),
        ("EPSG:32631", "EPSG:3857", "FORWARD"),
        ("EPSG:4326", "EPSG:4277", "FORWARD"),
        ("EPSG:4267", "EPSG:4269", "FORWARD"),
    ],
)
def test_transform_buffers_enqueues_complete_pipeline_in_supplied_stream(
    monkeypatch, source, target, direction
):
    transformer = Transformer.from_crs(source, target)
    fake_cp = _FakeCupy()
    fake_cp.ndarray = np.ndarray
    stream = _FakeStream(99)
    coordinate = SimpleNamespace(size=1, device=SimpleNamespace(id=0))
    out_x = np.empty(1)
    out_y = np.empty(1)
    observed = []

    monkeypatch.setattr("vibeproj.transformer.get_array_module", lambda value: fake_cp)

    def fake_transform(pipeline_self, x, y, xp, **kwargs):
        observed.append((pipeline_self.mode, stream.entered, kwargs["stream"] is stream))
        return kwargs["out_x"], kwargs["out_y"]

    monkeypatch.setattr("vibeproj.pipeline.TransformPipeline.transform", fake_transform)
    transformer.transform_buffers(
        coordinate,
        coordinate,
        direction=direction,
        out_x=out_x,
        out_y=out_y,
        stream=stream,
    )
    assert observed[0][1:] == (True, True)


@pytest.mark.parametrize(
    ("source", "target", "direction"),
    [
        ("EPSG:4326", "EPSG:4277", "FORWARD"),  # Helmert-only longlat
        ("EPSG:4326", "EPSG:27700", "FORWARD"),
        ("EPSG:4326", "EPSG:27700", "INVERSE"),
        ("EPSG:27700", "EPSG:32631", "FORWARD"),
    ],
)
def test_gpu_preallocated_correction_paths_call_no_empty_after_warmup(
    monkeypatch, source, target, direction
):
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except (ImportError, RuntimeError, OSError):
        pytest.skip("CuPy/CUDA unavailable")

    transformer = Transformer.from_crs(source, target)
    z = cp.asarray([10.0], dtype=cp.float64)
    if source == "EPSG:27700":
        x = cp.asarray([470000.0], dtype=cp.float64)
        y = cp.asarray([235000.0], dtype=cp.float64)
    else:
        x = cp.asarray([-1.0], dtype=cp.float64)
        y = cp.asarray([52.0], dtype=cp.float64)
    if direction == "INVERSE":
        x, y, z = transformer.transform_buffers(x, y, z)

    input_snapshots = tuple(value.copy() for value in (x, y, z))
    outputs = tuple(cp.empty_like(value) for value in (x, y, z))
    transformer.transform_buffers(
        x,
        y,
        z,
        direction=direction,
        out_x=outputs[0],
        out_y=outputs[1],
        out_z=outputs[2],
    )
    cp.cuda.get_current_stream().synchronize()

    empty_calls = []
    original_empty = cp.empty

    def counted_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(cp, "empty", counted_empty)
    result = transformer.transform_buffers(
        x,
        y,
        z,
        direction=direction,
        out_x=outputs[0],
        out_y=outputs[1],
        out_z=outputs[2],
    )
    cp.cuda.get_current_stream().synchronize()

    assert empty_calls == []
    assert all(actual is expected for actual, expected in zip(result, outputs, strict=True))
    for actual, expected in zip((x, y, z), input_snapshots, strict=True):
        cp.testing.assert_array_equal(actual, expected)


def _cupy_or_skip():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
        return cp
    except (ImportError, RuntimeError, OSError):
        pytest.skip("CuPy/CUDA unavailable")


def test_gpu_svd_only_forward_honors_out_z_on_null_stream():
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4267", "EPSG:26918")
    assert transformer._svd_correction is not None
    assert transformer._helmert is None
    x = cp.asarray([-75.0, -74.5], dtype=cp.float64)
    y = cp.asarray([40.0, 40.5], dtype=cp.float64)
    z = cp.asarray([10.0, 20.0], dtype=cp.float64)
    outputs = tuple(cp.empty_like(value) for value in (x, y, z))

    result = transformer.transform_buffers(
        x,
        y,
        z,
        out_x=outputs[0],
        out_y=outputs[1],
        out_z=outputs[2],
        stream=cp.cuda.Stream.null,
    )
    cp.cuda.Stream.null.synchronize()

    assert all(actual is expected for actual, expected in zip(result, outputs, strict=True))
    cp.testing.assert_array_equal(outputs[2], z)


def test_gpu_chunk_workspace_reuses_allocations_and_is_thread_safe(monkeypatch):
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    inputs = [
        (np.linspace(-4.0, -1.0, 2048), np.linspace(50.0, 54.0, 2048)),
        (np.linspace(-3.0, 0.0, 2048), np.linspace(51.0, 55.0, 2048)),
    ]
    expected = [transformer.transform_chunked(*pair, chunk_size=256) for pair in inputs]
    workspace = transformer._chunk_workspaces[int(cp.cuda.runtime.getDevice())]
    stream_ids = tuple(id(stream) for stream in workspace.streams)

    empty_calls = []
    original_empty = cp.empty

    def counted_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(cp, "empty", counted_empty)
    repeated = transformer.transform_chunked(*inputs[0], chunk_size=256)
    assert empty_calls == []
    assert tuple(id(stream) for stream in workspace.streams) == stream_ids
    for actual, reference in zip(repeated, expected[0], strict=True):
        np.testing.assert_array_equal(actual, reference)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transformer.transform_chunked, *pair, chunk_size=256) for pair in inputs
        ]
        concurrent = [future.result() for future in futures]
    for result, reference in zip(concurrent, expected, strict=True):
        for actual, wanted in zip(result, reference, strict=True):
            np.testing.assert_array_equal(actual, wanted)


def test_gpu_scratch_cache_stays_bounded_after_many_streams_and_growth():
    cp = _cupy_or_skip()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    streams = []
    for index in range(20):
        stream = cp.cuda.Stream(non_blocking=True)
        streams.append(stream)
        size = 8 << (index % 5)
        x = cp.full(size, -1.0, dtype=cp.float64)
        y = cp.full(size, 52.0, dtype=cp.float64)
        transformer.transform_buffers(
            x,
            y,
            out_x=cp.empty_like(x),
            out_y=cp.empty_like(y),
            stream=stream,
        )

    assert len(transformer._scratch_slots) <= 8
    for stream in streams:
        stream.synchronize()
    with transformer._scratch_slots_lock:
        transformer._prune_scratch_retired_locked()
    assert len(transformer._scratch_slots) <= 8
    assert transformer._scratch_retired == []
