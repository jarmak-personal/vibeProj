# Transcendental policy

`transcendentals` controls how GPU kernels evaluate functions such as sine,
cosine, `atan2`, and `asinh`. It is independent of `precision`: precision
selects the arithmetic representation, while the transcendental policy selects
an implementation of special functions within that representation.

The keyword is available on `transform()`, `transform_buffers()`,
`transform_chunked()`, `transform_bounds()`, `compile()`, and module-level
`warm_up()`. These entry points also accept an independent `precision` choice.
Neither choice is stored by `Transformer.from_crs()` or included in a
serialized transformer's state.

| Policy | Meaning |
|---|---|
| `"auto"` | Select a qualified implementation for the current device, transform, and workload size. Below a measured crossover it stays native. |
| `"native"` | Use CUDA/libdevice special functions. Adjacent sine and cosine calls may use native paired `sincos`; that is still native policy. |
| `"accelerated"` | Request a qualified accelerated implementation. If no qualified implementation covers the device, family, direction, precision, or domain, explicitly fall back to native. |

```python
from vibeproj import Transformer

t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Arithmetic precision and transcendental policy are separate choices.
x, y = t.transform_buffers(
    longitude,
    latitude,
    precision="fp64",
    transcendentals="accelerated",
)
```

`"accelerated"` means “use a qualified accelerated implementation where one
exists,” not “fail unless every operation is accelerated.” It is an explicit
override and ignores the automatic workload-size crossover, so it may be slower
than native for small arrays. Inspect the resolved decisions when it matters
whether a particular call actually accelerated:

```python
explanation = t.explain_strategy(
    transcendentals="auto",
    workload_size=1_000_000,
)
for decision in explanation.decisions:
    print(decision.family, decision.operation, decision.implementation_id)
    if decision.fallback:
        print("  native fallback:", decision.reason)
```

The implementation ID, not a kernel's private function name, is the stable
identifier to record in benchmark results. `list_transcendental_strategies()`
returns the immutable registry of known implementations.

Transform calls always resolve `auto` with the concrete input size. An
`explain_strategy()` or `compile()` call with no workload size describes or
precompiles the hardware-qualified variant without imposing the crossover;
pass `workload_size=` to make introspection match a planned transform exactly.

## Qualified hardware and coverage

The initial accelerated coverage is deliberately small:

| Implementation ID | Family and direction | Operations | Qualified automatic device | Auto minimum elements |
|---|---|---|---|---:|
| `native.libdevice` | Every fused family/direction and Helmert | CUDA native special math, including paired native `sincos` where implemented | All devices and CPU/no-GPU fallback | n/a |
| `helmert.fixed_q62` | Helmert datum shift | Bounded sine/cosine only; ECEF, square root, and `atan2` remain native fp64 | Ada `sm_89` consumer GPUs | 131,072 |
| `tmerc.forward.fixed_q62` | Forward UTM only | Bounded sine/cosine, the TM latitude correction, and bounded `asinh`; remaining math stays fp64 | Ada `sm_89` consumer GPUs | 256 |

On a qualified RTX 4090, `"auto"` resolves the specialized implementations at
or above the listed sizes; below them it resolves native. Explicit
`"accelerated"` can select the specialized implementation at any size. Generic
Transverse Mercator, inverse UTM,
all other projection families, unsupported precision combinations, unknown
devices, and inputs outside an implementation's guarded domain resolve or
fall back to `native.libdevice`.

H100, Hopper, and other datacenter GPUs remain native. Their much stronger
native fp64 throughput changes the performance trade-off, and acceleration
will not be enabled from RTX 4090 results alone. H100 support requires accuracy
and performance measurements collected on H100 hardware.

## Accuracy and guarded domains

Accelerated implementations preserve fp64 coordinate I/O and are qualified
against native policy. The current release gates are:

- Q1.62 sine/cosine: finite angles in `[-pi, pi]`, maximum absolute error
  `7e-16` per output, and maximum `|sin² + cos² - 1| < 1e-15`.
- Helmert: longitude/latitude outside the Q1.62 domain, non-finite angles, and
  the near-pole height-recovery guard use native math. Qualified comparisons
  require `< 1e-8 m` maximum horizontal difference from native and, for 3-D,
  `< 2e-7 m` maximum height difference.
- Forward UTM: the accelerated path requires `|longitude - central_meridian|
  <= 0.06 rad`; its `asinh` series additionally requires `|x| <= 0.06`.
  The `asinh` approximation has maximum absolute error `2e-17`. The reframed
  `atan2` correction has `|delta| <= 9.01e-4` and maximum absolute angular
  error `2.3e-16 rad` over the full guard. (The tighter `6.9e-4` correction
  bound applies to normal UTM's `+/-3 degree` zone.) The complete projected
  result must differ from native by `< 1e-8 m`.
- Non-finite, out-of-domain, wide-TM, near-pole, and otherwise unsupported
  coordinates take the native branch per coordinate.

These are native-equivalence contracts for the implementation change. The
existing projection accuracy requirements against pyproj still apply; an
accelerated implementation may not spend that error budget merely because it
passes its native-equivalence bound.
