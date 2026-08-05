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
returns the immutable registry of known implementations. Projection entries
with a uniform scale guard expose it as
`accuracy.max_physical_scale_m`.

Transform calls always resolve `auto` with the concrete input size. An
`explain_strategy()` or `compile()` call with no workload size describes or
precompiles the hardware-qualified variant without imposing the crossover;
pass `workload_size=` to make introspection match a planned transform exactly.
Module-level `warm_up(["tmerc"])` has no CRS domain to inspect, so on a
qualified `auto` or `accelerated` device it compiles both generic-TM native and
forward-UTM accelerated variants, plus the native inverse. A Transformer's
`compile()` uses its concrete CRS domains and compiles only the deduplicated
variants reachable in either transform direction.

## Qualified hardware and coverage

The initial accelerated coverage is deliberately small:

| Implementation ID | Family and direction | Operations | Qualified automatic device | Auto minimum elements |
|---|---|---|---|---:|
| `native.libdevice` | Every fused family/direction and Helmert | CUDA native special math, including paired native `sincos` where implemented | All devices and CPU/no-GPU fallback | n/a |
| `helmert.fixed_q62` | Helmert datum shift | Bounded sine/cosine only; ECEF, square root, and `atan2` remain native fp64 | Ada `sm_89` consumer GPUs | 131,072 |
| `tmerc.forward.fixed_q62` | Forward UTM only | Bounded sine/cosine, the TM latitude correction, and bounded `asinh`; remaining math stays fp64 | Ada `sm_89` consumer GPUs | 256 |
| `sinu.forward.fixed_q62` | Sinusoidal forward only | Guarded Q1.62 cosine; remaining arithmetic stays fp64 | Ada `sm_89` consumer GPUs | 524,288 |
| `ortho.forward.fixed_q62` | Orthographic forward only | Atomically guarded Q1.62 sine/cosine pairs; remaining arithmetic stays fp64 | Ada `sm_89` consumer GPUs | 262,144 |
| `ortho.inverse.guarded_reframe` | Spherical equatorial Orthographic inverse only (after CRS setup canonicalization) | Guarded algebraic reframe removes one `asin` and one `sincos`; ill-conditioned inputs use native fp64 | Ada `sm_89` consumer GPUs | 524,288 |
| `stere.inverse.fixed_q62` | Polar Stereographic inverse, public ellipsoidal A/B north/south and C south modes | Q1.62 sine inside the conformal-latitude iteration; scale, eccentricity, and iterative-angle guard failures use native fp64 | Ada `sm_89` consumer GPUs | 1,000,000 |
| `geos.forward.fixed_q62` | Geostationary forward, sphere/ellipsoid and sweep x/y | Paired Q1.62 trig for geocentric latitude and longitude; uncertain limb visibility recomputes complete native output | Ada `sm_89` consumer GPUs | 2,097,152 |
| `laea.forward.polar.fixed_q62` | Spherical polar LAEA forward, north/south origins | Q1.62 longitude sine/cosine; authalic and inverse paths remain native fp64 | Ada `sm_89` consumer GPUs | 1,048,576 |

On a qualified RTX 4090, `"auto"` resolves the specialized implementations at
or above the listed sizes; below them it resolves native. Explicit
`"accelerated"` can select the specialized implementation at any size. Generic
Transverse Mercator, inverse UTM, sinusoidal inverse, Orthographic inverse
outside the exact spherical-equatorial origin domain, LAEA outside spherical
polar forward, Polar Stereographic outside the five listed inverse domains,
GEOS inverse, all other projection families, unsupported
precision combinations, and unknown devices resolve or fall back to
`native.libdevice`. Guarded input values do not
change the host decision: a selected fixed `StrategyDecision` remains selected
while its kernel executes the native branch for those values. These
projection-specific accelerated implementations
are fp64-only; `precision="fp32"` and `precision="ds"` stay native. Planning
calls with `precision="auto"` may select them because the corresponding fused
kernel resolves to fp64.

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
- Sinusoidal forward: Q1.62 cosine is used only at physical scale
  `0 < scale <= 6,400,000 m`, for finite latitude in
  `[-pi/2, pi/2]` with wrapped longitude in `[-pi, pi]`. Other coordinates use
  native cosine. The complete projected result must differ from native by
  `< 1e-8 m`; final WGS84 qualification measured 3.725/1.863 nm maximum/p99.
- Orthographic forward: at physical scale `0 < scale <= 6,400,000 m`, both
  latitude and wrapped-longitude sine/cosine pairs use one atomic guard over
  `[-pi/2, pi/2]` and `[-pi, pi]`. If either argument is invalid, both pairs
  use native math. The complete projected result must differ from native by
  `< 1e-8 m`; final WGS84 qualification measured 2.033106/1.396984 nm
  maximum/p99.
- Spherical equatorial Orthographic inverse: for `0 < scale <= 6,400,000 m`,
  finite non-axis points with normalized `1e-16 < rho^2 <= 0.99` use the identity
  `q=sqrt(1-rho^2)`, `phi=asin(y)`, and `lambda=atan2(x,q)`. The center,
  near-center `rho^2 <= 1e-16` band, axes/signed zero, horizon/outside-disk,
  non-finite inputs, and the
  `|phi_argument| > 0.95` conditioning band execute the exact native formula.
  Final RTX 4090 full-valid-disk qualification measured 6.328/1.584 nm
  maximum/p99 error.
- Polar Stereographic inverse: Q1.62 sine is used inside the ellipsoidal
  conformal-latitude iteration for public variants A/B north/south and variant
  C south. The uniform guard requires `0.05 <= e <= 0.2` and
  `0 < a <= 6,400,000 m`; scale/eccentricity failures and non-finite or
  out-of-range iterative angles use native sine. Representative and
  adversarial maximum native-relative errors were 1.582 and 3.164 nm.
- Geostationary forward: the Q1.62 pairs cover both geocentric latitude and
  wrapped longitude for spherical and ellipsoidal geometry with sweep x or y.
  A launch-uniform guard requires finite valid satellite geometry and
  `0 < a <= 6,400,000 m`; otherwise the exact native trig path runs. The
  line-of-sight denominator is at least the satellite height, so its factor
  cancels the final height output scale rather than amplifying angular error.
  Coordinates whose Q1.62 visibility residual lies in the proved uncertainty
  band recompute native trig, visibility classification, and output atomically;
  exact and adjacent limb sentinels therefore match native policy.
- Spherical polar LAEA forward: north- and south-pole origins use Q1.62 paired
  longitude trig for finite wrapped longitude at
  `0 < scale <= 6,400,000 m`. Ellipsoidal polar, equatorial, oblique, and all
  inverse domains remain native.
- Non-finite, out-of-domain, wide-TM, near-pole, and otherwise unsupported
coordinates take the native branch per coordinate.

Host strategy fallback and kernel guards are intentionally distinct. An
unsupported backend, device, direction, precision, or domain produces an
observable `StrategyDecision` for `native.libdevice`. A qualified fixed
projection decision remains on its implementation ID when its uniform scale
or parameter guard fails, while the kernel executes native math. This keeps
dispatch inspectable without claiming the guarded inputs were accelerated.

These are native-equivalence contracts for the implementation change. The
existing projection accuracy requirements against pyproj still apply; an
accelerated implementation may not spend that error budget merely because it
passes its native-equivalence bound.

The exact IDs, guarded-domain contracts, and automatic thresholds above are
public. On the qualified RTX 4090 at 5,000,000 randomized coordinates, the
final enforced public benchmark measured 1.096x synchronized-wall speedup for
sinusoidal forward, 1.341x for orthographic forward, and a minimum 1.056x for
Polar Stereographic inverse across its five exact domains. These measurements
qualify the listed hardware and thresholds; they are not a performance promise
for other devices or workloads.
