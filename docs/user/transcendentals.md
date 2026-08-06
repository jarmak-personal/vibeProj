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
Module-level `warm_up(["tmerc"])` has no CRS domain to inspect. On a qualified
device it compiles generic-TM native plus the native inverse for `auto`; an
explicit `accelerated` warm-up also includes the forward-UTM accelerated
variant. A Transformer's
`compile()` uses its concrete CRS domains and compiles only the deduplicated
variants reachable in either transform direction.

## Qualified hardware and coverage

The initial accelerated coverage is deliberately small:

| Implementation ID | Family and direction | Operations | Qualified device/policy | Auto minimum elements |
|---|---|---|---|---:|
| `native.libdevice` | Every fused family/direction and Helmert | CUDA native special math, including paired native `sincos` where implemented | All devices and CPU/no-GPU fallback | n/a |
| `helmert.fixed_q62` | Helmert datum shift | Bounded sine/cosine only; ECEF, square root, and `atan2` remain native fp64 | Ada `sm_89` consumer GPUs | 131,072 |
| `tmerc.forward.fixed_q62` | Forward UTM only | Bounded sine/cosine, the TM latitude correction, and bounded `asinh`; remaining math stays fp64 | Ada `sm_89` consumer GPUs; explicit `accelerated` only | n/a |
| `sinu.forward.fixed_q62` | Spherical Sinusoidal forward only | Guarded Q1.62 cosine; remaining arithmetic stays fp64 | Ada `sm_89` consumer GPUs | 524,288 |
| `sinu.inverse.convergent_newton` | Ellipsoidal Sinusoidal inverse | Native ten-step Newton expressions with a `1e-14` convergence break; full native coordinate/sentinel domain | Ada `sm_89` consumer GPUs; `auto` only | 1 |
| `sinu.inverse.meridional_recurrence` | Ellipsoidal Sinusoidal inverse hot domain | Two recurrence Newton steps, one native-shaped correction, and paired final `sincos`; a cold lane makes the complete warp native | Ada `sm_89` consumer GPUs; explicit `accelerated` only | n/a |
| `ortho.forward.fixed_q62` | Orthographic forward only | Atomically guarded Q1.62 sine/cosine pairs; remaining arithmetic stays fp64 | Ada `sm_89` consumer GPUs | 262,144 |
| `ortho.inverse.guarded_reframe` | Spherical equatorial Orthographic inverse only (after CRS setup canonicalization) | Guarded algebraic reframe removes one `asin` and one `sincos`; ill-conditioned inputs use native fp64 | Ada `sm_89` consumer GPUs | 524,288 |
| `gnom.inverse.guarded_rsqrt_reframe` | Spherical Gnomonic inverse, equatorial and bounded oblique origins | Guarded reciprocal-square-root reframe for finite non-axis `1e-24 < rho^2 <= 0.02`; every cold coordinate uses exact native fp64 | Ada `sm_89` consumer GPUs | n/a (explicit only) |
| `stere.inverse.fixed_q62` | Polar Stereographic inverse, public ellipsoidal A/B north/south and C south modes | Q1.62 sine inside the conformal-latitude iteration; scale, eccentricity, and iterative-angle guard failures use native fp64 | Ada `sm_89` consumer GPUs | 1,000,000 |
| `geos.forward.fixed_q62` | Geostationary forward, sphere/ellipsoid and sweep x/y | Paired Q1.62 trig for geocentric latitude and longitude; uncertain limb visibility recomputes complete native output | Ada `sm_89` consumer GPUs | 2,097,152 |
| `laea.forward.polar.fixed_q62` | Spherical polar LAEA forward, north/south origins | Q1.62 longitude sine/cosine; authalic and inverse paths remain native fp64 | Ada `sm_89` consumer GPUs | 1,048,576 |
| `merc.forward.spherical.product_poly` | Spherical regular Mercator forward, variants A/B | Removes the zero-exponent `pow` while preserving the native clamp/log-tan result exactly | Ada `sm_89` consumer GPUs | 262,144 |
| `merc.forward.ellipsoidal.product_poly` | Ellipsoidal regular Mercator forward, variants A/B | Native clamp/log-tan with a product reframe and degree-eight polynomial; polar cap uses exact native math | Ada `sm_89` consumer GPUs; explicit `accelerated` only | n/a |
| `merc.inverse.exp_series` | Regular Mercator inverse, spherical/ellipsoidal variants A/B | Native conformal `exp`/`atan` seed followed by the shared sixth-order Poder/Engsager recovery series | Ada `sm_89` consumer GPUs | 65,536 |
| `lcc.forward.conformal_reframe` | Lambert Conformal Conic forward, spherical/ellipsoidal 1SP/2SP regular cones | Spherical log/exp power reframe; ellipsoidal native outer power with stable `exp`/`atanh` conformal correction | Ada `sm_89` consumer GPUs | 65,536 |
| `lcc.inverse.conformal_reframe` | Lambert Conformal Conic inverse, spherical/ellipsoidal 1SP/2SP | Spherical logarithmic latitude reconstruction; ellipsoidal native outer power with bounded six-step `exp`/`atanh` recovery and a final contraction correction when step six has not reached `1e-14` | Ada `sm_89` consumer GPUs | 128 |
| `krovak.inverse.guarded_log_ratio` | Standard-Bessel Krovak inverse, regular and north oriented | Guarded log-ratio conformal seed and sixth-order recovery; a cold lane makes the complete warp exact native | Ada `sm_89` consumer GPUs; explicit `accelerated` only | n/a |

On the qualifying RTX 4090 at five million coordinates, spherical Mercator
forward measured about 1.13x native, explicit ellipsoidal forward about 1.26x,
spherical inverse about 1.80x, and ellipsoidal inverse about 8.35x. Maximum
native-relative error was zero for spherical paths, 7.451 nm for ellipsoidal
forward, and 6.328 nm for ellipsoidal inverse.

On a qualified RTX 4090, `"auto"` resolves the automatically qualified specialized implementations at
or above the listed sizes; below them it resolves native. Explicit
`"accelerated"` can select the specialized implementation at any size. Generic
Transverse Mercator, inverse UTM, spherical Sinusoidal inverse, Orthographic inverse
outside the exact spherical-equatorial origin domain, LAEA outside spherical
polar forward, Polar Stereographic outside the five listed inverse domains,
GEOS inverse, Web Mercator in both directions, all unlisted projection domains, unsupported
precision combinations, and unknown devices resolve or fall back to
`native.libdevice`. Guarded input values do not
change the host decision: a selected fixed `StrategyDecision` remains selected
while its kernel executes the native branch for those values. These
projection-specific accelerated implementations
are fp64-only; `precision="fp32"` and `precision="ds"` stay native. Planning
calls with `precision="auto"` may select them because the corresponding fused
kernel resolves to fp64.

Forward UTM is an explicit-only strategy. Its fixed-Q62 path wins when inputs
stay in the normal in-zone domain (`|longitude - central_meridian| <= 0.06`
radians; a UTM zone's usual +/-3 degrees fits), but broad or incorrectly zoned
inputs can spend enough time in the native guard fallback to lose the advantage.
Use `transcendentals="accelerated"` when that input-domain invariant is known;
`"auto"` remains native for Tmerc at every workload size. The measured launch
crossover is 256 coordinates, so expert callers should also keep smaller calls
on native math.

Gnomonic inverse is a deliberate exception to automatic selection. Its bounded
hot domain is materially faster, but host dispatch cannot inspect each coordinate's
normalized radius. A random 10% mixture outside the guard measured about 0.92x native
on RTX 4090, so `"auto"` remains native at every size. Expert callers who know their
inputs are concentrated in the documented hot domain may opt in with
`transcendentals="accelerated"`; high-origin and polar CRS domains remain native.

Standard-Bessel Krovak inverse is also explicit-only. Its guarded log-ratio
path covers the six listed public regular/north-oriented CRSs, but `"auto"`
remains native at every size. Supported custom and spherical setups plus
invalid-setup and forward Krovak domains remain native. Modified Krovak CRS
methods are unsupported.

Ellipsoidal Sinusoidal inverse deliberately assigns different implementations
to the two policies. `"auto"` uses `sinu.inverse.convergent_newton` for every
nonempty qualified Ada fp64 workload. It preserves the native expressions and
ten-step cap but stops after a correction smaller than `1e-14`, including for
poles, off-image values, and non-finite inputs. Explicit `"accelerated"` instead
selects `sinu.inverse.meridional_recurrence`. That hybrid is intended for hot
inputs recovered within ±89.9° and wrapped relative longitude within ±π; any
cold lane makes its complete warp execute the exact native inverse.

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
- Spherical Sinusoidal forward: Q1.62 cosine is used only at physical scale
  `0 < scale <= 6,400,000 m`, for finite latitude in
  `[-pi/2, pi/2]` with wrapped longitude in `[-pi, pi]`. Other coordinates use
  native cosine. The complete projected result must differ from native by
  `< 1e-8 m`; final WGS84 qualification measured 3.725/1.863 nm maximum/p99.
- Spherical regular Mercator forward: automatic selection removes the
  zero-exponent `pow` and is bitwise exact across the full finite domain.
- Ellipsoidal regular Mercator forward: explicit acceleration requires finite parameters and
  nonzero units, `0 < e <= 0.1`, `0 < k0 <= 1`, and
  `0 < a <= 6,400,000 m`. Raw and derived longitude/latitude values must be
  finite and the hot latitude is bounded to `+/-89.9` degrees. The remaining
  polar cap uses the exact native expression. The native latitude clamp and
  log-tan form are unchanged; the complete projected result must differ from
  native by `< 1e-8 m`. It is excluded from `"auto"`: random workloads with
  10-50% of coordinates in the polar cap make warp-wide fallback slower than
  native-only execution.
- Regular Mercator inverse: the same finite setup, unit, eccentricity, and
  scale guards apply, while any finite `k0 > 0` is accepted. Raw coordinates,
  normalized coordinates, longitude, and all six conformal-series coefficients
  must remain finite. The complete geographic result must differ from native
  by `< 1e-8 m` after angular error is scaled by the ellipsoid radius.
- Lambert Conformal Conic forward: spherical and ellipsoidal 1SP/2SP setups
  require finite nonzero cone constants and units, exact `k0 == 1`,
  `0 <= e <= 0.1`, `0 < a <= 6,400,000 m`, and `|n| >= 0.2`. Exact poles,
  non-finite values, unbounded cone angles, and any cold lane make the complete
  warp use native math. Near-equator cones remain native. Complete projected
  output must differ from native by `< 1e-8 m` with no pyproj regression.
- Lambert Conformal Conic inverse: the same setup bounds apply without the
  forward `|n|` restriction. Exact apex, nonpositive radius ratios, and
  non-finite raw or normalized inputs use complete-warp native math. Huge finite
  coordinates whose ratio becomes positive infinity remain qualified and
  converge to the finite pole limit. Angular error scaled by the ellipsoid
  radius must be `< 1e-8 m`, with no pyproj regression. Ellipsoidal recovery
  applies at most six fixed-point steps; only when the sixth delta remains at
  least `1e-14`, that final delta receives a contraction correction. This adds
  no seventh transcendental evaluation and closes the exact `e=0.1` boundary.
- Standard-Bessel Krovak inverse: explicit `accelerated` covers the six public
  EPSG CRSs 2065/5221/5513/5514/8352/8353 in their exact regular or
  north-oriented setup domains. Regular public inputs remain X=Southing and
  Y=Westing even with `always_xy=True`. The setup guard pins the Bessel
  ellipsoid, derived cone/conformal scalars, method, and axis signs while
  allowing either finite central meridian; finite nonzero signed units are
  required. Coordinates need a positive finite radius, finite intermediates,
  and recovered `|latitude| <= 80 degrees`; otherwise the complete warp uses
  exact native math. Supported custom and spherical setups plus invalid,
  forward, fp32, double-single, and non-Ada cases remain native; Modified
  Krovak CRS methods are unsupported. `"auto"` is native at every size.
  Retained research measured about 2.583x gain and no more than 7.12 nm
  native-relative horizontal error. The formal six-CRS public run passed every
  gate with at least 2.6901x synchronized-wall and 2.6976x CUDA-event speedup at
  five million coordinates; maximum/p99 native-relative horizontal error was
  7.9089/4.7453 nm. N=1/2/5,000,000 explicit rows required every wall/device
  repeat to reach 1.05x and passed with worst repeats of 1.4264x/1.4262x/2.6898x
  respectively; every complete-warp cold sweep passed, while `"auto"` remained
  bitwise native.
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
- Ellipsoidal Sinusoidal inverse: both implementations require finite setup,
  nonzero signed unit factors, `0 < es <= 0.012`, and
  `0 < a <= 6,400,000 m`. The automatic convergence implementation covers the
  complete native coordinate domain. The explicit recurrence additionally
  requires finite normalized inputs, recovered `|latitude| <= 89.9°`, finite
  positive longitude denominator, and wrapped `|longitude| <= pi`; guard
  failure is complete-warp and bitwise native.
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

For ellipsoidal Sinusoidal inverse, the automatic convergence kernel was already
above the 1.05 gate at the exact nonempty endpoint N=1, where its minimum wall
speedup across WGS84 and the exact `es=.012`, `a=6,400,000 m` boundary was
1.6435x, and remained above it at N=2 and 5,000,000, including randomized 0%,
0.1%, 1%, 10%, 50%, and 100% cold mixtures. In the final public
five-million-coordinate run, the automatic kernel measured 3.026x wall /
3.034x device speedup in both domains. The explicit recurrence hybrid measured
4.621x / 4.643x on WGS84 and 4.623x / 4.644x at the boundary. The formal
suite's maximum/p99 native-relative horizontal error was 6.583/0 nm; the exact
boundary maximum was 6.582903846 nm. At N=1 the recurrence hybrid's minimum
wall repeat across both domains was 1.8215x, but the WGS84 random 10% cold and
all-cold median wall speedups were only 1.0238x and 1.0032x. Those retained
negative results are why the hybrid is not an AUTO choice.
