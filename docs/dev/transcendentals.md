# Transcendental inventory and qualification

This page is the static inventory and release contract for special-function
calls in fused CUDA projections and the Helmert datum-shift kernel. It describes
reachable projection math, not unused helpers that happen to be concatenated
into an NVRTC source string.

## Reading the inventory

For forward kernels, finite longitude offsets `lambda` are wrapped to
approximately `[-pi, pi]`. Latitude `phi` is not guarded inside each kernel, so
its bound is **unknown at the kernel boundary**. For inverse kernels, normalized
projected coordinates are likewise unknown/unbounded unless the row states an
explicit clamp. “Clamped” inverse-trigonometric arguments are provably in
`[-1, 1]`. All other derived bounds are marked unknown rather than inferred
from typical CRS usage.

The risk tier applies to changing the current implementation:

- **T0**: no transcendental operation.
- **T1**: retain native math; use the implemented same-argument native
  `sincos` pairing where both results are needed.
- **T2**: a bounded, guarded replacement with operation and coordinate error
  contracts exists.
- **T3**: unknown/unbounded, singular, or iterative domain. Keep native until a
  separately guarded domain and error proof are added.

`native.libdevice` includes separate native calls and paired native `sincos`.
Pairing is an exact-library baseline optimization, not approximate accelerated
coverage. “Pair” below means both outputs for the same argument are reachable
in the same kernel section, and every pair listed is implemented through the
shared native pairing helper (or the equivalent double-single helper).

## Complete fused-kernel inventory

| Family | Direction | Reachable operations | Argument/domain bound | Native same-argument pairs | Current implementation | Allowed action | Risk |
|---|---|---|---|---|---|---|---|
| `eqc` | forward | none | n/a | none | arithmetic only | none | T0 |
| `eqc` | inverse | none | n/a | none | arithmetic only | none | T0 |
| `sinu` | forward | `cos(phi)` | kernel-boundary `phi`: unknown; accelerated guard requires finite `|phi| <= pi/2`, wrapped `|lambda| <= pi`, and `0 < scale <= 6,400,000 m` | none | `native.libdevice`, or `sinu.forward.fixed_q62` | qualified guarded implementation; native otherwise | T2 |
| `sinu` | inverse | `cos(phi)` | inverse `phi`: unknown | none | `native.libdevice` | native only | T3 |
| `merc` | forward | `sin`, `tan`, `pow`, `log` | `phi` and singular distance to poles: unknown | none | `native.libdevice` | native only | T3 |
| `merc` | inverse | `exp`, `atan`, iterative `sin`/`pow` | normalized northing and exponent: unbounded | none | `native.libdevice` | native only | T3 |
| `webmerc` | forward | `tan`, `log` | `phi` and singular distance to poles: unknown | none | `native.libdevice` | native only | T3 |
| `webmerc` | inverse | `exp`, `atan` | normalized northing and exponent: unbounded | none | `native.libdevice` | native only | T3 |
| `tmerc` | forward | three `sin`+`cos` arguments, `rsqrt`, `atan2`, `asinh` | general TM derived arguments: unknown; qualified UTM guard: `|lambda| <= 0.06`, `|asinh_arg| <= 0.06` with error `<= 2e-17`, `|atan_delta| <= 9.01e-4` with error `<= 2.3e-16 rad`; Q1.62 angles guarded to `[-pi, pi]` | pair `2phi`, Gaussian latitude, and `lambda` | native paired `sincos`/libdevice, or `tmerc.forward.fixed_q62` | qualified guarded implementation; native otherwise | T2 |
| `tmerc` | inverse | `sin`/`cos`, `exp`, `sinh`, two `atan2`, `hypot` | inverse complex coordinates: unknown/unbounded | pair `2Cn`; pair `Cn` | `native.libdevice` | paired native only | T3 |
| `lcc` | forward | `sin`, `tan`, `pow`, output `sin`/`cos` | `phi`: unknown; `theta = n*lambda` depends on CRS parameter | pair `theta` | `native.libdevice` | paired native only | T3 |
| `lcc` | inverse | `sqrt`, `atan2`, `pow`, iterative `atan`/`sin`/`pow` | projected radius and iteration arguments: unknown | none | `native.libdevice` | native only | T3 |
| `stere` | forward | `sin`, `tan`, `pow`, output `sin`/`cos` | adjusted `phi`: unknown; `lambda` finite wrapped | pair `lambda` | `native.libdevice` | paired native only | T3 |
| `stere` | inverse | `sqrt`, `atan2`, iterative `atan`/`sin`/`pow` | projected radius and iteration arguments: unknown | none | `native.libdevice` | native only | T3 |
| `aea` | forward | `sin`, `atanh`, `sqrt`, output `sin`/`cos` | `phi`: unknown; negative radicand is clamped to zero; `theta=n*lambda` depends on CRS | pair `theta` | `native.libdevice` | paired native only | T3 |
| `aea` | inverse | `sqrt`, `atan2`, iterative `atanh`, `asin` | projected radius unknown; exact authalic poles accepted within the shared `1e-10` representational band; material `|q|>qp` is atomically invalid | none | `native.libdevice` | native only | T3 |
| `laea` | forward | `sin`, `atanh`, clamped `asin`, `sqrt`, `sin`/`cos` | finite latitude restricted to `[-pi/2,pi/2]`; spherical north/south polar acceleration guards wrapped `|lambda| <= pi` and `0 < scale <= 6,400,000 m`; antipode and non-finite behavior remain exact | pair authalic latitude; pair `lambda` | `native.libdevice`, or `laea.forward.polar.fixed_q62` in the exact spherical-polar domains | qualified guarded longitude pair for spherical polar modes; native otherwise | T2 |
| `laea` | inverse | `sqrt`, clamped `asin`, `sin`/`cos`, `atan2`, iterative `atanh` | exact normalized disk enforced for oblique/equatorial and polar modes; center and poles explicit | pair central angle | `native.libdevice` | paired native only | T3 |
| `eqearth` | forward | `sin`, `atanh`, two clamped `asin`, `cos`, `sqrt` | kernel-boundary `phi`: unknown; inverse-trig inputs clamped; spherical `e=0` uses exact `q=2sin(phi)` | none | `native.libdevice` | native only | T3 |
| `eqearth` | inverse | iterative polynomial, `sin`/`cos`, clamped `asin`, iterative `atanh` | authalic inverse input clamped; spherical and ellipsoidal q inversion share the same helper; component-wise non-finite behavior matches fused execution | pair `theta` | `native.libdevice` | paired native only | T3 |
| `cea` | forward | `sin`, `atanh` | `phi`: unknown; spherical `e=0` uses exact `q=2sin(phi)` | none | `native.libdevice` | native only | T3 |
| `cea` | inverse | iterative `atanh`, `asin` | exact authalic poles accepted within the shared `1e-10` representational band; material `|q|>qp` is atomically invalid | none | `native.libdevice` | native only | T3 |
| `ortho` | forward | `sin`/`cos` of `phi` and `lambda` | kernel-boundary `phi`: unknown; accelerated atomic guard requires finite `|phi| <= pi/2`, wrapped `|lambda| <= pi`, and `0 < scale <= 6,400,000 m` | pair `phi`; pair `lambda` | `native.libdevice`, or `ortho.forward.fixed_q62` | qualified guarded implementation; native otherwise | T2 |
| `ortho` | inverse | `sqrt`, clamped `asin`, paired `sin`/`cos`, `atan2` | exact spherical-equatorial domain after CRS setup canonicalization; accelerated reframe requires finite non-axis `1e-16 < rho^2 <= 0.99`, `|phi_argument| <= 0.95`, and `0 < scale <= 6,400,000 m`; projected radius is otherwise unknown | algebraic `q=sqrt(1-rho^2)` removes the radial `asin` and paired trig | `native.libdevice`, or `ortho.inverse.guarded_reframe` | qualified only when canonical setup yields the equatorial origin scalar; center/near-center and other guard failures use exact native cold fallback | T2 |
| `gnom` | forward | `sin`/`cos` of `phi` and `lambda` | `phi`: unknown; horizon denominator can approach zero | pair `phi`; pair `lambda` | `native.libdevice` | paired native only | T3 |
| `gnom` | inverse | `sqrt`, `atan`, `sin`/`cos`, `asin`, `atan2` | projected radius unbounded; derived `asin` input not explicitly clamped | pair central angle | `native.libdevice` | paired native only | T3 |
| `moll` | forward | `sin`; iterative `sin`/`cos`; output `sin`/`cos` | `phi`: unknown; converged `theta` bound not enforced in kernel | pair `2theta`; pair output `theta` | `native.libdevice` | paired native only | T3 |
| `moll` | inverse | two clamped `asin`, `sin`, `cos` | both `asin` inputs clamped; division by `cos(theta)` can approach singularity | none | `native.libdevice` | native only | T3 |
| `omerc` | forward | `sin`, `tan`, `pow`, `log`, `cos`, `atan2` | `U` clamped below `|1|`; remaining derived arguments depend on CRS/domain | pair `B*lambda` | `native.libdevice` | paired native only | T3 |
| `omerc` | inverse | `exp`, `sin`/`cos`, `sqrt`, `pow`, iterative `atan`/`sin`/`pow`, `atan2` | `U'` clamped below `|1|`; exponential and projected inputs unbounded | pair `B*u/A` | `native.libdevice` | paired native only | T3 |
| `krovak` | forward | `sin`, `tan`, `pow`, `atan`, `sin`/`cos`, two `asin` | inverse-trig inputs not explicitly clamped; `phi` unknown | pair `U`; pair `V`; pair cone angle | `native.libdevice` | paired native only | T3 |
| `krovak` | inverse | `sqrt`, `atan2`, `atan`, `pow`, `sin`/`cos`, `asin`, `tan`, iterative special math | radius guarded away from zero; remaining projected/iterative arguments unknown | pair `T`; pair `D`; iterative pair `phi` | `native.libdevice` | paired native only | T3 |
| `eck4` | forward | `sin`; iterative `sin`/`cos`; output `sin`/`cos` | `phi`: unknown; iteration bound not enforced | pair iterative/output `theta` | `native.libdevice` | paired native only | T3 |
| `eck4` | inverse | clamped `asin`, `sin`/`cos`, clamped `asin` | both inverse-trig arguments clamped | pair `theta` | `native.libdevice` | paired native only | T1 |
| `eck6` | forward | `sin`; iterative/output `sin`/`cos` | `phi`: unknown; iteration bound not enforced | pair `theta` | `native.libdevice` | paired native only | T3 |
| `eck6` | inverse | `sin`/`cos`, clamped `asin` | `asin` argument clamped; projected `theta` otherwise unknown | pair `theta` | `native.libdevice` | paired native only | T3 |
| `sterea` | forward | `sin`, `pow`, `asin`, paired `sin`/`cos` | conformal `asin` ratio is algebraically in `(-1,1)` for finite positive `w`; kernel does not guard invalid `w` | pair conformal latitude; pair scaled longitude | `native.libdevice` | paired native only | T3 |
| `sterea` | inverse | `sqrt`, `atan`, `sin`/`cos`, clamped `asin`, `atan2`, `log`, `exp`, `tan`, `pow` | one `asin` input clamped; projected and iterative arguments otherwise unknown | pair central angle; iterative pair `phi` | `native.libdevice` | paired native only | T3 |
| `geos` | forward | `tan`, `atan`, `sin`/`cos`, `sqrt`, `atan2`, clamped `asin` | Sweep X/Y explicit; finite latitude bounded; Q1.62 guards require valid satellite geometry and `0 < a <= 6,400,000 m`; an analytically bounded visibility-uncertainty band recomputes native trig, classification, and output atomically at the limb | pair geocentric latitude; pair `lambda` | `native.libdevice`, or `geos.forward.fixed_q62` for sphere/ellipsoid and sweep x/y | qualified guarded pairs; exact native fallback for uncertain visibility, invalid parameters, or scale | T2 |
| `geos` | inverse | paired `sin`/`cos`, `sqrt`, `atan2`, `atan` | principal scan angles enforced; CPU uses a strict zero discriminant tolerance, while CUDA admits only the measured dtype-specific tangent bin (`4e-8` fp32, `2e-15` fp64); Sweep X/Y rays explicit | pair x scan angle; pair y scan angle | `native.libdevice` | paired native only | T3 |
| `robin` | forward | none (table polynomial) | n/a | none | arithmetic/table only | none | T0 |
| `robin` | inverse | none (table polynomial/Newton) | n/a | none | arithmetic/table only | none | T0 |
| `wintri` | forward | `sin`/`cos`, clamped `acos`, `sin` | `acos` input clamped; `phi` otherwise unknown; `lambda` finite wrapped | pair `phi`; pair `lambda/2` | `native.libdevice` | paired native only | T3 |
| `wintri` | inverse | iterative `sin`/`cos`, clamped `acos`, `sin` | projected initial values and iteration domain unknown; `acos` input clamped | pair `phi`; pair `lambda/2` | `native.libdevice` | paired native only | T3 |
| `natearth` | forward | none (polynomial) | n/a | none | arithmetic only | none | T0 |
| `natearth` | inverse | none (polynomial/Newton) | n/a | none | arithmetic only | none | T0 |
| `aeqd` | forward | paired `sin`/`cos`, clamped `acos`, `sin` | explicit spherical CRS only; antipode is singular; ellipsoidal, Modified, and Guam methods are rejected before dispatch | pair `phi`; pair `lambda` | `native.libdevice` | paired native only | T3 |
| `aeqd` | inverse | `sqrt`, `sin`/`cos`, clamped `asin`, `atan2` | explicit spherical CRS only; projected radius unknown; ellipsoidal, Modified, and Guam methods are rejected before dispatch | pair central distance | `native.libdevice` | paired native only | T3 |
| `helmert` | datum shift (forward or inverse pipeline) | paired `sin`/`cos` for source latitude/longitude, `sqrt`, `atan2`, iterative `sin`, optional final paired `sin`/`cos` | Q1.62 guard `|angle| <= pi`; non-finite/outside use native; angles within `0.02 rad` of a pole use native for height conditioning | pair source latitude; pair source longitude; pair final latitude with height | native paired `sincos`/libdevice, or `helmert.fixed_q62` for bounded trig only | qualified guarded implementation; native otherwise | T2 |

AEA, LAEA, Equal Earth, and CEA use one neutral CPU/xp authalic helper for
`q`, pole snapping, and `q` inversion. Its spherical branch is exact, its
ellipsoidal branch uses the stable `atanh(e sin(phi))/e` form, and only NumPy
may terminate Newton iteration early. Other array namespaces execute a fixed
iteration count without reading a device scalar or synchronizing to the host.

## Coverage matrix

The user-facing coverage matrix is intentionally narrower than the inventory:

| ID | Family | Direction | Precision | Device qualification | Auto min elements | Guarded operations |
|---|---|---|---|---|---:|---|
| `native.libdevice` | `*` | forward/inverse/Helmert | all supported | universal fallback | 0 | all special math |
| `helmert.fixed_q62` | `helmert` | datum shift | all public modes (Helmert kernel stays fp64) | Ada `sm_89`, weak-native-fp64 consumer class | 131,072 | `sin`, `cos`; `|angle| <= pi`, near-pole native guard |
| `tmerc.forward.fixed_q62` | `tmerc` | forward UTM | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 256 | paired `sin`/`cos`, TM `atan2` correction, `asinh`; per-operation guards above |
| `sinu.forward.fixed_q62` | `sinu` | forward | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 524,288 | Q1.62 `cos(phi)`; valid latitude/wrapped longitude and `0 < scale <= 6,400,000 m`, native otherwise |
| `ortho.forward.fixed_q62` | `ortho` | forward | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 262,144 | Q1.62 paired `sin`/`cos`; atomic argument guard and `0 < scale <= 6,400,000 m`, native otherwise |
| `ortho.inverse.guarded_reframe` | `ortho` | inverse, spherical equatorial only | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 524,288 | guarded algebraic reframe for normalized `1e-16 < rho^2 <= 0.99`; center/near-center, axes, horizon/outside, non-finite, scale, and conditioning failures call the exact native expression |
| `geos.forward.fixed_q62` | `geos` | forward, sphere/ellipsoid, sweep x/y | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 2,097,152 | Q1.62 pairs for geocentric latitude and longitude; complete-native visibility-uncertainty fallback plus launch-uniform finite geometry/`a <= 6,400,000 m` guard |
| `laea.forward.polar.fixed_q62` | `laea` | forward, spherical north/south polar only | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 1,048,576 | Q1.62 paired longitude trig; exact pole/antipode/non-finite handling and `a <= 6,400,000 m` guard |

`tests/test_transcendental_coverage.py` binds these IDs and capabilities to
the public registry and resolver. Adding a documented accelerated row without
registry support, or registry support without updating the contract, fails the
test. Native-only inventory rows do not imply an accelerated registry entry.
For a domain with multiple future hardware variants, the resolver filters
policy, backend, compute capability, fp32:fp64 ratio, and precision first, then
chooses the highest explicit registry priority. An equal-priority eligible tie
is rejected as ambiguous; registration order is never a tie-breaker.

## Qualification and benchmark protocol

Run the end-to-end policy benchmark, not only an isolated operation benchmark:

```console
uv run python benchmarks/bench_transcendental_policy.py \
  --case all --n 5000000 --warmup 10 --iterations 30 --repeats 3 \
  --json benchmark-results.json --enforce-gates
```

Each result records the GPU and compute capability, requested policy and
resolved implementation IDs, workload size, family/direction, compute
precision, point count, warmup and iteration counts, interleaved CUDA-event
samples and distribution, synchronized public-dispatch wall samples and
distribution, throughput, speedup against native, coordinate errors against
native and pyproj, and normal plus guard-edge domains. CUDA-event intervals
measure device execution only. Synchronized wall intervals use a host monotonic
clock around the warmed public `transform_buffers()` call and completion, so
they include normalization, device detection, cached strategy resolution,
argument preparation, submission, and synchronization. Pre-allocate and reuse
all device buffers. Separate untimed steady-state calls wrap CuPy's allocator directly,
so even an allocation satisfied from the memory pool is counted. A CUDA stream
capture reports every kernel, memcpy, and memset graph node and the returned
objects are checked against the caller's output buffers. CUDA graph labels are
reported verbatim; the benchmark does not infer a host/device transfer
direction from a memory-pool delta or a hard-coded constant. `--enforce-gates`
exits nonzero if any required gate fails.

For every registry entry with a physical-scale ceiling, untimed public probes
at `nextafter(max_scale, +inf)` and `1e12 m` mechanically require the fixed
StrategyDecision to remain selected while both output arrays match native
bit-for-bit. This distinguishes host strategy fallback from the kernel's
uniform scale guard; normal per-coordinate argument guards follow the same
principle without changing the selected ID.

GEOS forward additionally proves its satellite geometry independently of
coordinate lanes. Its uniform guard accepts only positive finite `h` and `H`,
requires `H != h` and the exact setup relation `H == h + a`, then passes `a`
through the shared `0 < scale <= 6,400,000 m` gate. The setup contract bounds
`h / a <= 1e10`; tests span valid `h` from 1 m through that ceiling. Because
the perturbed line-of-sight denominator is at least `h`, the final `h` output
scale cancels rather than magnifying the Q1.62 angular error. Tests at the
scale limit and `nextafter(6,400,000, +inf)` require accelerated and bit-exact
native behavior respectively.

The default workload grid covers 1 through 5,000,000 elements on logarithmic
steps for every accelerated qualification case. Each case also injects its
exact `min_elements - 1` and
`min_elements` boundary sizes. `auto` must resolve native below the registry's
threshold and the qualified implementation at or above it; `native` must
remain native, and explicit `accelerated` must resolve the exact qualified ID.
The small-size gate permits at most 2% aggregate and 5% per-repeat
wall variation between identical native implementations; this is measurement
noise tolerance, not a performance claim. At and above `min_elements`, every
tested grid size must show at least 5% wall-clock improvement in all three
repeats. Explicit `accelerated` ignores `min_elements` and is measured at every
size, but no small-array speed claim is made for that override.

Expected kernel counts follow pipeline topology, not a universal one-launch
claim:

| Topology | Expected fused kernel nodes |
|---|---:|
| one projection, no datum work | 1 |
| geographic-only Helmert or SVD correction | 1 |
| Helmert or SVD plus one projection | 2 |
| projected-to-projected with one datum stage | 3 |
| Helmert plus SVD plus one projection | 3 |
| projected-to-projected with Helmert plus SVD | 4 |

Preallocated final outputs belong only to the final stage. Multi-stage paths
use four-array scratch slots per Transformer, keyed by input CUDA device and
normalized stream pointer. Slots are locked while host work is enqueued and
grow by doubling. The cache is LRU-bounded to eight leased slots per device.
Growth or eviction retains in-flight arrays behind recorded completion events
and prunes them only after an event query reports completion; it never
synchronizes. If all eight entries are leased, overflow scratch is transient
and event-retired instead of entering the cache. A warmed call at or below a
slot's capacity must not allocate again. Every supplied final `out_x`, `out_y`,
and (when `z` is supplied) `out_z` is written and returned as the identical
object. No topology may add copy or memset nodes merely to route intermediate
x/y coordinates.

`stream=None` resolves the current stream inside the input array's CUDA device.
The legacy null stream (`device_id == -1`) binds through that enclosing device
context; an explicit stream with a nonnegative, mismatched device ID is
rejected. This applies before scratch selection, so cache keys cannot silently
mix devices.

Each Transformer owns one persistent chunk workspace per CUDA device. Its lock
serializes a complete `transform_chunked()` call on that Transformer/device;
separate devices or Transformer instances remain independent. Within the call,
two persistent non-blocking streams and their pinned-host/device slots preserve
double-buffer overlap. Streams and slots are reused across calls and grow only
when capacity or z support increases. This makes repeated calls allocation-free
for internal workspace while preventing concurrent callers from racing over
the same staging buffers.

The older scripts have narrower purposes:

- `bench_int64_sincos.py` compares isolated paired trig experiments.
- `bench_alt_transcendentals.py` retains rejected/research `atan2` and `asinh`
  variants; they are not production coverage.
- `bench_tmerc_int64.py` and `bench_helmert_int64.py` retain raw-kernel and
  alternative fixed-core evidence. Their experimental modes are not public
  policies.

Eckert IV's Wave 1 inverse trig-pair experiment remains research-only. Replacing
the native inverse `sin(theta), cos(theta)` pair with Q1.62 measured a benign
1.0891x device speedup, but at `cy / C_y = -0.999994` and `lambda = pi` its
sub-ULP pair error amplifies to about 271.916 nm, failing the 10 nm gate. Run
`python benchmarks/eck4_inverse_rejection.py` or the matching deterministic
coverage test to reproduce the edge. No `eck4` implementation ID, registry row,
or public threshold is permitted without a new proof and full qualification.

### Ada qualification evidence

The public-policy benchmark was run on an RTX 4090 (`sm_89`, fp32:fp64 ratio
64) with 5,000,000 coordinates, 10 warm-up rounds, 30 interleaved iterations,
and three repeats. These measurements qualify that hardware; they are not a
performance promise for other devices. The first table is CUDA-event device
execution, not public-API wall time.

| Case | Native device median | Accelerated device median | Device speedup | Repeat speedups | Max / p99 error vs native |
|---|---:|---:|---:|---|---:|
| forward UTM | 2.0411 ms | 1.3128 ms | 1.555x | 1.556x, 1.556x, 1.555x | 3.73 nm / 1.86 nm |
| inverse UTM | 2.3357 ms | 2.3357 ms | 1.000x (native fallback) | 1.000x, 1.000x, 1.000x | exact |
| forward Helmert | 4.0755 ms | 3.7622 ms | 1.083x | 1.083x, 1.083x, 1.083x | 3.16 nm / 3.16 nm |
| inverse Helmert | 4.0746 ms | 3.7622 ms | 1.083x | 1.083x, 1.083x, 1.083x | 3.16 nm / 3.16 nm |

The corresponding synchronized public-dispatch wall measurements include the
warmed public call and completion synchronization:

| Case | Native wall p05 / p50 / p95 | Accelerated wall p05 / p50 / p95 | Wall speedup | Repeat speedups |
|---|---:|---:|---:|---|
| forward UTM | 2.0560 / 2.0577 / 2.0665 ms | 1.3279 / 1.3294 / 1.3432 ms | 1.548x | 1.548x, 1.548x, 1.548x |
| inverse UTM | 2.3501 / 2.3513 / 2.3536 ms | 2.3499 / 2.3511 / 2.3608 ms | 1.000x (native fallback) | 1.000x, 1.000x, 1.000x |
| forward Helmert | 4.0851 / 4.0866 / 4.0889 ms | 3.7725 / 3.7735 / 3.7753 ms | 1.083x | 1.083x, 1.083x, 1.083x |
| inverse Helmert | 4.0846 / 4.0861 / 4.0878 ms | 3.7724 / 3.7734 / 3.7750 ms | 1.083x | 1.083x, 1.083x, 1.083x |

The size grid qualified the conservative automatic crossovers below. Every
smaller grid row resolved native and passed the no-wall-regression noise bound;
every row from the threshold through 5,000,000 elements exceeded 1.05x in all
three repeats.

| Operation | Last tested size below threshold | Auto minimum | Native / auto wall p50 at threshold | Threshold repeat speedups | Auto speedup at 5M |
|---|---:|---:|---:|---|---:|
| forward UTM | 128 | 256 | 0.03019 / 0.02764 ms | 1.091x, 1.089x, 1.094x | 1.548x |
| forward Helmert | 65,536 | 131,072 | 0.11974 / 0.11187 ms | 1.068x, 1.072x, 1.071x | 1.083x |
| inverse Helmert | 65,536 | 131,072 | 0.11977 / 0.11185 ms | 1.071x, 1.071x, 1.072x | 1.083x |

All four cases reported matching non-finite behavior, passing guard-edge
comparisons, and no error regression relative to the native policy's pyproj
comparison. Five separately instrumented steady-state calls for every policy
reported zero allocator calls and exact caller-output identity. CUDA graph
capture reported one expected kernel and zero memcpy/memset nodes for every
standalone case. The untimed topology probes likewise passed with exact kernel
counts of 2 for Helmert plus projection, 1 for 3-D geographic Helmert, 1 for
geographic SVD, and 3 for projection-to-SVD-to-projection; each reported zero
allocator, memcpy, and memset activity. A zero memory-pool delta alone is not
used as evidence for these claims. The Helmert pyproj absolute difference is
dominated by the selected datum operation and missing optional OSTN15 grid;
the qualification gate is that the accelerated result does not regress the
same native baseline.

### Wave 1 projection qualification evidence

The final public-policy benchmark ran serially on the qualified RTX 4090 with
5,000,000 randomized coordinates, 10 warm-up rounds, 50 interleaved iterations,
and three repeats. Both cases passed every enforced normal, guard-edge,
non-finite, pyproj-regression, public-resolution, allocation, caller-output,
CUDA-graph topology, threshold-boundary, and full-grid gate.

| Case | Native / accelerated device p50 | Device speedup | Native / accelerated wall p50 | Wall speedup | Max / p99 error vs native |
|---|---:|---:|---:|---:|---:|
| sinusoidal forward | 0.427008 / 0.389120 ms | 1.097x | 0.405246 / 0.369870 ms | 1.096x | 3.725 / 1.863 nm |
| orthographic forward | 0.703488 / 0.520192 ms | 1.352x | 0.661692 / 0.493275 ms | 1.341x | 2.033 / 1.397 nm |

Sinusoidal's initial 262,144 candidate threshold had one randomized wall repeat
below 1.05x. The final 524,288 threshold passed at 1.069x, 1.065x, and 1.066x;
`auto` remained native at 524,287. Orthographic passed at 262,144 with 1.204x,
1.206x, and 1.194x wall speedups; `auto` remained native at 262,143. Every
larger tested grid size through 5,000,000 passed all three repeats. Companion
inverse paths remain `native.libdevice`.

At the exact 6,400,000 m scale ceiling, a separate two-million-coordinate
dense proof measured 3.725290298 nm maximum error for sinusoidal and
2.832507030 nm across orthographic origins `{-90, -45, 0, 45, 90}`. The exact
ceiling and its nextafter-below value retain Q1.62; nextafter-above, `1e12`,
zero, negative, infinity, and NaN scales execute bitwise-native uniformly.

### Wave 2A Orthographic inverse qualification evidence

The spherical-equatorial inverse reframe passed on the RTX 4090 over a
three-repeat synchronized public-call and CUDA-event full-valid-disk grid from
1 through 5,000,000 coordinates. At 5,000,000 coordinates, public repeat
speedups were 1.1344x, 1.1349x, and 1.1341x, with a 1.1341x median kernel
speedup. The maximum and p99 native-relative errors were 6.328 nm and 1.584 nm.
The candidate uses
36 registers, 56 bytes local memory, zero shared memory, and retained 100%
calculated thread occupancy at 256 threads/block (native: 35 registers and
40 bytes local).

The conservative automatic threshold is 524,288, the first tested size whose
three actual-public repeats all reached the 1.05x gate (1.0626x, 1.0570x, and
1.0600x); every larger tested size also passed. The final confirmation run
measured 1.0581x, 1.0608x, and 1.0588x at that boundary. Smaller and mid-sized inputs
failed that per-repeat gate, so `auto` stays native below 524,288. Scale sweeps through the exact
6,400,000 m ceiling preserve the 10 nm contract; nextafter-above and larger
scales, plus center, axes/signed zero, horizon/outside-disk, and non-finite
inputs, are bitwise-native. Production-shaped AEQD forward/inverse and bounded
rational Orthographic alternatives were slower (and both AEQD forward forms
also exceeded 10 nm), so their CUDA paths were removed and only their no-go
measurements remain in the benchmark artifact.

A warp-uniform vote prevents sparse invalid lanes from serializing the fast and
native bodies. Random-lane fallback sweeps at 0%, 0.1%, 1%, 10%, 50%, and 100%
measured synchronized speedups of 1.496x, 1.446x, 1.236x, 0.925x, 0.918x, and
0.912x, respectively, with every fallback lane bitwise-native. These adversarial
mixtures are retained as observability evidence; normal qualified interior data
drives the public crossover.

### Wave 2B forward qualification evidence

The final post-fallback public benchmark used the same RTX 4090, 5,000,000
coordinates, 10 warm-ups, 30 iterations, and three repeats. GEOS threshold
confirmation retained only `2,097,151`, `2,097,152`, and `5,000,000` in its
final boundary grid after a lower candidate crossover failed. LAEA used the
complete default grid. All six cases passed native-relative accuracy,
non-finite classification, pyproj-regression, scale/parameter guards,
allocation, output-identity, and graph-topology gates.

| Case | Auto minimum | Auto wall repeats at threshold | Device / wall speedup at 5M | Max / p99 error | Accelerated resources |
|---|---:|---|---|---|---|
| GEOS spherical sweep x | 2,097,152 | 1.057145x, 1.057123x, 1.057050x | 1.059619x / 1.059254x | 3.840 / 1.863 nm | 40 registers, 40 B local, 0 shared, 100% occupancy |
| GEOS spherical sweep y | 2,097,152 | 1.057008x, 1.057392x, 1.057516x | 1.060093x / 1.059748x | 3.754 / 1.863 nm | 40 registers, 40 B local, 0 shared, 100% occupancy |
| GEOS ellipsoidal sweep x | 2,097,152 | 1.059550x, 1.059209x, 1.059011x | 1.062527x / 1.061849x | 3.790 / 1.679 nm | 40 registers, 40 B local, 0 shared, 100% occupancy |
| GEOS ellipsoidal sweep y | 2,097,152 | 1.059719x, 1.059914x, 1.059278x | 1.062383x / 1.061922x | 3.979 / 1.679 nm | 40 registers, 40 B local, 0 shared, 100% occupancy |
| LAEA spherical north-polar | 1,048,576 | 1.057317x, 1.055433x, 1.057350x | 1.063725x / 1.062513x | 4.165 / 2.634 nm | 42 registers, 40 B local, 0 shared, 83.3% occupancy |
| LAEA spherical south-polar | 1,048,576 | 1.056744x, 1.057126x, 1.056576x | 1.063628x / 1.062891x | 4.165 / 2.634 nm | 42 registers, 40 B local, 0 shared, 83.3% occupancy |

Every case captured one fused kernel with zero memcpy or memset nodes and
returned both caller-provided output buffers by identity. GEOS uses an
analytically bounded uncertainty band around a Q1.62 visibility residual;
coordinates in that band recompute the complete native trig, visibility, and
scan-angle output. The final benchmark includes 3,078 exact/nextafter analytic
limb probes and 5,120 randomized near-limb probes per GEOS domain. Focused GPU
tests add 24,576 analytic probes per domain, and the pre-land dense review found
zero finite-mask mismatches across 1.2 million probes per domain. At
1,048,576, one post-fallback GEOS screen measured only about 1.045x, so `auto`
stays native until the clean 2,097,152 boundary. GEOS inverse, non-polar or
ellipsoidal LAEA forward, and every LAEA inverse domain remain native.

### Go/no-go gates

A new implementation is **no-go** unless all of the following pass:

1. No correctness regression against native and the existing pyproj oracle
   thresholds, including normal, boundary-adjacent, non-finite, and
   out-of-domain inputs.
2. Every replaced operation has an explicit numeric bound over its guarded
   domain, plus an end-to-end coordinate-error bound. Unknown/unbounded inputs
   take native behavior.
3. Median device-execution and synchronized public-dispatch wall speedups are
   each at least 5% (`speedup >= 1.05`) in three independent repeats at every
   size enabled for `auto`. Report all samples and p05/p50/p95; one favorable
   run is insufficient. Below that size, `auto` remains native and must pass
   the no-wall-regression noise bound.
4. Native and accelerated policies match the expected stage topology. Direct
   allocator instrumentation reports zero steady-state allocation calls, CUDA
   graph capture reports no unplanned copy/memset/kernel nodes, and returned
   objects are the caller's preallocated final buffers.
5. Native behavior is available on every device and for every unqualified
   family, direction, precision, domain, and non-finite input.
6. Hardware qualification is literal. H100 acceleration requires three stable
   H100 repeats and H100 accuracy results; RTX 4090/Ada evidence cannot qualify
   Hopper or a generic “datacenter” class.

### Adding a reusable implementation

1. Give the implementation a stable, descriptive ID. Do not encode transient
   benchmark numbers or a private kernel symbol in it.
2. Describe family, direction, operation set, precision, device predicate,
   guarded domain, error contract, and fallback in the central immutable
   registry.
3. Keep resolution pure and inspectable through `explain_strategy()`. Both
   `auto` and explicit `accelerated` must explain native fallback.
4. Reuse the implementation only where the argument proof is valid. Similar
   syntax (`sin(x)`) is not evidence that two projection domains are the same.
5. Extend the coverage-contract test and this inventory, then run the
   three-repeat end-to-end benchmark on every architecture being enabled.
6. Preserve one fused launch per mathematical stage and device residency,
   reuse intermediate scratch, and complete the GPU and pre-land review
   checklists before landing.
