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
| `sinu` | forward | `cos(phi)` | `phi`: unknown | none | `native.libdevice` | native only | T1 |
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
| `aea` | forward | `sin`, `log`, `sqrt`, output `sin`/`cos` | `phi`: unknown; negative radicand is clamped to zero; `theta=n*lambda` depends on CRS | pair `theta` | `native.libdevice` | paired native only | T3 |
| `aea` | inverse | `sqrt`, `atan2`, clamped `asin`, iterative `sin`/`cos`/`log` | projected radius: unknown; `asin` argument clamped | iterative pair `phi` | `native.libdevice` | paired native only | T3 |
| `laea` | forward | `sin`, `log`, clamped `asin`, `sqrt`, `sin`/`cos` | authalic `asin` input clamped; other derived denominators depend on mode/domain | pair authalic latitude; pair `lambda` | `native.libdevice` | paired native only | T3 |
| `laea` | inverse | `sqrt`, clamped `asin`, `sin`/`cos`, `atan2`, iterative `sin`/`cos`/`log` | central-angle `asin` input clamped; projected radius otherwise unknown | pair central angle; iterative pair `phi` | `native.libdevice` | paired native only | T3 |
| `eqearth` | forward | `sin`, `log`, two clamped `asin`, `cos`, `sqrt` | inverse-trig inputs clamped; `phi` otherwise unknown | none | `native.libdevice` | native only | T3 |
| `eqearth` | inverse | iterative polynomial, `sin`/`cos`, clamped `asin`, iterative `sin`/`cos`/`log` | authalic inverse input clamped; projected coordinates otherwise unknown | pair `theta`; iterative pair `phi` | `native.libdevice` | paired native only | T3 |
| `cea` | forward | `sin`, `log` | `phi`: unknown | none | `native.libdevice` | native only | T3 |
| `cea` | inverse | clamped `asin`, iterative `sin`/`cos`/`log` | initial `asin` argument clamped; projected northing otherwise unknown | iterative pair `phi` | `native.libdevice` | paired native only | T3 |
| `ortho` | forward | `sin`/`cos` of `phi` and `lambda` | `phi`: unknown; `lambda` finite wrapped | pair `phi`; pair `lambda` | `native.libdevice` | paired native only | T1 |
| `ortho` | inverse | `sqrt`, clamped `asin`, `sin`/`cos`, `atan2` | central-angle `asin` input clamped; projected radius otherwise unknown | pair central angle | `native.libdevice` | paired native only | T3 |
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
| `geos` | forward | `tan`, `atan`, `sin`/`cos`, `sqrt`, `atan2`, clamped `asin` | output `asin` input clamped; visibility and `phi` bounds not enforced here | pair geocentric latitude; pair `lambda` | `native.libdevice` | paired native only | T3 |
| `geos` | inverse | paired `sin`/`cos`, `sqrt`, `atan2`, `atan` | scan angles/projected coordinates unknown; discriminant clamped non-negative | pair x scan angle; pair y scan angle | `native.libdevice` | paired native only | T3 |
| `robin` | forward | none (table polynomial) | n/a | none | arithmetic/table only | none | T0 |
| `robin` | inverse | none (table polynomial/Newton) | n/a | none | arithmetic/table only | none | T0 |
| `wintri` | forward | `sin`/`cos`, clamped `acos`, `sin` | `acos` input clamped; `phi` otherwise unknown; `lambda` finite wrapped | pair `phi`; pair `lambda/2` | `native.libdevice` | paired native only | T3 |
| `wintri` | inverse | iterative `sin`/`cos`, clamped `acos`, `sin` | projected initial values and iteration domain unknown; `acos` input clamped | pair `phi`; pair `lambda/2` | `native.libdevice` | paired native only | T3 |
| `natearth` | forward | none (polynomial) | n/a | none | arithmetic only | none | T0 |
| `natearth` | inverse | none (polynomial/Newton) | n/a | none | arithmetic only | none | T0 |
| `aeqd` | forward | paired `sin`/`cos`, clamped `acos`, `sin` | `acos` input clamped; antipode is singular | pair `phi`; pair `lambda` | `native.libdevice` | paired native only | T3 |
| `aeqd` | inverse | `sqrt`, `sin`/`cos`, clamped `asin`, `atan2` | projected radius unknown; `asin` input clamped | pair central distance | `native.libdevice` | paired native only | T3 |
| `helmert` | datum shift (forward or inverse pipeline) | paired `sin`/`cos` for source latitude/longitude, `sqrt`, `atan2`, iterative `sin`, optional final paired `sin`/`cos` | Q1.62 guard `|angle| <= pi`; non-finite/outside use native; angles within `0.02 rad` of a pole use native for height conditioning | pair source latitude; pair source longitude; pair final latitude with height | native paired `sincos`/libdevice, or `helmert.fixed_q62` for bounded trig only | qualified guarded implementation; native otherwise | T2 |

## Coverage matrix

The user-facing coverage matrix is intentionally narrower than the inventory:

| ID | Family | Direction | Precision | Device qualification | Auto min elements | Guarded operations |
|---|---|---|---|---|---:|---|
| `native.libdevice` | `*` | forward/inverse/Helmert | all supported | universal fallback | 0 | all special math |
| `helmert.fixed_q62` | `helmert` | datum shift | all public modes (Helmert kernel stays fp64) | Ada `sm_89`, weak-native-fp64 consumer class | 131,072 | `sin`, `cos`; `|angle| <= pi`, near-pole native guard |
| `tmerc.forward.fixed_q62` | `tmerc` | forward UTM | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 256 | paired `sin`/`cos`, TM `atan2` correction, `asinh`; per-operation guards above |

`tests/test_transcendental_coverage.py` binds these IDs and capabilities to
the public registry and resolver. Adding a documented accelerated row without
registry support, or registry support without updating the contract, fails the
test. Native-only inventory rows do not imply an accelerated registry entry.

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

The default workload grid covers 32 through 5,000,000 elements on logarithmic
steps for forward UTM and both Helmert directions. `auto` must resolve native
below the registry's `min_elements` and the qualified implementation at or
above it. The small-size gate permits at most 2% aggregate and 5% per-repeat
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
