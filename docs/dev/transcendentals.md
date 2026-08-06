# Transcendental production contract

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
| `sinu` | forward | `cos(phi)` plus ellipsoidal meridional series | kernel-boundary `phi`: unknown; spherical accelerated guard requires finite `|phi| <= pi/2`, wrapped `|lambda| <= pi`, and `0 < scale <= 6,400,000 m`; all setup requires finite `0 < a <= 6,400,000 m`, and ellipsoidal setup additionally requires finite `0 < es <= 0.012` | none | `native.libdevice`, or spherical-only `sinu.forward.fixed_q62` | qualified guarded spherical implementation; supported ellipsoidal domains native; larger/more-eccentric custom bodies rejected during setup | T2 |
| `sinu` | inverse | `cos(phi)` plus ellipsoidal meridional inverse | qualified ellipsoidal setup requires eight finite coefficients with `c0>0`, finite nonzero signed units, `0<es<=0.012`, and `0<a<=6,400,000 m`; explicit recurrence additionally requires finite raw/normalized coordinates, `|cy|<=M(89.9°)`, recovered `|phi|<=89.9°`, positive finite denominator, and `|lambda|<=pi` | final paired `sin`/`cos` in recurrence variant | `native.libdevice`, automatic `sinu.inverse.convergent_newton`, or explicit `sinu.inverse.meridional_recurrence` | convergence break preserves the complete native coordinate domain; recurrence guard failure is complete-warp exact native | T2 |
| `merc` | forward | `sin`, `tan`, `pow`, `log` | spherical automatic path requires `e=0`; ellipsoidal explicit path requires `0<e<=0.1` and hot `|phi|<=89.9 degrees`; both require finite raw/derived coordinates, nonzero units, `0<k0<=1`, and `0<a<=6,400,000 m` | none | `native.libdevice`, `merc.forward.spherical.product_poly`, or explicit-only `merc.forward.ellipsoidal.product_poly` | spherical zero-exponent `pow` removal is exact; ellipsoidal product polynomial has complete-warp exact native fallback | T2 |
| `merc` | inverse | `exp`, `atan`, iterative `sin`/`pow` | accelerated setup requires finite nonzero units, `0 <= e <= 0.1`, finite `k0 > 0`, and `0 < a <= 6,400,000 m`; raw and normalized coordinates, longitude, and series coefficients finite | none | `native.libdevice`, or `merc.inverse.exp_series` in spherical/ellipsoidal A/B domains | shared sixth-order conformal-to-geodetic series; complete-warp exact native fallback | T2 |
| `webmerc` | forward | `tan`, `log` | `phi` and singular distance to poles: unknown | none | `native.libdevice` | native only | T3 |
| `webmerc` | inverse | `exp`, `atan` | normalized northing and exponent: unbounded | none | `native.libdevice` | native only | T3 |
| `tmerc` | forward | three `sin`+`cos` arguments, `rsqrt`, `atan2`, `asinh` | general TM derived arguments: unknown; qualified UTM guard: `|lambda| <= 0.06`, `|asinh_arg| <= 0.06` with error `<= 2e-17`, `|atan_delta| <= 9.01e-4` with error `<= 2.3e-16 rad`; Q1.62 angles guarded to `[-pi, pi]` | pair `2phi`, Gaussian latitude, and `lambda` | native paired `sincos`/libdevice, or `tmerc.forward.fixed_q62` | qualified guarded implementation; native otherwise | T2 |
| `tmerc` | inverse | `sin`/`cos`, `exp`, `sinh`, two `atan2`, `hypot` | inverse complex coordinates: unknown/unbounded | pair `2Cn`; pair `Cn` | `native.libdevice` | paired native only | T3 |
| `lcc` | forward | `sin`, `tan`, `pow`, output `sin`/`cos` | spherical/ellipsoidal 1SP/2SP regular cones require finite setup, signed nonzero units, exact `k0==1`, `0<=e<=0.1`, `0<a<=6,400,000 m`, and `abs(n)>=0.2`; finite non-pole coordinates and bounded `theta` | pair `theta` | `native.libdevice`, or `lcc.forward.conformal_reframe` | spherical log/exp power; ellipsoidal native outer power with exp/atanh inner correction; complete-warp native fallback | T2 |
| `lcc` | inverse | `sqrt`, `atan2`, `pow`, iterative `atan`/`sin`/`pow` | same setup except no `abs(n)` floor; finite raw/normalized coordinates and positive radius ratio; huge finite raw coordinates may yield positive-infinite ratio and the pole limit | none | `native.libdevice`, or `lcc.inverse.conformal_reframe` | spherical logarithmic reconstruction; ellipsoidal native outer power plus bounded six-step exp/atanh recovery and a sixth-step contraction correction when `abs(dphi)>=1e-14`; apex/nonpositive/non-finite complete-warp native fallback | T2 |
| `stere` | forward | `sin`, `tan`, `pow`, output `sin`/`cos` | adjusted `phi`: unknown; `lambda` finite wrapped | pair `lambda` | `native.libdevice` | paired native only | T3 |
| `stere` | inverse | `sqrt`, `atan2`, iterative `atan`/`sin`/`pow` | public ellipsoidal A/B north/south and C south domains; accelerated uniform guard requires `0.05 <= e <= 0.2` and `0 < a <= 6,400,000 m`; shared helper guards each iterative sine angle | none | `native.libdevice`, or `stere.inverse.fixed_q62` | qualified Q1.62 iterative sine in five exact domains; native otherwise | T2 |
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
| `gnom` | forward | `sin`/`cos` of `phi` and `lambda` | exact spherical equatorial/north-pole/south-pole/oblique origin domains; visible only when `cos(c) >= 1e-10`; ellipsoidal setup rejected | pair `phi`; pair `lambda` | `native.libdevice` | paired native only; horizon and hidden hemisphere return non-finite coordinates | T3 |
| `gnom` | inverse | `hypot`, clamped `asin`, `atan2` | exact spherical origin domains; explicit bounded path requires equatorial or `|cos(phi0)| >= 0.5`, finite non-axis `1e-24 < rho^2 <= 0.02`, and `0 < scale <= 6,400,000 m`; ellipsoidal setup rejected | reciprocal-square-root normalization removes both `hypot` calls | `native.libdevice`, or explicit-only `gnom.inverse.guarded_rsqrt_reframe` | guarded implementation is expert opt-in; every guard failure calls the exact native expression and `auto` always stays native | T2 |
| `moll` | forward | `sin`; iterative `sin`/`cos`; output `sin`/`cos` | `phi`: unknown; converged `theta` bound not enforced in kernel | pair `2theta`; pair output `theta` | `native.libdevice` | paired native only | T3 |
| `moll` | inverse | two clamped `asin`, `sin`, `cos` | both `asin` inputs clamped; division by `cos(theta)` can approach singularity | none | `native.libdevice` | native only | T3 |
| `omerc` | forward | `sin`, `tan`, `pow`, `log`, `cos`, `atan2` | `U` clamped below `|1|`; remaining derived arguments depend on CRS/domain | pair `B*lambda` | `native.libdevice` | paired native only | T3 |
| `omerc` | inverse | `exp`, `sin`/`cos`, `sqrt`, `pow`, iterative `atan`/`sin`/`pow`, `atan2` | `U'` clamped below `|1|`; exponential and projected inputs unbounded | pair `B*u/A` | `native.libdevice` | paired native only | T3 |
| `krovak` | forward | `sin`, `tan`, `pow`, `atan`, `sin`/`cos`, two `asin` | inverse-trig inputs not explicitly clamped; `phi` unknown | pair `U`; pair `V`; pair cone angle | `native.libdevice` | paired native only | T3 |
| `krovak` | inverse | `sqrt`, `atan2`, `atan`, `pow`, paired `sin`/`cos`, `asin`, `tan`, iterative `sin`/`pow`; guarded reframe adds `log`, `tanh`, and a conformal series | exact standard-Bessel regular/north-oriented setup; accelerated coordinates require positive finite radius and finite intermediates with recovered `|phi| <= 80 degrees` | pair `T`; pair `D`; native inverse iteration uses `sin(phi)` only, not a trig pair | `native.libdevice`, or explicit-only `krovak.inverse.guarded_log_ratio` | guarded log-ratio and sixth-order conformal recovery; either cold vote makes the complete warp exactly native; `auto` always remains native | T2 |
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
| `tmerc.forward.fixed_q62` | `tmerc` | forward UTM | fp64 | Ada `sm_89`, weak-native-fp64 consumer class; explicit `accelerated` only | n/a | paired `sin`/`cos`, TM `atan2` correction, `asinh`; per-operation guards above; measured explicit crossover 256 elements |
| `sinu.forward.fixed_q62` | `sinu` | spherical forward | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 524,288 | Q1.62 `cos(phi)`; valid latitude/wrapped longitude and `0 < scale <= 6,400,000 m`, native otherwise |
| `sinu.inverse.convergent_newton` | `sinu` | ellipsoidal inverse | fp64 | Ada `sm_89`, weak-native-fp64 consumer class; `auto` only | 1 | native meridional/derivative expressions and ten-step cap with `fabs(delta)<1e-14` convergence termination; full native coordinate and sentinel domain |
| `sinu.inverse.meridional_recurrence` | `sinu` | ellipsoidal inverse hot domain | fp64 | Ada `sm_89`, weak-native-fp64 consumer class; explicit `accelerated` only | n/a | two `sincos(2phi)` recurrence steps, one native-shaped correction, final paired `sincos`; exact setup and ±89.9°/wrapped-longitude complete-warp guards |
| `ortho.forward.fixed_q62` | `ortho` | forward | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 262,144 | Q1.62 paired `sin`/`cos`; atomic argument guard and `0 < scale <= 6,400,000 m`, native otherwise |
| `ortho.inverse.guarded_reframe` | `ortho` | inverse, spherical equatorial only | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 524,288 | guarded algebraic reframe for normalized `1e-16 < rho^2 <= 0.99`; center/near-center, axes, horizon/outside, non-finite, scale, and conditioning failures call the exact native expression |
| `gnom.inverse.guarded_rsqrt_reframe` | `gnom` | inverse, spherical equatorial/bounded-oblique only | fp64 | Ada `sm_89`, weak-native-fp64 consumer class; explicit `accelerated` only | n/a | guarded reciprocal-square-root reframe for `1e-24 < rho^2 <= 0.02`; `auto` is intentionally native because a random 10% cold mixture regresses to about 0.92x, while hot-only explicit workloads measure about 1.31-1.47x |
| `stere.inverse.fixed_q62` | `stere` | inverse, ellipsoidal A/B north/south and C south | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 1,000,000 | Q1.62 sine inside `phi2`; CRS setup proves positive `akm1` and sign, uniform `0.05 <= e <= 0.2` and `a <= 6,400,000 m` guards, per-angle native fallback |
| `geos.forward.fixed_q62` | `geos` | forward, sphere/ellipsoid, sweep x/y | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 2,097,152 | Q1.62 pairs for geocentric latitude and longitude; complete-native visibility-uncertainty fallback plus launch-uniform finite geometry/`a <= 6,400,000 m` guard |
| `laea.forward.polar.fixed_q62` | `laea` | forward, spherical north/south polar only | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 1,048,576 | Q1.62 paired longitude trig; exact pole/antipode/non-finite handling and `a <= 6,400,000 m` guard |
| `merc.forward.spherical.product_poly` | `merc` | spherical forward A/B | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 262,144 | removes the zero-exponent `pow`; `e==0` device guard and full-domain bitwise-native result |
| `merc.forward.ellipsoidal.product_poly` | `merc` | ellipsoidal forward A/B | fp64 | Ada `sm_89`, weak-native-fp64 consumer class; explicit `accelerated` only | n/a | product reframe and degree-eight exponential polynomial; `0<e<=0.1`, `|phi|<=89.9 degrees`, and bounded setup guards; `auto` disabled by polar-cap mixture regression |
| `merc.inverse.exp_series` | `merc` | inverse, spherical/ellipsoidal A/B | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 65,536 | native `exp`/`atan` conformal seed followed by reusable sixth-order Poder/Engsager recovery; finite raw/normalized/setup/coefficients guard |
| `lcc.forward.conformal_reframe` | `lcc` | spherical/ellipsoidal 1SP/2SP forward regular cones | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 65,536 | log/exp or native-outer-power conformal reframe; exact finite setup, `k0==1`, `abs(n)>=0.2`, pole/non-finite/theta and complete-warp native guards |
| `lcc.inverse.conformal_reframe` | `lcc` | spherical/ellipsoidal 1SP/2SP inverse | fp64 | Ada `sm_89`, weak-native-fp64 consumer class | 128 | logarithmic or native-outer-power conformal reconstruction; six-step `1e-14` recovery bound with a final contraction correction only when step six has not converged; apex/nonpositive/non-finite complete-warp native guards |
| `krovak.inverse.guarded_log_ratio` | `krovak` | standard-Bessel inverse, regular and north oriented | fp64 | Ada `sm_89`, weak-native-fp64 consumer class; explicit `accelerated` only | n/a | setup-derived exact Bessel domain, guarded log-ratio and sixth-order conformal recovery; positive finite radius, finite intermediates, `|phi| <= 80 degrees`, and complete-warp exact native fallback |

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

## Qualification record

Dated benchmark results, accepted and rejected experiments, and wave-by-wave
decision history live in
[qualification record](https://github.com/jarmak-personal/vibeProj/blob/main/experiments/transcendentals/qualification.md).
Potential future work is tracked separately in
the [experimental roadmap](https://github.com/jarmak-personal/vibeProj/blob/main/experiments/transcendentals/roadmap.md).

Those experimental documents explain why the registry has its current shape.
They do not define production behavior; this inventory, the central immutable
registry, and `tests/test_transcendental_coverage.py` do.

## Go/no-go gates

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

## Adding a reusable implementation

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
