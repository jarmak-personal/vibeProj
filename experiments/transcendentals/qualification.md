# Transcendental qualification record

This is the historical engineering record for vibeProj's transcendental
acceleration work. It preserves the hardware-specific measurements, rejected
approaches, and reasoning behind the current production registry.

The maintained production inventory, resolver contract, and release gates live
in [`docs/dev/transcendentals.md`](../../docs/dev/transcendentals.md). Values
here describe the named qualification runs; they are not promises for other
devices or future releases.

## Ada qualification evidence

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

The size grid established the conservative crossovers below. The UTM crossover
is guidance for explicit callers: a later broad-longitude suite showed enough
guard fallback to regress, so it is not an automatic threshold. Helmert remains
automatic. Every row from the stated minimum through 5,000,000 elements
exceeded 1.05x in all three repeats.

| Operation | Last tested size below threshold | Policy minimum | Native / selected wall p50 at threshold | Threshold repeat speedups | Selected speedup at 5M |
|---|---:|---:|---:|---|---:|
| forward UTM (explicit only) | 128 | 256 | 0.03019 / 0.02764 ms | 1.091x, 1.089x, 1.094x | 1.548x |
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

## Wave 1 projection qualification evidence

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
below 1.05x. After adding its exact ellipsoidal baseline, the spherical cos-only
branch was formally requalified at 524,288 with 10 warm-ups, 30 iterations, and
three repeats: `auto` wall speedups were 1.055536x, 1.057237x, and 1.055013x;
`auto` remained native at 524,287. The accelerated kernel used 31 registers,
40 bytes local memory, zero shared memory, and retained 100% occupancy.
Orthographic passed at 262,144 with 1.204x,
1.206x, and 1.194x wall speedups; `auto` remained native at 262,143. Every
larger tested grid size through 5,000,000 passed all three repeats. Companion
inverse paths remain `native.libdevice`.

At the exact 6,400,000 m scale ceiling, a separate two-million-coordinate
dense proof measured 3.725290298 nm maximum error for sinusoidal and
2.832507030 nm across orthographic origins `{-90, -45, 0, 45, 90}`. The exact
ceiling and its nextafter-below value retain Q1.62; nextafter-above, `1e12`,
zero, negative, infinity, and NaN scales execute bitwise-native uniformly.

Ellipsoidal Sinusoidal uses a seventh-order meridional series only for finite
`0 < es <= 0.012` and finite `0 < a <= 6,400,000 m`; setup rejects larger or
malformed bodies before CPU/GPU dispatch. A 16,385-latitude dense PROJ comparison
at the inclusive `es=0.012`, `a=6,400,000 m` boundary measured 5.588/2.945 nm forward maximum/p99
coordinate error and 9.241/4.649 nm inverse physical maximum/p99 error.
`nextafter(0.012, +inf)`, `nextafter(6,400,000 m, +inf)`, nonpositive or
non-finite scale, negative/non-finite eccentricity squared, and representative
large or high-eccentricity custom bodies raise `UnsupportedProjectionError`.

### Eckert IV inverse rejection

Replacing the native inverse `sin(theta), cos(theta)` pair with Q1.62 measured
a 1.0891x device speedup, but at `cy / C_y = -0.999994` and `lambda = pi` its
sub-ULP pair error amplified to approximately 271.916 nm and failed the 10 nm
gate. `python benchmarks/eck4_inverse_rejection.py` and its deterministic
coverage test reproduce the edge. No accelerated Eckert IV inverse strategy
was registered.

## Wave 2A Orthographic inverse qualification evidence

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

## Wave 2B forward qualification evidence

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

## Wave 2C stereographic evidence

Polar Stereographic inverse qualified on the RTX 4090 through normal public
projected-CRS-to-geodetic `transform_buffers()` calls with preallocated
outputs. The timed distribution covers full longitude and polar latitude
magnitudes from 45 through 90 degrees. All five public exact domains—variants
A and B in both hemispheres and variant C south—passed three interleaved
30-iteration device and synchronized-wall repeats at 1,000,000, 2,000,000,
and 5,000,000 coordinates. The minimum wall repeat was 1.055206x at the
threshold and 1.056216x at 5,000,000; 999,999 coordinates resolved native and
showed no regression. Representative maximum error was 1.582008 nm.

The kernel replaces only the iterative `phi2` sine. CRS setup proves positive
`akm1` and exact sign for the registered methods. A launch-uniform guard
requires `0.05 <= e <= 0.2` and `0 < a <= 6,400,000 m`, while the shared helper
routes non-finite or out-of-range iterative angles to native sine. Scale at the
exact ceiling passes the coordinate contract; nextafter above, `1e12 m`,
near-spherical eccentricity, and high eccentricity are bitwise native. The two
eccentricity fallback screens retained 1.013–1.015x synchronized wall ratios,
so the selected-but-native kernel branch has no material regression.
Separate logarithmic projected-radius accuracy probes from `1e-15` through
`1e140`, centers, axes, huge values, and non-finite inputs reached 3.164 nm
with matching classifications. Those adversarial data are correctness
evidence, not a universal performance claim; the deliberately huge-radius mix
missed 1.05x at 1,000,000.

Polar forward Q1.62 and both Oblique Stereographic forward approaches were
rejected. At 2,097,152 full-domain coordinates their best wall repeats were
about 0.941x, 0.896x, and 0.901x native. An outlined oblique algebraic form was
1.112–1.212x in a direct RD bounding-box kernel, but its public path reached
only about 1.044x there and about 0.958x over the full globe. Coordinate
locality is not an exact dispatch domain, so `auto` does not depend on input
ordering or inspect data on the host. `benchmarks/wave2c_rejection.py` retains
these forward no-go results and deterministic guard/warp profiles. The earlier
Oblique Stereographic inverse screen was reversed twice and is intentionally
absent from all retained claims.

## Wave 3A bounded Gnomonic inverse evidence

The reciprocal-square-root inverse reframe qualified only as an explicit
strategy for spherical equatorial and bounded-oblique origins with normalized
`1e-24 < rho^2 <= 0.02`. Homogeneous hot workloads measured approximately
1.31–1.47x native, while a random 10% cold mixture measured approximately
0.92x. Because normal dispatch cannot prove the coordinate-radius bound without
inspecting device data, `auto` remains native and every guard failure executes
the exact native expression.

## Wave 3B regular Mercator evidence

The final public replay covered spherical and ellipsoidal variants A/B in both
directions on the qualified RTX 4090. It used preallocated public
`transform_buffers()` calls, 5,000,000 coordinates, 10 warm-ups, three repeats
of 30 iterations, and device-event plus synchronized-wall timing. The workload
grid added N-1/N for automatic crossovers and 262,144/5,000,000 for the
explicit ellipsoidal-forward performance contract. All eight main cases, all
22 grid rows, setup/coordinate/scale guards, pyproj comparisons, graph
topology, allocation, output-identity, and retained mixture sweeps passed.

| Case group | Policy / minimum | Device / wall at 5M | Max / p99 error | Minimum qualified repeat, device / wall |
|---|---|---:|---:|---:|
| spherical forward A/B | auto, 262,144 | 1.129x / 1.128x | exact / exact | 1.124x / 1.105x |
| ellipsoidal forward A/B | explicit only, performance screened from 262,144 | 1.261x / 1.259x | 7.451 / 0.699 nm | 1.224x / 1.188x |
| spherical inverse A/B | auto, 65,536 | 1.805x / 1.792x | exact / exact | 1.625x / 1.302x |
| ellipsoidal inverse A/B | auto, 65,536 | 8.349x / 8.243x | 6.328 / 3.164 nm | 6.983x / 4.232x |

Both forward kernels use 36 registers, 40 bytes local memory, zero shared
memory, and calculated full occupancy. The inverse series reduces native's 38
registers to 36 while retaining 40 bytes local and zero shared. The native
inverse ABI remains unchanged; six reusable conformal-to-geodetic coefficients
are added only to the accelerated inverse source and argument pack.

Ellipsoidal forward is deliberately excluded from `auto`. A uniform
`e=0.1`, `a=6,400,000 m` five-million-point screen found 14.901 nm one-ULP
differences inside the extreme polar cap. Native `exp(-correction)` and a
degree-ten polynomial reproduced the same four failures; subtracting the
correction after `log` produced 280. The production hot domain therefore ends
at `|latitude|=89.9 degrees`, and a warp containing any coordinate beyond that
bound executes the exact native expression. At both 262,144 and 5,000,000,
retained random-lane cap fractions of 0%, 0.1%, 1%, 10%, 50%, and 100% kept
every cold lane bitwise-native and the complete output below 10 nm. Hot-only
speedup was about 1.27x, but 10% and 50% mixtures measured about 0.962-0.969x;
that negative result is an enforced benchmark gate and is why only explicit
callers may select the ellipsoidal-forward ID.

Replay the immutable qualification artifact with:

```console
flock /tmp/vibeproj-wave3-gpu.lock -c 'uv run python \
  benchmarks/bench_transcendental_policy.py --case merc --n 5000000 \
  --workload-sizes 5000000 --warmup 10 --iterations 30 --repeats 3 \
  --oracle-n 100000 --precision fp64 --seed 42 --enforce-gates \
  --json /tmp/wave3b_merc_public_qualification.json'
```

## Wave 3C Lambert Conformal Conic evidence

Wave 3C qualifies spherical and ellipsoidal LCC 1SP/2SP in both directions on
Ada `sm_89`. Forward uses the accelerated path only for the exact regular-cone
domain `abs(n)>=0.2`; inverse also qualifies near-equator cones. Both require
finite setup, signed nonzero units, exact `k0==1`, `0<=e<=0.1`, and
`0<a<=6,400,000 m`. The thresholds are 65,536 forward and 128 inverse.

The public artifact includes northern, southern, and zero-standard-parallel
cones, both geometry modes and variants, exact `e=0.1`/`a=6,400,000 m`
boundaries, nextafter setup boundaries, exact poles/apex, non-finite values,
huge finite inverse inputs, and random opposite-cone extreme fractions of 0%,
0.1%, 1%, 10%, 50%, and 100%. Guarded fallbacks are complete-warp and bitwise
native. The accelerated forward kernel currently compiles to 38 registers and
the accelerated inverse kernel to 40 registers. Both use 56 bytes of local
memory, zero shared memory, six active 256-thread blocks per SM, and full
calculated occupancy on the qualification toolchain.

Forward crossover screens at 4,096, 8,192, 16,384, and 32,768 were retained as
negative threshold evidence: device execution was faster, but one or more
synchronized public-wall repeats lacked the required 1.05 margin across the
full regular/mixed matrix. At 65,536 the worst candidate wall repeat exceeded
1.07 in the crossover screen, so the library uses the single conservative
65,536 threshold for every qualified forward domain. Inverse remains robust at
128. CUDA-event ratios for below-threshold AUTO/native identity calls are
informational because timer quantization dominates at those sizes; exact
same-ID cache resolution and synchronized-wall no-regression remain gated.

Numeric no-go evidence for the rejected forward log-tan and explicit-zero-exp
variants and inaccurate two-step inverse iterations is retained in the central
benchmark metadata. Native LCC source and ABI remain unchanged; the guarded
helpers and reframe symbols exist only in accelerated sources.
The ellipsoidal inverse performs at most six exp/atanh fixed-point steps using
the native `1e-14` stopping rule. When the sixth delta has not converged, it is
divided by the local contraction complement
`1 - e^2(1-sin^2(phi))/(1-e^2 sin^2(phi))`. This final-step correction requires
no seventh transcendental evaluation and is what keeps the closed `e=0.1`,
`a=6,400,000 m` boundary below 10 nm.

Replay the scoped 3x30 public qualification with:

```console
flock /tmp/vibeproj-wave3-gpu.lock -c 'uv run python \
  benchmarks/bench_transcendental_policy.py --case lcc --n 5000000 \
  --workload-sizes 5000000 --warmup 10 --iterations 30 --repeats 3 \
  --oracle-n 100000 --precision fp64 --seed 42 --enforce-gates \
  --json /tmp/wave3c_lcc_public_qualification.json'
```

## Wave 3D ellipsoidal Sinusoidal inverse evidence

Wave 3D assigns two implementations to the same exact ellipsoidal inverse setup
domain through disjoint policies. `auto` selects
`sinu.inverse.convergent_newton`, which leaves every native meridional arc,
derivative, final longitude, invalid-image, pole, and non-finite expression in
place and adds only a break after an applied Newton correction falls below
`1e-14`. Because it has no coordinate guard, it covers the complete native
inverse image. The exact automatic threshold is one coordinate: public 3x30
screens at N=1 passed for WGS84 and the closed `es=.012`, `a=6,400,000 m`
boundary, with N=2 retained as adjacent confirmation.

Explicit `accelerated` selects `sinu.inverse.meridional_recurrence`. Its first
two Newton steps evaluate the seven meridional harmonics from one
`sincos(2*phi)` call and a sine recurrence; one direct native-shaped correction
then restores the production meridional expression, followed by paired final
`sincos`. The pre-recurrence warp vote requires finite raw/normalized values and
`|cy|<=M(89.9 degrees)`. A second vote requires recovered
`|phi|<=89.9 degrees`, a positive finite longitude denominator, and wrapped
`|lambda|<=pi`. Failure at either vote sends the complete warp through an
outlined copy of the exact native ellipsoidal inverse.

On the RTX 4090, the final public qualification covered both WGS84 and the exact
`es=.012`, `a=6,400,000 m` boundary. At five million coordinates, the automatic
kernel measured 3.026x wall / 3.034x device speedup in both domains. The
recurrence hybrid measured 4.621x wall / 4.643x device on WGS84 and 4.623x /
4.644x at the boundary. At the exact N=1 endpoint, the minimum wall repeats
across both domains were 1.6435x and 1.8215x respectively. Qualification also
included 0%, 0.1%, 1%, 10%, 50%, and 100% randomized
pole/non-finite/off-image mixtures; finite classes matched, and the formal
suite's maximum/p99 native-relative horizontal error was 6.583/0 nm. The exact
boundary maximum was 6.582903846 nm. In the WGS84 fallback sweep, random 10%
cold and all-cold recurrence workloads had median wall speedups of only 1.0238x
and 1.0032x, below the automatic 1.05 contract; these retained negative results
are why only explicit callers can select the recurrence ID.

Both kernels compile to 38 registers and zero shared memory on the qualification
toolchain. The automatic kernel uses 40 bytes of local memory and the recurrence
kernel 56 bytes. Native source and ABI remain byte-for-byte unchanged. The
automatic accelerated ABI matches native; only the recurrence source adds its
setup-derived `M(89.9 degrees)` scalar. Both remain one fused kernel launch with
preallocated-buffer, independent-stream, allocation, and CUDA-graph residency
coverage.

Replay the scoped public qualification with:

```console
flock /tmp/vibeproj-wave3-gpu.lock -c 'uv run python \
  benchmarks/bench_transcendental_policy.py --case sinu-inverse --n 5000000 \
  --workload-sizes 1,2,5000000 --warmup 10 --iterations 30 --repeats 3 \
  --oracle-n 100000 --precision fp64 --seed 42 --enforce-gates \
  --json /tmp/wave3d_sinu_inverse_public_qualification.json'
```

The final formal artifact has SHA-256
`62e9db49f7a3fcac9f99a1aa6c64861f0ebd27eba5d2552bdd7d204368c82937`.

## Wave 3E standard-Bessel Krovak inverse qualification

Wave 3E registers one explicit-only strategy,
`krovak.inverse.guarded_log_ratio`. Its host domain is derived from the complete
CRS setup: ellipsoidal geometry; operation method `Krovak` with exact `(-1,-1)`
axis signs or `Krovak (North Orientated)` with exact `(+1,+1)` signs; finite,
nonzero signed units; and finite launch scalars. The standard Bessel semi-major
axis and eccentricity use absolute `math.isclose` tolerances of `1e-9 m` and
`2e-16`. `B`, `k`, `n`, normalized radius, pseudo-parallel tangent, cone-axis
sine/cosine, `log(k)`, and all six conformal-series coefficients use relative
and absolute tolerances of `2e-15`. The central meridian and false offsets need
only be finite: `lam0` is deliberately not pinned because the Ferro and
Greenwich realizations differ.

The two exact domains cover EPSG:2065/5513/8352 regular public
X=Southing/Y=Westing coordinates and EPSG:5221/5514/8353 north-oriented
X=Easting/Y=Northing coordinates. Each projected CRS is qualified against its
own geodetic CRS so the benchmark does not introduce WGS 84 datum operations.
Supported custom and spherical setups, non-finite setup, mismatched-sign, or
otherwise invalid setups resolve native. Modified Krovak CRS methods remain
unsupported. Forward Krovak, fp32/double-single, non-Ada devices, and native
policy also remain native.

The guarded kernel replaces iterative conformal recovery with a log-ratio seed
and the shared sixth-order conformal-to-geodetic series. Positive finite radius,
finite derived values, and recovered `|phi| <= 80 degrees` are required. A cold
lane at either vote sends its complete warp through the outlined exact native
inverse. Fixed-six iteration and per-lane fallback variants are rejected
research and are not registered. The retained research screen measured about
2.583x gain with at most 7.12 nm native-relative horizontal error, but these are
not the formal production qualification metrics. `auto` remains
`native.libdevice` for every workload size.

The formal public run must cover main correctness, kernel resources, exact
setup, signed units/axis semantics, scale rejection, restored-state replay,
N=1/2/5,000,000 timing, and homogeneous complete-warp cold sweeps at
0/0.1/1/10/50/100%. The integrated formal run used:

```console
flock /tmp/vibeproj-wave3-gpu.lock -c 'uv run python \
  benchmarks/bench_transcendental_policy.py --case krovak-inverse --n 5000000 \
  --workload-sizes 1,2,5000000 --warmup 10 --iterations 30 --repeats 3 \
  --oracle-n 100000 --precision fp64 --seed 42 --enforce-gates \
  --json /tmp/wave3e_krovak_inverse_public_qualification.json'
```

All six cases passed every enforced gate. At five million coordinates, the
minimum synchronized-wall accelerated speedup across the six main cases was
2.6901x and the minimum CUDA-event speedup was 2.6976x. The formal maximum/p99
native-relative horizontal error was 7.9089/4.7453 nm. Every N=1, N=2, and
N=5,000,000 explicit timing row required its median and all three wall/device
repeats to reach 1.05x; the worst wall repeats were 1.4264x, 1.4262x, and
2.6898x respectively, and every `auto` row resolved `native.libdevice`. All
homogeneous 0/0.1/1/10/50/100% cold-warp sweeps, setup,
unit/sign, scale, replay, allocation, topology, resource, and pyproj-regression
gates passed. Native/candidate kernels used 38/40 registers and 40/56 bytes of
local memory respectively, with zero shared memory. The formal artifact SHA-256
is `1cddab8c4850768cc25d942c8b2761143c21d2a7a4a2def739608578a5f5a22e`.


## Wave 3F Transverse Mercator inverse no-go

A bounded Q1.62-trig and hyperbolic-series prototype reached approximately
1.061x on homogeneous hot UTM inputs with no more than 3.589 nm observed error.
It did not qualify: a 1% cold mixture missed the 1.05 gate, broad-coordinate and
10%-cold workloads regressed, and setup-only dispatch cannot prove the required
coordinate bounds. The remaining proof and branching surface did not justify
an explicit-only production strategy. Inverse Transverse Mercator therefore
remains `native.libdevice`; the landed domain-baseline tests preserve the
correctness target for future work.

## Retained research scripts

- `benchmarks/bench_int64_sincos.py` compares isolated paired-trig experiments.
- `benchmarks/bench_alt_transcendentals.py` retains rejected/research `atan2`
  and `asinh` variants.
- `benchmarks/bench_tmerc_int64.py` and `benchmarks/bench_helmert_int64.py`
  retain raw-kernel and alternative fixed-core evidence.

These experimental modes are not public policies.
