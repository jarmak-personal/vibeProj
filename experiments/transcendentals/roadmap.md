# Transcendental acceleration roadmap

This roadmap describes possible future work. It is deliberately separate from
the production contract: an idea listed here has no user-visible status until
it passes the release gates and is registered.

## Current frontier

| Status | Projection directions |
|---|---|
| Automatic on qualified Ada hardware | Helmert; Sinusoidal forward and ellipsoidal inverse; Orthographic forward and spherical-equatorial inverse; polar Stereographic inverse; GEOS forward; spherical-polar LAEA forward; regular Mercator spherical forward and spherical/ellipsoidal inverse; regular LCC forward/inverse |
| Explicit accelerated | UTM forward; bounded Gnomonic inverse; ellipsoidal Mercator forward; Sinusoidal meridional-recurrence inverse; standard-Bessel Krovak inverse |
| Arithmetic/table only | Equidistant Cylindrical, Robinson, Natural Earth |
| Native after a production no-go | Transverse Mercator inverse, Web Mercator alternatives, CEA acceleration, AEQD, forward Stereographic, forward Oblique Stereographic, Eckert IV inverse fixed trig |

The authoritative implementation IDs, exact domains, and thresholds are in
[`docs/dev/transcendentals.md`](../../docs/dev/transcendentals.md).

## Candidate waves

### Reusable math

- Test range-reduced integer or double-single `atan2`, `asinh`, `atanh`,
  `exp`, and `log` only where an end-to-end profile shows enough leverage.
- Revisit CORDIC or polynomial/rational decompositions when a projection can
  prove a tight argument interval without host inspection.
- Prefer shared implementations with projection-specific domain proofs over
  projection-local copies.

### Native projection families

The broad remaining candidates are Albers Equal Area, Equal Earth, Mollweide,
Oblique Mercator, Eckert IV forward, Eckert VI, and Winkel Tripel. Krovak
forward, Gnomonic forward, LAEA inverse/non-polar forward, and the native
directions of already-partial families may be reconsidered only with a new
formulation or a materially different workload contract.

### Hardware expansion

- Re-run the complete accuracy and three-repeat public-wall qualification on
  Hopper/H100; Ada results do not predict its stronger native FP64 behavior.
- Add hardware-specific registry entries only when measured crossover and
  accuracy data justify them. A faster native path may correctly mean no
  accelerated entry.
- Requalify thresholds when compiler, CUDA, or kernel-resource changes alter
  occupancy or public dispatch cost.

## Promotion rule

Every candidate must meet the production go/no-go gates in
[`docs/dev/transcendentals.md`](../../docs/dev/transcendentals.md). Record
failed attempts in [`qualification.md`](qualification.md) so the roadmap
contains opportunities rather than repeated dead ends.
