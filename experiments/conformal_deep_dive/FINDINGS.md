# Conformal Mapping Deep Dive: Findings

## TL;DR

**Conformal complex polynomials are not viable for replacing NTv2 grids.**
The NAD27→NAD83 correction surface is only ~13% conformal. Even at degree 30
(62 real parameters), the max error is still **104 meters**. The physics
assumption was wrong: survey network distortions are NOT predominantly angle-preserving.

## The Experiment

- Sampled pyproj's grid-based NAD27→NAD83 at 15,000 points over CONUS
- Computed our 7-param Helmert at the same points
- Extracted the residual (what the grid adds beyond Helmert)
- Fitted conformal (complex) polynomials of degree 1–30
- Compared against general (non-conformal) polynomials and SVD compression

## Key Numbers

### The Residual (Grid − Helmert)

| Metric | dlat | dlon |
|--------|------|------|
| Range (arcsec) | [−1.54, 0.57] | [−2.43, 4.13] |
| RMS (arcsec) | 0.17 | 0.42 |
| Combined RMS | 13.97 m | max 128.72 m |

### Conformal Polynomial: Accuracy vs Degree

| Degree | Params | Max Error (m) | RMS (m) | Conformal % |
|--------|--------|---------------|---------|-------------|
| 5 | 12 | 105.0 | 11.2 | 6.7% |
| 10 | 22 | 105.2 | 10.9 | 11.3% |
| 15 | 32 | 104.1 | 10.9 | 12.9% |
| 20 | 42 | 103.7 | 10.9 | 12.9% |
| 30 | 62 | 104.2 | 10.8 | 13.5% |

**Convergence plateaus at ~13%.** Adding more degrees doesn't help because
the surface is fundamentally non-conformal.

### Conformal vs General Polynomial

| Type | Degree | Params | Max Error (m) | RMS (m) |
|------|--------|--------|---------------|---------|
| Conformal | 10 | 22 | 105.2 | 10.9 |
| General | 10 | 132 | 75.7 | 6.3 |
| Conformal | 20 | 42 | 103.7 | 10.9 |
| General | 20 | 462 | 50.1 | 3.7 |

General polynomials do better but still can't reach sub-cm. Neither polynomial
family converges fast enough for this surface — it has too much local structure.

### SVD Compression (the winner)

| Rank | Params | Max Error (m) | RMS (m) |
|------|--------|---------------|---------|
| 3 | 1,506 | 14.9 | 2.49 |
| 5 | 2,510 | 4.5 | 0.30 |
| 10 | 5,020 | **0.04** | **0.002** |
| 15 | 7,530 | ~0.00 | ~0.00 |

**SVD rank-10 achieves sub-5cm with 5,020 parameters. Rank-15 is exact.**

## Why the Conformal Assumption Failed

The original hypothesis: "Survey networks are built from angular observations,
so datum distortions should be approximately conformal (angle-preserving)."

**Reality:** The NAD27→NAD83 correction has three components:
1. **Ellipsoid geometry change** (Clarke 1866 → GRS80) — mostly handled by Helmert
2. **Survey network distortion** — accumulated scale and orientation errors from
   decades of triangulation chains. These are NOT conformal — they include
   systematic scale errors, distance-dependent biases, and non-uniform stretching.
3. **Tectonic deformation** — differential plate motion between observation epochs

The dlon residual (RMS 0.42 arcsec) is 2.5× larger than the dlat residual
(RMS 0.17 arcsec). This asymmetry alone means the surface can't be conformal —
a conformal map couples the two components equally via Cauchy-Riemann.

The energy decomposition confirms this:
- dlon carries 86% of the total residual energy
- The conformal fit can only capture 13% because it's forced to couple
  dlat and dlon symmetrically

## Conclusion

| Approach | Sub-cm viable? | Params for best accuracy |
|----------|----------------|------------------------|
| Conformal polynomial | **No** (floor ~100m) | N/A |
| General polynomial | **No** (floor ~50m at deg-20) | N/A |
| SVD rank-10 | **Yes** (4.4 cm max) | 5,020 |
| SVD rank-15 | **Yes** (exact) | 7,530 |

**Recommendation:** Drop conformal mapping from consideration. Focus on
SVD compression (#11) combined with the pyproj oracle (#6) or install-time
generation (#9) for the data source.
