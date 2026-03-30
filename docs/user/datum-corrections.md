# Custom Datum Corrections

vibeProj ships with baked SVD corrections for common datum pairs (e.g. NAD27 to NAD83),
but you can generate your own for any datum pair that pyproj supports.

## How it works

The fitting tool samples pyproj on a dense lat/lon grid, subtracts the Helmert prediction
(if available), and fits a truncated SVD to the residual. The result is a compact set of
coefficients (~5,000 floats per component) that replace multi-megabyte NTv2/NADCON grid files.

At runtime, vibeProj evaluates the SVD correction via bilinear interpolation on the GPU
(or NumPy on CPU) as an additive correction after the Helmert shift.

## Generating corrections

The fitting tool lives at `tools/fit_datum_corrections.py`. It requires pyproj and NumPy
(both already vibeProj dependencies).

### Basic usage

```bash
uv run python tools/fit_datum_corrections.py \
    --src-crs EPSG:4267 --dst-crs EPSG:4269
```

This auto-detects the bounding box from the source CRS area of use, samples at a 100x150
grid, fits a rank-10 SVD, and validates against 5,000 random test points.

### Tuning parameters

```bash
uv run python tools/fit_datum_corrections.py \
    --src-crs EPSG:4267 --dst-crs EPSG:4269 \
    --n-lat 100 --n-lon 150 --rank 10 --target-accuracy 0.05
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-lat` | 100 | Grid rows (latitude) |
| `--n-lon` | 150 | Grid columns (longitude) |
| `--rank` | 10 | SVD truncation rank (higher = more accurate, more coefficients) |
| `--target-accuracy` | 0.05 | Target max error in meters; warns if exceeded |
| `--n-test-points` | 5000 | Random validation points |
| `--bbox` | auto | Override bounding box: `LAT_MIN LAT_MAX LON_MIN LON_MAX` |

### Interpreting output

The tool prints accuracy statistics, then outputs a `DatumCorrectionData` Python block:

```
Fitting SVD correction: EPSG:4267 -> EPSG:4269
Grid: 100 x 150, rank: 10
...
  Max error:  0.0312 m  (3.12 cm)
  P95 error:  0.0015 m  (0.15 cm)

================================================================
# Paste the following into src/vibeproj/_datum_corrections.py
================================================================
```

If the max error exceeds your target, try increasing `--rank` or grid density.

## Installing the correction

1. Paste the generated `DatumCorrectionData(...)` block into
   `src/vibeproj/_datum_corrections.py`
2. Add the `_register(...)` call at the bottom (the tool outputs this too)
3. Run the datum correction tests to verify:

```bash
uv run pytest tests/test_datum_corrections.py -v
```

The correction is automatically picked up by `Transformer.from_crs()` for
the registered datum pair.

## Tips

- **Rank vs accuracy**: rank 10 is usually sufficient for sub-5cm. Start there and only
  increase if the validation error is too high.
- **Grid density**: 100x150 covers CONUS well. For smaller regions (e.g. a single country),
  you can use fewer grid points. For global transforms, increase both.
- **Custom bbox**: use `--bbox` when the auto-detected area of use is too broad
  (e.g. NAD27 includes Alaska, but you only need CONUS).
- **No Helmert available**: the tool still works — it fits the full correction instead of
  just the residual. Accuracy may require a higher rank.
