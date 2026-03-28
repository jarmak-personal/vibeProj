# Testing

## Running tests

```bash
# All tests (CPU + GPU if available)
uv run pytest

# CPU-only tests
uv run pytest tests/test_transformer.py tests/test_crs.py

# GPU kernel tests (requires CuPy + NVIDIA GPU)
uv run pytest tests/test_fused_kernels.py

# Verbose output
uv run pytest -v

# Single test
uv run pytest tests/test_transformer.py::test_wgs84_to_utm_one_point
```

## Test structure

| File | Tests | Requires GPU |
|---|---|---|
| `tests/test_crs.py` | CRS parsing and resolution | No |
| `tests/test_transformer.py` | CPU-path transforms for all 20 projections | No |
| `tests/test_fused_kernels.py` | GPU fused kernel correctness | Yes |
| `tests/test_helmert.py` | Helmert datum shift math, extraction, cross-datum integration | No |
| `tests/test_datum_corrections.py` | SVD datum corrections: coefficient evaluation, accuracy vs pyproj, fallback chain | No |

GPU tests are automatically skipped when CuPy is not available
(`pytest.importorskip("cupy")`).

## Test patterns

### Forward test against pyproj

For projections with EPSG codes, validate output against pyproj:

```python
def test_my_forward():
    pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:XXXX")
    t = Transformer.from_crs("EPSG:4326", "EPSG:XXXX")

    lat, lon = np.array([40.0]), np.array([-74.0])
    exp_x, exp_y = pp.transform(lat, lon)
    vp_x, vp_y = t.transform(lat, lon)

    assert_allclose(vp_x, exp_x, atol=0.01)
    assert_allclose(vp_y, exp_y, atol=0.01)
```

### Roundtrip test

Forward then inverse should recover the original coordinates:

```python
def test_my_roundtrip():
    t = Transformer.from_crs("EPSG:4326", "EPSG:XXXX")
    lat, lon = 40.0, -74.0
    x, y = t.transform(lat, lon)
    lat2, lon2 = t.transform(x, y, direction="INVERSE")
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)
```

### GPU vs CPU comparison

Fused kernel tests compare GPU output against the CPU element-wise path:

```python
def test_my_fused_matches_numpy():
    lat = np.array([40.0, -30.0, 60.0])
    lon = np.array([-74.0, 20.0, 140.0])
    _run_forward_gpu_vs_cpu("EPSG:4326", "EPSG:XXXX", lat, lon, atol=0.01)
```

### Custom CRS roundtrip (no EPSG)

For projections without EPSG codes:

```python
def test_my_roundtrip():
    from vibeproj.crs import ProjectionParams
    from vibeproj.ellipsoid import WGS84
    from vibeproj.pipeline import TransformPipeline

    params = ProjectionParams(
        projection_name="myprojection", ellipsoid=WGS84,
        lon_0=0.0, lat_0=45.0, north_first=False,
    )
    src = ProjectionParams(
        projection_name="longlat", ellipsoid=WGS84, north_first=True,
    )
    pipe = TransformPipeline(src, params)
    lat = np.array([40.0, 50.0])
    lon = np.array([-5.0, 5.0])
    x, y = pipe.transform(lat, lon, np)

    inv_pipe = TransformPipeline(params, src)
    lat2, lon2 = inv_pipe.transform(x, y, np)
    assert_allclose(lat2, lat, atol=1e-7)
    assert_allclose(lon2, lon, atol=1e-7)
```

## Tolerance guidelines

| Test type | Typical tolerance | Reason |
|---|---|---|
| Forward vs pyproj | 0.01 m | Minor implementation differences |
| Roundtrip | 1e-7 degrees | Machine precision for fp64 |
| GPU vs CPU | 1e-4 to 0.01 m | Should be identical; allows for fp64 associativity |
| Iterative inverse (Winkel Tripel) | 0.005 degrees | Newton convergence limit |
| Cross-datum vs pyproj | 10 m | Helmert variant differences; pyproj may use grid shifts |
| Helmert roundtrip (fwd+inv) | 1e-4 degrees | Linearized rotation matrix is approximate |
| Helmert z roundtrip (fwd+inv) | 0.02 m | ~14mm due to linearized rotation matrix |
| SVD correction vs pyproj | 0.05 m | Sub-5cm P95 over baked coverage area |

## Linting

```bash
# Check for errors
uv run ruff check src/ tests/

# Check formatting
uv run ruff format --check src/ tests/

# Auto-fix
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

## CI

GitHub Actions runs on every push/PR to `main`:

- Tests on Python 3.12 and 3.13
- Ruff lint and format checks

GPU tests are not run in CI (no GPU available). They must be run
locally before merging GPU-affecting changes.
