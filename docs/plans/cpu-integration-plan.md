# vibeProj CPU Library Integration Plan

> Generated 2026-03-24. Four-agent analysis: Python Engineer, Maintainability Reviewer, Performance Reviewer, Ecosystem Agent.

## Architectural Decisions (resolved 2026-03-24)

1. **z packs into the 2-stream double buffer — no third stream.** z is not independent; the Helmert
   kernel reads x, y, z together in one launch. A third stream would require cross-stream event
   synchronization for zero benefit (PCIe is a shared bus — concurrent `cudaMemcpyAsync` calls
   serialize on the link). Each stream slot grows from 4 to 6 device arrays when `chunk_z=True`.
   When z is not needed, the z buffers are never allocated (zero overhead preserved).

2. **Chunk size stays at 1M default, already configurable.** 1M points fits in L2 cache for all
   target hardware (A100 40MB, H100 50MB, RTX 4090 72MB) in both 2D and 3D configurations.
   No auto-tuning needed — pipeline overlap dominates the performance picture.

3. **Compat signatures: explicit `chunk_size` + `**kw` → `from_crs()`.** `chunk_size` is the only
   execution-side knob worth exposing (performance tuning, no correctness impact). `direction` is
   always FORWARD (implicit from src/dst CRS). `precision`/`stream` are internal — not exposed.
   `**kw` passes to `from_crs()` which is strict about kwargs (typos raise `TypeError`).

4. **CPU fallback works transparently.** `transform_chunked()` already catches `ImportError` on
   CuPy and falls back to the NumPy xp path. No code changes needed in compat. `chunk_size`
   becomes a harmless no-op on CPU. Silent fallback, no warnings.

5. **Target Shapely 2.x only.** New library, no concern for older support.

6. **Rely on CuPy GC for pinned buffer cleanup.** No explicit `.close()` or context manager needed.

## Design Principles

- **Defend the array boundary.** `transform(x, y) -> (x', y')` is the API. Every integration reduces to "extract arrays, call transform, put arrays back."
- **No damage to vibeProj's simplicity.** 11 core files, 7 public exports, 2 runtime deps. Keep it that way.
- **Transfer is the bottleneck, not compute.** Kernels run 100-350x faster than pyproj. The entire optimization challenge is H<->D management.
- **vibeProj is a library others call, not a framework that orchestrates others.**

---

## Tier 1: Internal Optimization — `transform_chunked()` Pipeline

> The rising tide that lifts all boats. Benefits every integration without adding API surface.

### Context

Current `transform_chunked()` processes chunks serially with pageable memory:
```
[H2D 1.3ms] [kernel 0.5ms] [D2H 1.3ms] [H2D 1.3ms] [kernel 0.5ms] [D2H 1.3ms] ...
```

Target: double-buffered pipeline with pinned memory and 2 CUDA streams:
```
Stream A: [H2D chunk 0] [compute 0] [D2H chunk 0]   [H2D chunk 2] ...
Stream B:               [H2D chunk 1] [compute 1] [D2H chunk 1]   ...
```

Expected improvement: **~2.4x throughput** (1.3ms/chunk amortized vs 3.05ms/chunk serial).

### Tasks

- [ ] **Pinned host memory staging buffers**
  - Allocate pinned buffers via `cupy.cuda.alloc_pinned_memory` for H->D and D->H staging
  - Wrap as NumPy arrays via `numpy.frombuffer` for zero-copy host-side access
  - ~50% faster than pageable `cudaMemcpy` and enables async transfers
  - Pool pinned buffers on the Transformer instance (avoid repeated alloc overhead ~100us each)

- [ ] **Double-buffered CUDA stream pipeline**
  - Create 2 `cupy.cuda.Stream(non_blocking=True)` instances
  - Alternate chunks between streams: chunk N on stream A, chunk N+1 on stream B
  - Use `cudaMemcpyAsync` (via CuPy stream-ordered ops) to overlap transfer with compute
  - Synchronize only at the end (or when reading output of a completed stream slot)

- [ ] **Pre-allocated device buffer pairs (one per stream)**
  - 2D: 2 stream slots x (dev_x, dev_y, dev_ox, dev_oy) = 8 device arrays
  - 3D (when `chunk_z=True`): 2 stream slots x (dev_x, dev_y, dev_z, dev_ox, dev_oy, dev_oz) = 12 device arrays
  - z buffers allocated conditionally — zero overhead when z not provided
  - Allocated once per `transform_chunked()` call, reused across chunks
  - Current code already pre-allocates 1 set; extend to 2

- [ ] **Benchmarks: before/after**
  - Measure `transform_chunked()` throughput at 1M, 10M, 100M points
  - Compare serial vs pipelined on PCIe 3.0 and 4.0 if available
  - Update `benchmarks/results.json` with chunked pipeline numbers

### Key Numbers

| Metric | Current | Target |
|--------|---------|--------|
| Per-chunk time (1M pts, tmerc, PCIe 4.0) | ~3.05 ms | ~1.28 ms |
| 100M point transform (tmerc) | ~305 ms | ~128 ms |
| Pinned alloc overhead | N/A (pageable) | ~100us once, amortized |
| Stream count | 0 (synchronous) | 2 |

---

## Tier 2: Thin Compatibility Layer — `compat.py`

> A single file, max 150 lines, not re-exported from `__init__.py`. Users opt in explicitly via `from vibeproj.compat import ...`.

### Rules

- Every function lazily imports its third-party dependency
- Never imported by any core vibeProj module (one-way dependency arrow)
- Not added to `__init__.py` or `__all__`
- No optional dependencies added to `pyproject.toml`
- Line count cap: 150 lines
- No abstract base classes, plugin registries, or extension points

### Tasks

- [ ] **Create `src/vibeproj/compat.py`**

- [ ] **`reproject_geodataframe(gdf, dst_crs, *, chunk_size=1_000_000, **kw)`**
  - Lazy-import geopandas, shapely
  - Extract coords via `shapely.get_coordinates(gdf.geometry.values)`
  - Bulk transform via `Transformer.transform_chunked()`
  - Reconstruct geometries via `shapely.set_coordinates()`
  - Handle 2D and 3D geometries (check `shapely.has_z`)
  - Raise `ValueError` if `gdf.crs is None`
  - `chunk_size` controls GPU chunking (no-op on CPU fallback)
  - `**kw` passed to `Transformer.from_crs()` (supports `always_xy`, `datum_shift`, `epoch`)

- [ ] **`make_shapely_transform(src_crs, dst_crs, *, chunk_size=1_000_000, **kw)`**
  - Return a callable compatible with `shapely.transform(geom, func)` (Shapely 2.x API)
  - Accepts (N, 2) or (N, 3) coordinate arrays
  - Handles z-coordinate pass-through
  - `**kw` passed to `Transformer.from_crs()`

- [ ] **`reproject_geometries(geometries, src_crs, dst_crs, *, chunk_size=1_000_000, **kw)`**
  - Accept single geometry, list, or numpy array of geometries
  - Bulk extract -> transform -> reconstruct via get/set_coordinates
  - Return same type as input
  - `**kw` passed to `Transformer.from_crs()`

- [ ] **Tests: `tests/test_compat.py`**
  - Guard imports with `pytest.importorskip("geopandas")` / `pytest.importorskip("shapely")`
  - Test: Point GeoDataFrame reprojection (validate against pyproj)
  - Test: Polygon GeoDataFrame reprojection (geometry type preserved)
  - Test: Mixed geometry types
  - Test: 3D geometries with Z
  - Test: No-CRS GeoDataFrame raises ValueError
  - Test: Single shapely geometry round-trip
  - Test: `make_shapely_transform` with `shapely.transform()`
  - Test: Large batch (100K+ points) — verify no per-geometry loop

---

## Tier 3: Documentation Recipes

> Complete, tested examples. No new code in vibeProj — these go in docs.

### Tasks

- [ ] **Create `docs/recipes/` directory**

- [ ] **Recipe: Shapely reprojection** (`docs/recipes/shapely.md`)
  - Shapely 2.x `shapely.transform()` API:
    ```python
    import shapely
    from vibeproj.compat import make_shapely_transform
    func = make_shapely_transform("EPSG:4326", "EPSG:32631")
    new_geom = shapely.transform(geom, func)
    ```
  - Batch path using `compat.reproject_geometries()`
  - When to use GPU vs CPU (breakeven: ~100-2000 pairs depending on projection)

- [ ] **Recipe: GeoPandas reprojection** (`docs/recipes/geopandas.md`)
  - Using `compat.reproject_geodataframe()`
  - Manual 15-line recipe using `shapely.get_coordinates()` directly
  - Performance comparison vs `gdf.to_crs()` at various scales
  - GeoArrow future note (zero-copy when geopandas adopts native GeoArrow storage)

- [ ] **Recipe: Rasterio raster reprojection** (`docs/recipes/rasterio.md`)
  - Key insight: vibeProj transforms the coordinate grid, not pixels
  - Step 1: Build destination coordinate meshgrid
  - Step 2: Inverse-project to source CRS via `transform_chunked()`
  - Step 3: Resample pixels via `scipy.ndimage.map_coordinates()`
  - GPU-native grid generation (zero H->D transfer):
    ```python
    cols = cp.arange(width, dtype=cp.float64) * affine.a + affine.c
    rows = cp.arange(height, dtype=cp.float64) * affine.e + affine.f
    grid_x, grid_y = cp.meshgrid(cols, rows)
    src_x, src_y = t.transform_buffers(grid_x.ravel(), grid_y.ravel())
    ```
  - Tile-based processing for large rasters (memory-bounded)
  - Performance table:

    | Tile size | Coord pairs | GPU (ms) | CPU (ms) | Speedup |
    |-----------|-------------|----------|----------|---------|
    | 256x256 | 65K | ~0.07 | ~9 | 129x |
    | 4096x4096 | 16.8M | ~18 | ~2,200 | 122x |

- [ ] **Recipe: QGIS Processing plugin** (`docs/recipes/qgis.md`)
  - Skeleton `QgsProcessingProvider` + `QgsProcessingAlgorithm`
  - Note: Python-C++ boundary limits deep integration
  - Batch vector reprojection algorithm example
  - Recommendation: separate repository (`vibeproj-qgis`)

- [ ] **Recipe: Dask distributed reprojection** (`docs/recipes/dask.md`)
  - `dask.array.map_blocks` + `transform_chunked()`
  - Leverage existing Transformer pickle support
  - Multi-GPU partitioning pattern

---

## Future Roadmap (Post-Integration)

> These are strategic priorities identified by the ecosystem analysis. Not part of the current plan, but tracked here for sequencing.

### Near-Term

- [ ] **NTv2 grid shift support**
  - Required for Canadian (NAD27->NAD83), Australian (GDA94->GDA2020), European datums
  - Load NTv2 grids to GPU, implement bilinear interpolation kernel
  - Eliminates the main accuracy gap vs pyproj

- [ ] **cuDF/GeoArrow integration (separate package: `vibeproj-cudf`)**
  - Read geometry columns from cuDF GeoDataFrame
  - Transform coordinates on-device (zero host round-trip)
  - Fills the biggest gap in the RAPIDS spatial stack

- [ ] **Dask integration (separate package or recipe)**
  - `dask.array.map_blocks` + Transformer pickle serialization
  - Out-of-core reprojection across multi-GPU nodes

### Long-Term

- [ ] **GPU raster warping (full pipeline)**
  - Not just coordinate transform — also GPU-based bilinear/cubic resampling
  - Essentially a GPU `rasterio.warp.reproject()`
  - Biggest long-term prize, biggest engineering investment

- [ ] **PDAL point cloud pipeline integration**
  - LiDAR: billions of independent points — ideal GPU workload
  - Standalone `vibeproj-pdal` package

- [ ] **QGIS Processing plugin (separate repo)**
  - `vibeproj-qgis` with Processing Provider
  - Limited by QGIS Python-C++ boundary for interactive workflows
  - Useful for batch reprojection of large vector layers

---

## Anti-Patterns Checklist

> Review before each PR. If any box is checked, the PR needs rework.

- [ ] Added a `vibeproj/integrations/` subpackage (subpackages invite growth)
- [ ] Added `Transformer.transform_geodataframe()` or similar (Transformer must not know about pandas)
- [ ] Added optional deps to `pyproject.toml` (creates release coupling)
- [ ] Added plugin registry or ABC for integrations (3 integrations don't need a framework)
- [ ] Imported rasterio/geopandas/shapely at module load time (200-500ms import overhead)
- [ ] Modified `transformer.py`, `pipeline.py`, `crs.py`, or `fused_kernels.py` for integration concerns
- [ ] Re-exported compat functions from `__init__.py` (must be explicit opt-in)
- [ ] `compat.py` exceeded 150 lines (scope creep)
- [ ] Per-geometry H<->D transfers anywhere (must batch)
- [ ] Synchronous copies in a hot loop without stream overlap

---

## Reference Numbers

### Breakeven Points (GPU vs CPU pyproj)

| Projection | Breakeven N (PCIe 4.0) | Breakeven N (PCIe 3.0) |
|------------|------------------------|------------------------|
| tmerc | ~100 | ~150 |
| krovak | ~30 | ~50 |
| lcc | ~200 | ~400 |
| webmerc | ~700 | ~1,500 |
| eqc | ~2,000 | ~5,000 |

### End-to-End Integration Performance

| Scenario | GPU | CPU | Speedup |
|----------|-----|-----|---------|
| 1M tmerc (kernel only) | 0.49 ms | 139 ms | 284x |
| 1M tmerc (with current H<->D) | ~3 ms | 139 ms | 46x |
| 1M tmerc (optimized pipeline) | ~1.3 ms | 139 ms | 107x |
| 10Kx10K raster coord grid | ~250 ms | ~14 s | 56x |
| 1M-geom GeoDataFrame (50 verts/geom) | ~40 ms | ~7 s | 175x |

### Competitive Landscape

| Library | Projections | GPU | Datum Shifts | Status |
|---------|-------------|-----|-------------|--------|
| **vibeProj** | 24 | Yes (NVRTC) | 7+15 param Helmert | Active |
| cuProj (RAPIDS) | 1 (UTM only) | Yes (C++/CUDA) | None | Maintenance mode |
| pyproj | All PROJ | No | Full (incl. NTv2, geoid) | Active, CPU-only |
| rasterio.warp | All PROJ | No | Full (via GDAL) | Active, CPU-only |
