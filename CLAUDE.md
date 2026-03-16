# vibeProj

GPU-accelerated coordinate projection library. 20 projections, each with a fused NVRTC kernel.

## Architecture

- **Fused NVRTC kernels** (`src/vibeproj/fused_kernels.py`) — 40 CUDA kernels (20 projections × fwd/inv).
  Each runs the full pipeline in one kernel launch: axis swap → deg/rad → central meridian →
  projection math → scale/offset → axis swap output. Compiled at runtime via CuPy `RawKernel`.
- **xp fallback path** (`src/vibeproj/projections/`) — NumPy/CuPy element-wise implementations.
  Used on CPU and as reference for testing. Each projection is a class with `setup()`, `forward()`, `inverse()`.
- **Pipeline** (`src/vibeproj/pipeline.py`) — chains pre/post ops. `_try_fused()` fast-path intercepts
  CuPy arrays and dispatches to fused kernels when available.
- **CRS resolution** (`src/vibeproj/crs.py`) — uses pyproj to extract projection parameters from EPSG codes.
  Maps pyproj method names → internal projection names via `_METHOD_MAP`.
- **GPU detection** (`src/vibeproj/gpu_detect.py`) — queries `SingleToDoublePrecisionPerfRatio` to classify
  consumer (1:64) vs datacenter (1:2) GPU. Auto precision always uses fp64 (projection math is SFU-bound).
- **Double-single arithmetic** (`src/vibeproj/_ds_device_fns.py`) — experimental ds fp32 pair arithmetic
  for fp64-equivalent accuracy at fp32 throughput. Available via `precision="ds"`.

## Key conventions

- Kernel I/O is always `double*` (fp64 storage per ADR-0002). Compute precision is parameterized via `{real_t}`.
- Fused kernel preambles handle axis order (CRS-dependent `north_first` flags).
- Tests validate against pyproj. GPU tests compare fused kernel output against NumPy xp path.
- `transform_buffers()` is the zero-copy API for vibeSpatial integration (pre-allocated output arrays).

## Adding a new projection

1. Create `src/vibeproj/projections/<name>.py` with a class that has `setup()`, `forward()`, `inverse()`
2. Register it: `register("<name>", MyProjection())` at module bottom
3. Import it in `src/vibeproj/projections/__init__.py`
4. Add pyproj method name mapping in `src/vibeproj/crs.py` `_METHOD_MAP`
5. Add fused kernel source strings in `src/vibeproj/fused_kernels.py`:
   - Forward + inverse kernel source using `_FWD_PREAMBLE`/`_FWD_POSTAMBLE` etc.
   - Add to `_SOURCE_MAP`, `_SUPPORTED`, and `fused_transform()` arg packing
6. Add tests in `tests/test_fused_kernels.py`

## Running tests

```bash
uv run pytest                              # all 85 tests
uv run pytest tests/test_fused_kernels.py  # GPU kernel tests (needs CuPy + GPU)
uv run pytest tests/test_transformer.py    # CPU xp path tests
```

## vibeSpatial integration

```python
from vibeproj import Transformer

t = Transformer.from_crs(src_crs, dst_crs)
# Zero-copy: reads device buffers, writes to pre-allocated device buffers
t.transform_buffers(buf.x, buf.y, out_x=new_x, out_y=new_y)
```
