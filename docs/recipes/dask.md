# Dask Distributed Reprojection

Distribute coordinate reprojection across Dask workers. vibeProj's
`Transformer` is pickle-safe and its `transform_chunked()` method handles
GPU memory management internally, making Dask integration straightforward.

## Prerequisites

```
pip install dask[distributed] cupy
```

---

## 1. Basic: `dask.array.map_blocks` with `transform_chunked()`

The simplest pattern: wrap `transform_chunked()` in a function that Dask can
serialize and dispatch to workers.

```python
import dask.array as da
import numpy as np
from vibeproj import Transformer

# Build the transformer on the client — it will be pickled to each worker.
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Host-resident input arrays (e.g. loaded from Parquet or HDF5)
lon = da.from_array(np.random.uniform(0, 6, 50_000_000), chunks=5_000_000)
lat = da.from_array(np.random.uniform(43, 49, 50_000_000), chunks=5_000_000)


def reproject_chunk(lon_chunk, lat_chunk, block_info=None):
    """Reproject one Dask chunk on a single worker."""
    # transform_chunked() handles H->D transfer, GPU compute, and D->H
    # internally using a double-buffered pinned-memory pipeline.
    x, y = t.transform_chunked(lon_chunk, lat_chunk, chunk_size=1_000_000)
    return x


# map_blocks applies the function to each chunk independently.
# We need separate calls for x and y because map_blocks returns one array.
easting = da.map_blocks(
    lambda lo, la: t.transform_chunked(lo, la, chunk_size=1_000_000)[0],
    lon, lat,
    dtype=np.float64,
)
northing = da.map_blocks(
    lambda lo, la: t.transform_chunked(lo, la, chunk_size=1_000_000)[1],
    lon, lat,
    dtype=np.float64,
)

# Trigger computation
easting_result, northing_result = da.compute(easting, northing)
```

To avoid computing the transform twice (once for x, once for y), use
`dask.delayed` instead:

```python
import dask

@dask.delayed
def reproject_block(lon_block, lat_block):
    return t.transform_chunked(lon_block, lat_block, chunk_size=1_000_000)

# Build the task graph
results = []
for i in range(0, len(lon_arr), block_size):
    lo = lon_arr[i : i + block_size]
    la = lat_arr[i : i + block_size]
    results.append(reproject_block(lo, la))

# Execute — Dask distributes blocks across workers
computed = dask.compute(*results)
easting = np.concatenate([r[0] for r in computed])
northing = np.concatenate([r[1] for r in computed])
```

---

## 2. Transformer Pickle Support

`Transformer` implements `__getstate__` / `__setstate__`. Pickle serializes
only the lightweight CRS identifiers and configuration flags -- not the
compiled GPU kernels or device buffers:

```python
import pickle

t = Transformer.from_crs("EPSG:4326", "EPSG:32631", datum_shift="accurate")

# What gets serialized:
# - crs_from, crs_to (EPSG codes or strings -- not resolved pyproj objects)
# - always_xy, datum_shift, epoch
data = pickle.dumps(t)

# On the worker: re-resolves CRS, re-extracts Helmert params, rebuilds pipeline.
# Kernel compilation happens on first transform call (or call t.compile()).
t2 = pickle.loads(data)
```

This means:

- **Dask can ship Transformers to any worker** via its default pickle
  serializer. No custom serialization hooks needed.
- **Each worker compiles kernels independently** on first use. Call
  `t.compile()` in a worker setup function to front-load compilation if
  latency matters.
- **pyproj must be installed on every worker** since CRS resolution happens
  at deserialization time.

---

## 3. Multi-GPU Partitioning

When workers have access to multiple GPUs, pin each Dask partition to a
specific device using `cupy.cuda.Device`. This prevents all partitions from
contending on GPU 0.

### With `dask.delayed`

```python
import cupy as cp
import dask
from vibeproj import Transformer

num_gpus = cp.cuda.runtime.getDeviceCount()
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")


@dask.delayed
def reproject_on_device(lon_block, lat_block, device_id):
    """Pin this task to a specific GPU."""
    with cp.cuda.Device(device_id):
        # transform_chunked handles all device memory on the active GPU.
        return t.transform_chunked(
            lon_block, lat_block, chunk_size=1_000_000
        )


# Round-robin partitions across GPUs
results = []
for i, (lo, la) in enumerate(blocks):
    gpu = i % num_gpus
    results.append(reproject_on_device(lo, la, gpu))

computed = dask.compute(*results)
```

### With Dask Distributed workers pinned to GPUs

For clusters where each worker owns exactly one GPU (the typical
`dask-cuda` deployment), the device is already set per worker. No
explicit `Device()` context is needed:

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# One worker per GPU, each with CUDA_VISIBLE_DEVICES set automatically.
cluster = LocalCUDACluster()
client = Client(cluster)

# Submit work normally — each worker uses its assigned GPU.
futures = client.map(
    lambda args: t.transform_chunked(args[0], args[1], chunk_size=1_000_000),
    blocks,
)
results = client.gather(futures)
```

### Staying on-device with `transform_buffers()`

If data is already on the GPU (e.g. loaded via cuDF or RAPIDS), use
`transform_buffers()` to avoid any host round-trip:

```python
@dask.delayed
def reproject_device_buffers(dev_x, dev_y, device_id):
    """Zero-copy transform for data already on the GPU."""
    with cp.cuda.Device(device_id):
        out_x = cp.empty_like(dev_x)
        out_y = cp.empty_like(dev_y)
        t.transform_buffers(dev_x, dev_y, out_x=out_x, out_y=out_y)
        return out_x, out_y
```

---

## Performance Notes

- **Chunk size tuning.** `transform_chunked(chunk_size=N)` controls the GPU
  chunk size within each Dask partition. The Dask partition size controls how
  much data ships to each worker. Keep Dask partitions large (5--50M points)
  and let `transform_chunked` subdivide internally for GPU memory management.
- **Pinned buffer reuse.** `transform_chunked()` pools pinned host memory and
  device buffers on the Transformer instance. Reusing the same Transformer
  across calls (as Dask naturally does with the pickled instance) avoids
  repeated allocation overhead.
- **Kernel compilation.** Each worker compiles NVRTC kernels on first use
  (~200ms one-time cost). For latency-sensitive workloads, call `t.compile()`
  in a worker initialization callback.
- **CPU fallback.** If CuPy is not installed on a worker, `transform_chunked()`
  silently falls back to the NumPy CPU path. This enables mixed CPU/GPU
  clusters without code changes.
