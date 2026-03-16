# Quick Start

## Basic usage

```python
from vibeproj import Transformer

# Create a transformer between two coordinate systems
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Transform a single point (scalar)
x, y = t.transform(49.0, 2.0)

# Transform arrays
import numpy as np
lat = np.array([49.0, 48.8566, 40.7128])
lon = np.array([2.0, 2.3522, -74.006])
x, y = t.transform(lat, lon)

# Inverse transform
lat2, lon2 = t.transform(x, y, direction="INVERSE")
```

## CRS input formats

`from_crs()` accepts several formats:

```python
# EPSG integer
t = Transformer.from_crs(4326, 32631)

# Authority string
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Tuple
t = Transformer.from_crs(("EPSG", 4326), ("EPSG", 32631))

# Plain string
t = Transformer.from_crs("4326", "32631")
```

## GPU acceleration

When CuPy is available and input arrays are CuPy arrays, transforms
run on the GPU automatically using fused NVRTC kernels:

```python
import cupy as cp

lat = cp.array([49.0, 48.8566, 40.7128], dtype=cp.float64)
lon = cp.array([2.0, 2.3522, -74.006], dtype=cp.float64)

t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
x, y = t.transform(lat, lon)  # runs on GPU
```

NumPy arrays always use the CPU path. CuPy arrays always use the GPU path.
There is no explicit device selection -- the array type determines the backend.

## Zero-copy transforms

For high-throughput pipelines, `transform_buffers()` avoids all intermediate
allocations by writing directly into pre-allocated output arrays:

```python
import cupy as cp

t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Input arrays (already on GPU)
lat = cp.asarray(lat_data, dtype=cp.float64)
lon = cp.asarray(lon_data, dtype=cp.float64)

# Pre-allocate output
out_x = cp.empty_like(lat)
out_y = cp.empty_like(lon)

# Transform in-place -- no intermediate allocation
t.transform_buffers(lat, lon, out_x=out_x, out_y=out_y)
```

The returned arrays are the same objects as `out_x` and `out_y`.

## Projected-to-projected transforms

vibeProj handles projected-to-projected transforms by chaining through
a geographic intermediate:

```python
# UTM Zone 31N -> Web Mercator (no need to go through EPSG:4326 manually)
t = Transformer.from_crs("EPSG:32631", "EPSG:3857")
x_webmerc, y_webmerc = t.transform(x_utm, y_utm)
```

On GPU, this executes as two fused kernel calls (inverse + forward) with
no host round-trip between them.
