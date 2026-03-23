# vibeProj

```{raw} html
<div class="cp-hero">
  <div class="cp-hero-content">
    <h1 class="cp-hero-title" data-glitch="vibeProj">vibe<span class="accent">Proj</span></h1>
    <p class="cp-hero-subtitle">
      GPU-accelerated coordinate projection. 20 projections, each with a fused
      NVRTC kernel that runs the full transform pipeline in a single launch.
    </p>
    <div class="cp-hero-actions">
      <a class="cp-btn cp-btn--primary" href="user/index.html">User Guide &rarr;</a>
      <a class="cp-btn cp-btn--secondary" href="dev/index.html">Developer Guide &rarr;</a>
    </div>
  </div>
</div>
```

```{raw} html
<div class="cp-features">
  <div class="cp-card cp-reveal">
    <h3>Fused Kernels</h3>
    <p>One kernel launch per transform. Axis swap, deg/rad, projection math, scale/offset — all fused into a single NVRTC kernel. No intermediate buffers.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>20 Projections</h3>
    <p>Transverse Mercator, Lambert, Albers, Stereographic, and 16 more. Forward and inverse. Each projection has a handwritten CUDA kernel.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Zero-Copy Integration</h3>
    <p>Reads CuPy device buffers, writes to pre-allocated output. No host↔device round-trips. Designed for vibeSpatial's buffer protocol.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>CRS Resolution</h3>
    <p>Pass an EPSG code, get a transformer. Uses pyproj under the hood to extract parameters, then dispatches to the right fused kernel.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Precision Control</h3>
    <p>fp64 by default. Experimental double-single (ds) mode pairs two fp32 values for fp64-equivalent accuracy at fp32 throughput on consumer GPUs.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>CPU Fallback</h3>
    <p>NumPy-backed element-wise path for every projection. Same API, same results. Tests validate GPU output against the CPU reference.</p>
  </div>
</div>
```

## Quick Example

```python
from vibeproj import Transformer

t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
x, y = t.transform(49.0, 2.0)           # scalar
x, y = t.transform(lat_array, lon_array) # NumPy or CuPy arrays
```

```{raw} html
<div class="cp-links" style="text-align: center; margin: 2rem 0;">
  <a href="https://github.com/jarmak-personal/vibeProj" style="margin: 0 1rem;">GitHub</a> &middot;
  <a href="https://pypi.org/project/vibeproj/" style="margin: 0 1rem;">PyPI</a> &middot;
  <a href="https://github.com/jarmak-personal/vibeProj/issues" style="margin: 0 1rem;">Issues</a>
</div>
```

```{toctree}
:hidden:
:maxdepth: 2

user/index
dev/index
api/index
```
