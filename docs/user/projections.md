# Supported Projections

vibeProj supports 24 coordinate projections. Each has both a NumPy/CuPy
element-wise implementation and a fused NVRTC GPU kernel.

## Projection table

| Projection | Internal Name | EPSG Examples | Notes |
|---|---|---|---|
| Transverse Mercator / UTM | `tmerc` | 32601--32760, 27700 | 6th-order Krueger series |
| Web Mercator | `webmerc` | 3857 | Spherical Mercator |
| Mercator (ellipsoidal) | `merc` | 3395 | Variant A/B |
| Lambert Conformal Conic | `lcc` | 2154 | 1SP and 2SP |
| Albers Equal Area | `aea` | 5070 | Conic equal-area |
| Polar Stereographic | `stere` | 3031, 3413 | Variants A/B/C |
| Lambert Azimuthal Equal Area | `laea` | 3035 | Oblique/equatorial/polar |
| Oblique Stereographic | `sterea` | 28992 | Double projection via conformal sphere |
| Plate Carree | `eqc` | 4087 | Equidistant cylindrical |
| Sinusoidal | `sinu` | -- | Pseudocylindrical equal-area |
| Equal Earth | `eqearth` | 8857 | Polynomial pseudocylindrical |
| Cylindrical Equal Area | `cea` | 6933 | EASE-Grid 2.0 |
| Orthographic | `ortho` | -- | Globe view |
| Gnomonic | `gnom` | -- | Great circle navigation |
| Mollweide | `moll` | -- | Equal-area world maps |
| Robinson | `robin` | -- | Compromise world maps |
| Winkel Tripel | `wintri` | -- | National Geographic standard |
| Natural Earth | `natearth` | -- | Polynomial pseudocylindrical |
| Azimuthal Equidistant | `aeqd` | -- | Distance-from-center |
| Geostationary Satellite | `geos` | -- | Weather satellite view |
| Oblique Mercator (Hotine) | `omerc` | 3375 | Variants A/B |
| Krovak | `krovak` | 5514 | North-orientated variant |
| Eckert IV | `eck4` | -- | Pseudocylindrical equal-area |
| Eckert VI | `eck6` | -- | Pseudocylindrical equal-area |

## Using projections via EPSG codes

Most projections are resolved automatically from EPSG codes via pyproj:

```python
from vibeproj import Transformer

# UTM Zone 31N
t = Transformer.from_crs("EPSG:4326", "EPSG:32631")

# Netherlands national grid (oblique stereographic)
t = Transformer.from_crs("EPSG:4326", "EPSG:28992")

# LAEA Europe
t = Transformer.from_crs("EPSG:4326", "EPSG:3035")
```

## Using projections without EPSG codes

Some projections (orthographic, gnomonic, etc.) don't have standard EPSG
codes. Use them via the pipeline API directly:

```python
from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import WGS84
from vibeproj.pipeline import TransformPipeline

# Orthographic centered on Paris
params = ProjectionParams(
    projection_name="ortho",
    ellipsoid=WGS84,
    lon_0=2.35,
    lat_0=48.86,
    north_first=False,
)
src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)

pipe = TransformPipeline(src, params)
x, y = pipe.transform(lat_array, lon_array, np)  # or cp for GPU
```

## Datum shifting

When the source and destination CRS use different geodetic datums (e.g.,
WGS84 vs OSGB36), vibeProj applies a **Helmert 7-parameter transformation**
automatically. This covers ~80% of real-world cross-datum transforms with
~1--5 metre accuracy.

```python
# Cross-datum: WGS84 -> British National Grid (OSGB36 / Airy 1830)
t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
x, y = t.transform(-0.1278, 51.5074)
print(t.accuracy)  # "sub-meter"
```

Helmert parameters are extracted from pyproj's EPSG database at construction
time; the actual datum shift math runs on vibeProj's own GPU kernels (or
NumPy on CPU). Same-datum transforms have **zero overhead** -- the datum
shift code path is bypassed entirely when no Helmert is needed.

15-parameter time-dependent Helmert is also supported for sub-decimeter
accuracy on modern datum pairs (e.g. ITRF to ETRS89). Pass an explicit
``epoch`` or let vibeProj resolve it from the source CRS coordinate epoch:

```python
t = Transformer.from_crs("EPSG:4326", "EPSG:27700", epoch=2024.0)
print(t.accuracy)  # "sub-decimeter" when 15-param rates are present
```

### SVD-compressed datum corrections

For datum pairs where Helmert alone is insufficient (e.g. NAD27 to NAD83),
vibeProj includes baked SVD-compressed corrections fitted from public domain
grid data (NADCON5). These are applied automatically as an additive correction
after the Helmert shift, achieving sub-5cm accuracy without external grid files.

```python
# NAD27 → NAD83 (SVD correction applied automatically)
t = Transformer.from_crs("EPSG:4267", "EPSG:4269")
x, y = t.transform(-90.0, 40.0)
print(t.accuracy)  # "sub-5cm"
```

Currently baked pairs:

- **NAD27 to NAD83** (CONUS) — rank-10 SVD, P95 accuracy 0.15 cm vs pyproj

For datum pairs without a baked SVD correction or Helmert parameters, vibeProj
emits a ``RuntimeWarning`` and falls back to projection math without a datum
shift. Results may differ from pyproj by meters to hundreds of meters in these
cases.

**Not yet supported:**

- **Raw NTv2 / NADCON grid loading** — vibeProj does not load external grid
  files at runtime. Datum pairs not covered by baked SVD corrections or Helmert
  fall back to no datum shift. Use pyproj or rasterio directly if you need
  coverage beyond the baked pairs.

## Known limitations

- **Equal Earth** (`eqearth`): Uses the spherical polynomial formula on geodetic
  latitude rather than converting to authalic latitude first. This means absolute
  metre values differ from pyproj by ~15%. Roundtrip accuracy is exact
  (forward and inverse are self-consistent).

- **Oblique Stereographic** (`sterea`): The double-projection through a conformal
  sphere introduces ~130m systematic offset from pyproj's more rigorous method
  in the forward direction. The inverse conformal sphere conversion has a known
  accuracy limitation (~0.2 degrees). Roundtrip accuracy is sub-millimetre.

- **Geostationary** (`geos`): The inverse has limited accuracy for off-nadir
  points due to a simplified geocentric latitude conversion.

- **Winkel Tripel** (`wintri`): The inverse uses Newton iteration and converges
  to ~0.005 degrees rather than machine precision.
