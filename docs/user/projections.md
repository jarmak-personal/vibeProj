# Supported Projections

vibeProj supports 20 coordinate projections. Each has both a NumPy/CuPy
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

**Not yet supported:**

- NTv2 / NADCON grid-based shifts (sub-centimetre accuracy for national datums)
- Time-dependent Helmert (ITRF plate motion models)

If no Helmert transformation is available for a cross-datum pair (grid-only
datums), vibeProj warns and applies projection math without a datum shift.

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
