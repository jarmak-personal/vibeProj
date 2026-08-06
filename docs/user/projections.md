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
| Equal Earth | `eqearth` | 8857 | Spherical or ellipsoidal authalic form |
| Cylindrical Equal Area | `cea` | 6933 | EASE-Grid 2.0 |
| Orthographic | `ortho` | -- | Globe view |
| Gnomonic | `gnom` | -- | Great circle navigation |
| Mollweide | `moll` | -- | Equal-area world maps |
| Robinson | `robin` | -- | Compromise world maps |
| Winkel Tripel | `wintri` | -- | National Geographic standard |
| Natural Earth | `natearth` | -- | Polynomial pseudocylindrical |
| Azimuthal Equidistant (spherical) | `aeqd` | -- | Requires an explicit spherical CRS (`+R`) |
| Geostationary Satellite | `geos` | -- | Sweep X/Y and custom satellite height |
| Oblique Mercator (Hotine) | `omerc` | 3375 | Variants A/B |
| Krovak | `krovak` | 5513, 5514 | Regular Southing/Westing and north-oriented Easting/Northing |
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

Gnomonic is spherical-only. Define it with an explicit spherical ellipsoid
(`+R` in a CRS definition, or `SPHERE` in `ProjectionParams`). Ellipsoidal
Gnomonic definitions raise `UnsupportedProjectionError`; vibeProj does not
silently substitute spherical equations. Forward points at or behind the
horizon are outside the projection domain and return non-finite coordinates,
matching PROJ. The inverse remains stable at the projection center and for
very large finite projected radii.

Sinusoidal supports both explicit spherical definitions and ellipsoidal CRS
definitions such as `ESRI:54008`; the ellipsoidal path uses meridional distance
and the ellipsoidal prime-vertical radius in both CPU and fused CUDA execution.
The seventh-order meridional series is supported for finite eccentricity squared
in the inclusive range `0 <= es <= 0.012` and finite positive semi-major axes
no larger than `6,400,000 m`. This covers terrestrial Earth and Mars ellipsoids.
More eccentric or larger custom bodies raise `UnsupportedProjectionError`
during projection setup instead of returning an unqualified approximation.
Mercator honors the declared scale factor for variant A and derives the scale
at the natural origin from the standard parallel for variant B (for example,
EPSG:3994 and EPSG:3388).

Spherical Azimuthal Equidistant is available with an explicit radius, for
example `+proj=aeqd +lat_0=45 +lon_0=0 +R=6378137 +type=crs`. Ellipsoidal
Azimuthal Equidistant, Modified Azimuthal Equidistant, and Guam Projection are
not implemented. Constructing a Transformer for those methods raises
`UnsupportedProjectionError`; they never silently execute spherical formulas.

Geostationary CRS definitions retain both `+sweep=x` and `+sweep=y` plus the
declared satellite height `+h`. Height must be positive and no more than
`1e10` equatorial radii. Forward points behind the visible ellipsoid limb and
inverse points outside the Earth-intersection disk or principal scan-angle
range return non-finite sentinels.

Krovak uses one north-oriented mathematical core for both public variants.
EPSG:5514 exposes Easting/Northing. EPSG:5513 applies the declared negative
axes and exposes X=Southing, Y=Westing. Because those are explicit X/Y axes,
PROJ visualization order—and therefore `always_xy=True`—keeps Southing first
and Westing second rather than relabeling them as conventional E/N. Krovak's
false easting and northing parameters follow its method-specific subtractive
convention; metre and non-metre units retain positive public unit factors.
For custom nonzero offsets, vibeProj matches PROJ forward coordinates and
applies the same convention symmetrically in inverse transforms, so its inverse
exactly undoes its forward result. This intentionally avoids the asymmetric
custom-offset inverse behavior exposed by legacy PROJ pipelines through 9.5.
The authoritative EPSG:5513 and EPSG:5514 definitions use zero offsets and are
unaffected by this compatibility distinction.

## Datum shifting

When the source and destination CRS use different geodetic datums or
reference frames, vibeProj asks pyproj/PROJ for the coordinate-operation
plan instead of guessing from ellipsoid parameters. If the selected or
best supported operation contains a Helmert step, vibeProj applies a
**Helmert 7-parameter transformation** automatically.

```python
# Cross-datum: WGS84 -> British National Grid (OSGB36 / Airy 1830)
t = Transformer.from_crs("EPSG:4326", "EPSG:27700")
x, y = t.transform(-0.1278, 51.5074)
print(t.accuracy)  # "sub-meter"
```

Helmert parameters are extracted from pyproj's EPSG database at construction
time; the actual datum shift math runs on vibeProj's own GPU kernels (or
NumPy on CPU). Same-datum transforms have **zero overhead**. Some datum
pairs, such as common WGS84/NAD83 operations, are also represented by PROJ
as explicit no-op operations with meter-level expected accuracy; vibeProj
keeps those no-op transforms but reports them separately from same-datum
sub-millimeter transforms.

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

- **Oblique Stereographic** (`sterea`): The double-projection through a conformal
  sphere introduces ~130m systematic offset from pyproj's more rigorous method
  in the forward direction. The inverse conformal sphere conversion has a known
  accuracy limitation (~0.2 degrees). Roundtrip accuracy is sub-millimetre.

- **Winkel Tripel** (`wintri`): The inverse uses Newton iteration and converges
  to ~0.005 degrees rather than machine precision.
