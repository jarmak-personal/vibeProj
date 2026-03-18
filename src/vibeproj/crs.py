"""CRS resolution — extract projection type and parameters from EPSG codes.

Uses pyproj for CRS metadata extraction, then maps to our internal projection types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyproj import CRS
from pyproj.exceptions import CRSError

from vibeproj.ellipsoid import WGS84, GRS80, SPHERE, Ellipsoid
from vibeproj.exceptions import CRSResolutionError, UnsupportedProjectionError


@dataclass
class ProjectionParams:
    """Parameters extracted from a CRS needed to configure a projection."""

    projection_name: str  # internal name: "tmerc", "merc", "webmerc", "lcc", etc.
    ellipsoid: Ellipsoid = field(default_factory=lambda: WGS84)
    # Common parameters (not all apply to every projection)
    lon_0: float = 0.0  # central meridian (degrees)
    lat_0: float = 0.0  # latitude of origin (degrees)
    lat_1: float = 0.0  # first standard parallel (degrees)
    lat_2: float = 0.0  # second standard parallel (degrees)
    k_0: float = 1.0  # scale factor
    x_0: float = 0.0  # false easting (meters)
    y_0: float = 0.0  # false northing (meters)
    # UTM-specific
    utm_zone: int = 0
    south: bool = False
    # Axis order: True if first axis is northing (e.g. EPSG:3035 is N,E)
    north_first: bool = False
    # Extra params for specific projections
    extra: dict[str, Any] = field(default_factory=dict)


def parse_crs_input(crs_input) -> CRS:
    """Parse a CRS input to a pyproj CRS object.

    Accepts:
    - An EPSG integer code: 4326
    - An authority string: "EPSG:4326" or "epsg:4326"
    - A tuple: ("EPSG", 4326)
    - A pyproj CRS object
    """
    if isinstance(crs_input, CRS):
        return crs_input

    if isinstance(crs_input, tuple):
        authority, code = crs_input
        return CRS.from_authority(authority, code)

    if isinstance(crs_input, int):
        return CRS.from_epsg(crs_input)

    if isinstance(crs_input, str):
        # Try "EPSG:XXXX" format first
        s = crs_input.strip()
        if ":" in s:
            return CRS.from_user_input(s)
        # Try as plain integer string
        try:
            return CRS.from_epsg(int(s))
        except (ValueError, CRSError):
            pass
        return CRS.from_user_input(s)

    raise CRSResolutionError(f"Cannot parse CRS from {type(crs_input).__name__}: {crs_input}")


# Mapping from pyproj projection method names to our internal names
_METHOD_MAP = {
    "Transverse Mercator": "tmerc",
    "Popular Visualisation Pseudo Mercator": "webmerc",
    "Mercator (variant A)": "merc",
    "Mercator (variant B)": "merc",
    "Mercator (1SP)": "merc",
    "Mercator (2SP)": "merc",
    "Lambert Conic Conformal (1SP)": "lcc",
    "Lambert Conic Conformal (2SP)": "lcc",
    "Albers Equal Area": "aea",
    "Polar Stereographic (variant A)": "stere",
    "Polar Stereographic (variant B)": "stere",
    "Polar Stereographic (variant C)": "stere",
    "Lambert Azimuthal Equal Area": "laea",
    "Equidistant Cylindrical": "eqc",
    "Equidistant Cylindrical (Spherical)": "eqc",
    "Sinusoidal": "sinu",
    "Equal Earth": "eqearth",
    "Lambert Cylindrical Equal Area": "cea",
    "Lambert Cylindrical Equal Area (Spherical)": "cea",
    "Orthographic": "ortho",
    "Gnomonic": "gnom",
    "Mollweide": "moll",
    "Oblique Stereographic": "sterea",
    "Geostationary Satellite (Sweep Y)": "geos",
    "Geostationary Satellite (Sweep X)": "geos",
    "Robinson": "robin",
    "Winkel Tripel": "wintri",
    "Natural Earth": "natearth",
    "Modified Azimuthal Equidistant": "aeqd",
    "Azimuthal Equidistant": "aeqd",
    "Azimuthal Equidistant (Spherical)": "aeqd",
}


def _get_ellipsoid(crs: CRS) -> Ellipsoid:
    """Extract ellipsoid from CRS."""
    ell = crs.ellipsoid
    if ell is None:
        return WGS84
    a = ell.semi_major_metre
    inv_f = ell.inverse_flattening
    if inv_f == 0 or inv_f is None:
        # Sphere
        return Ellipsoid(a=a, b=a, f=0.0, e=0.0, es=0.0, n=0.0)
    # Check for known ellipsoids to reuse precomputed instances
    if abs(a - 6378137.0) < 0.1:
        if abs(inv_f - 298.257223563) < 1e-6:
            return WGS84
        if abs(inv_f - 298.257222101) < 1e-6:
            return GRS80
    return Ellipsoid.from_af(a, inv_f)


def _get_param(params_list, name, default=0.0):
    """Extract a numeric parameter from a CRS coordinate operation params list.

    Uses case-insensitive matching with spaces normalized.
    """
    name_norm = name.lower().replace("_", " ")
    for p in params_list:
        if p.name.lower() == name_norm:
            return p.value
    return default


def resolve_projection_params(crs: CRS) -> ProjectionParams:
    """Extract projection parameters from a CRS.

    Returns a ProjectionParams object describing the projection type and its parameters.
    For geographic CRS (lat/lon), returns projection_name="longlat".
    """
    # Detect axis order: is first axis northing/latitude?
    # Use abbreviation: 'Lat', 'N', 'Y' = northing-first; 'Lon', 'E', 'X' = easting-first
    axes = crs.axis_info
    first_is_north = len(axes) >= 1 and axes[0].abbrev in ("Lat", "N", "Y")

    if crs.is_geographic:
        return ProjectionParams(
            projection_name="longlat",
            ellipsoid=_get_ellipsoid(crs),
            north_first=first_is_north,
        )

    if not crs.is_projected:
        raise CRSResolutionError(f"Unsupported CRS type: {crs}")

    cf = crs.coordinate_operation
    if cf is None:
        raise CRSResolutionError(f"Cannot extract projection from CRS: {crs}")

    method_name = cf.method_name
    proj_name = _METHOD_MAP.get(method_name)

    if proj_name is None:
        raise UnsupportedProjectionError(
            f"Unsupported projection method: '{method_name}'. "
            f"Supported methods: {sorted(_METHOD_MAP.keys())}"
        )

    ellipsoid = _get_ellipsoid(crs)

    # Check for Web Mercator specifically (EPSG:3857)
    epsg = crs.to_epsg()
    if epsg == 3857:
        proj_name = "webmerc"
        ellipsoid = SPHERE

    pl = cf.params  # list of CoordinateOperationParameter objects

    # Extract parameters — names match pyproj exactly
    lon_0 = _get_param(
        pl,
        "Longitude of natural origin",
        _get_param(pl, "Longitude of false origin", _get_param(pl, "Longitude of origin", 0.0)),
    )
    lat_0 = _get_param(
        pl,
        "Latitude of natural origin",
        _get_param(
            pl, "Latitude of false origin", _get_param(pl, "Latitude of standard parallel", 0.0)
        ),
    )
    lat_1 = _get_param(pl, "Latitude of 1st standard parallel", 0.0)
    lat_2 = _get_param(pl, "Latitude of 2nd standard parallel", 0.0)
    k_0 = _get_param(
        pl, "Scale factor at natural origin", _get_param(pl, "Scale factor on initial line", 1.0)
    )
    x_0 = _get_param(pl, "False easting", _get_param(pl, "Easting at false origin", 0.0))
    y_0 = _get_param(pl, "False northing", _get_param(pl, "Northing at false origin", 0.0))

    params = ProjectionParams(
        projection_name=proj_name,
        ellipsoid=ellipsoid,
        lon_0=lon_0,
        lat_0=lat_0,
        lat_1=lat_1,
        lat_2=lat_2,
        k_0=k_0,
        x_0=x_0,
        y_0=y_0,
        north_first=first_is_north,
    )

    # UTM detection
    if proj_name == "tmerc" and epsg is not None:
        if 32601 <= epsg <= 32660:
            params.utm_zone = epsg - 32600
            params.south = False
        elif 32701 <= epsg <= 32760:
            params.utm_zone = epsg - 32700
            params.south = True

    return params


def resolve_transform(crs_from, crs_to) -> tuple[ProjectionParams, ProjectionParams, CRS, CRS]:
    """Resolve source and target CRS for a coordinate transform.

    Returns (src_params, dst_params, src_crs, dst_crs).
    """
    src_crs = parse_crs_input(crs_from)
    dst_crs = parse_crs_input(crs_to)
    src_params = resolve_projection_params(src_crs)
    dst_params = resolve_projection_params(dst_crs)
    return src_params, dst_params, src_crs, dst_crs
