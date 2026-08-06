"""CRS resolution — extract projection type and parameters from EPSG codes.

Uses pyproj for CRS metadata extraction, then maps to our internal projection types.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from pyproj import CRS
from pyproj.exceptions import CRSError

from vibeproj.ellipsoid import WGS84, GRS80, SPHERE, Ellipsoid
from vibeproj.exceptions import CRSResolutionError, UnsupportedProjectionError

type CRSInput = int | str | tuple[str, int] | CRS


@dataclass
class ProjectionParams:
    """Parameters extracted from a CRS needed to configure a projection."""

    projection_name: str  # internal name: "tmerc", "merc", "webmerc", "lcc", etc.
    ellipsoid: Ellipsoid = field(default_factory=lambda: WGS84)
    # Common parameters (not all apply to every projection)
    lon_0: float = 0.0  # central meridian (degrees)
    lat_0: float = 0.0  # latitude of origin (degrees)
    # ``None`` means the coordinate operation omitted the parameter.  Zero is
    # a valid, explicit standard parallel and must not be used as a sentinel.
    lat_1: float | None = None  # first standard parallel (degrees)
    lat_2: float | None = None  # second standard parallel (degrees)
    k_0: float = 1.0  # scale factor
    x_0: float = 0.0  # false easting (meters)
    y_0: float = 0.0  # false northing (meters)
    x_unit_to_m: float = 1.0  # projected easting unit -> meters
    y_unit_to_m: float = 1.0  # projected northing unit -> meters
    # UTM-specific
    utm_zone: int = 0
    south: bool = False
    # Axis order: True if first axis is northing (e.g. EPSG:3035 is N,E)
    north_first: bool = False
    # Extra params for specific projections
    extra: dict[str, Any] = field(default_factory=dict)
    # Exact coordinate-operation method retained for strategy-domain planning.
    # Appended to preserve the public positional construction order above.
    operation_method: str | None = None


@dataclass(frozen=True)
class DatumOperationPlan:
    """Datum/reference-frame operation metadata from pyproj/PROJ.

    This is intentionally an internal planning layer, not a public
    TransformerPlan API.  It records whether a datum operation is needed and
    how well vibeProj can represent the operation with its current kernels.
    """

    cross_datum: bool
    source_datum: str | None = None
    target_datum: str | None = None
    operation: str | None = None
    expected_accuracy_m: float | None = None
    area_of_use: str | None = None
    best_available: bool | None = None
    missing_grids: tuple[str, ...] = ()
    best_has_helmert: bool = False
    has_available_helmert: bool = False
    best_is_noop: bool = False
    best_is_ballpark: bool = False
    warning_level: Literal["ok", "degraded", "unsupported"] = "ok"

    @property
    def uses_authoritative_noop(self) -> bool:
        """True when PROJ's best available operation is an explicit no-op."""
        return (
            self.cross_datum
            and self.best_available is True
            and self.best_is_noop
            and not self.best_is_ballpark
        )


def parse_crs_input(crs_input: CRSInput) -> CRS:
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
    "Lambert Azimuthal Equal Area (Spherical)": "laea",
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
    "Azimuthal Equidistant": "aeqd",
    "Azimuthal Equidistant (Spherical)": "aeqd",
    "Hotine Oblique Mercator (variant A)": "omerc",
    "Hotine Oblique Mercator (variant B)": "omerc",
    "Krovak": "krovak",
    "Krovak (North Orientated)": "krovak",
    "Eckert IV": "eck4",
    "Eckert VI": "eck6",
}

# Recognized so callers receive a stable typed error instead of a generic
# unknown-method failure. These methods are intentionally absent from the
# supported/public method map above.
_RECOGNIZED_UNSUPPORTED_METHODS = frozenset({"Modified Azimuthal Equidistant", "Guam Projection"})
_NONPUBLIC_PROJECTION_METHODS = frozenset(
    {"Azimuthal Equidistant", *_RECOGNIZED_UNSUPPORTED_METHODS}
)


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


def _get_linear_param(params_list, name, default=0.0):
    """Extract a linear parameter and normalize it to meters."""
    name_norm = name.lower().replace("_", " ")
    for p in params_list:
        if p.name.lower() == name_norm:
            factor = getattr(p, "unit_conversion_factor", None) or 1.0
            return p.value * factor
    return default


def _get_axis_unit_factors(crs: CRS, *, first_is_north: bool) -> tuple[float, float]:
    """Return (easting_unit_to_m, northing_unit_to_m) for projected CRS axes."""
    if not crs.is_projected:
        return 1.0, 1.0

    x_unit_to_m = 1.0
    y_unit_to_m = 1.0
    axes = crs.axis_info

    for axis in axes:
        direction = axis.direction.lower()
        factor = axis.unit_conversion_factor or 1.0
        if direction in ("east", "west"):
            x_unit_to_m = factor
        elif direction in ("north", "south"):
            y_unit_to_m = factor

    # Fallback when the CRS axis directions are generic (e.g. X/Y).
    if len(axes) >= 2:
        first_factor = axes[0].unit_conversion_factor or 1.0
        second_factor = axes[1].unit_conversion_factor or 1.0
        if x_unit_to_m == 1.0 and y_unit_to_m == 1.0:
            if first_is_north:
                return second_factor, first_factor
            return first_factor, second_factor

    return x_unit_to_m, y_unit_to_m


def resolve_projection_params(crs: CRS) -> ProjectionParams:
    """Extract projection parameters from a CRS.

    Returns a ProjectionParams object describing the projection type and its parameters.
    For geographic CRS (lat/lon), returns projection_name="longlat".
    """
    # Detect axis order: is first axis northing/latitude?
    # Use abbreviation: 'Lat', 'N', 'Y' = northing-first; 'Lon', 'E', 'X' = easting-first
    axes = crs.axis_info
    first_is_north = len(axes) >= 1 and axes[0].abbrev in ("Lat", "N", "Y")
    x_unit_to_m, y_unit_to_m = _get_axis_unit_factors(crs, first_is_north=first_is_north)

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
    if method_name in _RECOGNIZED_UNSUPPORTED_METHODS:
        raise UnsupportedProjectionError(
            f"Projection method '{method_name}' is recognized but not implemented. "
            "vibeProj supports spherical Azimuthal Equidistant only; use an explicit "
            "spherical CRS (+R) when those semantics are intended."
        )
    proj_name = _METHOD_MAP.get(method_name)

    if proj_name is None:
        public_methods = sorted(
            method for method in _METHOD_MAP if method not in _NONPUBLIC_PROJECTION_METHODS
        )
        raise UnsupportedProjectionError(
            f"Unsupported projection method: '{method_name}'. Supported methods: {public_methods}"
        )

    ellipsoid = _get_ellipsoid(crs)
    if proj_name in ("aeqd", "gnom") and ellipsoid.es != 0.0:
        display_name = {
            "aeqd": "Azimuthal Equidistant",
            "gnom": "Gnomonic",
        }[proj_name]
        raise UnsupportedProjectionError(
            f"Ellipsoidal projection method '{method_name}' is not implemented. "
            f"vibeProj supports spherical {display_name} only; use an explicit "
            "spherical CRS (+R) when those semantics are intended."
        )

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
        _get_param(
            pl,
            "Longitude of false origin",
            _get_param(
                pl,
                "Longitude of origin",
                _get_param(pl, "Longitude of projection centre", 0.0),
            ),
        ),
    )
    lat_0 = _get_param(
        pl,
        "Latitude of natural origin",
        _get_param(
            pl,
            "Latitude of false origin",
            _get_param(
                pl,
                "Latitude of standard parallel",
                _get_param(pl, "Latitude of projection centre", 0.0),
            ),
        ),
    )
    lat_1 = _get_param(
        pl,
        "Latitude of 1st standard parallel",
        _get_param(
            pl,
            "Latitude of standard parallel",
            _get_param(pl, "Latitude of true scale", None),
        ),
    )
    lat_2 = _get_param(pl, "Latitude of 2nd standard parallel", None)
    k_0 = _get_param(
        pl,
        "Scale factor at natural origin",
        _get_param(
            pl,
            "Scale factor on initial line",
            _get_param(
                pl,
                "Scale factor at projection centre",
                _get_param(pl, "Scale factor on pseudo standard parallel", 1.0),
            ),
        ),
    )
    if proj_name == "merc" and method_name in ("Mercator (variant B)", "Mercator (2SP)"):
        if lat_1 is None:
            raise CRSResolutionError(
                f"Projection method '{method_name}' requires a latitude of standard parallel"
            )
        latitude_standard_parallel = math.radians(lat_1)
        sin_latitude = math.sin(latitude_standard_parallel)
        k_0 = math.cos(latitude_standard_parallel) / math.sqrt(
            1.0 - ellipsoid.es * sin_latitude * sin_latitude
        )
    x_0 = _get_linear_param(
        pl,
        "False easting",
        _get_linear_param(
            pl,
            "Easting at false origin",
            _get_linear_param(pl, "Easting at projection centre", 0.0),
        ),
    )
    y_0 = _get_linear_param(
        pl,
        "False northing",
        _get_linear_param(
            pl,
            "Northing at false origin",
            _get_linear_param(pl, "Northing at projection centre", 0.0),
        ),
    )

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
        x_unit_to_m=x_unit_to_m,
        y_unit_to_m=y_unit_to_m,
        north_first=first_is_north,
        operation_method=method_name,
    )

    if proj_name == "geos":
        height = _get_linear_param(pl, "Satellite Height", 35_785_831.0)
        if not math.isfinite(height) or height <= 0.0:
            raise CRSResolutionError(
                f"Geostationary satellite height must be finite and positive, got {height!r}"
            )
        params.extra["h"] = height
        params.extra["sweep_axis"] = (
            "x" if method_name == "Geostationary Satellite (Sweep X)" else "y"
        )

    # Oblique Mercator extra params
    if proj_name == "omerc":
        params.extra["alpha_c"] = _get_param(pl, "Azimuth at projection centre", 0.0)
        params.extra["gamma_c"] = _get_param(pl, "Angle from Rectified to Skew Grid", 0.0)
        # Variant A: FE/FN already encode u_c offset → skip u_c subtraction (no_uoff=True)
        # Variant B: Ec/Nc at projection centre → apply u_c subtraction (no_uoff=False)
        has_epc = any(p.name == "Easting at projection centre" for p in pl)
        params.extra["no_uoff"] = not has_epc

    # Krovak extra params
    if proj_name == "krovak":
        params.extra["alpha_c"] = _get_param(pl, "Co-latitude of cone axis", 30.28813975277778)
        params.extra["phi_p"] = _get_param(pl, "Latitude of pseudo standard parallel", 78.5)

    # UTM detection
    if proj_name == "tmerc" and epsg is not None:
        if 32601 <= epsg <= 32660:
            params.utm_zone = epsg - 32600
            params.south = False
        elif 32701 <= epsg <= 32760:
            params.utm_zone = epsg - 32700
            params.south = True

    return params


def resolve_transform(
    crs_from: CRSInput, crs_to: CRSInput
) -> tuple[ProjectionParams, ProjectionParams, CRS, CRS]:
    """Resolve source and target CRS for a coordinate transform.

    Returns (src_params, dst_params, src_crs, dst_crs).
    """
    src_crs = parse_crs_input(crs_from)
    dst_crs = parse_crs_input(crs_to)
    src_params = resolve_projection_params(src_crs)
    dst_params = resolve_projection_params(dst_crs)
    return src_params, dst_params, src_crs, dst_crs


def _safe_proj4(transformer) -> str:
    """Return a transformer's PROJ string, or an empty string if unavailable."""
    try:
        return transformer.to_proj4() or ""
    except Exception:
        return ""


def _transformer_has_helmert(transformer) -> bool:
    return "+proj=helmert" in _safe_proj4(transformer)


def _transformer_is_noop(transformer) -> bool:
    return _safe_proj4(transformer).strip() == "+proj=noop"


def _missing_grid_names(unavailable_operations) -> tuple[str, ...]:
    """Return unique missing grid names from TransformerGroup metadata."""
    names: list[str] = []
    seen: set[str] = set()
    for operation in unavailable_operations:
        grids = getattr(operation, "grids", ()) or ()
        for grid in grids:
            if getattr(grid, "available", True):
                continue
            name = getattr(grid, "short_name", "") or getattr(grid, "full_name", "")
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def build_datum_operation_plan(src_crs: CRS, dst_crs: CRS) -> DatumOperationPlan:
    """Inspect the datum/reference-frame operation needed between two CRS.

    pyproj/PROJ is the source of truth for datum operation metadata.  This
    function only plans and classifies operations; it does not execute them.
    """
    src_geo = src_crs.geodetic_crs
    dst_geo = dst_crs.geodetic_crs
    if src_geo is None or dst_geo is None:
        return DatumOperationPlan(cross_datum=False)

    src_datum = src_geo.datum.name if src_geo.datum else None
    dst_datum = dst_geo.datum.name if dst_geo.datum else None
    if src_geo.datum == dst_geo.datum:
        return DatumOperationPlan(
            cross_datum=False,
            source_datum=src_datum,
            target_datum=dst_datum,
        )

    from pyproj.transformer import TransformerGroup

    try:
        tg = TransformerGroup(src_geo, dst_geo)
    except Exception as exc:
        warnings.warn(
            f"Failed to query datum operation metadata: {exc}. "
            "Proceeding without datum shift — coordinates may be offset.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DatumOperationPlan(
            cross_datum=True,
            source_datum=src_datum,
            target_datum=dst_datum,
            warning_level="unsupported",
        )

    transformers = list(tg.transformers)
    best = transformers[0] if transformers else None
    best_description = getattr(best, "description", None) if best is not None else None
    best_accuracy = getattr(best, "accuracy", None) if best is not None else None
    if best_accuracy is not None and best_accuracy < 0:
        best_accuracy = None
    best_area = getattr(getattr(best, "area_of_use", None), "name", None)
    best_has_helmert = best is not None and _transformer_has_helmert(best)
    has_available_helmert = any(_transformer_has_helmert(t) for t in transformers)
    best_is_noop = best is not None and _transformer_is_noop(best)
    best_is_ballpark = bool(best_description and "ballpark" in best_description.lower())
    missing_grids = _missing_grid_names(tg.unavailable_operations)

    warning_level: Literal["ok", "degraded", "unsupported"] = "ok"
    if best is None:
        warning_level = "unsupported"
    elif best_has_helmert:
        warning_level = "ok"
    elif best_is_noop and not best_is_ballpark and tg.best_available:
        warning_level = "degraded"
    elif has_available_helmert:
        warning_level = "degraded"
    elif missing_grids or best_is_ballpark:
        warning_level = "unsupported"
    else:
        warning_level = "unsupported"

    return DatumOperationPlan(
        cross_datum=True,
        source_datum=src_datum,
        target_datum=dst_datum,
        operation=best_description,
        expected_accuracy_m=best_accuracy,
        area_of_use=best_area,
        best_available=tg.best_available,
        missing_grids=missing_grids,
        best_has_helmert=best_has_helmert,
        has_available_helmert=has_available_helmert,
        best_is_noop=best_is_noop,
        best_is_ballpark=best_is_ballpark,
        warning_level=warning_level,
    )


# ---------------------------------------------------------------------------
# Helmert extraction
# ---------------------------------------------------------------------------

_ARC_SECOND_TO_RAD = math.pi / (180.0 * 3600.0)


def _parse_helmert_from_proj4(proj4: str):
    """Parse Helmert parameters from a PROJ pipeline string.

    Looks for ``+proj=helmert`` steps and extracts the 7 parameters.
    Returns (tx, ty, tz, rx_as, ry_as, rz_as, ds_ppm, convention, is_inverse)
    or None.
    """
    steps = proj4.split("+step")
    for step in steps:
        if "+proj=helmert" not in step:
            continue

        tokens = step.split()
        params: dict[str, str] = {}
        is_inverse = False
        for token in tokens:
            if token == "+inv":
                is_inverse = True
            elif token.startswith("+") and "=" in token:
                key, _, val = token[1:].partition("=")
                params[key] = val

        tx = float(params.get("x", "0"))
        ty = float(params.get("y", "0"))
        tz = float(params.get("z", "0"))
        rx_as = float(params.get("rx", "0"))
        ry_as = float(params.get("ry", "0"))
        rz_as = float(params.get("rz", "0"))
        ds_ppm = float(params.get("s", "0"))
        convention = params.get("convention", "position_vector")

        # Time-dependent (15-parameter) rates
        dtx = float(params.get("dx", "0"))
        dty = float(params.get("dy", "0"))
        dtz = float(params.get("dz", "0"))
        drx_as = float(params.get("drx", "0"))
        dry_as = float(params.get("dry", "0"))
        drz_as = float(params.get("drz", "0"))
        dds_ppm = float(params.get("ds", "0"))
        t_epoch = float(params.get("t_epoch", "0"))

        return (
            tx,
            ty,
            tz,
            rx_as,
            ry_as,
            rz_as,
            ds_ppm,
            convention,
            is_inverse,
            dtx,
            dty,
            dtz,
            drx_as,
            dry_as,
            drz_as,
            dds_ppm,
            t_epoch,
        )

    return None


def extract_helmert(src_crs: CRS, dst_crs: CRS):
    """Extract Helmert 7-parameter datum shift between two CRS.

    Uses pyproj to determine the best available transformation pipeline,
    then extracts the Helmert operation parameters from the PROJ pipeline string.

    Returns
    -------
    HelmertParams or None
        None if same datum, no Helmert available, or identity transform.
    """
    from vibeproj.helmert import HelmertParams

    # Resolve to geographic CRS (strip projection)
    src_geo = src_crs.geodetic_crs
    dst_geo = dst_crs.geodetic_crs
    if src_geo is None or dst_geo is None:
        return None

    # Same datum — no shift needed
    if src_geo.datum == dst_geo.datum:
        return None

    # Query pyproj for available transformations
    from pyproj.transformer import TransformerGroup

    try:
        tg = TransformerGroup(src_geo, dst_geo)
    except Exception as exc:
        warnings.warn(
            f"Failed to query Helmert parameters: {exc}. "
            "Proceeding without datum shift — coordinates may be offset.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # Search through available transformers for one containing a Helmert step
    for transformer in tg.transformers:
        try:
            proj4 = transformer.to_proj4()
        except Exception as exc:
            warnings.warn(
                f"Failed to extract proj4 string: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        parsed = _parse_helmert_from_proj4(proj4)
        if parsed is None:
            continue

        (
            tx,
            ty,
            tz,
            rx_as,
            ry_as,
            rz_as,
            ds_ppm,
            convention,
            is_inverse,
            dtx,
            dty,
            dtz,
            drx_as,
            dry_as,
            drz_as,
            dds_ppm,
            t_epoch,
        ) = parsed

        rx_rad = rx_as * _ARC_SECOND_TO_RAD
        ry_rad = ry_as * _ARC_SECOND_TO_RAD
        rz_rad = rz_as * _ARC_SECOND_TO_RAD
        ds = 1.0 + ds_ppm * 1e-6

        # Convert rate rotations to radians/yr, rate scale to 1/yr
        drx_rad = drx_as * _ARC_SECOND_TO_RAD
        dry_rad = dry_as * _ARC_SECOND_TO_RAD
        drz_rad = drz_as * _ARC_SECOND_TO_RAD
        dds_scaled = dds_ppm * 1e-6

        # Coordinate Frame uses opposite rotation sign convention
        if convention == "coordinate_frame":
            rx_rad = -rx_rad
            ry_rad = -ry_rad
            rz_rad = -rz_rad
            drx_rad = -drx_rad
            dry_rad = -dry_rad
            drz_rad = -drz_rad

        src_ell = _get_ellipsoid(src_geo)
        dst_ell = _get_ellipsoid(dst_geo)

        # When +inv is set, the PROJ string contains the forward (A→B) params
        # but the pipeline applies them in reverse (B→A). We negate the params
        # to get the B→A direction, while keeping our CRS ellipsoid assignment.
        if is_inverse:
            tx, ty, tz = -tx, -ty, -tz
            rx_rad, ry_rad, rz_rad = -rx_rad, -ry_rad, -rz_rad
            ds = 1.0 / ds if ds != 0.0 else 1.0
            dtx, dty, dtz = -dtx, -dty, -dtz
            drx_rad, dry_rad, drz_rad = -drx_rad, -dry_rad, -drz_rad
            dds_scaled = -dds_scaled

        helmert = HelmertParams(
            tx=tx,
            ty=ty,
            tz=tz,
            rx=rx_rad,
            ry=ry_rad,
            rz=rz_rad,
            ds=ds,
            src_ellipsoid=src_ell,
            dst_ellipsoid=dst_ell,
            dtx=dtx,
            dty=dty,
            dtz=dtz,
            drx=drx_rad,
            dry=dry_rad,
            drz=drz_rad,
            dds=dds_scaled,
            t_epoch=t_epoch,
        )

        # Skip identity transforms (all params effectively zero)
        if (
            abs(helmert.tx) < 1e-6
            and abs(helmert.ty) < 1e-6
            and abs(helmert.tz) < 1e-6
            and abs(helmert.rx) < 1e-12
            and abs(helmert.ry) < 1e-12
            and abs(helmert.rz) < 1e-12
            and abs(helmert.ds - 1.0) < 1e-12
        ):
            return None

        return helmert

    # No Helmert operation found (e.g., grid-only transform)
    return None
