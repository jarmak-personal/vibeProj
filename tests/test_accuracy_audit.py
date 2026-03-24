"""Comprehensive accuracy audit: every projection vs pyproj.

Tests forward and inverse transforms using grids of points, comparing
vibeProj output against pyproj with tight tolerances.  The goal is to
find accuracy regressions and systematic errors across all 24 projections.
"""

import numpy as np
import pytest
from pyproj import Proj
from pyproj import Transformer as PyProjTransformer

from vibeproj import Transformer
from vibeproj.crs import ProjectionParams
from vibeproj.ellipsoid import WGS84
from vibeproj.pipeline import TransformPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(lat_range, lon_range, n=20):
    """Create an n×n grid of lat/lon points."""
    lat = np.linspace(lat_range[0], lat_range[1], n)
    lon = np.linspace(lon_range[0], lon_range[1], n)
    lat_g, lon_g = np.meshgrid(lat, lon)
    return lat_g.ravel(), lon_g.ravel()


def _max_error(a, b):
    """Max absolute error between two arrays, ignoring NaN."""
    diff = np.abs(a - b)
    return np.nanmax(diff)


# ---------------------------------------------------------------------------
# Part 1: EPSG-based projections — direct comparison against pyproj
#
# For same-datum projections we expect sub-millimeter forward accuracy.
# For cross-datum projections the Helmert approximation limits accuracy.
# ---------------------------------------------------------------------------


class TestEPSGProjections:
    """Forward + inverse accuracy against pyproj for every EPSG-addressable projection."""

    # --- Transverse Mercator (UTM) ---

    @pytest.mark.parametrize(
        "epsg, lat_range, lon_range",
        [
            ("EPSG:32610", (32.0, 42.0), (-126.0, -117.0)),  # UTM 10N - US West
            ("EPSG:32631", (44.0, 56.0), (0.0, 6.0)),  # UTM 31N - France/UK
            ("EPSG:32756", (-38.0, -28.0), (148.0, 156.0)),  # UTM 56S - Sydney
            ("EPSG:32618", (36.0, 44.0), (-78.0, -72.0)),  # UTM 18N - US East
            ("EPSG:32719", (-58.0, -50.0), (-72.0, -64.0)),  # UTM 19S - Patagonia
            ("EPSG:32648", (-2.0, 6.0), (102.0, 108.0)),  # UTM 48N - Singapore
        ],
    )
    def test_tmerc_forward_grid(self, epsg, lat_range, lon_range):
        lat, lon = _make_grid(lat_range, lon_range, n=25)
        pp = PyProjTransformer.from_crs("EPSG:4326", epsg)
        t = Transformer.from_crs("EPSG:4326", epsg, always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 1e-3, (
            f"tmerc {epsg} x error: {_max_error(vp_x, exp_x):.6e}"
        )
        assert _max_error(vp_y, exp_y) < 1e-3, (
            f"tmerc {epsg} y error: {_max_error(vp_y, exp_y):.6e}"
        )

    @pytest.mark.parametrize(
        "epsg, lat_range, lon_range",
        [
            ("EPSG:32631", (44.0, 56.0), (0.0, 6.0)),
            ("EPSG:32756", (-38.0, -28.0), (148.0, 156.0)),
        ],
    )
    def test_tmerc_inverse_grid(self, epsg, lat_range, lon_range):
        lat, lon = _make_grid(lat_range, lon_range, n=25)
        pp = PyProjTransformer.from_crs("EPSG:4326", epsg)
        t = Transformer.from_crs("EPSG:4326", epsg, always_xy=False)

        # Forward
        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        # Inverse
        from pyproj.enums import TransformDirection

        pp_lat, pp_lon = pp.transform(exp_x, exp_y, direction=TransformDirection.INVERSE)
        vp_lat, vp_lon = t.transform(vp_x, vp_y, direction="INVERSE")

        assert _max_error(vp_lat, pp_lat) < 1e-9, (
            f"tmerc inv lat error: {_max_error(vp_lat, pp_lat):.6e}"
        )
        assert _max_error(vp_lon, pp_lon) < 1e-9, (
            f"tmerc inv lon error: {_max_error(vp_lon, pp_lon):.6e}"
        )

    # --- Web Mercator ---

    def test_webmerc_forward_grid(self):
        # Web Mercator valid range: ~±85.06°
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3857")
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"webmerc x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"webmerc y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Mercator (ellipsoidal) ---

    def test_merc_forward_grid(self):
        lat, lon = _make_grid((-70.0, 70.0), (-170.0, 170.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3395")
        t = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"merc x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"merc y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Lambert Conformal Conic ---

    def test_lcc_forward_grid(self):
        # EPSG:2154 France Lambert 93
        lat, lon = _make_grid((42.0, 51.5), (-5.0, 9.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:2154")
        t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"lcc x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"lcc y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Albers Equal Area ---

    def test_aea_forward_grid(self):
        # EPSG:5070 NAD83 CONUS Albers
        lat, lon = _make_grid((25.0, 50.0), (-125.0, -65.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:5070")
        t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"aea x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"aea y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Polar Stereographic ---

    def test_stere_forward_grid(self):
        # EPSG:3031 Antarctic
        lat, lon = _make_grid((-89.0, -65.0), (-180.0, 180.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3031")
        t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.1, f"stere x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.1, f"stere y error: {_max_error(vp_y, exp_y):.6e}"

    # --- LAEA ---

    def test_laea_forward_grid(self):
        # EPSG:3035 LAEA Europe
        lat, lon = _make_grid((45.0, 65.0), (-10.0, 30.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3035")
        t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"laea x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"laea y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Equidistant Cylindrical ---

    def test_eqc_forward_grid(self):
        lat, lon = _make_grid((-70.0, 70.0), (-170.0, 170.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:4087")
        t = Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"eqc x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"eqc y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Equal Earth ---

    def test_eqearth_forward_grid(self):
        lat, lon = _make_grid((-70.0, 70.0), (-170.0, 170.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:8857")
        t = Transformer.from_crs("EPSG:4326", "EPSG:8857", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"eqearth x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"eqearth y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Cylindrical Equal Area ---

    def test_cea_forward_grid(self):
        lat, lon = _make_grid((-70.0, 70.0), (-170.0, 170.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:6933")
        t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"cea x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"cea y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Oblique Mercator ---

    def test_omerc_forward_grid(self):
        # EPSG:3168 Malaysia RSO (variant A) — same-datum
        lat, lon = _make_grid((1.0, 7.0), (100.0, 104.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:3168", always_xy=False)
        t = Transformer.from_crs("EPSG:4326", "EPSG:3168", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        assert _max_error(vp_x, exp_x) < 0.01, f"omerc x error: {_max_error(vp_x, exp_x):.6e}"
        assert _max_error(vp_y, exp_y) < 0.01, f"omerc y error: {_max_error(vp_y, exp_y):.6e}"

    # --- Oblique Stereographic (cross-datum) ---

    def test_sterea_forward_grid(self):
        # EPSG:28992 Amersfoort/RD New — cross-datum (WGS84→Bessel)
        lat, lon = _make_grid((51.0, 53.5), (3.0, 7.5), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)
        t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        err_x = _max_error(vp_x, exp_x)
        err_y = _max_error(vp_y, exp_y)
        # Cross-datum: Helmert ~5m accuracy
        assert err_x < 10.0, f"sterea x error: {err_x:.3f}m"
        assert err_y < 10.0, f"sterea y error: {err_y:.3f}m"

    # --- Krovak (cross-datum) ---

    def test_krovak_forward_grid(self):
        # EPSG:5514 S-JTSK / Krovak East North — cross-datum
        lat, lon = _make_grid((48.0, 51.0), (12.0, 18.0), n=20)
        pp = PyProjTransformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=False)
        t = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=False)

        exp_x, exp_y = pp.transform(lat, lon)
        vp_x, vp_y = t.transform(lat, lon)

        err_x = _max_error(vp_x, exp_x)
        err_y = _max_error(vp_y, exp_y)
        # Cross-datum: Helmert ~5-10m
        assert err_x < 15.0, f"krovak x error: {err_x:.3f}m"
        assert err_y < 15.0, f"krovak y error: {err_y:.3f}m"


# ---------------------------------------------------------------------------
# Part 2: Manual-pipeline projections — compare against pyproj +proj= strings
#
# For spherical-only projections (sinu, ortho, gnom, moll, eck4, eck6,
# robin, wintri, natearth, aeqd), vibeProj uses WGS84.a as sphere radius.
# We construct matching pyproj Proj objects with the same parameters.
#
# For ellipsoidal projections (geos, eqearth, cea), we use the full WGS84
# ellipsoid in pyproj.
# ---------------------------------------------------------------------------


def _vibeproj_forward(proj_name, lat, lon, lat_0=0.0, lon_0=0.0, extra=None):
    """Run a vibeProj forward transform via TransformPipeline."""
    params = ProjectionParams(
        projection_name=proj_name,
        ellipsoid=WGS84,
        lon_0=lon_0,
        lat_0=lat_0,
        north_first=False,
        extra=extra or {},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    pipe = TransformPipeline(src, params)
    return pipe.transform(lat, lon, np)


def _vibeproj_inverse(proj_name, x, y, lat_0=0.0, lon_0=0.0, extra=None):
    """Run a vibeProj inverse transform via TransformPipeline."""
    params = ProjectionParams(
        projection_name=proj_name,
        ellipsoid=WGS84,
        lon_0=lon_0,
        lat_0=lat_0,
        north_first=False,
        extra=extra or {},
    )
    src = ProjectionParams(projection_name="longlat", ellipsoid=WGS84, north_first=True)
    inv_pipe = TransformPipeline(params, src)
    return inv_pipe.transform(x, y, np)


class TestSphericalProjectionsVsPyproj:
    """Forward accuracy of spherical-only projections against pyproj.

    These projections use only the semi-major axis a from WGS84.
    We match that in pyproj with +R=6378137 (sphere of same radius).
    """

    WGS84_A = 6378137.0

    def _pyproj_forward(self, proj_name, lat, lon, lon_0=0.0, lat_0=0.0, **extra):
        """Forward transform via pyproj Proj with spherical params."""
        proj_str = f"+proj={proj_name} +lon_0={lon_0} +lat_0={lat_0} +R={self.WGS84_A} +units=m"
        for k, v in extra.items():
            proj_str += f" +{k}={v}"
        p = Proj(proj_str)
        return p(lon, lat)  # pyproj.Proj takes (lon, lat)

    # --- Sinusoidal ---

    def test_sinu_forward(self):
        lat, lon = _make_grid((-70.0, 70.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("sinu", lat, lon)
        pp_x, pp_y = self._pyproj_forward("sinu", lat, lon)

        assert _max_error(vp_x, pp_x) < 0.001, f"sinu x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"sinu y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Orthographic ---

    def test_ortho_forward(self):
        # Points must be within visible hemisphere (< 90° from center)
        lat, lon = _make_grid((20.0, 70.0), (-30.0, 30.0), n=20)
        vp_x, vp_y = _vibeproj_forward("ortho", lat, lon, lat_0=45.0)
        pp_x, pp_y = self._pyproj_forward("ortho", lat, lon, lat_0=45.0)

        assert _max_error(vp_x, pp_x) < 0.001, f"ortho x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"ortho y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Gnomonic ---

    def test_gnom_forward(self):
        # Gnomonic distorts rapidly — keep points within ~30° of center
        lat, lon = _make_grid((30.0, 60.0), (-15.0, 15.0), n=20)
        vp_x, vp_y = _vibeproj_forward("gnom", lat, lon, lat_0=45.0)
        pp_x, pp_y = self._pyproj_forward("gnom", lat, lon, lat_0=45.0)

        assert _max_error(vp_x, pp_x) < 0.001, f"gnom x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"gnom y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Mollweide ---

    def test_moll_forward(self):
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("moll", lat, lon)
        pp_x, pp_y = self._pyproj_forward("moll", lat, lon)

        assert _max_error(vp_x, pp_x) < 0.001, f"moll x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"moll y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Eckert IV ---

    def test_eck4_forward(self):
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("eck4", lat, lon)
        pp_x, pp_y = self._pyproj_forward("eck4", lat, lon)

        assert _max_error(vp_x, pp_x) < 0.001, f"eck4 x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"eck4 y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Eckert VI ---

    def test_eck6_forward(self):
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("eck6", lat, lon)
        pp_x, pp_y = self._pyproj_forward("eck6", lat, lon)

        assert _max_error(vp_x, pp_x) < 0.001, f"eck6 x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"eck6 y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Robinson ---

    def test_robin_forward(self):
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("robin", lat, lon)
        pp_x, pp_y = self._pyproj_forward("robin", lat, lon)

        # Robinson uses linear table interpolation; PROJ uses cubic.
        # ~13 km max at global scale is the interpolation approximation, not a formula bug.
        assert _max_error(vp_x, pp_x) < 15000, f"robin x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 5000, f"robin y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Winkel Tripel ---

    def test_wintri_forward(self):
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("wintri", lat, lon)
        pp_x, pp_y = self._pyproj_forward("wintri", lat, lon)

        assert _max_error(vp_x, pp_x) < 0.001, f"wintri x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"wintri y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Natural Earth ---

    def test_natearth_forward(self):
        lat, lon = _make_grid((-80.0, 80.0), (-170.0, 170.0), n=25)
        vp_x, vp_y = _vibeproj_forward("natearth", lat, lon)
        pp_x, pp_y = self._pyproj_forward("natearth", lat, lon)

        assert _max_error(vp_x, pp_x) < 0.001, f"natearth x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"natearth y error: {_max_error(vp_y, pp_y):.6e}"

    # --- Azimuthal Equidistant ---

    def test_aeqd_forward(self):
        # Points within ~60° of center
        lat, lon = _make_grid((10.0, 70.0), (-40.0, 40.0), n=20)
        vp_x, vp_y = _vibeproj_forward("aeqd", lat, lon, lat_0=45.0)
        pp_x, pp_y = self._pyproj_forward("aeqd", lat, lon, lat_0=45.0)

        assert _max_error(vp_x, pp_x) < 0.001, f"aeqd x error: {_max_error(vp_x, pp_x):.6e}"
        assert _max_error(vp_y, pp_y) < 0.001, f"aeqd y error: {_max_error(vp_y, pp_y):.6e}"


# ---------------------------------------------------------------------------
# Part 3: Geostationary — ellipsoidal, uses a and b
# ---------------------------------------------------------------------------


class TestGeostationaryVsPyproj:
    """Geostationary uses both a and b from WGS84."""

    def test_geos_forward(self):
        h = 35785831.0
        # Points visible from geostationary orbit at lon_0=0
        lat, lon = _make_grid((-60.0, 60.0), (-60.0, 60.0), n=20)

        vp_x, vp_y = _vibeproj_forward("geos", lat, lon, extra={"h": h})

        p = Proj(f"+proj=geos +lon_0=0 +h={h} +x_0=0 +y_0=0 +ellps=WGS84 +units=m")
        pp_x, pp_y = p(lon, lat)

        # Filter out points that pyproj marks as inf (outside visible disk)
        mask = np.isfinite(pp_x) & np.isfinite(pp_y) & np.isfinite(vp_x) & np.isfinite(vp_y)
        assert mask.sum() > 0, "No valid points for geostationary comparison"

        err_x = _max_error(vp_x[mask], pp_x[mask])
        err_y = _max_error(vp_y[mask], pp_y[mask])

        assert err_x < 0.01, f"geos x error: {err_x:.6e}"
        assert err_y < 0.01, f"geos y error: {err_y:.6e}"


# ---------------------------------------------------------------------------
# Part 4: Roundtrip accuracy — forward then inverse should recover input
# ---------------------------------------------------------------------------


class TestRoundtripAccuracy:
    """Forward+inverse roundtrip must recover input coordinates."""

    @pytest.mark.parametrize(
        "proj_name, lat_0, lon_0, lat_range, lon_range, atol",
        [
            ("sinu", 0.0, 0.0, (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("ortho", 45.0, 0.0, (20.0, 70.0), (-30.0, 30.0), 1e-9),
            ("gnom", 45.0, 0.0, (30.0, 60.0), (-15.0, 15.0), 1e-9),
            ("moll", 0.0, 0.0, (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("eck4", 0.0, 0.0, (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("eck6", 0.0, 0.0, (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("robin", 0.0, 0.0, (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("natearth", 0.0, 0.0, (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("aeqd", 45.0, 0.0, (10.0, 70.0), (-40.0, 40.0), 1e-9),
            ("wintri", 0.0, 0.0, (-70.0, 70.0), (-150.0, 150.0), 0.02),  # Newton limited
        ],
    )
    def test_manual_pipeline_roundtrip(self, proj_name, lat_0, lon_0, lat_range, lon_range, atol):
        lat, lon = _make_grid(lat_range, lon_range, n=20)
        x, y = _vibeproj_forward(proj_name, lat, lon, lat_0=lat_0, lon_0=lon_0)

        # Filter out any NaN/inf from projections with limited domains
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() == 0:
            pytest.skip(f"No finite forward results for {proj_name}")

        lat2, lon2 = _vibeproj_inverse(proj_name, x[mask], y[mask], lat_0=lat_0, lon_0=lon_0)

        assert _max_error(lat2, lat[mask]) < atol, (
            f"{proj_name} roundtrip lat error: {_max_error(lat2, lat[mask]):.6e}"
        )
        assert _max_error(lon2, lon[mask]) < atol, (
            f"{proj_name} roundtrip lon error: {_max_error(lon2, lon[mask]):.6e}"
        )

    @pytest.mark.parametrize(
        "epsg, lat_range, lon_range, atol",
        [
            ("EPSG:32631", (44.0, 56.0), (0.0, 6.0), 1e-9),
            ("EPSG:3857", (-80.0, 80.0), (-170.0, 170.0), 1e-9),
            ("EPSG:3395", (-70.0, 70.0), (-170.0, 170.0), 1e-9),
            ("EPSG:2154", (42.0, 51.5), (-5.0, 9.0), 1e-9),
            ("EPSG:5070", (25.0, 50.0), (-125.0, -65.0), 1e-9),
            ("EPSG:3031", (-89.0, -65.0), (-180.0, 180.0), 1e-8),
            ("EPSG:3035", (45.0, 65.0), (-10.0, 30.0), 1e-9),
            ("EPSG:4087", (-70.0, 70.0), (-170.0, 170.0), 1e-9),
            ("EPSG:8857", (-70.0, 70.0), (-170.0, 170.0), 1e-9),
            ("EPSG:6933", (-70.0, 70.0), (-170.0, 170.0), 1e-9),
            ("EPSG:3168", (1.0, 7.0), (100.0, 104.0), 5e-9),
        ],
    )
    def test_epsg_roundtrip(self, epsg, lat_range, lon_range, atol):
        lat, lon = _make_grid(lat_range, lon_range, n=20)
        t = Transformer.from_crs("EPSG:4326", epsg, always_xy=False)

        x, y = t.transform(lat, lon)
        lat2, lon2 = t.transform(x, y, direction="INVERSE")

        assert _max_error(lat2, lat) < atol, (
            f"{epsg} roundtrip lat error: {_max_error(lat2, lat):.6e}"
        )
        assert _max_error(lon2, lon) < atol, (
            f"{epsg} roundtrip lon error: {_max_error(lon2, lon):.6e}"
        )


# ---------------------------------------------------------------------------
# Part 5: Error magnitude report — not a pass/fail test, just prints errors
# ---------------------------------------------------------------------------


class TestErrorReport:
    """Collect and print maximum errors for all projections.

    This is a diagnostic test: it always passes but prints a table of
    max errors so we can spot regressions and prioritize fixes.
    """

    def test_print_error_summary(self, capsys):
        results = []

        # EPSG-based projections
        epsg_cases = [
            ("tmerc", "EPSG:32631", (48.0, 52.0), (0.0, 6.0)),
            ("webmerc", "EPSG:3857", (-60.0, 60.0), (-150.0, 150.0)),
            ("merc", "EPSG:3395", (-60.0, 60.0), (-150.0, 150.0)),
            ("lcc", "EPSG:2154", (43.0, 51.0), (-4.0, 8.0)),
            ("aea", "EPSG:5070", (28.0, 48.0), (-120.0, -70.0)),
            ("stere", "EPSG:3031", (-89.0, -65.0), (-170.0, 170.0)),
            ("laea", "EPSG:3035", (45.0, 65.0), (-5.0, 25.0)),
            ("eqc", "EPSG:4087", (-60.0, 60.0), (-150.0, 150.0)),
            ("eqearth", "EPSG:8857", (-60.0, 60.0), (-150.0, 150.0)),
            ("cea", "EPSG:6933", (-60.0, 60.0), (-150.0, 150.0)),
            ("omerc", "EPSG:3168", (1.0, 7.0), (100.0, 104.0)),
            ("sterea", "EPSG:28992", (51.0, 53.5), (3.0, 7.5)),
            ("krovak", "EPSG:5514", (48.0, 51.0), (12.0, 18.0)),
        ]

        for name, epsg, lat_r, lon_r in epsg_cases:
            lat, lon = _make_grid(lat_r, lon_r, n=15)
            pp = PyProjTransformer.from_crs("EPSG:4326", epsg)
            t = Transformer.from_crs("EPSG:4326", epsg, always_xy=False)

            exp_x, exp_y = pp.transform(lat, lon)
            vp_x, vp_y = t.transform(lat, lon)

            err_x = _max_error(vp_x, exp_x)
            err_y = _max_error(vp_y, exp_y)
            results.append((name, epsg, err_x, err_y))

        # Manual-pipeline projections vs pyproj
        sphere_cases = [
            ("sinu", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("ortho", {}, (20.0, 70.0), (-30.0, 30.0), 45.0, 0.0),
            ("gnom", {}, (30.0, 60.0), (-15.0, 15.0), 45.0, 0.0),
            ("moll", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("eck4", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("eck6", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("robin", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("wintri", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("natearth", {}, (-70.0, 70.0), (-170.0, 170.0), 0.0, 0.0),
            ("aeqd", {}, (10.0, 70.0), (-40.0, 40.0), 45.0, 0.0),
        ]

        for name, extra_proj, lat_r, lon_r, lat_0, lon_0 in sphere_cases:
            lat, lon = _make_grid(lat_r, lon_r, n=15)
            vp_x, vp_y = _vibeproj_forward(name, lat, lon, lat_0=lat_0, lon_0=lon_0)

            proj_str = f"+proj={name} +lon_0={lon_0} +lat_0={lat_0} +R=6378137 +units=m"
            p = Proj(proj_str)
            pp_x, pp_y = p(lon, lat)

            mask = np.isfinite(vp_x) & np.isfinite(vp_y) & np.isfinite(pp_x) & np.isfinite(pp_y)
            if mask.sum() == 0:
                results.append((name, "manual", float("inf"), float("inf")))
                continue

            err_x = _max_error(vp_x[mask], pp_x[mask])
            err_y = _max_error(vp_y[mask], pp_y[mask])
            results.append((name, "manual", err_x, err_y))

        # Geostationary
        h = 35785831.0
        lat, lon = _make_grid((-50.0, 50.0), (-50.0, 50.0), n=15)
        vp_x, vp_y = _vibeproj_forward("geos", lat, lon, extra={"h": h})
        p = Proj(f"+proj=geos +lon_0=0 +h={h} +x_0=0 +y_0=0 +ellps=WGS84 +units=m")
        pp_x, pp_y = p(lon, lat)
        mask = np.isfinite(vp_x) & np.isfinite(vp_y) & np.isfinite(pp_x) & np.isfinite(pp_y)
        if mask.sum() > 0:
            results.append(
                (
                    "geos",
                    "manual",
                    _max_error(vp_x[mask], pp_x[mask]),
                    _max_error(vp_y[mask], pp_y[mask]),
                )
            )

        # Print summary
        print("\n" + "=" * 72)
        print("ACCURACY AUDIT — max forward error vs pyproj (meters)")
        print("=" * 72)
        print(f"{'Projection':<12} {'CRS':<14} {'Max ΔX (m)':>14} {'Max ΔY (m)':>14}")
        print("-" * 72)
        for name, crs, ex, ey in results:
            print(f"{name:<12} {crs:<14} {ex:>14.6e} {ey:>14.6e}")
        print("=" * 72)
