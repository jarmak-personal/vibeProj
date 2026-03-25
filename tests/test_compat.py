"""Tests for vibeproj.compat — Shapely/GeoPandas integration layer."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

from pyproj import Transformer as PyProjTransformer  # noqa: E402
from shapely import LineString, MultiPoint, Point, box  # noqa: E402

from vibeproj.compat import (  # noqa: E402
    make_shapely_transform,
    reproject_geodataframe,
    reproject_geometries,
)

SRC_CRS = "EPSG:4326"
DST_CRS = "EPSG:32631"  # UTM zone 31N


def _pyproj_ref(lon, lat):
    """Reference transform via pyproj for validation."""
    pp = PyProjTransformer.from_crs(SRC_CRS, DST_CRS, always_xy=True)
    return pp.transform(lon, lat)


# ---------------------------------------------------------------------------
# reproject_geodataframe
# ---------------------------------------------------------------------------


class TestReprojectGeoDataFrame:
    def test_point_reprojection(self):
        """Point GeoDataFrame reprojection validated against pyproj."""
        lons = [2.0, 3.0, 4.0]
        lats = [48.0, 49.0, 50.0]
        gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
            crs=SRC_CRS,
        )
        result = reproject_geodataframe(gdf, DST_CRS)
        coords = shapely.get_coordinates(result.geometry.values)

        for i, (lon, lat) in enumerate(zip(lons, lats)):
            exp_x, exp_y = _pyproj_ref(lon, lat)
            assert_allclose(coords[i, 0], exp_x, atol=0.01)
            assert_allclose(coords[i, 1], exp_y, atol=0.01)

        assert str(result.crs) != str(gdf.crs)

    def test_polygon_preserved(self):
        """Polygon geometry type is preserved after reprojection."""
        gdf = gpd.GeoDataFrame(
            geometry=[box(2.0, 48.0, 3.0, 49.0), box(3.0, 49.0, 4.0, 50.0)],
            crs=SRC_CRS,
        )
        result = reproject_geodataframe(gdf, DST_CRS)
        for geom in result.geometry:
            assert geom.geom_type == "Polygon"

    def test_mixed_geometry_types(self):
        """Mixed geometry types are all reprojected."""
        gdf = gpd.GeoDataFrame(
            geometry=[
                Point(2.0, 48.0),
                LineString([(2.0, 48.0), (3.0, 49.0)]),
                box(2.0, 48.0, 3.0, 49.0),
            ],
            crs=SRC_CRS,
        )
        result = reproject_geodataframe(gdf, DST_CRS)
        assert result.geometry.iloc[0].geom_type == "Point"
        assert result.geometry.iloc[1].geom_type == "LineString"
        assert result.geometry.iloc[2].geom_type == "Polygon"

    def test_3d_geometries(self):
        """3D geometries with Z coordinates are handled."""
        gdf = gpd.GeoDataFrame(
            geometry=[Point(2.0, 48.0, 100.0), Point(3.0, 49.0, 200.0)],
            crs=SRC_CRS,
        )
        result = reproject_geodataframe(gdf, DST_CRS)
        coords = shapely.get_coordinates(result.geometry.values, include_z=True)
        assert coords.shape[1] == 3
        # Z should be present (passthrough — same datum, no Helmert)
        assert_allclose(coords[0, 2], 100.0, atol=0.01)
        assert_allclose(coords[1, 2], 200.0, atol=0.01)

    def test_no_crs_raises(self):
        """GeoDataFrame with no CRS raises ValueError."""
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        with pytest.raises(ValueError, match="no CRS"):
            reproject_geodataframe(gdf, DST_CRS)

    def test_attributes_preserved(self):
        """Non-geometry columns survive reprojection."""
        gdf = gpd.GeoDataFrame(
            {"name": ["a", "b"], "val": [1, 2]},
            geometry=[Point(2.0, 48.0), Point(3.0, 49.0)],
            crs=SRC_CRS,
        )
        result = reproject_geodataframe(gdf, DST_CRS)
        assert list(result["name"]) == ["a", "b"]
        assert list(result["val"]) == [1, 2]


# ---------------------------------------------------------------------------
# make_shapely_transform
# ---------------------------------------------------------------------------


class TestMakeShapelyTransform:
    def test_with_shapely_transform(self):
        """Callable works with shapely.transform()."""
        func = make_shapely_transform(SRC_CRS, DST_CRS)
        pt = Point(2.0, 48.0)
        result = shapely.transform(pt, func)
        exp_x, exp_y = _pyproj_ref(2.0, 48.0)
        c = shapely.get_coordinates(result)
        assert_allclose(c[0, 0], exp_x, atol=0.01)
        assert_allclose(c[0, 1], exp_y, atol=0.01)

    def test_3d_coords(self):
        """3D coordinates pass through shapely.transform."""
        func = make_shapely_transform(SRC_CRS, DST_CRS)
        pt = Point(2.0, 48.0, 100.0)
        result = shapely.transform(pt, func, include_z=True)
        c = shapely.get_coordinates(result, include_z=True)
        assert c.shape[1] == 3
        assert_allclose(c[0, 2], 100.0, atol=0.01)

    def test_multigeometry(self):
        """Works with multi-geometries."""
        func = make_shapely_transform(SRC_CRS, DST_CRS)
        mp = MultiPoint([(2.0, 48.0), (3.0, 49.0)])
        result = shapely.transform(mp, func)
        assert result.geom_type == "MultiPoint"
        assert len(result.geoms) == 2


# ---------------------------------------------------------------------------
# reproject_geometries
# ---------------------------------------------------------------------------


class TestReprojectGeometries:
    def test_single_geometry(self):
        """Single geometry in, single geometry out."""
        pt = Point(2.0, 48.0)
        result = reproject_geometries(pt, SRC_CRS, DST_CRS)
        assert result.geom_type == "Point"
        c = shapely.get_coordinates(result)
        exp_x, exp_y = _pyproj_ref(2.0, 48.0)
        assert_allclose(c[0, 0], exp_x, atol=0.01)
        assert_allclose(c[0, 1], exp_y, atol=0.01)

    def test_list_of_geometries(self):
        """List in, list out."""
        geoms = [Point(2.0, 48.0), Point(3.0, 49.0)]
        result = reproject_geometries(geoms, SRC_CRS, DST_CRS)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_numpy_array_of_geometries(self):
        """Numpy array in, numpy array out."""
        geoms = np.array([Point(2.0, 48.0), Point(3.0, 49.0)])
        result = reproject_geometries(geoms, SRC_CRS, DST_CRS)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_roundtrip(self):
        """Forward + inverse roundtrip preserves coordinates."""
        pt = Point(2.0, 48.0)
        projected = reproject_geometries(pt, SRC_CRS, DST_CRS)
        recovered = reproject_geometries(projected, DST_CRS, SRC_CRS)
        c = shapely.get_coordinates(recovered)
        assert_allclose(c[0, 0], 2.0, atol=1e-6)
        assert_allclose(c[0, 1], 48.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Performance: large batch
# ---------------------------------------------------------------------------


class TestLargeBatch:
    def test_100k_points_bulk(self):
        """100K+ points are handled in bulk (no per-geometry loop)."""
        n = 100_000
        rng = np.random.default_rng(42)
        lons = rng.uniform(0, 6, n)
        lats = rng.uniform(44, 52, n)
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lons, lats),
            crs=SRC_CRS,
        )
        result = reproject_geodataframe(gdf, DST_CRS, chunk_size=50_000)
        assert len(result) == n
        # Spot-check a few points against pyproj
        coords = shapely.get_coordinates(result.geometry.values)
        for i in [0, n // 2, n - 1]:
            exp_x, exp_y = _pyproj_ref(lons[i], lats[i])
            assert_allclose(coords[i, 0], exp_x, atol=0.01)
            assert_allclose(coords[i, 1], exp_y, atol=0.01)
