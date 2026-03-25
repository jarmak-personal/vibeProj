"""Thin compatibility layer for Shapely 2.x and GeoPandas integration.

Not re-exported from vibeproj.__init__. Use explicitly:
    from vibeproj.compat import reproject_geodataframe
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from vibeproj.transformer import Transformer

if TYPE_CHECKING:
    import geopandas as gpd

    from vibeproj.crs import CRSInput


def _transform_coords(
    coords: np.ndarray,
    t: Transformer,
    chunk_size: int,
) -> np.ndarray:
    """Bulk-transform an (N, 2) or (N, 3) coordinate array."""
    x, y = coords[:, 0], coords[:, 1]
    has_z = coords.shape[1] >= 3
    if has_z:
        z = coords[:, 2]
        rx, ry, rz = t.transform_chunked(x, y, z=z, chunk_size=chunk_size)
        return np.column_stack([rx, ry, rz])
    rx, ry = t.transform_chunked(x, y, chunk_size=chunk_size)
    return np.column_stack([rx, ry])


def reproject_geodataframe(
    gdf: gpd.GeoDataFrame,
    dst_crs: CRSInput,
    *,
    transformer: Transformer | None = None,
    chunk_size: int = 1_000_000,
    **kw: Any,
) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame using vibeProj's bulk transform.

    Returns a new GeoDataFrame with the target CRS set.  Pass a pre-built
    ``transformer`` to reuse CRS resolution and pinned-buffer pools across calls.
    """
    import shapely

    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set. Assign gdf.crs before reprojecting.")

    geom_arr = gdf.geometry.values
    has_z = bool(shapely.has_z(geom_arr).any())
    coords = shapely.get_coordinates(geom_arr, include_z=has_z)

    t = transformer or Transformer.from_crs(str(gdf.crs), dst_crs, **kw)
    new_coords = _transform_coords(coords, t, chunk_size)

    new_geoms = shapely.set_coordinates(geom_arr.copy(), new_coords)
    result = gdf.copy()
    result[gdf.geometry.name] = new_geoms
    result = result.set_geometry(gdf.geometry.name)
    result = result.set_crs(dst_crs, allow_override=True)
    return result


def make_shapely_transform(
    src_crs: CRSInput,
    dst_crs: CRSInput,
    *,
    chunk_size: int = 1_000_000,
    **kw: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable for use with ``shapely.transform(geom, func)``.

    The returned function accepts an (N, 2) or (N, 3) coordinate array.
    """
    t = Transformer.from_crs(src_crs, dst_crs, **kw)

    def _transform(coords: np.ndarray) -> np.ndarray:
        return _transform_coords(coords, t, chunk_size)

    return _transform


def reproject_geometries(
    geometries: Any,
    src_crs: CRSInput,
    dst_crs: CRSInput,
    *,
    transformer: Transformer | None = None,
    chunk_size: int = 1_000_000,
    **kw: Any,
) -> Any:
    """Bulk-reproject Shapely geometries via coordinate extraction.

    Accepts a single geometry, list, or numpy array. Returns same type.
    Pass a pre-built ``transformer`` to amortise CRS resolution across calls.
    """
    import shapely

    single = False
    if hasattr(geometries, "geom_type"):
        # Single Shapely geometry
        single = True
        geom_arr = np.array([geometries])
    elif isinstance(geometries, np.ndarray):
        geom_arr = geometries
    else:
        geom_arr = np.array(list(geometries))

    has_z = bool(shapely.has_z(geom_arr).any())
    coords = shapely.get_coordinates(geom_arr, include_z=has_z)

    t = transformer or Transformer.from_crs(src_crs, dst_crs, **kw)
    new_coords = _transform_coords(coords, t, chunk_size)

    result = shapely.set_coordinates(geom_arr.copy(), new_coords)
    if single:
        return result[0]
    if isinstance(geometries, np.ndarray):
        return result
    return list(result)
