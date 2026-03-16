"""High-level Transformer API — drop-in replacement for pyproj.Transformer.

Usage:
    from vibeproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    x, y = t.transform(lon, lat)           # always_xy=True (default)
    lon, lat = t.transform(x, y, direction="INVERSE")
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vibeproj.crs import resolve_transform
from vibeproj.pipeline import TransformPipeline
from vibeproj.runtime import get_array_module, to_device


class Transformer:
    """GPU-accelerated coordinate transformer.

    - transform(x, y) where x=lon, y=lat for geographic CRS (always_xy=True default)
    - direction="FORWARD" or "INVERSE"
    - Accepts scalars, lists, numpy arrays, or cupy arrays

    When CuPy is available and inputs are on GPU, transforms run on GPU.
    Otherwise falls back to NumPy on CPU.
    """

    def __init__(self, crs_from, crs_to, *, always_xy=False):
        src_params, dst_params = resolve_transform(crs_from, crs_to)

        # always_xy=True forces (x, y) = (lon, lat) / (easting, northing) order,
        # matching shapely/geopandas conventions regardless of CRS native axis order.
        if always_xy:
            src_params = dataclasses.replace(src_params, north_first=False)
            dst_params = dataclasses.replace(dst_params, north_first=False)

        self._pipeline = TransformPipeline(src_params, dst_params)
        self._src_params = src_params
        self._dst_params = dst_params
        self._always_xy = always_xy
        # Build the inverse pipeline lazily
        self._inv_pipeline = None

    @staticmethod
    def from_crs(crs_from, crs_to, *, always_xy=False) -> Transformer:
        """Create a Transformer from source and target CRS.

        Parameters
        ----------
        crs_from, crs_to :
            EPSG integer (4326), string ("EPSG:4326"), or tuple (("EPSG", 4326)).
        always_xy : bool, default False
            If True, input/output axis order is always (x, y) — i.e.
            (longitude, latitude) for geographic CRS and (easting, northing)
            for projected CRS. This matches shapely and geopandas conventions.
            If False, uses the CRS native axis order (pyproj default).
        """
        return Transformer(crs_from, crs_to, always_xy=always_xy)

    def transform(self, x, y, direction="FORWARD"):
        """Transform coordinates.

        Parameters
        ----------
        x, y : scalar, list, numpy array, or cupy array
            Input coordinates. With always_xy=True (default): x=longitude, y=latitude
            for geographic CRS. With always_xy=False: native CRS axis order.
        direction : str
            "FORWARD" or "INVERSE".

        Returns
        -------
        tuple of arrays (or scalars if scalar input)
            Transformed (x, y) coordinates.
        """
        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        # Detect scalar input
        is_scalar = isinstance(x, (int, float)) and isinstance(y, (int, float))

        # Determine array module from input
        xp = get_array_module(x)
        if xp is np:
            xp = get_array_module(y)

        if is_scalar:
            x = xp.asarray([x], dtype="f8")
            y = xp.asarray([y], dtype="f8")
        else:
            x = to_device(x, xp)
            y = to_device(y, xp)
            # Ensure float dtype
            if not xp.issubdtype(x.dtype, xp.floating):
                x = x.astype(xp.float64)
            if not xp.issubdtype(y.dtype, xp.floating):
                y = y.astype(xp.float64)

        if direction == "FORWARD":
            rx, ry = self._pipeline.transform(x, y, xp)
        else:
            if self._inv_pipeline is None:
                self._inv_pipeline = TransformPipeline(self._dst_params, self._src_params)
            rx, ry = self._inv_pipeline.transform(x, y, xp)

        if is_scalar:
            # Convert back to Python floats
            if hasattr(rx, "get"):
                rx, ry = float(rx.get()[0]), float(ry.get()[0])
            else:
                rx, ry = float(rx[0]), float(ry[0])

        return rx, ry

    def transform_buffers(
        self, x, y, *, direction="FORWARD", out_x=None, out_y=None, precision="auto"
    ):
        """Zero-overhead transform for device-resident arrays.

        Designed for integration with vibeSpatial's OwnedGeometryArray.
        Skips scalar detection, dtype conversion, and array module inference.

        Parameters
        ----------
        x, y : cupy.ndarray or numpy.ndarray
            Coordinate arrays (fp64 storage per ADR-0002).
        direction : str
            "FORWARD" or "INVERSE".
        out_x, out_y : cupy.ndarray or numpy.ndarray, optional
            Pre-allocated fp64 output arrays. Avoids allocation.
        precision : str
            "fp64" = full double precision.
            "fp32" = fp32 compute with fp64 I/O (ADR-0002 mixed precision).
                     Gives ~32x throughput on consumer GPUs for projection math.
            "auto" = fp64 (projection math is trig-dominated / SFU-bound).

        Returns
        -------
        tuple of arrays
            Transformed (out_x, out_y). Same objects if pre-allocated.
        """
        xp = get_array_module(x)

        if direction == "FORWARD":
            pipeline = self._pipeline
        else:
            if self._inv_pipeline is None:
                self._inv_pipeline = TransformPipeline(self._dst_params, self._src_params)
            pipeline = self._inv_pipeline

        return pipeline.transform(x, y, xp, out_x=out_x, out_y=out_y, precision=precision)
