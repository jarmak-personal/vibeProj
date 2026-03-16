"""Base class for map projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from vibeproj.crs import ProjectionParams


class Projection:
    """Base class for all map projections.

    Subclasses implement forward/inverse transforms that operate on arrays
    of coordinates. The `xp` parameter is the array module (numpy or cupy).

    Projection math operates in radians on the ellipsoid. The pipeline
    handles degree/radian conversion, axis swapping, and false easting/northing.
    """

    name: str = ""

    def setup(self, params: ProjectionParams) -> dict:
        """Compute derived parameters from projection params.

        Called once at construction time. Returns a dict of computed params
        that will be passed to forward/inverse.
        """
        return {}

    def forward(
        self, lam: object, phi: object, params: ProjectionParams, computed: dict, xp: ModuleType
    ) -> tuple:
        """Forward projection: geographic (lon, lat in radians) -> projected (x, y in meters).

        lam: longitude relative to central meridian (radians), array
        phi: latitude (radians), array
        Returns (x, y) in projection-native units (before false easting/northing and scale).
        """
        raise NotImplementedError

    def inverse(
        self, x: object, y: object, params: ProjectionParams, computed: dict, xp: ModuleType
    ) -> tuple:
        """Inverse projection: projected (x, y) -> geographic (lon, lat in radians).

        x, y: projection-native units (after removing false easting/northing and scale).
        Returns (lam, phi) in radians, with lam relative to central meridian.
        """
        raise NotImplementedError
