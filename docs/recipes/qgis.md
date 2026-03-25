# QGIS Processing Plugin

Skeleton recipe for wrapping vibeProj as a QGIS Processing algorithm.
This provides a starting point for batch vector reprojection of large
layers. A production plugin should live in a separate repository
(`vibeproj-qgis`).

## Minimal Processing provider

A QGIS Processing provider registers one or more algorithms that appear
in the Processing Toolbox and can be used in graphical models and batch
jobs.

```python
"""vibeProj Processing provider for QGIS.

Drop this file (and the algorithm file below) into
~/.local/share/QGIS/QGIS3/profiles/default/processing/scripts/
or package as a proper plugin with metadata.txt.
"""

from qgis.core import QgsProcessingProvider


class VibeProjectionProvider(QgsProcessingProvider):
    def id(self):
        return "vibeproj"

    def name(self):
        return "vibeProj"

    def longName(self):
        return "vibeProj GPU-Accelerated Reprojection"

    def loadAlgorithms(self):
        self.addAlgorithm(VibeProjectionBatchReproject())
```

## Batch vector reprojection algorithm

The algorithm reads all features from an input layer, extracts
coordinates in bulk, transforms them with vibeProj, and writes the
output layer.

```python
import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterCrs,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsWkbTypes,
)


class VibeProjectionBatchReproject(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET_CRS = "TARGET_CRS"
    OUTPUT = "OUTPUT"

    def name(self):
        return "batchreproject"

    def displayName(self):
        return "Batch Reproject (vibeProj)"

    def group(self):
        return "Reprojection"

    def groupId(self):
        return "reprojection"

    def createInstance(self):
        return VibeProjectionBatchReproject()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT, "Input layer",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS, "Target CRS",
                defaultValue="EPSG:4326",
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT, "Reprojected layer",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from vibeproj import Transformer

        source = self.parameterAsSource(parameters, self.INPUT, context)
        target_crs = self.parameterAsCrs(
            parameters, self.TARGET_CRS, context,
        )

        # Build the vibeProj transformer from EPSG codes.
        src_auth = source.sourceCrs().authid()   # e.g. "EPSG:4326"
        dst_auth = target_crs.authid()
        t = Transformer.from_crs(src_auth, dst_auth)

        # Prepare output sink with the target CRS.
        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            source.fields(), source.wkbType(), target_crs,
        )

        # --- Bulk coordinate extraction ---
        features = list(source.getFeatures())
        total = len(features)
        if total == 0:
            return {self.OUTPUT: dest_id}

        # Collect all vertex coordinates from all geometries.
        all_x, all_y = [], []
        vertex_counts = []
        for feat in features:
            geom = feat.geometry()
            vertices = geom.constGet().coordinateSequence()
            count = 0
            for ring_group in vertices:
                for ring in ring_group:
                    for pt in ring:
                        all_x.append(pt.x())
                        all_y.append(pt.y())
                        count += 1
            vertex_counts.append(count)

        # --- Bulk transform ---
        x_arr = np.array(all_x, dtype=np.float64)
        y_arr = np.array(all_y, dtype=np.float64)

        # GPU path via transform_chunked, or CPU fallback (automatic).
        out_x, out_y = t.transform_chunked(x_arr, y_arr)

        # --- Reconstruct geometries ---
        offset = 0
        for i, feat in enumerate(features):
            if feedback.isCanceled():
                break

            n = vertex_counts[i]
            geom = feat.geometry()
            new_geom = _replace_vertices(
                geom, out_x[offset:offset + n], out_y[offset:offset + n],
            )
            offset += n

            out_feat = QgsFeature(feat)
            out_feat.setGeometry(new_geom)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            feedback.setProgress(int((i + 1) / total * 100))

        return {self.OUTPUT: dest_id}


def _replace_vertices(geom, new_x, new_y):
    """Replace all vertices in a QgsGeometry with new coordinates.

    This is a simplified version that handles points and linestrings.
    A production plugin would need to handle all geometry types
    (polygons, multi-geometries, curves) and preserve Z/M values.
    """
    # For a production plugin, manipulate the QgsAbstractGeometry
    # directly rather than round-tripping through WKT.
    if geom.type() == QgsWkbTypes.PointGeometry:
        return QgsGeometry.fromPointXY(QgsPointXY(new_x[0], new_y[0]))

    # Fallback: rebuild from coordinate list.
    points = [QgsPointXY(float(new_x[j]), float(new_y[j]))
              for j in range(len(new_x))]
    if geom.type() == QgsWkbTypes.LineGeometry:
        return QgsGeometry.fromPolylineXY(points)

    # Polygon and multi-geometry reconstruction is more involved.
    # A real plugin should walk the geometry rings properly.
    return QgsGeometry.fromPolygonXY([points])
```

## The Python-C++ boundary problem

QGIS stores geometries as C++ `QgsGeometry` objects. Every access from
Python crosses the SIP binding layer:

- `feat.geometry()` copies the C++ geometry into a Python wrapper.
- Iterating vertices with `coordinateSequence()` creates Python objects
  per-vertex.
- Reconstructing a geometry from Python coordinates copies back into C++.

This means **every coordinate crosses the Python-C++ boundary twice**
(once to extract, once to reconstruct), regardless of how fast the
transform itself is. For a layer with 50M vertices, the extraction and
reconstruction loops dominate runtime -- the GPU transform in the middle
takes a fraction of a millisecond.

There is no public C++ API in QGIS Processing that gives direct access
to coordinate buffers. The `QgsGeometry` memory layout is internal to
the GEOS/QgsAbstractGeometry hierarchy. Unlike Shapely 2.x (which
exposes `get_coordinates()` / `set_coordinates()` for bulk NumPy
access), QGIS has no equivalent batch coordinate API on the Python side.

Practical impact:

| Layer size | Extract (Python loop) | vibeProj transform | Reconstruct (Python loop) |
|---|---|---|---|
| 100K vertices | ~200 ms | < 1 ms | ~300 ms |
| 10M vertices | ~20 s | ~3 ms | ~30 s |

The transform is never the bottleneck. The Python loops are.

### Mitigation strategies

1. **Batch at the layer level, not feature level.** Extract all vertices
   from all features into flat arrays (as shown above), transform once,
   then redistribute. This at least avoids creating a `Transformer` per
   feature.

2. **Use Shapely as an intermediary.** If the layer can be exported to
   GeoPackage or GeoJSON, load it with GeoPandas/Shapely, use the bulk
   `shapely.get_coordinates()` path (see the [Shapely recipe](shapely.md)
   and [GeoPandas recipe](geopandas.md)), then re-import.

3. **Write a C++ QGIS plugin.** A native C++ Processing algorithm could
   access `QgsAbstractGeometry` coordinate arrays directly, memcpy to a
   pinned host buffer, and call vibeProj's CUDA kernels via the C ABI.
   This eliminates the Python boundary entirely but requires building
   against the QGIS C++ SDK.

## Recommendation: separate `vibeproj-qgis` repository

A proper QGIS plugin requires:

- `metadata.txt` with version, QGIS min/max version, description
- Plugin installer/uninstaller hooks
- GUI dialogs (optional, but expected for non-Processing plugins)
- Dependency bundling (vibeProj, NumPy, optionally CuPy)
- Testing against multiple QGIS versions (LTR + latest)

None of this belongs in the vibeProj core repository. Create a separate
`vibeproj-qgis` repository that depends on `vibeproj` as a pip
dependency and packages the Processing provider as a standard QGIS
plugin zip.

Suggested structure:

```
vibeproj-qgis/
    metadata.txt
    __init__.py                    # Plugin entry point
    provider.py                    # VibeProjectionProvider
    algorithms/
        batch_reproject.py         # The algorithm above, production-quality
        raster_reproject.py        # Coordinate grid transform for rasters
    tests/
        test_batch_reproject.py    # Uses qgis.testing fixtures
```

The plugin should detect CuPy at runtime and fall back to CPU
transparently. Users with a GPU get the speedup; users without still
get a working plugin backed by vibeProj's NumPy path.
