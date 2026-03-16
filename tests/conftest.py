"""Shared test fixtures and constants."""

# Grid corners shared across test_transformer and test_fused_kernels
GRID_CORNERS = [
    # (min_corner, max_corner, target_crs)
    ((37.7081, -122.5149), (37.8324, -122.3573), "EPSG:32610"),  # San Francisco
    ((-33.9, 151.2), (-33.7, 151.3), "EPSG:32756"),  # Sydney
    ((51.3, -0.5), (51.6, 0.3), "EPSG:32630"),  # London
    ((40.4774, -74.2591), (40.9176, -73.7004), "EPSG:32618"),  # NYC
    ((-54.9, -68.4), (-54.7, -68.1), "EPSG:32719"),  # Ushuaia
    ((1.2, 103.6), (1.5, 104.0), "EPSG:32648"),  # Singapore
]
