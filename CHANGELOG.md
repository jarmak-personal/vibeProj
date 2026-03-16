# Changelog

## 0.1.0 — 2026-03-15

Initial release.

- 20 coordinate projections with fused NVRTC GPU kernels
- pyproj-compatible `Transformer` API (`from_crs`, `transform`)
- Zero-copy `transform_buffers()` API for vibeSpatial integration
- NumPy fallback for all projections (CPU path)
- Double-single (ds) experimental precision mode for Transverse Mercator
- Automatic GPU detection and precision selection
