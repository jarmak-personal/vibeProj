# Precision and GPU Behaviour

## I/O precision

All input/output arrays use **fp64** (`double`) storage. This is a hard
convention (ADR-0002): coordinate data is always stored at full double
precision regardless of the compute precision used internally.

## Compute precision modes

The `transform_buffers()` method accepts a `precision` parameter:

| Mode | Compute type | Accuracy | Use case |
|---|---|---|---|
| `"fp64"` | `double` | Full | Default. Exact to machine epsilon. |
| `"fp32"` | `float` | ~1m | Expert opt-in. Raw fp32 projection math. |
| `"ds"` | double-single | ~fp64 | Experimental. fp32 pair arithmetic. |
| `"auto"` | `double` | Full | Same as fp64 (trig-dominated). |

```python
# Full precision (default)
t.transform_buffers(lat, lon, precision="fp64")

# Experimental double-single arithmetic (TM only)
t.transform_buffers(lat, lon, precision="ds")
```

## Why auto always uses fp64

Projection math is dominated by transcendental functions (`sin`, `cos`,
`atan2`, `asinh`) which use the GPU's Special Function Unit (SFU). The
SFU fp64:fp32 throughput ratio is ~1:4, not 1:64 like ALU operations.

This means fp32 compute gives only ~4x speedup for projection kernels,
not the theoretical 32x. Since the GPU is already 100--300x faster than
CPU at fp64, the accuracy trade-off is not worthwhile. Auto mode therefore
always selects fp64.

## Double-single arithmetic

The `"ds"` precision mode uses pairs of fp32 values to represent ~48-bit
mantissa (~14 decimal digits). This is implemented for Transverse Mercator
and gives fp64-equivalent accuracy using fp32 FMA instructions.

On consumer GPUs (RTX series, 1:64 fp64:fp32 ratio):
- `ds_add`: ~10x faster than fp64 add
- `ds_mul`: ~16x faster than fp64 mul

In practice, the SFU bottleneck means ds provides no speedup for
trig-heavy projection kernels. The ds path exists for experimentation.

## Consumer vs datacenter GPUs

vibeProj queries `SingleToDoublePrecisionPerfRatio` to classify the GPU:

- **Consumer** (RTX 4090, etc.): ratio = 1:64 for ALU, but ~1:4 for SFU
- **Datacenter** (A100, H100): ratio = 1:2

Both types run fp64 by default. Datacenter GPUs will see higher absolute
throughput due to their better fp64 hardware.
