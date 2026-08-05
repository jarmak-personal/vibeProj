# Precision and GPU Behaviour

## I/O precision

All input/output arrays use **fp64** (`double`) storage. This is a hard
convention (ADR-0002): coordinate data is always stored at full double
precision regardless of the compute precision used internally.

## Compute precision modes

The `transform_buffers()` method accepts a `precision` parameter:

| Mode | Compute type | Accuracy | Use case |
|---|---|---|---|
| `"fp64"` | `double` | Full | Default. Validated fp64-equivalent projection accuracy. |
| `"fp32"` | `float` | ~1m | Expert opt-in. Raw fp32 projection math. |
| `"ds"` | double-single | ~fp64 | Experimental. fp32 pair arithmetic. |
| `"auto"` | `double` | Full | Same arithmetic precision as fp64. |

```python
# Full precision (default)
t.transform_buffers(lon, lat, precision="fp64")

# Experimental double-single arithmetic (TM only)
t.transform_buffers(lon, lat, precision="ds")
```

## Automatic device strategies

The public `transcendentals="auto"|"native"|"accelerated"` policy is
independent of these compute-precision modes. See
[Transcendental policy](transcendentals.md) for resolution, fallback, hardware
qualification, and introspection.

Projection arithmetic and all coordinate I/O remain fp64 by default. Accurate
double transcendental functions expand to argument reduction and native
instruction sequences, which are expensive on consumer GPUs with weak fp64
throughput.

On validated Ada `sm_89` consumer GPUs, auto dispatch accelerates bounded
Helmert and forward UTM transcendentals with table-free Q1.62 trig. Forward UTM
also uses bounded fp64-accurate correction polynomials for
`atan2` and `asinh`. Inputs outside those domains fall back per-coordinate to
native fp64. Datacenter, unknown, and future GPUs conservatively use native
fp64 until independently benchmarked.

## Double-single arithmetic

The `"ds"` precision mode uses pairs of fp32 values to represent ~48-bit
mantissa (~14 decimal digits). This is implemented for Transverse Mercator
and gives fp64-equivalent accuracy using fp32 FMA instructions.

On consumer GPUs (RTX series, 1:64 fp64:fp32 ratio):
- `ds_add`: ~10x faster than fp64 add
- `ds_mul`: ~16x faster than fp64 mul

Current DS transcendental wrappers convert to double and call native fp64
functions. Their cost plus DS normalization means the DS TM kernel provides no
speedup in practice; the path remains available for experimentation.

## Consumer vs datacenter GPUs

vibeProj queries `SingleToDoublePrecisionPerfRatio` to classify the GPU:

- **Consumer** (RTX 4090, etc.): ratio = 1:64 for native fp64 arithmetic
- **Datacenter** (A100, H100): ratio = 1:2

Both types retain fp64 projection arithmetic. Validated consumer GPUs may use
the bounded internal strategies above; datacenter GPUs use native fp64.
