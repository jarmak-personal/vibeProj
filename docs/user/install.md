# Installation

## Requirements

- Python 3.12+
- NumPy 2+
- pyproj 3.7+

For GPU acceleration:

- NVIDIA GPU with CUDA 12.x
- CuPy 13+

## Install with uv

```bash
# CPU-only (NumPy fallback)
uv sync

# With GPU support
uv sync --group gpu
```

## Install with pip

```bash
# CPU-only
pip install vibeproj

# With CuPy for GPU acceleration
pip install vibeproj cupy-cuda12x
```

## Verifying GPU support

```python
from vibeproj.runtime import gpu_available

print(gpu_available())  # True if CuPy + GPU detected
```

When CuPy is not installed or no GPU is available, vibeProj automatically
falls back to NumPy. No code changes required.
