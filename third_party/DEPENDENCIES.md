# Third-Party Dependencies

vibeProj depends on the following third-party packages, installed via pip at
runtime. These are **not** bundled or vendored — license obligations for their
distribution are fulfilled by their respective package maintainers and PyPI.

This file is provided for license-audit convenience.

## Required

| Package | License | PyPI |
|---------|---------|------|
| NumPy >= 2 | BSD-3-Clause (primary) | https://pypi.org/project/numpy/ |
| pyproj >= 3.7 | MIT | https://pypi.org/project/pyproj/ |

pyproj bundles the PROJ C library (MIT license).

## Optional — GPU (`pip install vibeproj[cu12]` or `vibeproj[cu13]`)

| Package | License | PyPI |
|---------|---------|------|
| CuPy | MIT | https://pypi.org/project/cupy-cuda12x/ |
| cuda-python | Mixed (see below) | https://pypi.org/project/cuda-python/ |
| cuda-cccl | Apache-2.0 | https://pypi.org/project/cuda-cccl/ |

**Note on cuda-python licensing:** The
[cuda-python repository](https://github.com/NVIDIA/cuda-python) is a monorepo
with per-package licensing:

| Sub-package | License |
|-------------|---------|
| `cuda.core` | Apache-2.0 |
| `cuda.pathfinder` | Apache-2.0 |
| `cuda.bindings` | NVIDIA Software License |
| `cuda.python` | NVIDIA Software License |

The NVIDIA Software License is not an OSI-approved open-source license.
Users install these packages themselves as optional dependencies; vibeProj does
not redistribute them.
