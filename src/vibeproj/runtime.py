"""GPU/CPU runtime detection and array module selection."""

from __future__ import annotations

import numpy as np


def get_array_module(x=None):
    """Return the array module (numpy or cupy) for the given array.

    If x is None or not a CuPy array, returns numpy.
    """
    if x is not None:
        xp = _array_module_for(x)
        if xp is not None:
            return xp
    return np


def _array_module_for(x):
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp
    except ImportError:
        pass
    return None


def gpu_available() -> bool:
    """Check if CuPy is available and a GPU is accessible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, RuntimeError, OSError):
        return False


def to_device(x, xp):
    """Ensure array x is on the device managed by xp."""
    return xp.asarray(x)
