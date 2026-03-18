"""vibeProj — GPU-accelerated coordinate projection library."""

__version__ = "0.1.2"

from vibeproj.transformer import Transformer


def list_projections() -> dict[str, dict]:
    """Return supported projections and their metadata.

    Returns
    -------
    dict[str, dict]
        Keys are internal projection names. Each value has:
        - "methods": list of pyproj method names that map to this projection
        - "fused": True if a GPU-accelerated fused kernel is available
    """
    from vibeproj.crs import _METHOD_MAP
    from vibeproj.fused_kernels import _SUPPORTED
    from vibeproj.projections import PROJECTION_REGISTRY

    inverse_map: dict[str, list[str]] = {}
    for method, name in _METHOD_MAP.items():
        inverse_map.setdefault(name, []).append(method)

    result = {}
    for name in sorted(PROJECTION_REGISTRY):
        result[name] = {
            "methods": sorted(inverse_map.get(name, [])),
            "fused": (name, "forward") in _SUPPORTED,
        }
    return result


def warm_up(projections=None, *, precision="auto"):
    """Pre-compile fused NVRTC kernels to eliminate first-call latency.

    Parameters
    ----------
    projections : list of str, optional
        Projection names to compile (e.g. ["tmerc", "webmerc"]).
        If None, compiles all supported projections.
    precision : str
        Compute precision: "auto"/"fp64"/"fp32"/"ds".

    Examples
    --------
    >>> import vibeproj
    >>> vibeproj.warm_up(["tmerc", "webmerc"])  # selective
    >>> vibeproj.warm_up()                       # all projections
    """
    from vibeproj.fused_kernels import compile_kernels

    compile_kernels(projections, precision=precision)


__all__ = ["Transformer", "list_projections", "warm_up", "__version__"]
