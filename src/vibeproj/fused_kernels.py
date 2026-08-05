"""Fused NVRTC kernels for GPU-accelerated coordinate projection.

Each kernel runs the full transform pipeline (axis swap, deg/rad, central meridian,
projection math, scale, offset) in a single kernel launch — one thread per coordinate.
This eliminates ~20 intermediate kernel launches and array round-trips compared to
the CuPy element-wise path.

Uses CuPy RawKernel for NVRTC compilation and caching.
"""

from __future__ import annotations

from collections.abc import Iterable
import threading
import warnings

import numpy as np

from vibeproj.transcendentals import (
    HELMERT_FIXED_Q62,
    NATIVE_LIBDEVICE,
    ORTHO_FORWARD_FIXED_Q62,
    SINU_FORWARD_FIXED_Q62,
    TMERC_FIXED_Q62,
)

GEOS_FP32_DISCRIMINANT_TOLERANCE = 4e-8
GEOS_FP64_DISCRIMINANT_TOLERANCE = 2e-15
GEOS_SCAN_ANGLE_LIMIT = np.pi / 2.0

_GEOS_DEVICE_NUMERIC_CONTRACT = (
    f"#define VP_GEOS_FP32_DISCRIMINANT_TOLERANCE "
    f"{GEOS_FP32_DISCRIMINANT_TOLERANCE:.17g}f\n"
    f"#define VP_GEOS_FP64_DISCRIMINANT_TOLERANCE "
    f"{GEOS_FP64_DISCRIMINANT_TOLERANCE:.17g}\n"
    f"#define VP_GEOS_SCAN_ANGLE_LIMIT {GEOS_SCAN_ANGLE_LIMIT:.17g}\n"
)

# Kernel cache: (projection, direction, dtype, implementation_id) -> RawKernel
# Protected by _kernel_cache_lock for thread-safe compilation.
_kernel_cache: dict[tuple[str, str, str, str], object] = {}
_kernel_cache_lock = threading.RLock()

# Projections with fused kernel support
_SUPPORTED = {
    ("tmerc", "forward"),
    ("tmerc", "inverse"),
    ("webmerc", "forward"),
    ("webmerc", "inverse"),
    ("merc", "forward"),
    ("merc", "inverse"),
    ("lcc", "forward"),
    ("lcc", "inverse"),
    ("aea", "forward"),
    ("aea", "inverse"),
    ("stere", "forward"),
    ("stere", "inverse"),
    ("laea", "forward"),
    ("laea", "inverse"),
    ("eqc", "forward"),
    ("eqc", "inverse"),
    ("sinu", "forward"),
    ("sinu", "inverse"),
    ("eqearth", "forward"),
    ("eqearth", "inverse"),
    ("cea", "forward"),
    ("cea", "inverse"),
    ("ortho", "forward"),
    ("ortho", "inverse"),
    ("gnom", "forward"),
    ("gnom", "inverse"),
    ("moll", "forward"),
    ("moll", "inverse"),
    ("omerc", "forward"),
    ("omerc", "inverse"),
    ("krovak", "forward"),
    ("krovak", "inverse"),
    ("eck4", "forward"),
    ("eck4", "inverse"),
    ("eck6", "forward"),
    ("eck6", "inverse"),
    ("sterea", "forward"),
    ("sterea", "inverse"),
    ("geos", "forward"),
    ("geos", "inverse"),
    ("robin", "forward"),
    ("robin", "inverse"),
    ("wintri", "forward"),
    ("wintri", "inverse"),
    ("natearth", "forward"),
    ("natearth", "inverse"),
    ("aeqd", "forward"),
    ("aeqd", "inverse"),
}


def can_fuse(projection_name: str, direction: str) -> bool:
    """Check if a fused kernel is available for this projection + direction."""
    return (projection_name, direction) in _SUPPORTED


# ===================================================================
# Double-single (ds) arithmetic CUDA functions (for consumer GPUs)
# ===================================================================

from vibeproj._ds_device_fns import DS_DEVICE_FNS as _DS_ARITH  # noqa: E402
from vibeproj._transcendental_device_fns import (  # noqa: E402
    HELMERT_FIXED_Q62_DEVICE_FNS as _HELMERT_FIXED_Q62_DEVICE_FNS,
    NATIVE_PAIRED_SINCOS_DEVICE_FNS as _NATIVE_PAIRED_SINCOS_DEVICE_FNS,
    PROJECTION_BOUNDED_Q62_DEVICE_FNS as _PROJECTION_BOUNDED_Q62_DEVICE_FNS,
    TM_UTM_QUALIFIED_DEVICE_FNS as _TM_UTM_QUALIFIED_DEVICE_FNS,
)

# ds gatg + ds clenshaw_complex for TM
_DS_TM_DEVICE_FNS = (
    _DS_ARITH
    + """
__device__ __forceinline__ void ds_sincos(ds_t angle, ds_t* sin_out, ds_t* cos_out) {{
    double sin_value, cos_value;
    sincos(ds_to_double(angle), &sin_value, &cos_value);
    *sin_out = ds_from_double(sin_value);
    *cos_out = ds_from_double(cos_value);
}}

// ds_gatg: Clenshaw-type series evaluation in ds arithmetic
__device__ inline ds_t ds_gatg(
    ds_t p0, ds_t p1, ds_t p2, ds_t p3, ds_t p4, ds_t p5,
    ds_t B, ds_t cos_2B, ds_t sin_2B
) {{
    ds_t two_cos_2B = ds_mul(ds_from_float(2.0f), cos_2B);
    ds_t h2 = ds_from_float(0.0f);
    ds_t h1 = p5;
    ds_t h;
    h = ds_add(ds_add(ds_sub(ds_from_float(0.0f), h2), ds_mul(two_cos_2B, h1)), p4); h2 = h1; h1 = h;
    h = ds_add(ds_add(ds_sub(ds_from_float(0.0f), h2), ds_mul(two_cos_2B, h1)), p3); h2 = h1; h1 = h;
    h = ds_add(ds_add(ds_sub(ds_from_float(0.0f), h2), ds_mul(two_cos_2B, h1)), p2); h2 = h1; h1 = h;
    h = ds_add(ds_add(ds_sub(ds_from_float(0.0f), h2), ds_mul(two_cos_2B, h1)), p1); h2 = h1; h1 = h;
    h = ds_add(ds_add(ds_sub(ds_from_float(0.0f), h2), ds_mul(two_cos_2B, h1)), p0); h2 = h1; h1 = h;
    return ds_add(B, ds_mul(h, sin_2B));
}}

// ds_clenshaw_complex: complex Clenshaw summation in ds
__device__ inline void ds_clenshaw_complex(
    ds_t a0, ds_t a1, ds_t a2, ds_t a3, ds_t a4, ds_t a5,
    ds_t sin_r, ds_t cos_r, ds_t sinh_i, ds_t cosh_i,
    ds_t* out_R, ds_t* out_I
) {{
    ds_t r = ds_mul(ds_mul(ds_from_float(2.0f), cos_r), cosh_i);
    ds_t im; im.hi = -2.0f; im.lo = 0.0f;
    im = ds_mul(ds_mul(im, sin_r), sinh_i);
    ds_t hr = a5, hi = ds_from_float(0.0f);
    ds_t hr1 = ds_from_float(0.0f), hi1 = ds_from_float(0.0f);
    ds_t hr2, hi2;
    #define DS_CLEN_STEP(coeff) \\
        hr2=hr1; hi2=hi1; hr1=hr; hi1=hi; \\
        hr=ds_add(ds_add(ds_sub(ds_from_float(0.0f),hr2), ds_sub(ds_mul(r,hr1), ds_mul(im,hi1))), coeff); \\
        hi=ds_add(ds_sub(ds_from_float(0.0f),hi2), ds_add(ds_mul(im,hr1), ds_mul(r,hi1)));
    DS_CLEN_STEP(a4)
    DS_CLEN_STEP(a3)
    DS_CLEN_STEP(a2)
    DS_CLEN_STEP(a1)
    DS_CLEN_STEP(a0)
    #undef DS_CLEN_STEP
    r = ds_mul(sin_r, cosh_i);
    im = ds_mul(cos_r, sinh_i);
    *out_R = ds_sub(ds_mul(r, hr), ds_mul(im, hi));
    *out_I = ds_add(ds_mul(r, hi), ds_mul(im, hr));
}}
"""
)

# ===================================================================
# Shared CUDA device function blocks
# ===================================================================

# -- Transverse Mercator helpers (gatg, clenshaw_complex) --
_TM_DEVICE_FNS = """
__device__ inline {real_t} gatg(
    {real_t} p0, {real_t} p1, {real_t} p2, {real_t} p3, {real_t} p4, {real_t} p5,
    {real_t} B, {real_t} cos_2B, {real_t} sin_2B
) {{
    {real_t} two_cos_2B = ({real_t})2.0 * cos_2B;
    {real_t} h2 = ({real_t})0.0, h1 = p5, h;
    h = -h2 + two_cos_2B * h1 + p4; h2 = h1; h1 = h;
    h = -h2 + two_cos_2B * h1 + p3; h2 = h1; h1 = h;
    h = -h2 + two_cos_2B * h1 + p2; h2 = h1; h1 = h;
    h = -h2 + two_cos_2B * h1 + p1; h2 = h1; h1 = h;
    h = -h2 + two_cos_2B * h1 + p0; h2 = h1; h1 = h;
    return B + h * sin_2B;
}}
__device__ inline void clenshaw_complex(
    {real_t} a0, {real_t} a1, {real_t} a2, {real_t} a3, {real_t} a4, {real_t} a5,
    {real_t} sin_r, {real_t} cos_r, {real_t} sinh_i, {real_t} cosh_i,
    {real_t}* out_R, {real_t}* out_I
) {{
    {real_t} r = ({real_t})2.0 * cos_r * cosh_i;
    {real_t} im = ({real_t})-2.0 * sin_r * sinh_i;
    {real_t} hr = a5, hi = ({real_t})0.0, hr1 = ({real_t})0.0, hi1 = ({real_t})0.0, hr2, hi2;
    hr2=hr1; hi2=hi1; hr1=hr; hi1=hi; hr=-hr2+r*hr1-im*hi1+a4; hi=-hi2+im*hr1+r*hi1;
    hr2=hr1; hi2=hi1; hr1=hr; hi1=hi; hr=-hr2+r*hr1-im*hi1+a3; hi=-hi2+im*hr1+r*hi1;
    hr2=hr1; hi2=hi1; hr1=hr; hi1=hi; hr=-hr2+r*hr1-im*hi1+a2; hi=-hi2+im*hr1+r*hi1;
    hr2=hr1; hi2=hi1; hr1=hr; hi1=hi; hr=-hr2+r*hr1-im*hi1+a1; hi=-hi2+im*hr1+r*hi1;
    hr2=hr1; hi2=hi1; hr1=hr; hi1=hi; hr=-hr2+r*hr1-im*hi1+a0; hi=-hi2+im*hr1+r*hi1;
    r = sin_r * cosh_i; im = cos_r * sinh_i;
    *out_R = r * hr - im * hi;
    *out_I = r * hi + im * hr;
}}
"""

# -- Conic/Stereographic helpers (tsfn, phi2) --
_CONIC_DEVICE_FNS = """
#define HALF_PI (({real_t})0.5 * {pi})
__device__ inline {real_t} tsfn({real_t} phi, {real_t} sin_phi, {real_t} e) {{
    {real_t} esp = e * sin_phi;
    return tan(({real_t})0.5 * (HALF_PI - phi)) / pow(((({real_t})1.0 - esp) / (({real_t})1.0 + esp)), ({real_t})0.5 * e);
}}
__device__ inline {real_t} phi2({real_t} ts, {real_t} e) {{
    {real_t} half_e = ({real_t})0.5 * e;
    {real_t} phi = HALF_PI - ({real_t})2.0 * atan(ts);
    for (int i = 0; i < 15; i++) {{
        {real_t} e_sin = e * sin(phi);
        {real_t} dphi = HALF_PI - ({real_t})2.0 * atan(ts * pow((({real_t})1.0 - e_sin) / (({real_t})1.0 + e_sin), half_e)) - phi;
        phi += dphi;
        if (fabs(dphi) < {tol}) break;
    }}
    return phi;
}}
"""

# -- Equal-area helpers (qsfn, phi_from_q) --
_EA_DEVICE_FNS = """
#define VP_EA_Q_POLE_SNAP_D 1e-10
__device__ inline {real_t} vp_ea_clamp_unit_preserve_nan({real_t} value) {{
    return isnan(value)
        ? value
        : fmin(fmax(value, ({real_t})-1.0), ({real_t})1.0);
}}
__device__ inline {real_t} qsfn({real_t} sin_phi, {real_t} e) {{
    if (e == ({real_t})0.0) return ({real_t})2.0 * sin_phi;
    {real_t} e_sin = e * sin_phi;
    {real_t} one_minus_e2 = ({real_t})1.0 - e * e;
    return one_minus_e2 * (sin_phi / (({real_t})1.0 - e_sin * e_sin)
           + atanh(e_sin) / e);
}}
__device__ inline {real_t} phi_from_q(
    {real_t} q, {real_t} e, {real_t} es, {real_t} qp
) {{
    if (!isfinite(q)) return q;
    if (q >= qp) return ({real_t})0.5 * {pi};
    if (q <= -qp) return ({real_t})-0.5 * {pi};
    // Solve in s=sin(phi), where dq/ds remains finite at both poles.
    // Newton directly in phi loses convergence as cos(phi) approaches zero.
    {real_t} s = vp_ea_clamp_unit_preserve_nan(q / qp);
    for (int i = 0; i < 15; i++) {{
        {real_t} one_minus = ({real_t})1.0 - es * s * s;
        {real_t} ds = (qsfn(s, e) - q) * one_minus * one_minus
            / (({real_t})2.0 * (({real_t})1.0 - es));
        s = vp_ea_clamp_unit_preserve_nan(s - ds);
        if (fabs(ds) < {tol}) break;
    }}
    return asin(s);
}}
"""

# ===================================================================
# Forward/inverse preamble/postamble macros (reduce repetition)
# ===================================================================

# Mixed-precision design (ADR-0002 compliant):
# - I/O arrays are always double* (fp64 storage, canonical precision)
# - Compute type is {real_t} (float for fp32 mode, double for fp64 mode)
# - Forward postamble: scale/offset in double to preserve sub-meter precision
# - Inverse preamble: offset/scale removal in double before cast to compute_t
# This means fp32 mode reads fp64, computes trig/series in fp32 (32x throughput
# on consumer GPUs), then does final scale+offset in fp64 for output.

_FWD_PREAMBLE = """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double d_arg1 = in_x[idx], d_arg2 = in_y[idx];
    double d_lat, d_lon;
    if (src_north_first) {{ d_lat = d_arg1; d_lon = d_arg2; }} else {{ d_lon = d_arg1; d_lat = d_arg2; }}
    {real_t} phi = ({real_t})(d_lat * 0.017453292519943295);
    {real_t} lam = ({real_t})(d_lon * 0.017453292519943295 - (double)lam0);
    const {real_t} TWO_PI = ({real_t})6.283185307179586;
    if (isfinite(lam)) lam = lam - TWO_PI * rint(lam / TWO_PI);
"""

_FWD_POSTAMBLE = """
    double d_easting_out = (double)easting / x_unit_to_m;
    double d_northing_out = (double)northing / y_unit_to_m;
    if (dst_north_first) {{ out_x[idx] = d_northing_out; out_y[idx] = d_easting_out; }}
    else                 {{ out_x[idx] = d_easting_out;  out_y[idx] = d_northing_out; }}
"""

_INV_PREAMBLE = """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double d_arg1 = in_x[idx], d_arg2 = in_y[idx];
    double d_northing_in, d_easting_in;
    if (src_north_first) {{ d_northing_in = d_arg1; d_easting_in = d_arg2; }} else {{ d_easting_in = d_arg1; d_northing_in = d_arg2; }}
    double d_easting = d_easting_in * x_unit_to_m;
    double d_northing = d_northing_in * y_unit_to_m;
    // Offset/scale removal in fp64 before cast to compute precision
    {real_t} cx = ({real_t})((d_easting - (double)x0) / (double)a);
    {real_t} cy = ({real_t})((d_northing - (double)y0) / (double)a);
"""

_INV_POSTAMBLE = """
    if (isfinite(lam)) {{
        lam = lam + ({real_t})lam0;
        const {real_t} TWO_PI_i = ({real_t})6.283185307179586;
        lam = lam - TWO_PI_i * rint(lam / TWO_PI_i);
    }}
    double d_lat = (double)phi * 57.29577951308232;
    double d_lon = (double)lam * 57.29577951308232;
    if (dst_north_first) {{ out_x[idx] = d_lat; out_y[idx] = d_lon; }}
    else                 {{ out_x[idx] = d_lon; out_y[idx] = d_lat; }}
"""

_FWD_SIGNATURE = """
extern "C" __global__ void __launch_bounds__(256) {func}(
    const double* __restrict__ in_x, const double* __restrict__ in_y,
    double* __restrict__ out_x, double* __restrict__ out_y,
"""

_INV_SIGNATURE = _FWD_SIGNATURE  # same

_KERNEL_UNIT_SIGNATURE_NEEDLE = "int src_north_first, int dst_north_first, int n"
_KERNEL_UNIT_SIGNATURE_REPLACEMENT = (
    "double x_unit_to_m, double y_unit_to_m,\n    int src_north_first, int dst_north_first, int n"
)


def _inject_linear_unit_args(source: str) -> str:
    """Inject projected-unit ABI params into compiled kernel source."""
    return source.replace(_KERNEL_UNIT_SIGNATURE_NEEDLE, _KERNEL_UNIT_SIGNATURE_REPLACEMENT)


# ===================================================================
# Plate Carrée kernels
# ===================================================================

_EQC_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="eqc_forward", real_t="{real_t}")
    + """
    {real_t} cos_lat_ts, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    double easting  = (double)(lam * cos_lat_ts) * (double)a + (double)x0;
    double northing = (double)phi * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_EQC_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="eqc_inverse", real_t="{real_t}")
    + """
    {real_t} cos_lat_ts, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} lam = cx / cos_lat_ts;
    {real_t} phi = cy;
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Sinusoidal kernels
# ===================================================================

_SINU_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="sinu_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    double easting  = (double)(lam * cos(phi)) * (double)a + (double)x0;
    double northing = (double)phi * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_SINU_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="sinu_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} phi = cy;
    {real_t} lam = cx / cos(phi);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Mercator (ellipsoidal) kernels
# ===================================================================

_MERC_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="merc_forward", real_t="{real_t}")
    + """
    {real_t} e, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} e_sin_phi = e * sin(phi);
    {real_t} y_proj = log(tan(({real_t})0.25 * {pi} + ({real_t})0.5 * phi)
                     * pow((({real_t})1.0 - e_sin_phi) / (({real_t})1.0 + e_sin_phi), ({real_t})0.5 * e));
    double easting  = (double)lam * (double)a + (double)x0;
    double northing = (double)y_proj * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_MERC_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="merc_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} lam = cx;
    {real_t} phi = ({real_t})2.0 * atan(exp(cy)) - ({real_t})0.5 * {pi};
    for (int i = 0; i < 15; i++) {{
        {real_t} e_sin = e * sin(phi);
        {real_t} phi_new = ({real_t})2.0 * atan(exp(cy) * pow((({real_t})1.0 + e_sin) / (({real_t})1.0 - e_sin), ({real_t})0.5 * e)) - ({real_t})0.5 * {pi};
        if (fabs(phi_new - phi) < {tol}) {{ phi = phi_new; break; }}
        phi = phi_new;
    }}
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Web Mercator kernels
# ===================================================================

_WEBMERC_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="webmerc_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    double easting  = (double)lam * (double)a + (double)x0;
    double northing = (double)log(tan(({real_t})0.25 * {pi} + ({real_t})0.5 * phi)) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_WEBMERC_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="webmerc_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} lam = cx;
    {real_t} phi = ({real_t})2.0 * atan(exp(cy)) - ({real_t})0.5 * {pi};
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Transverse Mercator kernels
# ===================================================================

_TM_FORWARD_SOURCE = (
    _TM_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="tm_forward", real_t="{real_t}")
    + """
    {real_t} cbg0, {real_t} cbg1, {real_t} cbg2, {real_t} cbg3, {real_t} cbg4, {real_t} cbg5,
    {real_t} gtu0, {real_t} gtu1, {real_t} gtu2, {real_t} gtu3, {real_t} gtu4, {real_t} gtu5,
    {real_t} Qn, {real_t} Zb, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_two_phi, cos_two_phi;
    vp_native_sincos(({real_t})2.0 * phi, &sin_two_phi, &cos_two_phi);
    {real_t} Cn = gatg(cbg0, cbg1, cbg2, cbg3, cbg4, cbg5,
                       phi, cos_two_phi, sin_two_phi);
    {real_t} sin_Cn, cos_Cn;
    vp_native_sincos(Cn, &sin_Cn, &cos_Cn);
    {real_t} sin_Ce, cos_Ce;
    vp_native_sincos(lam, &sin_Ce, &cos_Ce);
    {real_t} cos_Cn_cos_Ce = cos_Cn * cos_Ce;
    Cn = atan2(sin_Cn, cos_Cn_cos_Ce);
    {real_t} inv_denom = rsqrt(sin_Cn * sin_Cn + cos_Cn_cos_Ce * cos_Cn_cos_Ce);
    {real_t} tan_Ce = sin_Ce * cos_Cn * inv_denom;
    {real_t} Ce = asinh(tan_Ce);
    {real_t} two_inv = ({real_t})2.0 * inv_denom;
    {real_t} two_inv_sq = two_inv * inv_denom;
    {real_t} tmp_r = cos_Cn_cos_Ce * two_inv_sq;
    {real_t} sin_arg_r = sin_Cn * tmp_r;
    {real_t} cos_arg_r = cos_Cn_cos_Ce * tmp_r - ({real_t})1.0;
    {real_t} sinh_arg_i = tan_Ce * two_inv;
    {real_t} cosh_arg_i = two_inv_sq - ({real_t})1.0;
    {real_t} dCn, dCe;
    clenshaw_complex(gtu0, gtu1, gtu2, gtu3, gtu4, gtu5,
                     sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);
    Cn += dCn; Ce += dCe;
    double easting  = (double)(Qn * Ce) * (double)a + (double)x0;
    double northing = (double)(Qn * Cn + Zb) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

# Dedicated fp64 TM forward source. Native sine/cosine is paired on every GPU;
# validated Ada consumer GPUs can instead use bounded Q1.62 trig and replace
# general atan2/asinh with UTM-domain correction polynomials. All other TM
# arithmetic and kernel I/O remain fp64.
_TM_FORWARD_FP64_BODY = (
    r"""
#ifndef VP_TM_FORWARD_KERNEL_NAME
#define VP_TM_FORWARD_KERNEL_NAME tm_forward_fp64
#endif
#ifndef VP_TM_SINCOS
#define VP_TM_SINCOS(angle, lam, sin_out, cos_out) \
    sincos(angle, sin_out, cos_out)
#endif
#ifndef VP_TM_LATITUDE
#define VP_TM_LATITUDE(gaussian_lat, sin_lat, cos_lat, cos_lon, product, lam) \
    atan2(sin_lat, product)
#endif
#ifndef VP_TM_ASINH
#define VP_TM_ASINH(value) asinh(value)
#endif
"""
    + _FWD_SIGNATURE.format(func="VP_TM_FORWARD_KERNEL_NAME", real_t="double")
    + """
    double cbg0, double cbg1, double cbg2, double cbg3, double cbg4, double cbg5,
    double gtu0, double gtu1, double gtu2, double gtu3, double gtu4, double gtu5,
    double Qn, double Zb, double lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {"""
    + _FWD_PREAMBLE.format(real_t="double")
    + r"""
    double sin_two_phi, cos_two_phi;
    VP_TM_SINCOS(2.0 * phi, lam, &sin_two_phi, &cos_two_phi);
    double Cn = gatg(
        cbg0, cbg1, cbg2, cbg3, cbg4, cbg5,
        phi, cos_two_phi, sin_two_phi
    );
    const double gaussian_lat = Cn;
    double sin_Cn, cos_Cn;
    VP_TM_SINCOS(Cn, lam, &sin_Cn, &cos_Cn);
    double sin_Ce, cos_Ce;
    VP_TM_SINCOS(lam, lam, &sin_Ce, &cos_Ce);
    double cos_Cn_cos_Ce = cos_Cn * cos_Ce;
    Cn = VP_TM_LATITUDE(
        gaussian_lat, sin_Cn, cos_Cn, cos_Ce, cos_Cn_cos_Ce, lam
    );
    double inv_denom = rsqrt(
        sin_Cn * sin_Cn + cos_Cn_cos_Ce * cos_Cn_cos_Ce
    );
    double tan_Ce = sin_Ce * cos_Cn * inv_denom;
    double Ce = VP_TM_ASINH(tan_Ce);
    double two_inv = 2.0 * inv_denom;
    double two_inv_sq = two_inv * inv_denom;
    double tmp_r = cos_Cn_cos_Ce * two_inv_sq;
    double sin_arg_r = sin_Cn * tmp_r;
    double cos_arg_r = cos_Cn_cos_Ce * tmp_r - 1.0;
    double sinh_arg_i = tan_Ce * two_inv;
    double cosh_arg_i = two_inv_sq - 1.0;
    double dCn, dCe;
    clenshaw_complex(
        gtu0, gtu1, gtu2, gtu3, gtu4, gtu5,
        sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe
    );
    Cn += dCn;
    Ce += dCe;
    double easting = Qn * Ce * a + x0;
    double northing = (Qn * Cn + Zb) * a + y0;
"""
    + _FWD_POSTAMBLE.format()
    + "}"
)

_TM_FORWARD_FP64_SOURCE = _TM_DEVICE_FNS.format(real_t="double") + _TM_FORWARD_FP64_BODY

_TM_FORWARD_FIXED_Q62_SOURCE = (
    _TM_UTM_QUALIFIED_DEVICE_FNS
    + _TM_DEVICE_FNS.format(real_t="double")
    + r"""
#define VP_TM_FORWARD_KERNEL_NAME tm_forward_fixed_q62
#define VP_TM_SINCOS(angle, lam, sin_out, cos_out) \
    vp_tm_utm_sincos(angle, lam, sin_out, cos_out)
#define VP_TM_LATITUDE(gaussian_lat, sin_lat, cos_lat, cos_lon, product, lam) \
    vp_tm_utm_latitude(gaussian_lat, sin_lat, cos_lat, cos_lon, product, lam)
#define VP_TM_ASINH(value) vp_tm_utm_asinh(value)
"""
    + _TM_FORWARD_FP64_BODY
)

_TM_INVERSE_SOURCE = (
    _TM_DEVICE_FNS
    + _INV_SIGNATURE.format(func="tm_inverse", real_t="{real_t}")
    + """
    {real_t} cgb0, {real_t} cgb1, {real_t} cgb2, {real_t} cgb3, {real_t} cgb4, {real_t} cgb5,
    {real_t} utg0, {real_t} utg1, {real_t} utg2, {real_t} utg3, {real_t} utg4, {real_t} utg5,
    {real_t} Qn, {real_t} Zb, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} Cn = (cy - Zb) / Qn, Ce = cx / Qn;
    {real_t} sin_arg_r, cos_arg_r;
    vp_native_sincos(({real_t})2.0 * Cn, &sin_arg_r, &cos_arg_r);
    {real_t} exp_2_Ce = exp(({real_t})2.0 * Ce);
    {real_t} half_inv = ({real_t})0.5 / exp_2_Ce;
    {real_t} sinh_arg_i = ({real_t})0.5 * exp_2_Ce - half_inv;
    {real_t} cosh_arg_i = ({real_t})0.5 * exp_2_Ce + half_inv;
    {real_t} dCn, dCe;
    clenshaw_complex(utg0, utg1, utg2, utg3, utg4, utg5,
                     sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);
    Cn += dCn; Ce += dCe;
    {real_t} sin_Cn, cos_Cn;
    vp_native_sincos(Cn, &sin_Cn, &cos_Cn);
    {real_t} sinhCe = sinh(Ce);
    Ce = atan2(sinhCe, cos_Cn);
    {real_t} modulus_Ce = hypot(sinhCe, cos_Cn);
    Cn = atan2(sin_Cn, modulus_Ce);
    {real_t} tmp = ({real_t})2.0 * modulus_Ce / (sinhCe * sinhCe + ({real_t})1.0);
    {real_t} phi = gatg(cgb0, cgb1, cgb2, cgb3, cgb4, cgb5, Cn, tmp * modulus_Ce - ({real_t})1.0, sin_Cn * tmp);
    {real_t} lam = Ce;
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Lambert Conformal Conic kernels
# ===================================================================

_LCC_FORWARD_SOURCE = (
    _CONIC_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="lcc_forward", real_t="{real_t}")
    + """
    {real_t} nn, {real_t} F, {real_t} rho0, {real_t} e, {real_t} k0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_phi = sin(phi);
    {real_t} ts = tsfn(phi, sin_phi, e);
    {real_t} rho = F * pow(ts, nn) * k0;
    {real_t} theta = nn * lam;
    {real_t} sin_theta, cos_theta;
    vp_native_sincos(theta, &sin_theta, &cos_theta);
    double easting  = (double)(rho * sin_theta) * (double)a + (double)x0;
    double northing = (double)(rho0 - rho * cos_theta) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_LCC_INVERSE_SOURCE = (
    _CONIC_DEVICE_FNS
    + _INV_SIGNATURE.format(func="lcc_inverse", real_t="{real_t}")
    + """
    {real_t} nn, {real_t} F, {real_t} rho0, {real_t} e, {real_t} k0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} dy = rho0 - cy;
    {real_t} rho = sqrt(cx * cx + dy * dy);
    if (nn < ({real_t})0.0) {{ rho = -rho; cx = -cx; dy = -dy; }}
    {real_t} lam = atan2(cx, dy) / nn;
    {real_t} ts = pow(rho / (F * k0), ({real_t})1.0 / nn);
    {real_t} phi = phi2(ts, e);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Polar Stereographic kernels
# ===================================================================

_STERE_FORWARD_SOURCE = (
    _CONIC_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="stere_forward", real_t="{real_t}")
    + """
    {real_t} akm1, {real_t} sign, {real_t} e,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} phi_adj = sign * phi;
    {real_t} sin_phi_adj = sin(phi_adj);
    {real_t} t = tsfn(phi_adj, sin_phi_adj, e);
    {real_t} rho = akm1 * t;
    {real_t} sin_lam, cos_lam;
    vp_native_sincos(lam, &sin_lam, &cos_lam);
    double easting  = (double)(rho * sin_lam) * (double)a + (double)x0;
    double northing = (double)(-sign * rho * cos_lam) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_STERE_INVERSE_SOURCE = (
    _CONIC_DEVICE_FNS
    + _INV_SIGNATURE.format(func="stere_inverse", real_t="{real_t}")
    + """
    {real_t} akm1, {real_t} sign, {real_t} e,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} y_adj = -sign * cy;
    {real_t} rho = sqrt(cx * cx + y_adj * y_adj);
    {real_t} ts = rho / akm1;
    {real_t} phi = sign * phi2(ts, e);
    {real_t} lam = atan2(cx, y_adj);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Albers Equal Area kernels
# ===================================================================

_AEA_FORWARD_SOURCE = (
    _EA_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="aea_forward", real_t="{real_t}")
    + """
    double d_nn, double d_C, double d_rho0,
    {real_t} e, {real_t} es, double d_qp,
    double lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    double easting, northing;
    if (fabs(d_lat) == 90.0 && isfinite(d_lon)) {{
        // Preserve an exact geographic pole through fp32 forward -> inverse
        // roundtrips without broadening the inverse q-domain tolerance.
        double q_pole = copysign(d_qp, d_lat);
        double rho_pole = sqrt(d_C - d_nn * q_pole) / d_nn;
        double d_lam_pole = d_lon * 0.017453292519943295 - lam0;
        const double D_TWO_PI = 6.283185307179586;
        d_lam_pole -= D_TWO_PI * rint(d_lam_pole / D_TWO_PI);
        double theta_pole = d_nn * d_lam_pole;
        double sin_theta_pole, cos_theta_pole;
        sincos(theta_pole, &sin_theta_pole, &cos_theta_pole);
        easting = rho_pole * sin_theta_pole * a + x0;
        northing = (d_rho0 - rho_pole * cos_theta_pole) * a + y0;
    }} else {{
        {real_t} nn = ({real_t})d_nn;
        {real_t} C = ({real_t})d_C;
        {real_t} rho0 = ({real_t})d_rho0;
        {real_t} q = qsfn(sin(phi), e);
        {real_t} rho_sq = C - nn * q;
        if (rho_sq < ({real_t})0.0) rho_sq = ({real_t})0.0;
        {real_t} rho = sqrt(rho_sq) / nn;
        {real_t} theta = nn * lam;
        {real_t} sin_theta, cos_theta;
        vp_native_sincos(theta, &sin_theta, &cos_theta);
        easting = (double)(rho * sin_theta) * a + x0;
        northing = (double)(rho0 - rho * cos_theta) * a + y0;
    }}
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_AEA_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="aea_inverse", real_t="{real_t}")
    + """
    double d_nn, double d_C, double d_rho0,
    {real_t} e, {real_t} es, double d_qp,
    {real_t} lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    double d_cx_domain = (d_easting - x0) / a;
    double d_cy_domain = (d_northing - y0) / a;
    double d_rho_domain = hypot(d_cx_domain, d_rho0 - d_cy_domain);
    double d_q = (d_C - (d_rho_domain * d_nn) * (d_rho_domain * d_nn)) / d_nn;
    // Match the CPU EPS_ANGLE pole snap: public scale/unit roundtrips can
    // reconstruct a valid forward pole a few fp64 ulps beyond +/-qp.
    if (isfinite(d_q) && fabs(fabs(d_q) - d_qp) <= VP_EA_Q_POLE_SNAP_D) {{
        d_q = copysign(d_qp, d_q);
    }}
    if (isfinite(d_arg1) && isfinite(d_arg2)
        && (!isfinite(d_q) || fabs(d_q) > d_qp)) {{
        out_x[idx] = out_y[idx] = 1.0 / 0.0;
        return;
    }}
    {real_t} nn = ({real_t})d_nn;
    {real_t} C = ({real_t})d_C;
    {real_t} rho0 = ({real_t})d_rho0;
    {real_t} qp = ({real_t})d_qp;
    {real_t} dy = rho0 - cy;
    {real_t} rho = sqrt(cx * cx + dy * dy);
    if (nn < ({real_t})0.0) {{ rho = -rho; cx = -cx; dy = -dy; }}
    {real_t} lam = atan2(cx, dy) / nn;
    {real_t} q = (C - (rho * nn) * (rho * nn)) / nn;
    {real_t} phi = phi_from_q(q, e, es, qp);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Lambert Azimuthal Equal Area kernels
# ===================================================================

_LAEA_FORWARD_SOURCE = (
    _EA_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="laea_forward", real_t="{real_t}")
    + """
    int mode, {real_t} Rq, {real_t} D, {real_t} qp,
    {real_t} sin_beta0, {real_t} cos_beta0, {real_t} phi0, {real_t} e, {real_t} es,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} q = qsfn(sin(phi), e);
    {real_t} beta = asin(vp_ea_clamp_unit_preserve_nan(q / qp));
    {real_t} sin_beta, cos_beta;
    vp_native_sincos(beta, &sin_beta, &cos_beta);
    if (fabs(phi) == ({real_t})0.5 * {pi}) {{
        sin_beta = copysign(({real_t})1.0, phi);
        cos_beta = ({real_t})0.0;
    }}
    {real_t} sin_lam, cos_lam;
    vp_native_sincos(lam, &sin_lam, &cos_lam);
    {real_t} ex, ey;
    if (mode == 0) {{ // oblique
        {real_t} b = ({real_t})1.0 + sin_beta0 * sin_beta + cos_beta0 * cos_beta * cos_lam;
        if (isnan(b)) {{
            ex = ey = ({real_t})nan("");
        }} else if (b <= ({real_t})1e-10) {{
            ex = ey = ({real_t})(1.0 / 0.0);
        }} else {{
            b = Rq * sqrt(({real_t})2.0 / b);
            ex = b * D * cos_beta * sin_lam;
            ey = (b / D) * (cos_beta0 * sin_beta - sin_beta0 * cos_beta * cos_lam);
        }}
    }} else if (mode == 1) {{ // equatorial
        {real_t} b = ({real_t})1.0 + cos_beta * cos_lam;
        if (isnan(b)) {{
            ex = ey = ({real_t})nan("");
        }} else if (b <= ({real_t})1e-10) {{
            ex = ey = ({real_t})(1.0 / 0.0);
        }} else {{
            b = Rq * sqrt(({real_t})2.0 / b);
            ex = b * D * cos_beta * sin_lam;
            ey = (b / D) * sin_beta;
        }}
    }} else if (mode == 2) {{ // north pole
        if (fabs(phi + ({real_t})0.5 * {pi}) <= ({real_t})1e-10) {{
            ex = ey = ({real_t})(1.0 / 0.0);
        }} else {{
            {real_t} q_diff = qp - q;
            if (q_diff < ({real_t})0.0) q_diff = ({real_t})0.0;
            {real_t} rho = sqrt(q_diff);
            ex = rho * sin_lam;
            ey = -rho * cos_lam;
        }}
    }} else {{ // south pole
        if (fabs(phi - ({real_t})0.5 * {pi}) <= ({real_t})1e-10) {{
            ex = ey = ({real_t})(1.0 / 0.0);
        }} else {{
            {real_t} q_diff = qp + q;
            if (q_diff < ({real_t})0.0) q_diff = ({real_t})0.0;
            {real_t} rho = sqrt(q_diff);
            ex = rho * sin_lam;
            ey = rho * cos_lam;
        }}
    }}
    if (isnan(d_lat) || isnan(d_lon)) {{
        ex = ey = ({real_t})nan("");
    }} else if (!isfinite(d_lat) || !isfinite(d_lon)
               || fabs(phi) > ({real_t})0.5 * {pi}) {{
        ex = ey = ({real_t})(1.0 / 0.0);
    }}
    double easting  = (double)ex * (double)a + (double)x0;
    double northing = (double)ey * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_LAEA_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="laea_inverse", real_t="{real_t}")
    + """
    int mode, {real_t} Rq, {real_t} D, {real_t} qp,
    {real_t} sin_beta0, {real_t} cos_beta0, {real_t} phi0, {real_t} e, {real_t} es,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} raw_rho = hypot(cx, cy);
    {real_t} phi, lam;
    if (isnan(raw_rho)) {{
        phi = lam = ({real_t})nan("");
    }} else if (raw_rho == ({real_t})0.0) {{
        // The projection center is independent of D and avoids 0/0 in the
        // azimuth formulas (especially for an exact polar origin).
        phi = phi0;
        lam = ({real_t})0.0;
    }} else {{
        {real_t} sin_beta;
        if (mode == 0 || mode == 1) {{ // oblique or equatorial
            {real_t} x_adj = cx / D, y_adj = cy * D;
            {real_t} rho = hypot(x_adj, y_adj);
            if (rho > ({real_t})2.0 * Rq) {{
                double invalid = 1.0 / 0.0;
                if (dst_north_first) {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
                else                 {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
                return;
            }}
            {real_t} ce = ({real_t})2.0 * asin(rho / (({real_t})2.0 * Rq));
            {real_t} sin_ce, cos_ce;
            vp_native_sincos(ce, &sin_ce, &cos_ce);
            if (mode == 0) {{
                sin_beta = cos_ce * sin_beta0 + y_adj * sin_ce * cos_beta0 / rho;
                lam = atan2(x_adj * sin_ce, rho * cos_beta0 * cos_ce - y_adj * sin_beta0 * sin_ce);
            }} else {{
                sin_beta = y_adj * sin_ce / rho;
                lam = atan2(x_adj * sin_ce, rho * cos_ce);
            }}
        }} else {{
            // Polar modes use the unadjusted normalized coordinates. Applying
            // D first corrupts exact +/-90-degree origins when host D is tiny.
            {real_t} rho = raw_rho;
            if (rho > sqrt(({real_t})2.0 * qp)) {{
                double invalid = 1.0 / 0.0;
                if (dst_north_first) {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
                else                 {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
                return;
            }}
            if (mode == 2) {{ // north pole
                sin_beta = ({real_t})1.0 - (rho * rho) / qp;
                lam = atan2(cx, -cy);
            }} else {{ // south pole
                sin_beta = (rho * rho) / qp - ({real_t})1.0;
                lam = atan2(cx, cy);
            }}
        }}
        {real_t} q = qp * vp_ea_clamp_unit_preserve_nan(sin_beta);
        phi = phi_from_q(q, e, es, qp);
    }}
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Equal Earth kernels (polynomial — no iteration for forward, Newton for inverse)
# ===================================================================

_EQEARTH_FORWARD_SOURCE = (
    _EA_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="eqearth_forward", real_t="{real_t}")
    + """
    {real_t} e, {real_t} qp, {real_t} rqda,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    const {real_t} A1=({real_t})1.340264, A2=({real_t})-0.081106, A3=({real_t})0.000893, A4=({real_t})0.003796;
    const {real_t} SQRT3_2 = ({real_t})0.86602540378443864676;
    const {real_t} M = ({real_t})1.1547005383792515;  // 2*sqrt(3)/3
    // Geodetic -> authalic latitude
    {real_t} q = qsfn(sin(phi), e);
    {real_t} beta = asin(vp_ea_clamp_unit_preserve_nan(q / qp));
    {real_t} theta = asin(SQRT3_2 * sin(beta));
    {real_t} t2 = theta * theta;
    {real_t} t6 = t2 * t2 * t2;
    {real_t} d = A1 + ({real_t})3.0*A2*t2 + t6*(({real_t})7.0*A3 + ({real_t})9.0*A4*t2);
    double easting  = (double)(rqda * M * lam * cos(theta) / d) * (double)a + (double)x0;
    double northing = (double)(rqda * theta * (A1 + A2*t2 + t6*(A3 + A4*t2))) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_EQEARTH_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="eqearth_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} es, {real_t} qp, {real_t} rqda,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    const {real_t} A1=({real_t})1.340264, A2=({real_t})-0.081106, A3=({real_t})0.000893, A4=({real_t})0.003796;
    const {real_t} M = ({real_t})1.1547005383792515;  // 2*sqrt(3)/3
    // Remove rqda scaling
    {real_t} cxs = cx / rqda, cys = cy / rqda;
    {real_t} theta = cys;
    for (int i = 0; i < 12; i++) {{
        {real_t} t2 = theta * theta;
        {real_t} t6 = t2 * t2 * t2;
        {real_t} fy = theta * (A1 + A2*t2 + t6*(A3 + A4*t2)) - cys;
        {real_t} fpy = A1 + ({real_t})3.0*A2*t2 + t6*(({real_t})7.0*A3 + ({real_t})9.0*A4*t2);
        {real_t} dtheta = fy / fpy;
        theta = theta - dtheta;
        if (fabs(dtheta) < {tol}) break;
    }}
    {real_t} t2 = theta * theta;
    {real_t} t6 = t2 * t2 * t2;
    {real_t} d = A1 + ({real_t})3.0*A2*t2 + t6*(({real_t})7.0*A3 + ({real_t})9.0*A4*t2);
    {real_t} lam = cxs * d / (M * cos(theta));
    // Recover authalic latitude, then convert to geodetic via q-inversion
    {real_t} sin_beta = vp_ea_clamp_unit_preserve_nan(
        sin(theta) * ({real_t})2.0 / ({real_t})1.7320508075688772935);
    {real_t} q = qp * sin_beta;
    {real_t} phi = phi_from_q(q, e, es, qp);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Cylindrical Equal Area kernels
# ===================================================================

_CEA_FORWARD_SOURCE = (
    _EA_DEVICE_FNS
    + _FWD_SIGNATURE.format(func="cea_forward", real_t="{real_t}")
    + """
    {real_t} e, double d_qp, double d_k0,
    double lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} k0 = ({real_t})d_k0;
    {real_t} sin_phi = sin(phi);
    {real_t} q = qsfn(sin_phi, e);
    double easting = (double)(lam * k0) * a + x0;
    double northing;
    if (fabs(d_lat) == 90.0) {{
        northing = (0.5 * copysign(d_qp, d_lat) / d_k0) * a + y0;
    }} else {{
        northing = (double)(({real_t})0.5 * q / k0) * a + y0;
    }}
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_CEA_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="cea_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} es, double d_qp, double d_k0,
    {real_t} lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    double d_cy_domain = (d_northing - y0) / a;
    double d_q = 2.0 * d_cy_domain * d_k0;
    if (isfinite(d_q) && fabs(fabs(d_q) - d_qp) <= VP_EA_Q_POLE_SNAP_D) {{
        d_q = copysign(d_qp, d_q);
    }}
    if (isfinite(d_arg1) && isfinite(d_arg2)
        && (!isfinite(d_q) || fabs(d_q) > d_qp)) {{
        out_x[idx] = out_y[idx] = 1.0 / 0.0;
        return;
    }}
    {real_t} qp = ({real_t})d_qp;
    {real_t} k0 = ({real_t})d_k0;
    {real_t} lam = cx / k0;
    {real_t} q = ({real_t})2.0 * cy * k0;
    {real_t} phi = phi_from_q(q, e, es, qp);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Orthographic kernels
# ===================================================================

_ORTHO_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="ortho_forward", real_t="{real_t}")
    + """
    {real_t} sin_phi0, {real_t} cos_phi0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_phi, cos_phi;
    vp_native_sincos(phi, &sin_phi, &cos_phi);
    {real_t} sin_lam, cos_lam;
    vp_native_sincos(lam, &sin_lam, &cos_lam);
    double easting  = (double)(cos_phi * sin_lam) * (double)a + (double)x0;
    double northing = (double)(cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_ORTHO_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="ortho_inverse", real_t="{real_t}")
    + """
    {real_t} sin_phi0, {real_t} cos_phi0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} rho = sqrt(cx*cx + cy*cy);
    {real_t} c = asin(fmin(fmax(rho, ({real_t})-1.0), ({real_t})1.0));
    {real_t} sin_c, cos_c;
    vp_native_sincos(c, &sin_c, &cos_c);
    {real_t} safe_rho = fmax(rho, ({real_t})1e-30);
    {real_t} phi = asin(cos_c * sin_phi0 + cy * sin_c * cos_phi0 / safe_rho);
    {real_t} lam = atan2(cx * sin_c, safe_rho * cos_phi0 * cos_c - cy * sin_phi0 * sin_c);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Gnomonic kernels
# ===================================================================

_GNOM_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="gnom_forward", real_t="{real_t}")
    + """
    {real_t} sin_phi0, {real_t} cos_phi0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_phi, cos_phi;
    vp_native_sincos(phi, &sin_phi, &cos_phi);
    {real_t} sin_lam, cos_lam;
    vp_native_sincos(lam, &sin_lam, &cos_lam);
    {real_t} cos_c = sin_phi0 * sin_phi + cos_phi0 * cos_phi * cos_lam;
    double easting  = (double)(cos_phi * sin_lam / cos_c) * (double)a + (double)x0;
    double northing = (double)((cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam) / cos_c) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_GNOM_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="gnom_inverse", real_t="{real_t}")
    + """
    {real_t} sin_phi0, {real_t} cos_phi0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} rho = sqrt(cx*cx + cy*cy);
    {real_t} c = atan(rho);
    {real_t} sin_c, cos_c;
    vp_native_sincos(c, &sin_c, &cos_c);
    {real_t} safe_rho = fmax(rho, ({real_t})1e-30);
    {real_t} phi = asin(cos_c * sin_phi0 + cy * sin_c * cos_phi0 / safe_rho);
    {real_t} lam = atan2(cx * sin_c, safe_rho * cos_phi0 * cos_c - cy * sin_phi0 * sin_c);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Mollweide kernels
# ===================================================================

_MOLL_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="moll_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    const {real_t} SQRT2 = ({real_t})1.4142135623730950488;
    {real_t} pi_sin_phi = {pi} * sin(phi);
    {real_t} theta = phi;
    for (int i = 0; i < 20; i++) {{
        {real_t} sin_two_theta, cos_two_theta;
        vp_native_sincos(
            ({real_t})2.0 * theta, &sin_two_theta, &cos_two_theta
        );
        {real_t} dtheta = -(({real_t})2.0 * theta + sin_two_theta - pi_sin_phi)
                         / (({real_t})2.0 + ({real_t})2.0 * cos_two_theta);
        theta += dtheta;
        if (fabs(dtheta) < {tol}) break;
    }}
    {real_t} sin_theta, cos_theta;
    vp_native_sincos(theta, &sin_theta, &cos_theta);
    double easting  = (double)(lam * ({real_t})2.0 * SQRT2 / {pi} * cos_theta) * (double)a + (double)x0;
    double northing = (double)(SQRT2 * sin_theta) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_MOLL_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="moll_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    const {real_t} SQRT2 = ({real_t})1.4142135623730950488;
    {real_t} theta = asin(fmin(fmax(cy / SQRT2, ({real_t})-1.0), ({real_t})1.0));
    {real_t} phi = asin(fmin(fmax((({real_t})2.0 * theta + sin(({real_t})2.0 * theta)) / {pi}, ({real_t})-1.0), ({real_t})1.0));
    {real_t} lam = cx * {pi} / (({real_t})2.0 * SQRT2 * cos(theta));
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Oblique Mercator (Hotine) kernels
# ===================================================================

_OMERC_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="omerc_forward", real_t="{real_t}")
    + """
    {real_t} e, {real_t} B, {real_t} A_norm, {real_t} H,
    {real_t} sin_g0, {real_t} cos_g0, {real_t} sin_gc, {real_t} cos_gc,
    {real_t} u_c,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    // Conformal latitude function t(phi)
    {real_t} sin_phi = sin(phi);
    {real_t} e_sin = e * sin_phi;
    {real_t} t = tan(({real_t})0.25 * {pi} - ({real_t})0.5 * phi)
              * pow((({real_t})1.0 + e_sin) / (({real_t})1.0 - e_sin), ({real_t})0.5 * e);

    {real_t} Q = H / pow(t, B);
    {real_t} S = (Q - ({real_t})1.0 / Q) * ({real_t})0.5;
    {real_t} T = (Q + ({real_t})1.0 / Q) * ({real_t})0.5;
    {real_t} sin_B_lam, cos_B_lam;
    vp_native_sincos(B * lam, &sin_B_lam, &cos_B_lam);
    {real_t} V = sin_B_lam;
    {real_t} U = (-V * cos_g0 + S * sin_g0) / T;
    // Clamp U
    if (U > ({real_t})0.9999999999) U = ({real_t})0.9999999999;
    if (U < ({real_t})-0.9999999999) U = ({real_t})-0.9999999999;

    {real_t} v_ob = A_norm / (({real_t})2.0 * B)
                  * log((({real_t})1.0 - U) / (({real_t})1.0 + U));
    {real_t} u_ob = (A_norm / B)
                  * atan2(S * cos_g0 + V * sin_g0, cos_B_lam)
                  - u_c;

    // Rectified grid rotation + scale + offset (fp64)
    double easting  = ((double)v_ob * (double)cos_gc + (double)u_ob * (double)sin_gc) * (double)a + (double)x0;
    double northing = ((double)u_ob * (double)cos_gc - (double)v_ob * (double)sin_gc) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_OMERC_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="omerc_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} B, {real_t} A_norm, {real_t} H,
    {real_t} sin_g0, {real_t} cos_g0, {real_t} sin_gc, {real_t} cos_gc,
    {real_t} u_c,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    // Undo rectified grid rotation (cx, cy are already normalised)
    {real_t} v_ob = cx * cos_gc - cy * sin_gc;
    {real_t} u_ob = cx * sin_gc + cy * cos_gc + u_c;

    // Oblique Mercator inverse
    {real_t} Qp = exp(-B * v_ob / A_norm);
    {real_t} Sp = (Qp - ({real_t})1.0 / Qp) * ({real_t})0.5;
    {real_t} Tp = (Qp + ({real_t})1.0 / Qp) * ({real_t})0.5;
    {real_t} sin_B_u, cos_B_u;
    vp_native_sincos(B * u_ob / A_norm, &sin_B_u, &cos_B_u);
    {real_t} Vp = sin_B_u;
    {real_t} Up = (Vp * cos_g0 + Sp * sin_g0) / Tp;
    if (Up > ({real_t})0.9999999999) Up = ({real_t})0.9999999999;
    if (Up < ({real_t})-0.9999999999) Up = ({real_t})-0.9999999999;

    // Recover t from Up
    {real_t} t = pow(H / sqrt((({real_t})1.0 + Up) / (({real_t})1.0 - Up)), ({real_t})1.0 / B);

    // Iterative conformal latitude inversion
    {real_t} phi = ({real_t})0.5 * {pi} - ({real_t})2.0 * atan(t);
    for (int i = 0; i < 15; i++) {{
        {real_t} e_sin = e * sin(phi);
        {real_t} phi_new = ({real_t})0.5 * {pi} - ({real_t})2.0 * atan(
            t * pow((({real_t})1.0 - e_sin) / (({real_t})1.0 + e_sin), ({real_t})0.5 * e));
        if (fabs(phi_new - phi) < {tol}) {{ phi = phi_new; break; }}
        phi = phi_new;
    }}

    // Recover lambda
    {real_t} lam = -atan2(Sp * cos_g0 - Vp * sin_g0,
                          cos_B_u) / B;
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Krovak kernels (oblique conformal conic)
# ===================================================================

_KROVAK_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="krovak_forward", real_t="{real_t}")
    + """
    {real_t} e, {real_t} B, {real_t} kk, {real_t} nn,
    {real_t} r0_norm, {real_t} tan_half_p,
    {real_t} sin_ac, {real_t} cos_ac,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    // Gaussian conformal sphere
    {real_t} sin_phi = sin(phi);
    {real_t} e_sin = e * sin_phi;
    {real_t} gfi = pow((({real_t})1.0 + e_sin) / (({real_t})1.0 - e_sin), B * e * ({real_t})0.5);
    {real_t} Q = kk * pow(tan(({real_t})0.25 * {pi} + phi * ({real_t})0.5), B) / gfi;
    {real_t} U = ({real_t})2.0 * atan(Q) - ({real_t})0.5 * {pi};
    {real_t} V = -B * lam;

    // Oblique coordinates
    {real_t} sin_U, cos_U;
    vp_native_sincos(U, &sin_U, &cos_U);
    {real_t} sin_V, cos_V;
    vp_native_sincos(V, &sin_V, &cos_V);
    {real_t} T = asin(cos_ac * sin_U + sin_ac * cos_U * cos_V);
    {real_t} D = asin(cos_U * sin_V / cos(T));

    // Oblique cone
    {real_t} theta = nn * D;
    {real_t} r_norm = r0_norm * pow(tan_half_p / tan(({real_t})0.25 * {pi} + T * ({real_t})0.5), nn);
    {real_t} sin_theta, cos_theta;
    vp_native_sincos(theta, &sin_theta, &cos_theta);

    // North Orientated: negate
    double easting  = (double)(-r_norm * sin_theta) * (double)a + (double)x0;
    double northing = (double)(-r_norm * cos_theta) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_KROVAK_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="krovak_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} B, {real_t} kk, {real_t} nn,
    {real_t} r0_norm, {real_t} tan_half_p,
    {real_t} sin_ac, {real_t} cos_ac,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    // Undo North Orientated negation (cx, cy already normalised)
    {real_t} r_norm = sqrt(cx * cx + cy * cy);
    {real_t} theta = atan2(-cx, -cy);

    // Inverse cone
    {real_t} D = theta / nn;
    {real_t} T = ({real_t})2.0 * atan(
        pow(r0_norm / fmax(r_norm, ({real_t})1e-30), ({real_t})1.0 / nn) * tan_half_p
    ) - ({real_t})0.5 * {pi};

    // Inverse oblique
    {real_t} sin_T, cos_T;
    vp_native_sincos(T, &sin_T, &cos_T);
    {real_t} sin_D, cos_D;
    vp_native_sincos(D, &sin_D, &cos_D);
    {real_t} U = asin(cos_ac * sin_T - sin_ac * cos_T * cos_D);
    {real_t} V = asin(cos_T * sin_D / cos(U));

    // Inverse Gaussian sphere: t = [tan(pi/4+U/2)/k]^(1/B)
    {real_t} t = pow(tan(({real_t})0.25 * {pi} + U * ({real_t})0.5) / kk, ({real_t})1.0 / B);
    {real_t} phi = ({real_t})2.0 * atan(t) - ({real_t})0.5 * {pi};
    for (int i = 0; i < 15; i++) {{
        {real_t} e_sin = e * sin(phi);
        {real_t} phi_new = ({real_t})2.0 * atan(
            t * pow((({real_t})1.0 + e_sin) / (({real_t})1.0 - e_sin), ({real_t})0.5 * e)
        ) - ({real_t})0.5 * {pi};
        if (fabs(phi_new - phi) < {tol}) {{ phi = phi_new; break; }}
        phi = phi_new;
    }}

    {real_t} lam = -V / B;
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Eckert IV kernels (pseudocylindrical equal-area)
# ===================================================================

_ECK4_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="eck4_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    const {real_t} C_x = ({real_t})0.42223820031577120149;
    const {real_t} C_y = ({real_t})1.32650042817700232218;
    const {real_t} C_p = ({real_t})3.57079632679489661923;
    {real_t} p = C_p * sin(phi);
    {real_t} theta = phi;
    for (int i = 0; i < 20; i++) {{
        {real_t} sin_t, cos_t;
        vp_native_sincos(theta, &sin_t, &cos_t);
        {real_t} V = theta + sin_t * cos_t + ({real_t})2.0 * sin_t - p;
        {real_t} dtheta = -V / (({real_t})1.0 + cos(({real_t})2.0 * theta) + ({real_t})2.0 * cos_t);
        theta += dtheta;
        if (fabs(dtheta) < {tol}) break;
    }}
    {real_t} sin_theta, cos_theta;
    vp_native_sincos(theta, &sin_theta, &cos_theta);
    double easting  = (double)(C_x * lam * (({real_t})1.0 + cos_theta)) * (double)a + (double)x0;
    double northing = (double)(C_y * sin_theta) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_ECK4_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="eck4_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    const {real_t} C_x = ({real_t})0.42223820031577120149;
    const {real_t} C_y = ({real_t})1.32650042817700232218;
    const {real_t} C_p = ({real_t})3.57079632679489661923;
    {real_t} theta = asin(fmin(fmax(cy / C_y, ({real_t})-1.0), ({real_t})1.0));
    {real_t} sin_t, cos_t;
    vp_native_sincos(theta, &sin_t, &cos_t);
    {real_t} phi = asin(fmin(fmax((theta + sin_t * cos_t + ({real_t})2.0 * sin_t) / C_p, ({real_t})-1.0), ({real_t})1.0));
    {real_t} lam = cx / (C_x * (({real_t})1.0 + cos_t));
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Eckert VI kernels (pseudocylindrical equal-area)
# ===================================================================

_ECK6_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="eck6_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    const {real_t} C_p = ({real_t})2.57079632679489661923;
    const {real_t} C_x = ({real_t})0.44101327172257790882;
    const {real_t} C_y = ({real_t})0.88202654344515581764;
    {real_t} p = C_p * sin(phi);
    {real_t} theta = phi;
    for (int i = 0; i < 20; i++) {{
        {real_t} sin_theta, cos_theta;
        vp_native_sincos(theta, &sin_theta, &cos_theta);
        {real_t} V = theta + sin_theta - p;
        {real_t} dtheta = -V / (({real_t})1.0 + cos_theta);
        theta += dtheta;
        if (fabs(dtheta) < {tol}) break;
    }}
    double easting  = (double)(C_x * lam * (({real_t})1.0 + cos(theta))) * (double)a + (double)x0;
    double northing = (double)(C_y * theta) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_ECK6_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="eck6_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    const {real_t} C_p = ({real_t})2.57079632679489661923;
    const {real_t} C_x = ({real_t})0.44101327172257790882;
    const {real_t} C_y = ({real_t})0.88202654344515581764;
    {real_t} theta = cy / C_y;
    {real_t} sin_theta, cos_theta;
    vp_native_sincos(theta, &sin_theta, &cos_theta);
    {real_t} phi = asin(fmin(fmax((theta + sin_theta) / C_p, ({real_t})-1.0), ({real_t})1.0));
    {real_t} lam = cx / (C_x * (({real_t})1.0 + cos_theta));
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Oblique Stereographic kernels (double projection)
# ===================================================================

_STEREA_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="sterea_forward", real_t="{real_t}")
    + """
    {real_t} e, {real_t} nn, {real_t} c, {real_t} R, {real_t} sin_chi0, {real_t} cos_chi0, {real_t} k0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_phi = sin(phi);
    {real_t} S = pow(((({real_t})1.0 + sin_phi) / (({real_t})1.0 - sin_phi)) * pow((({real_t})1.0 - e*sin_phi) / (({real_t})1.0 + e*sin_phi), e), nn);
    {real_t} w = c * S;
    {real_t} chi = asin((w - ({real_t})1.0) / (w + ({real_t})1.0));
    {real_t} lam_s = nn * lam;
    {real_t} sin_chi, cos_chi;
    vp_native_sincos(chi, &sin_chi, &cos_chi);
    {real_t} sin_lam_s, cos_lam_s;
    vp_native_sincos(lam_s, &sin_lam_s, &cos_lam_s);
    {real_t} k_den = ({real_t})1.0 + sin_chi0*sin_chi + cos_chi0*cos_chi*cos_lam_s;
    double easting  = (double)(({real_t})2.0 * R * k0 * cos_chi * sin_lam_s / k_den) * (double)a + (double)x0;
    double northing = (double)(({real_t})2.0 * R * k0 * (cos_chi0*sin_chi - sin_chi0*cos_chi*cos_lam_s) / k_den) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_STEREA_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="sterea_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} nn, {real_t} c, {real_t} R, {real_t} sin_chi0, {real_t} cos_chi0, {real_t} k0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} xs = cx / (({real_t})2.0 * R * k0), ys = cy / (({real_t})2.0 * R * k0);
    {real_t} rho = sqrt(xs*xs + ys*ys);
    {real_t} ce = ({real_t})2.0 * atan(rho);
    {real_t} sin_ce, cos_ce;
    vp_native_sincos(ce, &sin_ce, &cos_ce);
    {real_t} sin_chi = cos_ce*sin_chi0 + ys*sin_ce*cos_chi0 / fmax(rho, ({real_t})1e-30);
    {real_t} chi = asin(fmin(fmax(sin_chi, ({real_t})-1.0), ({real_t})1.0));
    {real_t} lam_s = atan2(xs*sin_ce, rho*cos_chi0*cos_ce - ys*sin_chi0*sin_ce);
    {real_t} lam = lam_s / nn;
    // Conformal sphere -> geodetic (iterative)
    {real_t} psi = ({real_t})0.5 * (log((({real_t})1.0 + sin_chi) / fmax(({real_t})1.0 - sin_chi, ({real_t})1e-30)) - log(c)) / nn;
    {real_t} phi = ({real_t})2.0 * atan(exp(psi)) - ({real_t})0.5 * {pi};
    for (int i = 0; i < 15; i++) {{
        {real_t} sp, cp;
        vp_native_sincos(phi, &sp, &cp);
        {real_t} es = e * sp;
        {real_t} psi_c = log(tan(({real_t})0.25*{pi} + ({real_t})0.5*phi) * pow((({real_t})1.0-es)/(({real_t})1.0+es), ({real_t})0.5*e));
        {real_t} dphi = (psi - psi_c) * cp * (({real_t})1.0 - e*e*sp*sp) / (({real_t})1.0 - e*e);
        phi += dphi;
        if (fabs(dphi) < {tol}) break;
    }}
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Geostationary Satellite kernels
# ===================================================================

_GEOS_FORWARD_SOURCE = (
    _GEOS_DEVICE_NUMERIC_CONTRACT
    + _FWD_SIGNATURE.format(func="geos_forward", real_t="{real_t}")
    + """
    {real_t} H, {real_t} h, {real_t} r_eq2, {real_t} r_pol2, int sweep_x,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} ex, ey;
    if (isnan(d_lat) || isnan(d_lon)) {{
        ex = ey = ({real_t})nan("");
    }} else if (!isfinite(d_lat) || !isfinite(d_lon)
               || fabs(phi) > ({real_t})VP_GEOS_SCAN_ANGLE_LIMIT) {{
        ex = ey = ({real_t})(1.0 / 0.0);
    }} else {{
        {real_t} phi_gc = atan(r_pol2 / r_eq2 * tan(phi));
        {real_t} sin_pgc, cos_pgc;
        vp_native_sincos(phi_gc, &sin_pgc, &cos_pgc);
        // Geocentric earth radius (CGMS standard).
        {real_t} r_pol = sqrt(r_pol2);
        {real_t} r_earth = r_pol / sqrt(({real_t})1.0 - (r_eq2 - r_pol2) / r_eq2 * cos_pgc * cos_pgc);
        {real_t} sin_l, cos_l;
        vp_native_sincos(lam, &sin_l, &cos_l);
        {real_t} Vx = r_earth * cos_pgc * cos_l;
        {real_t} Vy = r_earth * cos_pgc * sin_l;
        {real_t} Vz = r_earth * sin_pgc;
        {real_t} Sx = H - Vx;
        // Points behind the ellipsoid are outside the satellite view. Keep
        // PROJ's +inf sentinel instead of projecting them onto the limb.
        {real_t} visibility = Sx * Vx - Vy * Vy - (r_eq2 / r_pol2) * Vz * Vz;
        if (isnan(visibility)) {{
            ex = ey = ({real_t})nan("");
        }} else if (visibility < ({real_t})0.0) {{
            ex = ey = ({real_t})(1.0 / 0.0);
        }} else if (sweep_x) {{
            ex = atan2(Vy, hypot(Sx, Vz));
            ey = atan2(Vz, Sx);
        }} else {{
            ex = atan2(Vy, Sx);
            ey = atan2(Vz, hypot(Sx, Vy));
        }}
    }}
    double easting  = (double)ex * (double)h + (double)x0;
    double northing = (double)ey * (double)h + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_GEOS_INVERSE_SOURCE = (
    _GEOS_DEVICE_NUMERIC_CONTRACT
    + _INV_SIGNATURE.format(func="geos_inverse", real_t="{real_t}")
    + """
    {real_t} H, {real_t} h, {real_t} r_eq2, {real_t} r_pol2, int sweep_x,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    if (isnan(d_arg1) || isnan(d_arg2)) {{
        double invalid = nan("");
        if (dst_north_first) {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        else                 {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        return;
    }}
    if (!isfinite(d_arg1) || !isfinite(d_arg2)) {{
        double invalid = 1.0 / 0.0;
        if (dst_north_first) {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        else                 {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        return;
    }}
    // cx = (easting - x0) / a, but physical coords are h * angle, so angle = cx * a / h.
    {real_t} x_angle = cx * a / h, y_angle = cy * a / h;
    if (fabs(x_angle) >= ({real_t})VP_GEOS_SCAN_ANGLE_LIMIT
        || fabs(y_angle) >= ({real_t})VP_GEOS_SCAN_ANGLE_LIMIT) {{
        double invalid = 1.0 / 0.0;
        if (dst_north_first) {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        else                 {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        return;
    }}
    {real_t} sx, cx2, sy, cy2;
    vp_native_sincos(x_angle, &sx, &cx2);
    vp_native_sincos(y_angle, &sy, &cy2);
    {real_t} Vx = cy2 * cx2;
    {real_t} Vy = sweep_x ? sx : cy2 * sx;
    {real_t} Vz = sweep_x ? cx2 * sy : sy;
    {real_t} ac = Vx*Vx + Vy*Vy + Vz*Vz*r_eq2/r_pol2;
    // Normalize the quadratic by H before forming its discriminant. The
    // physical form subtracts two ~1e30 terms at the limb and can flip the
    // sign solely from contraction/rounding. Track both product residuals so
    // an exact tangent remains on-disk without clamping a real negative D.
    {real_t} a_over_H = a / H;
    {real_t} normalized_c = fma(-a_over_H, a_over_H, ({real_t})1.0);
    {real_t} vx2 = Vx * Vx;
    {real_t} vx2_error = fma(Vx, Vx, -vx2);
    {real_t} ac_c = ac * normalized_c;
    {real_t} ac_c_error = fma(ac, normalized_c, -ac_c);
    {real_t} disc_hi = vx2 - ac_c;
    {real_t} disc_lo = ((vx2 - disc_hi) - ac_c) + vx2_error - ac_c_error;
    {real_t} disc = disc_hi + disc_lo;
    // Native fp32 sincos/FMA leaves about -1.9e-8 at the quantized tangent;
    // the next resolved outside bin is below -4.3e-7. This threshold accepts
    // only that representational tangent bin, not a broad off-disk band.
    const {real_t} discriminant_tolerance = sizeof({real_t}) == sizeof(float)
        ? ({real_t})VP_GEOS_FP32_DISCRIMINANT_TOLERANCE
        : ({real_t})VP_GEOS_FP64_DISCRIMINANT_TOLERANCE;
    if (disc < -discriminant_tolerance || isnan(disc)) {{
        double invalid = 1.0 / 0.0;
        if (dst_north_first) {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        else                 {{ out_x[idx] = invalid; out_y[idx] = invalid; }}
        return;
    }}
    if (disc < ({real_t})0.0) disc = ({real_t})0.0;
    {real_t} rs = H * (Vx - sqrt(disc)) / ac;
    {real_t} Px = H - rs*Vx;
    {real_t} Py = rs*Vy;
    {real_t} Pz = rs*Vz;
    {real_t} lam = atan2(Py, Px);
    {real_t} phi = atan(Pz * r_eq2 / (sqrt(Px*Px+Py*Py) * r_pol2));
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Robinson kernels (table-based interpolation)
# ===================================================================

_ROBIN_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="robin_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{
    // Robinson cubic polynomial coefficients (from PROJ PJ_robin.c).
    // Variable z is offset within 5-degree interval IN DEGREES (0 <= z <= 5).
    // value = c0 + z*(c1 + z*(c2 + z*c3))
    const {real_t} XC0[18] = {{1.0,0.9986,0.9954,0.99,0.9822,0.973,0.96,0.9427,0.9216,0.8962,0.8679,0.835,0.7986,0.7597,0.7186,0.6732,0.6213,0.5722}};
    const {real_t} XC1[18] = {{2.2199e-17,-0.000482243,-0.00083103,-0.00135364,-0.00167442,-0.00214868,-0.00305085,-0.00382792,-0.00467746,-0.00536223,-0.00609363,-0.00698325,-0.00755338,-0.00798324,-0.00851367,-0.00986209,-0.010418,-0.00906601}};
    const {real_t} XC2[18] = {{-7.15515e-05,-2.4897e-05,-4.48605e-05,-5.9661e-05,-4.49547e-06,-9.03571e-05,-9.00761e-05,-6.53386e-05,-0.00010457,-3.23831e-05,-0.000113898,-6.40253e-05,-5.00009e-05,-3.5971e-05,-7.01149e-05,-0.000199569,8.83923e-05,0.000182}};
    const {real_t} XC3[18] = {{3.1103e-06,-1.3309e-06,-9.86701e-07,3.6777e-06,-5.72411e-06,1.8736e-08,1.64917e-06,-2.6154e-06,4.81243e-06,-5.43432e-06,3.32484e-06,9.34959e-07,9.35324e-07,-2.27626e-06,-8.6303e-06,1.91974e-05,6.24051e-06,6.24051e-06}};
    const {real_t} YC0[18] = {{-5.20417e-18,0.062,0.124,0.186,0.248,0.31,0.372,0.434,0.4958,0.5571,0.6176,0.6769,0.7346,0.7903,0.8435,0.8936,0.9394,0.9761}};
    const {real_t} YC1[18] = {{0.0124,0.0124,0.0124,0.0123999,0.0124002,0.0123992,0.0124029,0.0123893,0.0123198,0.0121916,0.0119938,0.011713,0.0113541,0.0109107,0.0103431,0.00969686,0.00840947,0.00616527}};
    const {real_t} YC2[18] = {{1.21431e-18,-1.26793e-09,5.07171e-09,-1.90189e-08,7.10039e-08,-2.64997e-07,9.88983e-07,-3.69093e-06,-1.02252e-05,-1.54081e-05,-2.41424e-05,-3.20223e-05,-3.97684e-05,-4.89042e-05,-6.4615e-05,-6.4636e-05,-0.000192841,-0.000256}};
    const {real_t} YC3[18] = {{-8.45284e-11,4.22642e-10,-1.60604e-09,6.00152e-09,-2.24e-08,8.35986e-08,-3.11994e-07,-4.35621e-07,-3.45523e-07,-5.82288e-07,-5.25327e-07,-5.16405e-07,-6.09052e-07,-1.04739e-06,-1.40374e-09,-8.547e-06,-4.2106e-06,-4.2106e-06}};
    const {real_t} FXC = ({real_t})0.8487, FYC = ({real_t})1.3523;
    const {real_t} R2D = ({real_t})57.29577951308232;
"""
    + _FWD_PREAMBLE
    + """
    {real_t} abs_phi = fabs(phi);
    int ti = (int)(abs_phi * ({real_t})11.45915590261646417544);
    if (ti >= 18) ti = 17;
    {real_t} z = abs_phi * R2D - ({real_t})5.0 * ({real_t})ti;
    {real_t} X = XC0[ti] + z * (XC1[ti] + z * (XC2[ti] + z * XC3[ti]));
    {real_t} Y = YC0[ti] + z * (YC1[ti] + z * (YC2[ti] + z * YC3[ti]));
    double easting  = (double)(FXC * X * lam) * (double)a + (double)x0;
    {real_t} sgn = phi < ({real_t})0.0 ? ({real_t})-1.0 : ({real_t})1.0;
    double northing = (double)(FYC * Y * sgn) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_ROBIN_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="robin_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{
    // Robinson cubic polynomial coefficients (from PROJ PJ_robin.c).
    // Variable z is offset within 5-degree interval IN DEGREES (0 <= z <= 5).
    const {real_t} XC0[19] = {{1.0,0.9986,0.9954,0.99,0.9822,0.973,0.96,0.9427,0.9216,0.8962,0.8679,0.835,0.7986,0.7597,0.7186,0.6732,0.6213,0.5722,0.5322}};
    const {real_t} XC1[19] = {{2.2199e-17,-0.000482243,-0.00083103,-0.00135364,-0.00167442,-0.00214868,-0.00305085,-0.00382792,-0.00467746,-0.00536223,-0.00609363,-0.00698325,-0.00755338,-0.00798324,-0.00851367,-0.00986209,-0.010418,-0.00906601,-0.00677797}};
    const {real_t} XC2[19] = {{-7.15515e-05,-2.4897e-05,-4.48605e-05,-5.9661e-05,-4.49547e-06,-9.03571e-05,-9.00761e-05,-6.53386e-05,-0.00010457,-3.23831e-05,-0.000113898,-6.40253e-05,-5.00009e-05,-3.5971e-05,-7.01149e-05,-0.000199569,8.83923e-05,0.000182,0.000275608}};
    const {real_t} XC3[19] = {{3.1103e-06,-1.3309e-06,-9.86701e-07,3.6777e-06,-5.72411e-06,1.8736e-08,1.64917e-06,-2.6154e-06,4.81243e-06,-5.43432e-06,3.32484e-06,9.34959e-07,9.35324e-07,-2.27626e-06,-8.6303e-06,1.91974e-05,6.24051e-06,6.24051e-06,6.24051e-06}};
    const {real_t} YC0[19] = {{-5.20417e-18,0.062,0.124,0.186,0.248,0.31,0.372,0.434,0.4958,0.5571,0.6176,0.6769,0.7346,0.7903,0.8435,0.8936,0.9394,0.9761,1.0}};
    const {real_t} YC1[19] = {{0.0124,0.0124,0.0124,0.0123999,0.0124002,0.0123992,0.0124029,0.0123893,0.0123198,0.0121916,0.0119938,0.011713,0.0113541,0.0109107,0.0103431,0.00969686,0.00840947,0.00616527,0.00328947}};
    const {real_t} YC2[19] = {{1.21431e-18,-1.26793e-09,5.07171e-09,-1.90189e-08,7.10039e-08,-2.64997e-07,9.88983e-07,-3.69093e-06,-1.02252e-05,-1.54081e-05,-2.41424e-05,-3.20223e-05,-3.97684e-05,-4.89042e-05,-6.4615e-05,-6.4636e-05,-0.000192841,-0.000256,-0.000319159}};
    const {real_t} YC3[19] = {{-8.45284e-11,4.22642e-10,-1.60604e-09,6.00152e-09,-2.24e-08,8.35986e-08,-3.11994e-07,-4.35621e-07,-3.45523e-07,-5.82288e-07,-5.25327e-07,-5.16405e-07,-6.09052e-07,-1.04739e-06,-1.40374e-09,-8.547e-06,-4.2106e-06,-4.2106e-06,-4.2106e-06}};
    const {real_t} FXC = ({real_t})0.8487, FYC = ({real_t})1.3523;
"""
    + _INV_PREAMBLE
    + """
    {real_t} abs_y = fabs(cy) / FYC;
    {real_t} phi, lam;
    if (abs_y >= ({real_t})1.0) {{
        // Pathologic case: |lat| >= 90
        phi = cy < ({real_t})0.0 ? ({real_t})(-1.5707963267948966) : ({real_t})1.5707963267948966;
        lam = cx / (FXC * fmax(XC0[18], ({real_t})1e-30));
    }} else {{
        // Find interval via linear search on YC0
        int ti = (int)(abs_y * ({real_t})18.0);
        if (ti < 0) ti = 0;
        if (ti >= 18) ti = 17;
        while (ti > 0 && YC0[ti] > abs_y) ti--;
        while (ti < 17 && YC0[ti+1] <= abs_y) ti++;
        // Linear initial guess for z (in degrees, 0..5)
        {real_t} z = ({real_t})5.0 * (abs_y - YC0[ti]) / fmax(YC0[ti+1] - YC0[ti], ({real_t})1e-30);
        // Newton-Raphson on cubic Y(z) — converges in 2-3 iterations from linear guess.
        {real_t} c0s = YC0[ti] - abs_y;
        for (int it = 0; it < 4; it++) {{
            {real_t} val = c0s + z * (YC1[ti] + z * (YC2[ti] + z * YC3[ti]));
            {real_t} deriv = YC1[ti] + z * (({real_t})2.0 * YC2[ti] + z * ({real_t})3.0 * YC3[ti]);
            if (fabs(deriv) < ({real_t})1e-30) break;
            {real_t} dz = val / deriv;
            z -= dz;
            if (fabs(dz) < ({real_t})1e-12) break;
        }}
        {real_t} phi_deg = ({real_t})5.0 * ({real_t})ti + z;
        {real_t} X = XC0[ti] + z * (XC1[ti] + z * (XC2[ti] + z * XC3[ti]));
        phi = phi_deg * ({real_t})0.017453292519943295;
        if (cy < ({real_t})0.0) phi = -phi;
        lam = cx / (FXC * fmax(X, ({real_t})1e-30));
    }}
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Winkel Tripel kernels
# ===================================================================

_WINTRI_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="wintri_forward", real_t="{real_t}")
    + """
    {real_t} cos_phi1,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_phi, cos_phi;
    vp_native_sincos(phi, &sin_phi, &cos_phi);
    {real_t} sin_half_lam, cos_half_lam;
    vp_native_sincos(lam * ({real_t})0.5, &sin_half_lam, &cos_half_lam);
    {real_t} alpha = acos(fmin(fmax(cos_phi * cos_half_lam, ({real_t})-1.0), ({real_t})1.0));
    {real_t} sinc_a = fabs(alpha) < ({real_t})1e-10 ? ({real_t})1.0 : sin(alpha) / alpha;
    double easting  = (double)((({real_t})2.0 * cos_phi * sin_half_lam / sinc_a + lam*cos_phi1) * ({real_t})0.5) * (double)a + (double)x0;
    double northing = (double)((sin_phi / sinc_a + phi) * ({real_t})0.5) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_WINTRI_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="wintri_inverse", real_t="{real_t}")
    + """
    {real_t} cos_phi1,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} lam = cx * ({real_t})2.0;
    {real_t} phi = cy;
    const {real_t} EPS = ({real_t})1e-12;
    for (int i = 0; i < 10; i++) {{
        {real_t} sp, cp;
        vp_native_sincos(phi, &sp, &cp);
        {real_t} sh, ch;
        vp_native_sincos(lam * ({real_t})0.5, &sh, &ch);
        {real_t} D = cp * ch;
        {real_t} al = acos(fmin(fmax(D, ({real_t})-1.0), ({real_t})1.0));
        {real_t} sa = sin(al);
        int small = fabs(al) < EPS;
        {real_t} sa_safe = small ? ({real_t})1.0 : sa;
        {real_t} rsinc = small ? ({real_t})1.0 : al / sa_safe;
        {real_t} sa3 = sa_safe * sa_safe * sa_safe;
        {real_t} G = small ? ({real_t})0.0 : (sa_safe - al * D) / sa3;
        {real_t} f1 = (({real_t})2.0 * cp * sh * rsinc + lam * cos_phi1) * ({real_t})0.5 - cx;
        {real_t} f2 = (sp * rsinc + phi) * ({real_t})0.5 - cy;
        {real_t} J11 = (cp * ch * rsinc + cp * cp * sh * sh * G + cos_phi1) * ({real_t})0.5;
        {real_t} J12 = sp * sh * (D * G - rsinc);
        {real_t} J21 = sp * cp * sh * G * ({real_t})0.25;
        {real_t} J22 = (cp * rsinc + sp * sp * ch * G + ({real_t})1.0) * ({real_t})0.5;
        {real_t} det = J11 * J22 - J12 * J21;
        if (fabs(det) < ({real_t})1e-30) break;
        {real_t} inv_det = ({real_t})1.0 / det;
        {real_t} dlam = (J22 * f1 - J12 * f2) * inv_det;
        {real_t} dphi = (J11 * f2 - J21 * f1) * inv_det;
        lam -= dlam;
        phi -= dphi;
        if (fabs(dlam) < EPS && fabs(dphi) < EPS) break;
    }}
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Natural Earth kernels (polynomial)
# ===================================================================

_NATEARTH_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="natearth_forward", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    const {real_t} A0=({real_t})0.8707,A1=({real_t})-0.131979,A2=({real_t})-0.013791,A3=({real_t})0.003971,A4=({real_t})-0.001529;
    const {real_t} B0=({real_t})1.007226,B1=({real_t})0.015085,B2=({real_t})-0.044475,B3=({real_t})0.028874,B4=({real_t})-0.005916;
    {real_t} p2 = phi*phi, p4 = p2*p2;
    double easting  = (double)(lam * (A0 + p2*(A1 + p2*(A2 + p4*p2*(A3 + p2*A4))))) * (double)a + (double)x0;
    double northing = (double)(phi * (B0 + p2*(B1 + p4*(B2 + p2*(B3 + p2*B4))))) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_NATEARTH_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="natearth_inverse", real_t="{real_t}")
    + """
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    const {real_t} A0=({real_t})0.8707,A1=({real_t})-0.131979,A2=({real_t})-0.013791,A3=({real_t})0.003971,A4=({real_t})-0.001529;
    const {real_t} B0=({real_t})1.007226,B1=({real_t})0.015085,B2=({real_t})-0.044475,B3=({real_t})0.028874,B4=({real_t})-0.005916;
    {real_t} phi = cy;
    for (int i = 0; i < 15; i++) {{
        {real_t} p2 = phi*phi, p4 = p2*p2;
        {real_t} fy = phi*(B0+p2*(B1+p4*(B2+p2*(B3+p2*B4)))) - cy;
        {real_t} fpy = B0+p2*(({real_t})3.0*B1+p4*(({real_t})7.0*B2+p2*(({real_t})9.0*B3+({real_t})11.0*p2*B4)));
        phi -= fy / fpy;
        if (fabs(fy) < {tol}) break;
    }}
    {real_t} p2 = phi*phi, p4 = p2*p2;
    {real_t} lam = cx / (A0 + p2*(A1 + p2*(A2 + p4*p2*(A3 + p2*A4))));
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Azimuthal Equidistant kernels
# ===================================================================

_AEQD_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="aeqd_forward", real_t="{real_t}")
    + """
    {real_t} sin_phi0, {real_t} cos_phi0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sp, cp, sl, cl;
    vp_native_sincos(phi, &sp, &cp);
    vp_native_sincos(lam, &sl, &cl);
    {real_t} cos_c = fmin(fmax(sin_phi0*sp + cos_phi0*cp*cl, ({real_t})-1.0), ({real_t})1.0);
    {real_t} c2 = acos(cos_c);
    {real_t} k = fabs(c2) < ({real_t})1e-10 ? ({real_t})1.0 : c2 / sin(c2);
    double easting  = (double)(k * cp * sl) * (double)a + (double)x0;
    double northing = (double)(k * (cos_phi0*sp - sin_phi0*cp*cl)) * (double)a + (double)y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_AEQD_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="aeqd_inverse", real_t="{real_t}")
    + """
    {real_t} sin_phi0, {real_t} cos_phi0,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} c2 = sqrt(cx*cx + cy*cy);
    {real_t} sin_c, cos_c;
    vp_native_sincos(c2, &sin_c, &cos_c);
    {real_t} safe_c = fmax(c2, ({real_t})1e-30);
    {real_t} phi = asin(fmin(fmax(cos_c*sin_phi0 + cy*sin_c*cos_phi0/safe_c, ({real_t})-1.0), ({real_t})1.0));
    {real_t} lam = atan2(cx*sin_c, safe_c*cos_phi0*cos_c - cy*sin_phi0*sin_c);
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Double-single TM forward kernel (consumer GPU path)
# ===================================================================

_TM_FORWARD_DS_SOURCE = (
    _DS_TM_DEVICE_FNS
    + """
extern "C" __global__ void __launch_bounds__(256) tm_forward_ds(
    const double* __restrict__ in_x, const double* __restrict__ in_y,
    double* __restrict__ out_x, double* __restrict__ out_y,
    double cbg0, double cbg1, double cbg2, double cbg3, double cbg4, double cbg5,
    double gtu0, double gtu1, double gtu2, double gtu3, double gtu4, double gtu5,
    double Qn, double Zb, double lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double d1 = in_x[idx], d2 = in_y[idx];
    double d_lat, d_lon;
    if (src_north_first) {{ d_lat = d1; d_lon = d2; }} else {{ d_lon = d1; d_lat = d2; }}

    // Convert to ds in radians, subtract central meridian
    ds_t phi = ds_from_double(d_lat * 0.017453292519943295);
    ds_t lam = ds_from_double(d_lon * 0.017453292519943295 - lam0);
    lam = ds_wrap_to_pi(lam);

    // Coefficients as ds
    ds_t c0=ds_from_double(cbg0),c1=ds_from_double(cbg1),c2=ds_from_double(cbg2);
    ds_t c3=ds_from_double(cbg3),c4=ds_from_double(cbg4),c5=ds_from_double(cbg5);
    ds_t g0=ds_from_double(gtu0),g1=ds_from_double(gtu1),g2=ds_from_double(gtu2);
    ds_t g3=ds_from_double(gtu3),g4=ds_from_double(gtu4),g5=ds_from_double(gtu5);

    // gatg: geodetic -> Gaussian latitude
    ds_t two_phi = ds_mul(ds_from_float(2.0f), phi);
    ds_t sin_two_phi, cos_two_phi;
    ds_sincos(two_phi, &sin_two_phi, &cos_two_phi);
    ds_t Cn = ds_gatg(c0,c1,c2,c3,c4,c5, phi, cos_two_phi, sin_two_phi);

    // Gaussian -> complex spherical
    ds_t sin_Cn, cos_Cn;
    ds_sincos(Cn, &sin_Cn, &cos_Cn);
    ds_t sin_Ce, cos_Ce;
    ds_sincos(lam, &sin_Ce, &cos_Ce);
    ds_t cos_Cn_cos_Ce = ds_mul(cos_Cn, cos_Ce);
    Cn = ds_atan2(sin_Cn, cos_Cn_cos_Ce);
    ds_t inv_denom = ds_div(ds_from_float(1.0f), ds_hypot(sin_Cn, cos_Cn_cos_Ce));
    ds_t tan_Ce = ds_mul(ds_mul(sin_Ce, cos_Cn), inv_denom);
    ds_t Ce = ds_asinh(tan_Ce);

    // Optimized trig for Clenshaw
    ds_t two_inv = ds_mul(ds_from_float(2.0f), inv_denom);
    ds_t two_inv_sq = ds_mul(two_inv, inv_denom);
    ds_t tmp_r = ds_mul(cos_Cn_cos_Ce, two_inv_sq);
    ds_t sin_arg_r = ds_mul(sin_Cn, tmp_r);
    ds_t cos_arg_r = ds_sub(ds_mul(cos_Cn_cos_Ce, tmp_r), ds_from_float(1.0f));
    ds_t sinh_arg_i = ds_mul(tan_Ce, two_inv);
    ds_t cosh_arg_i = ds_sub(two_inv_sq, ds_from_float(1.0f));

    // Clenshaw complex
    ds_t dCn, dCe;
    ds_clenshaw_complex(g0,g1,g2,g3,g4,g5, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);
    Cn = ds_add(Cn, dCn);
    Ce = ds_add(Ce, dCe);

    // Scale/offset in fp64
    double easting  = ds_to_double(ds_mul(ds_from_double(Qn), Ce)) * a + x0;
    double northing = (ds_to_double(ds_add(ds_mul(ds_from_double(Qn), Cn), ds_from_double(Zb)))) * a + y0;
    double easting_out = easting / x_unit_to_m;
    double northing_out = northing / y_unit_to_m;

    if (dst_north_first) {{ out_x[idx] = northing_out; out_y[idx] = easting_out; }}
    else                 {{ out_x[idx] = easting_out;  out_y[idx] = northing_out; }}
}}
"""
)

_TM_INVERSE_DS_SOURCE = (
    _DS_TM_DEVICE_FNS
    + """
extern "C" __global__ void __launch_bounds__(256) tm_inverse_ds(
    const double* __restrict__ in_x, const double* __restrict__ in_y,
    double* __restrict__ out_x, double* __restrict__ out_y,
    double cgb0, double cgb1, double cgb2, double cgb3, double cgb4, double cgb5,
    double utg0, double utg1, double utg2, double utg3, double utg4, double utg5,
    double Qn, double Zb, double lam0, double a, double x0, double y0,
    int src_north_first, int dst_north_first, int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double d1 = in_x[idx], d2 = in_y[idx];
    double d_n_in, d_e_in;
    if (src_north_first) {{ d_n_in = d1; d_e_in = d2; }} else {{ d_e_in = d1; d_n_in = d2; }}
    double d_e = d_e_in * x_unit_to_m;
    double d_n = d_n_in * y_unit_to_m;

    // Remove offset/scale in fp64, then convert to ds
    ds_t Cn = ds_from_double(((d_n - y0) / a - Zb) / Qn);
    ds_t Ce = ds_from_double((d_e - x0) / a / Qn);

    // Coefficients as ds
    ds_t c0=ds_from_double(cgb0),c1=ds_from_double(cgb1),c2=ds_from_double(cgb2);
    ds_t c3=ds_from_double(cgb3),c4=ds_from_double(cgb4),c5=ds_from_double(cgb5);
    ds_t u0=ds_from_double(utg0),u1=ds_from_double(utg1),u2=ds_from_double(utg2);
    ds_t u3=ds_from_double(utg3),u4=ds_from_double(utg4),u5=ds_from_double(utg5);

    // Clenshaw complex with utg
    ds_t two_Cn = ds_mul(ds_from_float(2.0f), Cn);
    ds_t sin_ar, cos_ar;
    ds_sincos(two_Cn, &sin_ar, &cos_ar);
    ds_t two_Ce = ds_mul(ds_from_float(2.0f), Ce);
    ds_t exp_2Ce = ds_exp(two_Ce);
    ds_t half_inv = ds_div(ds_from_float(0.5f), exp_2Ce);
    ds_t sinh_ai = ds_sub(ds_mul(ds_from_float(0.5f), exp_2Ce), half_inv);
    ds_t cosh_ai = ds_add(ds_mul(ds_from_float(0.5f), exp_2Ce), half_inv);

    ds_t dCn, dCe;
    ds_clenshaw_complex(u0,u1,u2,u3,u4,u5, sin_ar, cos_ar, sinh_ai, cosh_ai, &dCn, &dCe);
    Cn = ds_add(Cn, dCn);
    Ce = ds_add(Ce, dCe);

    // Complex spherical -> Gaussian
    ds_t sin_Cn, cos_Cn;
    ds_sincos(Cn, &sin_Cn, &cos_Cn);
    ds_t sinhCe = ds_sinh(Ce);
    Ce = ds_atan2(sinhCe, cos_Cn);
    ds_t modulus_Ce = ds_hypot(sinhCe, cos_Cn);
    Cn = ds_atan2(sin_Cn, modulus_Ce);

    // Gaussian -> geodetic via gatg
    ds_t tmp = ds_div(ds_mul(ds_from_float(2.0f), modulus_Ce),
                      ds_add(ds_mul(sinhCe, sinhCe), ds_from_float(1.0f)));
    ds_t sin_2Cn = ds_mul(sin_Cn, tmp);
    ds_t cos_2Cn = ds_sub(ds_mul(tmp, modulus_Ce), ds_from_float(1.0f));
    ds_t phi = ds_gatg(c0,c1,c2,c3,c4,c5, Cn, cos_2Cn, sin_2Cn);
    ds_t lam_out = Ce;

    // Add central meridian, convert to degrees
    double lat = ds_to_double(phi) * 57.29577951308232;
    double lon = (ds_to_double(lam_out) + lam0) * 57.29577951308232;

    if (dst_north_first) {{ out_x[idx] = lat; out_y[idx] = lon; }}
    else                 {{ out_x[idx] = lon; out_y[idx] = lat; }}
}}
"""
)

# ===================================================================
# Source template registry
# ===================================================================

_SOURCE_MAP = {
    ("tmerc", "forward"): (_TM_FORWARD_SOURCE, "tm_forward"),
    ("tmerc", "inverse"): (_TM_INVERSE_SOURCE, "tm_inverse"),
    ("webmerc", "forward"): (_WEBMERC_FORWARD_SOURCE, "webmerc_forward"),
    ("webmerc", "inverse"): (_WEBMERC_INVERSE_SOURCE, "webmerc_inverse"),
    ("merc", "forward"): (_MERC_FORWARD_SOURCE, "merc_forward"),
    ("merc", "inverse"): (_MERC_INVERSE_SOURCE, "merc_inverse"),
    ("lcc", "forward"): (_LCC_FORWARD_SOURCE, "lcc_forward"),
    ("lcc", "inverse"): (_LCC_INVERSE_SOURCE, "lcc_inverse"),
    ("aea", "forward"): (_AEA_FORWARD_SOURCE, "aea_forward"),
    ("aea", "inverse"): (_AEA_INVERSE_SOURCE, "aea_inverse"),
    ("stere", "forward"): (_STERE_FORWARD_SOURCE, "stere_forward"),
    ("stere", "inverse"): (_STERE_INVERSE_SOURCE, "stere_inverse"),
    ("laea", "forward"): (_LAEA_FORWARD_SOURCE, "laea_forward"),
    ("laea", "inverse"): (_LAEA_INVERSE_SOURCE, "laea_inverse"),
    ("eqc", "forward"): (_EQC_FORWARD_SOURCE, "eqc_forward"),
    ("eqc", "inverse"): (_EQC_INVERSE_SOURCE, "eqc_inverse"),
    ("sinu", "forward"): (_SINU_FORWARD_SOURCE, "sinu_forward"),
    ("sinu", "inverse"): (_SINU_INVERSE_SOURCE, "sinu_inverse"),
    ("eqearth", "forward"): (_EQEARTH_FORWARD_SOURCE, "eqearth_forward"),
    ("eqearth", "inverse"): (_EQEARTH_INVERSE_SOURCE, "eqearth_inverse"),
    ("cea", "forward"): (_CEA_FORWARD_SOURCE, "cea_forward"),
    ("cea", "inverse"): (_CEA_INVERSE_SOURCE, "cea_inverse"),
    ("ortho", "forward"): (_ORTHO_FORWARD_SOURCE, "ortho_forward"),
    ("ortho", "inverse"): (_ORTHO_INVERSE_SOURCE, "ortho_inverse"),
    ("gnom", "forward"): (_GNOM_FORWARD_SOURCE, "gnom_forward"),
    ("gnom", "inverse"): (_GNOM_INVERSE_SOURCE, "gnom_inverse"),
    ("moll", "forward"): (_MOLL_FORWARD_SOURCE, "moll_forward"),
    ("moll", "inverse"): (_MOLL_INVERSE_SOURCE, "moll_inverse"),
    ("omerc", "forward"): (_OMERC_FORWARD_SOURCE, "omerc_forward"),
    ("omerc", "inverse"): (_OMERC_INVERSE_SOURCE, "omerc_inverse"),
    ("krovak", "forward"): (_KROVAK_FORWARD_SOURCE, "krovak_forward"),
    ("krovak", "inverse"): (_KROVAK_INVERSE_SOURCE, "krovak_inverse"),
    ("eck4", "forward"): (_ECK4_FORWARD_SOURCE, "eck4_forward"),
    ("eck4", "inverse"): (_ECK4_INVERSE_SOURCE, "eck4_inverse"),
    ("eck6", "forward"): (_ECK6_FORWARD_SOURCE, "eck6_forward"),
    ("eck6", "inverse"): (_ECK6_INVERSE_SOURCE, "eck6_inverse"),
    ("sterea", "forward"): (_STEREA_FORWARD_SOURCE, "sterea_forward"),
    ("sterea", "inverse"): (_STEREA_INVERSE_SOURCE, "sterea_inverse"),
    ("geos", "forward"): (_GEOS_FORWARD_SOURCE, "geos_forward"),
    ("geos", "inverse"): (_GEOS_INVERSE_SOURCE, "geos_inverse"),
    ("robin", "forward"): (_ROBIN_FORWARD_SOURCE, "robin_forward"),
    ("robin", "inverse"): (_ROBIN_INVERSE_SOURCE, "robin_inverse"),
    ("wintri", "forward"): (_WINTRI_FORWARD_SOURCE, "wintri_forward"),
    ("wintri", "inverse"): (_WINTRI_INVERSE_SOURCE, "wintri_inverse"),
    ("natearth", "forward"): (_NATEARTH_FORWARD_SOURCE, "natearth_forward"),
    ("natearth", "inverse"): (_NATEARTH_INVERSE_SOURCE, "natearth_inverse"),
    ("aeqd", "forward"): (_AEQD_FORWARD_SOURCE, "aeqd_forward"),
    ("aeqd", "inverse"): (_AEQD_INVERSE_SOURCE, "aeqd_inverse"),
}

_PI_LITERALS = {
    "float64": "3.141592653589793238462643383279502884",
    "float32": "3.14159265f",
}

# Convergence tolerance per precision: fp64 can reach 1e-14, fp32 can only reach ~1e-7
_TOL_LITERALS = {
    "float64": "1e-14",
    "float32": "1e-7f",
}

_TYPE_MAP = {
    "float64": "double",
    "float32": "float",
}

# Exact implementation IDs target one projection-direction-precision tuple.
# The registry remains host-owned; this table only validates and materializes
# the exact implementation selected for a fused launch.
_PROJECTION_IMPLEMENTATION_TARGETS = {
    TMERC_FIXED_Q62: ("tmerc", "forward", "float64"),
    SINU_FORWARD_FIXED_Q62: ("sinu", "forward", "float64"),
    ORTHO_FORWARD_FIXED_Q62: ("ortho", "forward", "float64"),
}

_PROJECTION_FIXED_Q62_REWRITES = {
    SINU_FORWARD_FIXED_Q62: (
        "sinu_forward_fixed_q62",
        (
            (
                "cos(phi)",
                "vp_projection_fixed_cos("
                "phi, (fabs(phi) <= 0.5 * VP_PI_D && fabs(lam) <= VP_PI_D) "
                "? 0.5 * VP_PI_D : -1.0, a)",
            ),
        ),
    ),
    ORTHO_FORWARD_FIXED_Q62: (
        "ortho_forward_fixed_q62",
        (
            (
                "vp_native_sincos(phi, &sin_phi, &cos_phi);\n"
                "    double sin_lam, cos_lam;\n"
                "    vp_native_sincos(lam, &sin_lam, &cos_lam);",
                "double sin_lam, cos_lam;\n"
                "    vp_projection_fixed_sincos_pair(\n"
                "        phi, 0.5 * VP_PI_D, &sin_phi, &cos_phi,\n"
                "        lam, VP_PI_D, &sin_lam, &cos_lam, a\n"
                "    );",
            ),
        ),
    ),
}


def _build_projection_fixed_q62_source(
    projection_name: str,
    direction: str,
    implementation_id: str,
) -> tuple[str, str]:
    """Build one fp64 fused source from shared guarded Q1.62 primitives."""
    template, native_func_name = _SOURCE_MAP[(projection_name, direction)]
    function_name, replacements = _PROJECTION_FIXED_Q62_REWRITES[implementation_id]
    source = _inject_linear_unit_args(
        template.format(
            real_t="double",
            pi=_PI_LITERALS["float64"],
            tol=_TOL_LITERALS["float64"],
        )
    )
    if source.count(native_func_name) != 1:
        raise RuntimeError(
            f"Expected one {native_func_name!r} kernel while building {implementation_id!r}"
        )
    source = source.replace(native_func_name, function_name)
    for native_expression, accelerated_expression in replacements:
        if source.count(native_expression) != 1:
            raise RuntimeError(
                f"Expected one {native_expression!r} site while building {implementation_id!r}"
            )
        source = source.replace(native_expression, accelerated_expression)
    return _PROJECTION_BOUNDED_Q62_DEVICE_FNS + source, function_name


def _resolve_tmerc_forward_mode(mode: str) -> str:
    if mode not in ("fp64", "int64"):
        raise ValueError(
            f"Invalid TM forward mode (private): {mode!r}. Must be 'fp64' or 'int64'; "
            "resolve public policy through vibeproj.transcendentals."
        )
    return mode


def _tmerc_implementation_from_legacy_mode(mode: str) -> str:
    """Translate the private pre-registry override used by benchmarks/tests."""
    return TMERC_FIXED_Q62 if _resolve_tmerc_forward_mode(mode) == "int64" else NATIVE_LIBDEVICE


def _validate_projection_implementation(
    projection_name: str,
    direction: str,
    compute_dtype: str,
    implementation_id: str,
) -> None:
    if implementation_id == NATIVE_LIBDEVICE:
        return
    target = _PROJECTION_IMPLEMENTATION_TARGETS.get(implementation_id)
    if target is None:
        raise ValueError(
            f"Unsupported transcendental implementation for fused projection kernel: "
            f"{implementation_id!r}"
        )
    if (projection_name, direction, compute_dtype) != target:
        raise ValueError(
            f"{implementation_id!r} is qualified only for {target[2]} {target[0]}.{target[1]}"
        )


def _get_kernel(
    projection_name: str,
    direction: str,
    compute_dtype: str,
    *,
    transcendental_impl: str = NATIVE_LIBDEVICE,
    tmerc_mode: str | None = None,
):
    """Get or compile a fused kernel (thread-safe).

    Uses double-checked locking: the fast path (cache hit) is lock-free.
    The lock is only acquired on cache miss to serialize NVRTC compilation.

    compute_dtype: "float64", "float32", or "ds" (double-single fp32).
    I/O arrays are always double* regardless of compute precision.
    """
    import cupy as cp

    if tmerc_mode is not None and (projection_name, direction) == ("tmerc", "forward"):
        transcendental_impl = _tmerc_implementation_from_legacy_mode(tmerc_mode)
    _validate_projection_implementation(
        projection_name, direction, compute_dtype, transcendental_impl
    )
    key = (projection_name, direction, compute_dtype, transcendental_impl)
    # Fast path: lock-free read (dict reads are thread-safe in CPython)
    if key in _kernel_cache:
        return _kernel_cache[key]

    # Slow path: compile under lock
    with _kernel_cache_lock:
        # Re-check after acquiring lock (another thread may have compiled)
        if key in _kernel_cache:
            return _kernel_cache[key]

        if transcendental_impl in _PROJECTION_FIXED_Q62_REWRITES:
            source, func_name = _build_projection_fixed_q62_source(
                projection_name,
                direction,
                transcendental_impl,
            )
        elif (projection_name, direction, compute_dtype) == (
            "tmerc",
            "forward",
            "float64",
        ):
            if transcendental_impl == TMERC_FIXED_Q62:
                source = _TM_FORWARD_FIXED_Q62_SOURCE
                func_name = "tm_forward_fixed_q62"
            else:
                source = _TM_FORWARD_FP64_SOURCE
                func_name = "tm_forward_fp64"
            source = _inject_linear_unit_args(source)
        elif compute_dtype == "ds":
            ds_key = (projection_name, direction)
            if ds_key in _DS_SOURCE_MAP:
                source, func_name = _DS_SOURCE_MAP[ds_key]
                source = _inject_linear_unit_args(source.format())
            else:
                # Fallback to fp64 (RLock allows re-entrant acquisition)
                warnings.warn(
                    f"No double-single kernel for '{projection_name}' {direction}, "
                    f"falling back to fp64.",
                    stacklevel=3,
                )
                return _get_kernel(
                    projection_name,
                    direction,
                    "float64",
                    transcendental_impl=NATIVE_LIBDEVICE,
                )
        else:
            template, func_name = _SOURCE_MAP[(projection_name, direction)]
            source = _NATIVE_PAIRED_SINCOS_DEVICE_FNS + _inject_linear_unit_args(
                template.format(
                    real_t=_TYPE_MAP[compute_dtype],
                    pi=_PI_LITERALS[compute_dtype],
                    tol=_TOL_LITERALS[compute_dtype],
                )
            )

        kernel = cp.RawKernel(source, func_name)
        _kernel_cache[key] = kernel
        return kernel


# DS source map: only projections with ds-specific implementations
_DS_SOURCE_MAP = {
    ("tmerc", "forward"): (_TM_FORWARD_DS_SOURCE, "tm_forward_ds"),
    ("tmerc", "inverse"): (_TM_INVERSE_DS_SOURCE, "tm_inverse_ds"),
    # Other projections fall back to fp64 when ds is requested
}


# ===================================================================
# Public API
# ===================================================================


def compile_kernels(
    projections=None,
    *,
    precision="auto",
    projection_variants: Iterable[tuple[str, str, str]] | None = None,
    transcendental_impl=NATIVE_LIBDEVICE,
    tmerc_mode=None,
):
    """Pre-compile fused NVRTC kernels to eliminate first-call latency.

    Parameters
    ----------
    projections : list of str, optional
        Projection names to compile (e.g. ["tmerc", "webmerc"]).
        If None, compiles all supported projections.
    precision : str
        Compute precision: "auto"/"fp64"/"fp32"/"ds".
    projection_variants : iterable of tuple, optional
        Concrete ``(projection, direction, implementation_id)`` variants to
        compile. Duplicate triples are compiled once. This exact plan takes
        precedence over ``projections`` and the legacy scalar argument.
    transcendental_impl : str
        Legacy scalar exact implementation ID. It is applied only to its
        qualified projection-direction target.
    tmerc_mode : str, optional
        Private compatibility override for deterministic legacy benchmarks.
    """
    try:
        import cupy  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return
    compute_dtype = {"auto": "float64", "fp64": "float64", "fp32": "float32", "ds": "ds"}.get(
        precision, "float64"
    )
    if projection_variants is not None:
        concrete_targets = tuple(dict.fromkeys(projection_variants))
        for projection_name, direction, _implementation_id in concrete_targets:
            if (projection_name, direction) not in _SUPPORTED:
                raise ValueError(
                    f"Unsupported fused projection compile target: {(projection_name, direction)!r}"
                )
    elif projections is None:
        concrete_targets = tuple(
            (projection_name, direction, NATIVE_LIBDEVICE)
            for projection_name, direction in _SUPPORTED
        )
    else:
        concrete_targets = tuple(
            (projection_name, direction, NATIVE_LIBDEVICE)
            for projection_name in projections
            for direction in ("forward", "inverse")
            if (projection_name, direction) in _SUPPORTED
        )
    implementation_target = _PROJECTION_IMPLEMENTATION_TARGETS.get(transcendental_impl)
    if transcendental_impl != NATIVE_LIBDEVICE and implementation_target is None:
        raise ValueError(
            "Unsupported transcendental implementation for fused projection kernel: "
            f"{transcendental_impl!r}"
        )
    for proj_name, direction, planned_impl in concrete_targets:
        operation_impl = planned_impl
        if projection_variants is None and (
            implementation_target is not None
            and (proj_name, direction) == implementation_target[:2]
        ):
            operation_impl = transcendental_impl
        kernel = _get_kernel(
            proj_name,
            direction,
            compute_dtype,
            transcendental_impl=operation_impl,
            tmerc_mode=tmerc_mode,
        )
        # RawKernel construction is lazy. Explicit warm-up APIs must force
        # NVRTC now, under the same lock that protects cache population.
        with _kernel_cache_lock:
            kernel.compile()


def fused_transform(
    arg1,
    arg2,
    *,
    projection_name: str,
    direction: str,
    computed: dict,
    src_north_first: bool,
    dst_north_first: bool,
    xp,
    out_x=None,
    out_y=None,
    precision: str = "auto",
    stream=None,
    transcendental_impl: str = NATIVE_LIBDEVICE,
    tmerc_mode: str | None = None,
) -> tuple | None:
    """Execute a fused GPU kernel for the full transform pipeline.

    Parameters
    ----------
    out_x, out_y : cupy.ndarray, optional
        Pre-allocated output arrays. Pass these to avoid allocation.
    precision : str
        "fp64" = full double precision (default for fp64 input).
        "fp32" = fp32 compute with fp64 I/O (ADR-0002 mixed precision).
        "auto" = fp64 arithmetic with validated internal device strategies.
    transcendental_impl : str
        Exact implementation ID resolved by :mod:`vibeproj.transcendentals`.
        Selection is compile-time and forms part of the RawKernel cache key.
    tmerc_mode : str, optional
        Private compatibility override for deterministic legacy benchmarks.

    Mixed precision (fp32 compute, fp64 I/O) is ADR-0002 compliant:
    - Input/output arrays are always fp64 (canonical storage precision)
    - Projection math runs in fp32 for ~32x throughput on consumer GPUs
    - Final scale/offset always in fp64 for sub-meter output precision
    """
    try:
        import cupy as cp
    except ImportError:
        return None
    if xp is not cp:
        return None

    from vibeproj.exceptions import CoordinateValidationError

    n = arg1.size
    if arg2.size != n:
        raise CoordinateValidationError(
            f"arg1 and arg2 must have the same size, got {n} and {arg2.size}"
        )
    if out_x is not None and out_x.size < n:
        raise CoordinateValidationError(
            f"out_x too small: need at least {n} elements, got {out_x.size}"
        )
    if out_y is not None and out_y.size < n:
        raise CoordinateValidationError(
            f"out_y too small: need at least {n} elements, got {out_y.size}"
        )

    # Determine compute precision
    # Normalize: external names (fp64/fp32/ds/auto) → internal dtype keys (float64/float32/ds)
    if precision == "auto":
        compute_dtype = "float64"  # Internal bounded strategies preserve full accuracy.
    elif precision == "fp32":
        compute_dtype = "float32"  # raw fp32 (lossy — expert opt-in)
    elif precision == "ds":
        compute_dtype = "ds"  # double-single fp32 (fp64-equivalent accuracy)
    elif precision == "fp64":
        compute_dtype = "float64"
    else:
        raise ValueError(
            f"Invalid precision: {precision!r}. Must be 'fp64', 'fp32', 'ds', or 'auto'."
        )

    # I/O is always fp64 (ADR-0002: storage is always fp64)
    # Kernel reads double*, computes in real_t, writes double*
    io_dtype = np.float64
    if arg1.dtype != np.float64:
        arg1 = arg1.astype(np.float64)
    if arg2.dtype != np.float64:
        arg2 = arg2.astype(np.float64)

    # ds kernels take double params (the ds arithmetic is internal)
    real_t = np.float64 if compute_dtype in ("float64", "ds") else np.float32
    if (
        transcendental_impl == TMERC_FIXED_Q62
        and (projection_name, direction) == ("tmerc", "forward")
        and not computed.get("is_utm", False)
    ):
        raise ValueError(f"{TMERC_FIXED_Q62!r} is qualified only for UTM domains")
    kernel = _get_kernel(
        projection_name,
        direction,
        compute_dtype,
        transcendental_impl=transcendental_impl,
        tmerc_mode=tmerc_mode,
    )
    if out_x is None:
        out_x = cp.empty(n, dtype=io_dtype)
    elif out_x.dtype != np.float64:
        raise CoordinateValidationError(
            f"out_x must be float64 (kernel writes double*), got {out_x.dtype}"
        )
    if out_y is None:
        out_y = cp.empty(n, dtype=io_dtype)
    elif out_y.dtype != np.float64:
        raise CoordinateValidationError(
            f"out_y must be float64 (kernel writes double*), got {out_y.dtype}"
        )
    block = 256
    grid = max(1, (n + block - 1) // block)
    snf = np.int32(src_north_first)
    dnf = np.int32(dst_north_first)
    nn = np.int32(n)

    # Build args per projection
    base = (arg1, arg2, out_x, out_y)
    unit_args = (
        np.float64(computed.get("x_unit_to_m", 1.0)),
        np.float64(computed.get("y_unit_to_m", 1.0)),
    )

    def _with_units(*params):
        return base + (*params, *unit_args, snf, dnf, nn)

    try:
        if projection_name in ("webmerc", "sinu"):
            args = _with_units(
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "eqc":
            args = _with_units(
                real_t(computed["cos_lat_ts"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "merc":
            args = _with_units(
                real_t(computed["e"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "tmerc":
            if direction == "forward":
                c6 = [real_t(c) for c in computed["cbg"]]
                g6 = [real_t(c) for c in computed["gtu"]]
            else:
                c6 = [real_t(c) for c in computed["cgb"]]
                g6 = [real_t(c) for c in computed["utg"]]
            args = _with_units(
                *c6,
                *g6,
                real_t(computed["Qn"]),
                real_t(computed["Zb"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "lcc":
            args = _with_units(
                real_t(computed["n"]),
                real_t(computed["F"]),
                real_t(computed["rho0"]),
                real_t(computed["e"]),
                real_t(computed["k0"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "stere":
            args = _with_units(
                real_t(computed["akm1"]),
                real_t(computed["sign"]),
                real_t(computed["e"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "aea":
            if direction == "forward":
                args = _with_units(
                    np.float64(computed["n"]),
                    np.float64(computed["C"]),
                    np.float64(computed["rho0"]),
                    real_t(computed["e"]),
                    real_t(computed["es"]),
                    np.float64(computed["qp"]),
                    np.float64(computed["lam0"]),
                    np.float64(computed["a"]),
                    np.float64(computed["x0"]),
                    np.float64(computed["y0"]),
                )
            else:
                args = _with_units(
                    np.float64(computed["n"]),
                    np.float64(computed["C"]),
                    np.float64(computed["rho0"]),
                    real_t(computed["e"]),
                    real_t(computed["es"]),
                    np.float64(computed["qp"]),
                    real_t(computed["lam0"]),
                    np.float64(computed["a"]),
                    np.float64(computed["x0"]),
                    np.float64(computed["y0"]),
                )

        elif projection_name == "laea":
            mode_map = {"oblique": 0, "equatorial": 1, "north_pole": 2, "south_pole": 3}
            args = _with_units(
                np.int32(mode_map[computed["mode"]]),
                real_t(computed["Rq"]),
                real_t(computed["D"]),
                real_t(computed["qp"]),
                real_t(computed["sin_beta0"]),
                real_t(computed["cos_beta0"]),
                real_t(computed["phi0"]),
                real_t(computed["e"]),
                real_t(computed["es"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "eqearth":
            if direction == "forward":
                args = _with_units(
                    real_t(computed["e"]),
                    real_t(computed["qp"]),
                    real_t(computed["rqda"]),
                    real_t(computed["lam0"]),
                    real_t(computed["a"]),
                    real_t(computed["x0"]),
                    real_t(computed["y0"]),
                )
            else:
                args = _with_units(
                    real_t(computed["e"]),
                    real_t(computed["es"]),
                    real_t(computed["qp"]),
                    real_t(computed["rqda"]),
                    real_t(computed["lam0"]),
                    real_t(computed["a"]),
                    real_t(computed["x0"]),
                    real_t(computed["y0"]),
                )

        elif projection_name == "omerc":
            args = _with_units(
                real_t(computed["e"]),
                real_t(computed["B"]),
                real_t(computed["A_norm"]),
                real_t(computed["H"]),
                real_t(computed["sin_gamma0"]),
                real_t(computed["cos_gamma0"]),
                real_t(computed["sin_gamma_c"]),
                real_t(computed["cos_gamma_c"]),
                real_t(computed["u_c"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "krovak":
            args = _with_units(
                real_t(computed["e"]),
                real_t(computed["B"]),
                real_t(computed["k"]),
                real_t(computed["n"]),
                real_t(computed["r_0_norm"]),
                real_t(computed["tan_half_p"]),
                real_t(computed["sin_alpha_c"]),
                real_t(computed["cos_alpha_c"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name in ("moll", "eck4", "eck6"):
            args = _with_units(
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "cea":
            if direction == "forward":
                args = _with_units(
                    real_t(computed["e"]),
                    np.float64(computed["qp"]),
                    np.float64(computed["k0"]),
                    np.float64(computed["lam0"]),
                    np.float64(computed["a"]),
                    np.float64(computed["x0"]),
                    np.float64(computed["y0"]),
                )
            else:
                args = _with_units(
                    real_t(computed["e"]),
                    real_t(computed["es"]),
                    np.float64(computed["qp"]),
                    np.float64(computed["k0"]),
                    real_t(computed["lam0"]),
                    np.float64(computed["a"]),
                    np.float64(computed["x0"]),
                    np.float64(computed["y0"]),
                )

        elif projection_name in ("ortho", "gnom", "aeqd"):
            args = _with_units(
                real_t(computed["sin_phi0"]),
                real_t(computed["cos_phi0"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "sterea":
            args = _with_units(
                real_t(computed["e"]),
                real_t(computed["n"]),
                real_t(computed["c"]),
                real_t(computed["R"]),
                real_t(computed["sin_chi0"]),
                real_t(computed["cos_chi0"]),
                real_t(computed["k0"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "geos":
            args = _with_units(
                real_t(computed["H"]),
                real_t(computed["h"]),
                real_t(computed["r_eq2"]),
                real_t(computed["r_pol2"]),
                np.int32(computed["sweep_axis"] == "x"),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name in ("robin", "natearth"):
            args = _with_units(
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        elif projection_name == "wintri":
            args = _with_units(
                real_t(computed["cos_phi1"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
            )

        else:
            raise ValueError(f"Unrecognized fused kernel projection: {projection_name!r}")
    except KeyError as exc:
        raise KeyError(
            f"Missing computed parameter {exc} for projection {projection_name!r}. "
            f"Check that the projection's setup() populates all required keys."
        ) from exc

    if stream is not None:
        with stream:
            kernel((grid,), (block,), args)
        # No synchronize() — caller owns the stream lifecycle.
        # Synchronizing here would stall the pipeline between Helmert
        # and projection kernels that are sequenced on the same stream.
    else:
        kernel((grid,), (block,), args)
    return out_x, out_y


# ===================================================================
# Helmert datum shift kernel
# ===================================================================

_HELMERT_SHIFT_BODY = """\
#ifndef VP_HELMERT_KERNEL_NAME
#define VP_HELMERT_KERNEL_NAME helmert_shift
#endif
#ifndef VP_HELMERT_SINCOS
#define VP_HELMERT_SINCOS(angle, sin_out, cos_out) sincos(angle, sin_out, cos_out)
#endif
#ifndef VP_HELMERT_SIN
#define VP_HELMERT_SIN(angle) sin(angle)
#endif

extern "C" __global__ void __launch_bounds__(256) VP_HELMERT_KERNEL_NAME(
    const double* __restrict__ in_lat,
    const double* __restrict__ in_lon,
    double* __restrict__ out_lat,
    double* __restrict__ out_lon,
    const double* __restrict__ in_h,
    double* __restrict__ out_h,
    double src_a, double src_es,
    double dst_a, double dst_es,
    double tx, double ty, double tz,
    double rx, double ry, double rz,
    double ds,
    int n,
    int has_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double lat = in_lat[idx] * 0.017453292519943295;
    double lon = in_lon[idx] * 0.017453292519943295;

    /* Geodetic -> ECEF on source ellipsoid */
    double sin_lat, cos_lat, sin_lon, cos_lon;
    VP_HELMERT_SINCOS(lat, &sin_lat, &cos_lat);
    VP_HELMERT_SINCOS(lon, &sin_lon, &cos_lon);
    double N = src_a / sqrt(1.0 - src_es * sin_lat * sin_lat);

    double X, Y, Z;
    if (has_z) {
        double h_val = in_h[idx];
        X = (N + h_val) * cos_lat * cos_lon;
        Y = (N + h_val) * cos_lat * sin_lon;
        Z = (N * (1.0 - src_es) + h_val) * sin_lat;
    } else {
        X = N * cos_lat * cos_lon;
        Y = N * cos_lat * sin_lon;
        Z = N * (1.0 - src_es) * sin_lat;
    }

    /* Helmert: X' = ds * R * X + T  (Position Vector convention) */
    double X2 = ds * ( X - rz * Y + ry * Z) + tx;
    double Y2 = ds * ( rz * X +  Y - rx * Z) + ty;
    double Z2 = ds * (-ry * X + rx * Y +  Z) + tz;

    /* ECEF -> Geodetic on destination ellipsoid (Bowring iterative) */
    double p = sqrt(X2 * X2 + Y2 * Y2);
    double lon_out = atan2(Y2, X2);
    double lat_out = atan2(Z2, p * (1.0 - dst_es));

    for (int i = 0; i < 10; i++) {
        double sin_lat_i = VP_HELMERT_SIN(lat_out);
        double N_i = dst_a / sqrt(1.0 - dst_es * sin_lat_i * sin_lat_i);
        double lat_new = atan2(Z2 + dst_es * N_i * sin_lat_i, p);
        if (fabs(lat_new - lat_out) < 1e-14) { lat_out = lat_new; break; }
        lat_out = lat_new;
    }

    out_lat[idx] = lat_out * 57.29577951308232;
    out_lon[idx] = lon_out * 57.29577951308232;

    if (has_z) {
        /* Recover ellipsoidal height on destination ellipsoid */
        double sin_lat_f, cos_lat_f;
        VP_HELMERT_SINCOS(lat_out, &sin_lat_f, &cos_lat_f);
        double N_f = dst_a / sqrt(1.0 - dst_es * sin_lat_f * sin_lat_f);
        double h_out_val;
        if (fabs(cos_lat_f) < 1e-10) {
            /* Near-pole: use Z-based formula */
            h_out_val = fabs(Z2) / fabs(sin_lat_f) - N_f * (1.0 - dst_es);
        } else {
            h_out_val = p / cos_lat_f - N_f;
        }
        out_h[idx] = h_out_val;
    }
}
"""

_HELMERT_SHIFT_SOURCE = _HELMERT_SHIFT_BODY
_HELMERT_SHIFT_FIXED_Q62_SOURCE = (
    _HELMERT_FIXED_Q62_DEVICE_FNS
    + """
#define VP_HELMERT_KERNEL_NAME helmert_shift_fixed_q62
#define VP_HELMERT_SINCOS(angle, sin_out, cos_out) \\
    vp_helmert_fixed_q62_sincos(angle, sin_out, cos_out)
#define VP_HELMERT_SIN(angle) vp_helmert_fixed_q62_sin(angle)
"""
    + _HELMERT_SHIFT_BODY
)

_helmert_kernel_cache: dict[str, object] = {}
_helmert_kernel_lock = threading.Lock()


def _resolve_helmert_trig_mode(trig_mode: str) -> str:
    if trig_mode not in ("fp64", "int64"):
        raise ValueError(
            f"Invalid Helmert trig mode (private): {trig_mode!r}. Must be 'fp64' or 'int64'; "
            "resolve public policy through vibeproj.transcendentals."
        )
    return trig_mode


def _helmert_implementation_from_legacy_mode(mode: str) -> str:
    """Translate the private pre-registry override used by benchmarks/tests."""
    return HELMERT_FIXED_Q62 if _resolve_helmert_trig_mode(mode) == "int64" else NATIVE_LIBDEVICE


def _get_helmert_kernel(
    transcendental_impl: str = NATIVE_LIBDEVICE,
    *,
    trig_mode: str | None = None,
):
    """Get or compile a Helmert datum shift kernel (thread-safe)."""
    if trig_mode is not None:
        transcendental_impl = _helmert_implementation_from_legacy_mode(trig_mode)
    if transcendental_impl not in (NATIVE_LIBDEVICE, HELMERT_FIXED_Q62):
        raise ValueError(
            f"Unsupported transcendental implementation for Helmert kernel: {transcendental_impl!r}"
        )
    if transcendental_impl in _helmert_kernel_cache:
        return _helmert_kernel_cache[transcendental_impl]

    import cupy as cp

    with _helmert_kernel_lock:
        if transcendental_impl in _helmert_kernel_cache:
            return _helmert_kernel_cache[transcendental_impl]
        if transcendental_impl == HELMERT_FIXED_Q62:
            source = _HELMERT_SHIFT_FIXED_Q62_SOURCE
            function_name = "helmert_shift_fixed_q62"
        else:
            source = _HELMERT_SHIFT_SOURCE
            function_name = "helmert_shift"
        kernel = cp.RawKernel(source, function_name)
        _helmert_kernel_cache[transcendental_impl] = kernel
        return kernel


def compile_helmert_kernel(
    *,
    transcendental_impl: str = NATIVE_LIBDEVICE,
    trig_mode: str | None = None,
):
    """Pre-compile the selected Helmert datum shift kernel."""
    try:
        import cupy  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return
    kernel = _get_helmert_kernel(transcendental_impl, trig_mode=trig_mode)
    with _helmert_kernel_lock:
        kernel.compile()


def fused_helmert_shift(
    lat,
    lon,
    helmert_params,
    xp,
    h=None,
    out_lat=None,
    out_lon=None,
    out_h=None,
    stream=None,
    transcendental_impl: str = NATIVE_LIBDEVICE,
    trig_mode: str | None = None,
):
    """Execute the Helmert datum shift on GPU.

    Parameters
    ----------
    lat, lon : cupy.ndarray
        Geographic coordinates in degrees on the source ellipsoid.
    helmert_params : HelmertParams
        Transformation parameters.
    xp : module
        Array module (must be cupy).
    h : cupy.ndarray, optional
        Ellipsoidal height in meters. When provided, height is transformed.
    out_lat, out_lon : cupy.ndarray, optional
        Pre-allocated output arrays.
    out_h : cupy.ndarray, optional
        Pre-allocated output height array (only used when h is not None).
    stream : cupy.cuda.Stream, optional
        CUDA stream for async execution.
    transcendental_impl : str
        Exact implementation ID resolved by :mod:`vibeproj.transcendentals`.
        Selection is compile-time and forms part of the RawKernel cache key.
    trig_mode : str, optional
        Private compatibility override for deterministic legacy benchmarks.

    Returns
    -------
    (out_lat, out_lon) or (out_lat, out_lon, out_h) or None if not on GPU.
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    if xp is not cp:
        return None

    kernel = _get_helmert_kernel(transcendental_impl, trig_mode=trig_mode)

    from vibeproj.exceptions import CoordinateValidationError

    n = lat.size
    if lon.size != n:
        raise CoordinateValidationError(
            f"lat and lon must have the same size, got {n} and {lon.size}"
        )
    if out_lat is None:
        out_lat = cp.empty(n, dtype=cp.float64)
    elif out_lat.dtype != np.float64:
        raise CoordinateValidationError(
            f"out_lat must be float64 (kernel writes double*), got {out_lat.dtype}"
        )
    if out_lat is not None and out_lat.size < n:
        raise CoordinateValidationError(
            f"out_lat too small: need at least {n} elements, got {out_lat.size}"
        )
    if out_lon is None:
        out_lon = cp.empty(n, dtype=cp.float64)
    elif out_lon.dtype != np.float64:
        raise CoordinateValidationError(
            f"out_lon must be float64 (kernel writes double*), got {out_lon.dtype}"
        )
    if out_lon is not None and out_lon.size < n:
        raise CoordinateValidationError(
            f"out_lon too small: need at least {n} elements, got {out_lon.size}"
        )

    has_z = h is not None
    if has_z:
        if h.size != n:
            raise CoordinateValidationError(
                f"lat and h must have the same size, got {n} and {h.size}"
            )
        if out_h is None:
            out_h = cp.empty(n, dtype=cp.float64)
        elif out_h.dtype != np.float64:
            raise CoordinateValidationError(
                f"out_h must be float64 (kernel writes double*), got {out_h.dtype}"
            )
        if out_h is not None and out_h.size < n:
            raise CoordinateValidationError(
                f"out_h too small: need at least {n} elements, got {out_h.size}"
            )
        in_h = h
    else:
        # Dummy pointers — kernel will not read/write these when has_z=0
        in_h = lat
        out_h_dummy = out_lat

    block = 256
    grid = max(1, (n + block - 1) // block)

    src = helmert_params.src_ellipsoid
    dst = helmert_params.dst_ellipsoid

    args = (
        lat,
        lon,
        out_lat,
        out_lon,
        in_h,
        out_h if has_z else out_h_dummy,
        np.float64(src.a),
        np.float64(src.es),
        np.float64(dst.a),
        np.float64(dst.es),
        np.float64(helmert_params.tx),
        np.float64(helmert_params.ty),
        np.float64(helmert_params.tz),
        np.float64(helmert_params.rx),
        np.float64(helmert_params.ry),
        np.float64(helmert_params.rz),
        np.float64(helmert_params.ds),
        np.int32(n),
        np.int32(1 if has_z else 0),
    )

    if stream is not None:
        with stream:
            kernel((grid,), (block,), args)
        # No synchronize() — caller owns the stream lifecycle.
        # When Helmert feeds into a projection kernel on the same stream,
        # CUDA execution order is guaranteed without explicit sync.
    else:
        kernel((grid,), (block,), args)

    if has_z:
        return out_lat, out_lon, out_h
    return out_lat, out_lon


# ===================================================================
# SVD datum correction kernel
# ===================================================================

_SVD_CORRECTION_SOURCE = """\
extern "C" __global__ void __launch_bounds__(256) svd_correction(
    const double* __restrict__ in_lat,
    const double* __restrict__ in_lon,
    double* __restrict__ out_lat,
    double* __restrict__ out_lon,
    const double* __restrict__ u_lat,
    const double* __restrict__ s_lat,
    const double* __restrict__ vt_lat,
    const double* __restrict__ u_lon,
    const double* __restrict__ s_lon,
    const double* __restrict__ vt_lon,
    double lat_min, double lat_scale,
    double lon_min, double lon_scale,
    int n_lat_grid, int n_lon_grid,
    int rank,
    int n,
    int negate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double lat = in_lat[idx];
    double lon = in_lon[idx];

    /* Fractional grid indices */
    double lat_idx = (lat - lat_min) * lat_scale;
    double lon_idx = (lon - lon_min) * lon_scale;

    /* Clamp to valid grid range */
    if (lat_idx < 0.0) lat_idx = 0.0;
    if (lat_idx > (double)(n_lat_grid - 1)) lat_idx = (double)(n_lat_grid - 1);
    if (lon_idx < 0.0) lon_idx = 0.0;
    if (lon_idx > (double)(n_lon_grid - 1)) lon_idx = (double)(n_lon_grid - 1);

    /* Integer neighbors and interpolation fractions */
    int i0 = (int)lat_idx;
    int j0 = (int)lon_idx;
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    if (i1 >= n_lat_grid) i1 = n_lat_grid - 1;
    if (j1 >= n_lon_grid) j1 = n_lon_grid - 1;
    double t = lat_idx - (double)i0;   /* lat interpolation fraction */
    double s = lon_idx - (double)j0;   /* lon interpolation fraction */

    /* Accumulate SVD correction: sum_k S[k] * lerp(U_k, lat) * lerp(V_k, lon) */
    double dlat = 0.0;
    double dlon = 0.0;

    for (int k = 0; k < rank; k++) {
        int row_lat = k * n_lat_grid;
        int row_lon = k * n_lon_grid;

        /* Lat component: lerp U_lat_k at lat_idx, lerp Vt_lat_k at lon_idx */
        double u_lat_interp = (1.0 - t) * u_lat[row_lat + i0] + t * u_lat[row_lat + i1];
        double v_lat_interp = (1.0 - s) * vt_lat[row_lon + j0] + s * vt_lat[row_lon + j1];
        dlat += s_lat[k] * u_lat_interp * v_lat_interp;

        /* Lon component: lerp U_lon_k at lat_idx, lerp Vt_lon_k at lon_idx */
        double u_lon_interp = (1.0 - t) * u_lon[row_lat + i0] + t * u_lon[row_lat + i1];
        double v_lon_interp = (1.0 - s) * vt_lon[row_lon + j0] + s * vt_lon[row_lon + j1];
        dlon += s_lon[k] * u_lon_interp * v_lon_interp;
    }

    /* Convert arcseconds to degrees */
    dlat *= (1.0 / 3600.0);
    dlon *= (1.0 / 3600.0);

    /* Apply correction (add or subtract) */
    if (negate) {
        out_lat[idx] = lat - dlat;
        out_lon[idx] = lon - dlon;
    } else {
        out_lat[idx] = lat + dlat;
        out_lon[idx] = lon + dlon;
    }
}
"""

_svd_kernel_cache = None
_svd_kernel_lock = threading.Lock()


def _get_svd_kernel():
    """Get or compile the SVD datum correction kernel (thread-safe)."""
    global _svd_kernel_cache
    if _svd_kernel_cache is not None:
        return _svd_kernel_cache

    import cupy as cp

    with _svd_kernel_lock:
        if _svd_kernel_cache is not None:
            return _svd_kernel_cache
        kernel = cp.RawKernel(_SVD_CORRECTION_SOURCE, "svd_correction")
        _svd_kernel_cache = kernel
        return kernel


def compile_svd_kernel():
    """Pre-compile the SVD datum correction kernel.

    Call during warm-up to avoid first-use compilation latency.
    """
    try:
        import cupy  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return
    kernel = _get_svd_kernel()
    with _svd_kernel_lock:
        kernel.compile()


# Device array cache for SVD coefficients.
# Keyed by (CUDA device ID, id(DatumCorrectionData)). We store a strong
# correction reference and a readiness event alongside the arrays so IDs
# cannot be recycled and cross-stream first use remains ordered without a
# device-wide synchronization.
_svd_device_cache: dict[tuple[int, int], tuple] = {}
_svd_device_cache_lock = threading.Lock()


def _svd_device_cache_key(correction, device_id: int) -> tuple[int, int]:
    return device_id, id(correction)


def _validate_svd_coefficient_devices(arrays, device_id: int) -> None:
    """Reject a corrupt/cross-device coefficient cache entry."""
    names = ("u_lat", "s_lat", "vt_lat", "u_lon", "s_lon", "vt_lon")
    for name, array in zip(names, arrays, strict=True):
        array_device_id = int(array.device.id)
        if array_device_id != device_id:
            raise ValueError(
                f"SVD coefficient {name} is on CUDA device {array_device_id}, "
                f"expected device {device_id}"
            )


def _resolve_svd_stream(cp, stream, device_id: int):
    """Resolve a stream for *device_id*, accepting CUDA null/legacy streams."""
    with cp.cuda.Device(device_id):
        target_stream = stream if stream is not None else cp.cuda.get_current_stream()
    stream_device_id = int(target_stream.device_id)
    # CuPy reports -1 for the CUDA null/legacy stream. Its device is the
    # surrounding CUDA device context, so normalize it to the launch device.
    if stream_device_id not in (-1, device_id):
        raise ValueError(
            f"SVD stream is on CUDA device {stream_device_id}, expected device {device_id}"
        )
    return target_stream


def _wait_for_svd_coefficients(key, cached, target_stream, device_id: int):
    """Order incomplete first-use copies without polluting steady-state capture."""
    ready_event = cached[2]
    if ready_event is None:
        return cached

    import cupy as cp

    with cp.cuda.Device(device_id):
        if not ready_event.done:
            target_stream.wait_event(ready_event)
            return cached

    # Once readiness is observed, discard the event under the cache lock.
    # Subsequent steady-state calls perform no CUDA event operation, which is
    # required when the consuming transform is captured into a CUDA graph.
    completed = (cached[0], cached[1], None)
    with _svd_device_cache_lock:
        if _svd_device_cache.get(key) is cached:
            _svd_device_cache[key] = completed
            return completed
        return _svd_device_cache[key]


def _get_svd_device_arrays(correction, *, device_id=None, stream=None):
    """Lazily transfer SVD coefficients to device (thread-safe, cached).

    Returns a tuple of 6 CuPy arrays:
        (u_lat_d, s_lat_d, vt_lat_d, u_lon_d, s_lon_d, vt_lon_d)
    All are contiguous fp64 arrays on *device_id*. First-use transfers are
    enqueued on *stream*, and every consumer waits on the cached readiness
    event without synchronizing the device or host.
    """
    import cupy as cp

    if device_id is None:
        device_id = int(cp.cuda.runtime.getDevice())
    device_id = int(device_id)
    target_stream = _resolve_svd_stream(cp, stream, device_id)

    key = _svd_device_cache_key(correction, device_id)
    cached = _svd_device_cache.get(key)
    if cached is not None:
        _validate_svd_coefficient_devices(cached[1], device_id)
        cached = _wait_for_svd_coefficients(key, cached, target_stream, device_id)
        return cached[1]  # (strong_ref, arrays, ready_event) -> arrays

    with _svd_device_cache_lock:
        cached = _svd_device_cache.get(key)
        if cached is None:
            # Flatten rank x n into contiguous 1D arrays (row-major).
            u_lat_flat = np.array([v for row in correction.u_lat for v in row], dtype=np.float64)
            vt_lat_flat = np.array([v for row in correction.vt_lat for v in row], dtype=np.float64)
            s_lat_flat = np.array(correction.s_lat, dtype=np.float64)

            u_lon_flat = np.array([v for row in correction.u_lon for v in row], dtype=np.float64)
            vt_lon_flat = np.array([v for row in correction.vt_lon for v in row], dtype=np.float64)
            s_lon_flat = np.array(correction.s_lon, dtype=np.float64)

            with cp.cuda.Device(device_id), target_stream:
                arrays = (
                    cp.asarray(u_lat_flat),
                    cp.asarray(s_lat_flat),
                    cp.asarray(vt_lat_flat),
                    cp.asarray(u_lon_flat),
                    cp.asarray(s_lon_flat),
                    cp.asarray(vt_lon_flat),
                )
                ready_event = cp.cuda.Event()
                ready_event.record(target_stream)
            _validate_svd_coefficient_devices(arrays, device_id)
            cached = (correction, arrays, ready_event)
            _svd_device_cache[key] = cached

    _validate_svd_coefficient_devices(cached[1], device_id)
    cached = _wait_for_svd_coefficients(key, cached, target_stream, device_id)
    return cached[1]


def fused_svd_correction(
    lat,
    lon,
    correction_data,
    xp,
    *,
    negate=False,
    out_lat=None,
    out_lon=None,
    stream=None,
):
    """Execute SVD datum correction on GPU.

    Parameters
    ----------
    lat, lon : cupy.ndarray
        Geographic coordinates in degrees.
    correction_data : DatumCorrectionData
        SVD-compressed datum correction coefficients.
    xp : module
        Array module (must be cupy).
    negate : bool
        If True, subtract the correction (inverse direction).
    out_lat, out_lon : cupy.ndarray, optional
        Pre-allocated output arrays (zero-copy support).
    stream : cupy.cuda.Stream, optional
        CUDA stream for async execution.

    Returns
    -------
    (out_lat, out_lon) or None if not on GPU / compilation fails.
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    if xp is not cp:
        return None

    from vibeproj.exceptions import CoordinateValidationError

    n = lat.size
    if lon.size != n:
        raise CoordinateValidationError(
            f"lat and lon must have the same size, got {n} and {lon.size}"
        )

    launch_device_id = int(lat.device.id)
    lon_device_id = int(lon.device.id)
    if lon_device_id != launch_device_id:
        raise CoordinateValidationError(
            f"lon is on CUDA device {lon_device_id}, expected device {launch_device_id}"
        )

    try:
        target_stream = _resolve_svd_stream(cp, stream, launch_device_id)
    except ValueError as exc:
        raise CoordinateValidationError(str(exc)) from exc

    if out_lat is not None and int(out_lat.device.id) != launch_device_id:
        raise CoordinateValidationError(
            f"out_lat is on CUDA device {out_lat.device.id}, expected device {launch_device_id}"
        )
    if out_lon is not None and int(out_lon.device.id) != launch_device_id:
        raise CoordinateValidationError(
            f"out_lon is on CUDA device {out_lon.device.id}, expected device {launch_device_id}"
        )

    with cp.cuda.Device(launch_device_id), target_stream:
        if not lat.flags.c_contiguous:
            lat = cp.ascontiguousarray(lat)
        if not lon.flags.c_contiguous:
            lon = cp.ascontiguousarray(lon)

    if lat.dtype != np.float64:
        raise CoordinateValidationError(
            f"lat must be float64 (kernel reads double*), got {lat.dtype}"
        )
    if lon.dtype != np.float64:
        raise CoordinateValidationError(
            f"lon must be float64 (kernel reads double*), got {lon.dtype}"
        )

    if out_lat is None:
        with cp.cuda.Device(launch_device_id), target_stream:
            out_lat = cp.empty(n, dtype=cp.float64)
    elif out_lat.dtype != np.float64:
        raise CoordinateValidationError(
            f"out_lat must be float64 (kernel writes double*), got {out_lat.dtype}"
        )
    if out_lat.size < n:
        raise CoordinateValidationError(
            f"out_lat too small: need at least {n} elements, got {out_lat.size}"
        )

    if out_lon is None:
        with cp.cuda.Device(launch_device_id), target_stream:
            out_lon = cp.empty(n, dtype=cp.float64)
    elif out_lon.dtype != np.float64:
        raise CoordinateValidationError(
            f"out_lon must be float64 (kernel writes double*), got {out_lon.dtype}"
        )
    if out_lon.size < n:
        raise CoordinateValidationError(
            f"out_lon too small: need at least {n} elements, got {out_lon.size}"
        )

    # Get device arrays on the launch device. First-use transfers and the
    # consuming kernel are ordered on the supplied stream without host sync.
    try:
        u_lat_d, s_lat_d, vt_lat_d, u_lon_d, s_lon_d, vt_lon_d = _get_svd_device_arrays(
            correction_data,
            device_id=launch_device_id,
            stream=target_stream,
        )
        _validate_svd_coefficient_devices(
            (u_lat_d, s_lat_d, vt_lat_d, u_lon_d, s_lon_d, vt_lon_d),
            launch_device_id,
        )
    except ValueError as exc:
        raise CoordinateValidationError(str(exc)) from exc

    with cp.cuda.Device(launch_device_id), target_stream:
        try:
            kernel = _get_svd_kernel()
        except Exception:
            return None

    # Pre-compute grid scaling factors
    lat_min, lat_max, lon_min, lon_max = correction_data.bbox
    n_lat = correction_data.n_lat
    n_lon = correction_data.n_lon
    lat_scale = (n_lat - 1) / (lat_max - lat_min)
    lon_scale = (n_lon - 1) / (lon_max - lon_min)

    block = 256
    grid = max(1, (n + block - 1) // block)

    args = (
        lat,
        lon,
        out_lat,
        out_lon,
        u_lat_d,
        s_lat_d,
        vt_lat_d,
        u_lon_d,
        s_lon_d,
        vt_lon_d,
        np.float64(lat_min),
        np.float64(lat_scale),
        np.float64(lon_min),
        np.float64(lon_scale),
        np.int32(n_lat),
        np.int32(n_lon),
        np.int32(correction_data.rank),
        np.int32(n),
        np.int32(1 if negate else 0),
    )

    with cp.cuda.Device(launch_device_id), target_stream:
        kernel((grid,), (block,), args)

    return out_lat, out_lon
