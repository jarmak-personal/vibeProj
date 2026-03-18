"""Fused NVRTC kernels for GPU-accelerated coordinate projection.

Each kernel runs the full transform pipeline (axis swap, deg/rad, central meridian,
projection math, scale, offset) in a single kernel launch — one thread per coordinate.
This eliminates ~20 intermediate kernel launches and array round-trips compared to
the CuPy element-wise path.

Uses CuPy RawKernel for NVRTC compilation and caching.
"""

from __future__ import annotations

import threading

import numpy as np

# Kernel cache: (projection_name, direction, dtype_name) -> RawKernel
# Protected by _kernel_cache_lock for thread-safe compilation.
_kernel_cache: dict[tuple[str, str, str], object] = {}
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

# ds gatg + ds clenshaw_complex for TM
_DS_TM_DEVICE_FNS = (
    _DS_ARITH
    + """
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
        if (fabs(dphi) < ({real_t})1e-14) break;
    }}
    return phi;
}}
"""

# -- Equal-area helpers (qsfn, phi_from_q) --
_EA_DEVICE_FNS = """
__device__ inline {real_t} qsfn({real_t} sin_phi, {real_t} e) {{
    {real_t} e_sin = e * sin_phi;
    {real_t} one_minus_e2 = ({real_t})1.0 - e * e;
    return one_minus_e2 * (sin_phi / (({real_t})1.0 - e_sin * e_sin)
           - (({real_t})0.5 / e) * log((({real_t})1.0 - e_sin) / (({real_t})1.0 + e_sin)));
}}
__device__ inline {real_t} phi_from_q({real_t} q, {real_t} e, {real_t} es) {{
    {real_t} phi = asin(fmin(fmax(q / ({real_t})2.0, ({real_t})-1.0), ({real_t})1.0));
    for (int i = 0; i < 15; i++) {{
        {real_t} sin_phi = sin(phi);
        {real_t} e_sin = e * sin_phi;
        {real_t} one_minus = ({real_t})1.0 - e_sin * e_sin;
        {real_t} dphi = (one_minus * one_minus / (({real_t})2.0 * cos(phi)))
            * (q / (({real_t})1.0 - es) - sin_phi / one_minus
               + (({real_t})0.5 / e) * log((({real_t})1.0 - e_sin) / (({real_t})1.0 + e_sin)));
        phi += dphi;
        if (fabs(dphi) < ({real_t})1e-14) break;
    }}
    return phi;
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
    lam = lam - TWO_PI * rint(lam / TWO_PI);
"""

_FWD_POSTAMBLE = """
    if (dst_north_first) {{ out_x[idx] = (double)northing; out_y[idx] = (double)easting; }}
    else                 {{ out_x[idx] = (double)easting;  out_y[idx] = (double)northing; }}
"""

_INV_PREAMBLE = """
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double d_arg1 = in_x[idx], d_arg2 = in_y[idx];
    double d_northing, d_easting;
    if (src_north_first) {{ d_northing = d_arg1; d_easting = d_arg2; }} else {{ d_easting = d_arg1; d_northing = d_arg2; }}
    // Offset/scale removal in fp64 before cast to compute precision
    {real_t} cx = ({real_t})((d_easting - (double)x0) / (double)a);
    {real_t} cy = ({real_t})((d_northing - (double)y0) / (double)a);
"""

_INV_POSTAMBLE = """
    lam = lam + ({real_t})lam0;
    const {real_t} TWO_PI_i = ({real_t})6.283185307179586;
    lam = lam - TWO_PI_i * rint(lam / TWO_PI_i);
    double d_lat = (double)phi * 57.29577951308232;
    double d_lon = (double)lam * 57.29577951308232;
    if (dst_north_first) {{ out_x[idx] = d_lat; out_y[idx] = d_lon; }}
    else                 {{ out_x[idx] = d_lon; out_y[idx] = d_lat; }}
"""

_FWD_SIGNATURE = """
extern "C" __global__ void {func}(
    const double* __restrict__ in_x, const double* __restrict__ in_y,
    double* __restrict__ out_x, double* __restrict__ out_y,
"""

_INV_SIGNATURE = _FWD_SIGNATURE  # same

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
    {real_t} easting  = lam * cos_lat_ts * a + x0;
    {real_t} northing = phi * a + y0;
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
    {real_t} easting  = lam * cos(phi) * a + x0;
    {real_t} northing = phi * a + y0;
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
    {real_t} easting  = lam * a + x0;
    {real_t} northing = y_proj * a + y0;
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
    for (int i = 0; i < 7; i++) {{
        {real_t} e_sin = e * sin(phi);
        phi = ({real_t})2.0 * atan(exp(cy) * pow((({real_t})1.0 + e_sin) / (({real_t})1.0 - e_sin), ({real_t})0.5 * e)) - ({real_t})0.5 * {pi};
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
    {real_t} easting  = lam * a + x0;
    {real_t} northing = log(tan(({real_t})0.25 * {pi} + ({real_t})0.5 * phi)) * a + y0;
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
    {real_t} Cn = gatg(cbg0, cbg1, cbg2, cbg3, cbg4, cbg5,
                       phi, cos(({real_t})2.0 * phi), sin(({real_t})2.0 * phi));
    {real_t} sin_Cn = sin(Cn), cos_Cn = cos(Cn);
    {real_t} sin_Ce = sin(lam), cos_Ce = cos(lam);
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
    {real_t} easting  = Qn * Ce * a + x0;
    {real_t} northing = (Qn * Cn + Zb) * a + y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
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
    {real_t} sin_arg_r = sin(({real_t})2.0 * Cn), cos_arg_r = cos(({real_t})2.0 * Cn);
    {real_t} exp_2_Ce = exp(({real_t})2.0 * Ce);
    {real_t} half_inv = ({real_t})0.5 / exp_2_Ce;
    {real_t} sinh_arg_i = ({real_t})0.5 * exp_2_Ce - half_inv;
    {real_t} cosh_arg_i = ({real_t})0.5 * exp_2_Ce + half_inv;
    {real_t} dCn, dCe;
    clenshaw_complex(utg0, utg1, utg2, utg3, utg4, utg5,
                     sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);
    Cn += dCn; Ce += dCe;
    {real_t} sin_Cn = sin(Cn), cos_Cn = cos(Cn), sinhCe = sinh(Ce);
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
    {real_t} easting  = rho * sin(theta) * a + x0;
    {real_t} northing = (rho0 - rho * cos(theta)) * a + y0;
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
    {real_t} rho = sqrt(cx * cx + (rho0 - cy) * (rho0 - cy));
    if (nn < ({real_t})0.0) rho = -rho;
    {real_t} theta = atan2(cx, rho0 - cy);
    if (nn < ({real_t})0.0) theta = -theta;
    {real_t} lam = theta / nn;
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
    {real_t} lam_adj = sign * lam;
    {real_t} easting  = rho * sin(lam_adj) * a + x0;
    {real_t} northing = (-sign * rho * cos(lam_adj)) * a + y0;
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
    {real_t} lam = sign * atan2(cx, y_adj);
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
    {real_t} nn, {real_t} C, {real_t} rho0, {real_t} e, {real_t} es,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} q = qsfn(sin(phi), e);
    {real_t} rho_sq = C - nn * q;
    if (rho_sq < ({real_t})0.0) rho_sq = ({real_t})0.0;
    {real_t} rho = sqrt(rho_sq) / nn;
    {real_t} theta = nn * lam;
    {real_t} easting  = rho * sin(theta) * a + x0;
    {real_t} northing = (rho0 - rho * cos(theta)) * a + y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_AEA_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="aea_inverse", real_t="{real_t}")
    + """
    {real_t} nn, {real_t} C, {real_t} rho0, {real_t} e, {real_t} es,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} rho = sqrt(cx * cx + (rho0 - cy) * (rho0 - cy));
    if (nn < ({real_t})0.0) rho = -rho;
    {real_t} theta = atan2(cx, rho0 - cy);
    if (nn < ({real_t})0.0) theta = -theta;
    {real_t} lam = theta / nn;
    {real_t} q = (C - (rho * nn) * (rho * nn)) / nn;
    {real_t} phi = phi_from_q(q, e, es);
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
    {real_t} sin_beta0, {real_t} cos_beta0, {real_t} e, {real_t} es,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} q = qsfn(sin(phi), e);
    {real_t} beta = asin(fmin(fmax(q / qp, ({real_t})-1.0), ({real_t})1.0));
    {real_t} sin_beta = sin(beta), cos_beta = cos(beta);
    {real_t} sin_lam = sin(lam), cos_lam = cos(lam);
    {real_t} ex, ey;
    if (mode == 0) {{ // oblique
        {real_t} b = ({real_t})1.0 + sin_beta0 * sin_beta + cos_beta0 * cos_beta * cos_lam;
        b = Rq * sqrt(({real_t})2.0 / fmax(b, ({real_t})1e-30));
        ex = b * D * cos_beta * sin_lam;
        ey = (b / D) * (cos_beta0 * sin_beta - sin_beta0 * cos_beta * cos_lam);
    }} else if (mode == 1) {{ // equatorial
        {real_t} b = ({real_t})1.0 + cos_beta * cos_lam;
        b = Rq * sqrt(({real_t})2.0 / fmax(b, ({real_t})1e-30));
        ex = b * D * cos_beta * sin_lam;
        ey = (b / D) * sin_beta;
    }} else if (mode == 2) {{ // north pole
        {real_t} q_diff = qp - q;
        if (q_diff < ({real_t})0.0) q_diff = ({real_t})0.0;
        {real_t} rho = Rq * sqrt(q_diff);
        ex = rho * sin_lam;
        ey = -rho * cos_lam;
    }} else {{ // south pole
        {real_t} q_diff = qp + q;
        if (q_diff < ({real_t})0.0) q_diff = ({real_t})0.0;
        {real_t} rho = Rq * sqrt(q_diff);
        ex = rho * sin_lam;
        ey = rho * cos_lam;
    }}
    {real_t} easting  = ex * a + x0;
    {real_t} northing = ey * a + y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_LAEA_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="laea_inverse", real_t="{real_t}")
    + """
    int mode, {real_t} Rq, {real_t} D, {real_t} qp,
    {real_t} sin_beta0, {real_t} cos_beta0, {real_t} e, {real_t} es,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} x_adj = cx / D, y_adj = cy * D;
    {real_t} rho = sqrt(x_adj * x_adj + y_adj * y_adj);
    {real_t} sin_beta, lam;
    if (mode == 0 || mode == 1) {{ // oblique or equatorial
        {real_t} ce = ({real_t})2.0 * asin(fmin(fmax(rho / (({real_t})2.0 * Rq), ({real_t})-1.0), ({real_t})1.0));
        {real_t} sin_ce = sin(ce), cos_ce = cos(ce);
        if (mode == 0) {{
            sin_beta = cos_ce * sin_beta0 + y_adj * sin_ce * cos_beta0 / fmax(rho, ({real_t})1e-30);
            lam = atan2(x_adj * sin_ce, rho * cos_beta0 * cos_ce - y_adj * sin_beta0 * sin_ce);
        }} else {{
            sin_beta = y_adj * sin_ce / fmax(rho, ({real_t})1e-30);
            lam = atan2(x_adj * sin_ce, rho * cos_ce);
        }}
    }} else if (mode == 2) {{ // north pole
        sin_beta = ({real_t})1.0 - (rho * rho) / (Rq * Rq);
        lam = atan2(cx, -cy);
    }} else {{ // south pole
        sin_beta = (rho * rho) / (Rq * Rq) - ({real_t})1.0;
        lam = atan2(cx, cy);
    }}
    {real_t} q = qp * sin_beta;
    {real_t} phi = phi_from_q(q, e, es);
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
    {real_t} beta = asin(fmin(fmax(q / qp, ({real_t})-1.0), ({real_t})1.0));
    {real_t} theta = asin(SQRT3_2 * sin(beta));
    {real_t} t2 = theta * theta;
    {real_t} t6 = t2 * t2 * t2;
    {real_t} d = A1 + ({real_t})3.0*A2*t2 + t6*(({real_t})7.0*A3 + ({real_t})9.0*A4*t2);
    {real_t} easting  = (rqda * M * lam * cos(theta) / d) * a + x0;
    {real_t} northing = (rqda * theta * (A1 + A2*t2 + t6*(A3 + A4*t2))) * a + y0;
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
        theta = theta - fy / fpy;
    }}
    {real_t} t2 = theta * theta;
    {real_t} t6 = t2 * t2 * t2;
    {real_t} d = A1 + ({real_t})3.0*A2*t2 + t6*(({real_t})7.0*A3 + ({real_t})9.0*A4*t2);
    {real_t} lam = cxs * d / (M * cos(theta));
    // Recover authalic latitude, then convert to geodetic via q-inversion
    {real_t} sin_beta = fmin(fmax(sin(theta) * ({real_t})2.0 / ({real_t})1.7320508075688772935, ({real_t})-1.0), ({real_t})1.0);
    {real_t} q = qp * sin_beta;
    {real_t} phi = phi_from_q(q, e, es);
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
    {real_t} e, {real_t} k0, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} sin_phi = sin(phi);
    {real_t} q = qsfn(sin_phi, e);
    {real_t} easting  = lam * k0 * a + x0;
    {real_t} northing = (({real_t})0.5 * q / k0) * a + y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_CEA_INVERSE_SOURCE = (
    _EA_DEVICE_FNS
    + _INV_SIGNATURE.format(func="cea_inverse", real_t="{real_t}")
    + """
    {real_t} e, {real_t} es, {real_t} k0, {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} lam = cx / k0;
    {real_t} q = ({real_t})2.0 * cy * k0;
    {real_t} phi = phi_from_q(q, e, es);
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
    {real_t} sin_phi = sin(phi), cos_phi = cos(phi), cos_lam = cos(lam);
    {real_t} easting  = (cos_phi * sin(lam)) * a + x0;
    {real_t} northing = (cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam) * a + y0;
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
    {real_t} sin_c = sin(c), cos_c = cos(c);
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
    {real_t} sin_phi = sin(phi), cos_phi = cos(phi), cos_lam = cos(lam);
    {real_t} cos_c = sin_phi0 * sin_phi + cos_phi0 * cos_phi * cos_lam;
    {real_t} easting  = (cos_phi * sin(lam) / cos_c) * a + x0;
    {real_t} northing = ((cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_lam) / cos_c) * a + y0;
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
    {real_t} sin_c = sin(c), cos_c = cos(c);
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
        {real_t} dtheta = -(({real_t})2.0 * theta + sin(({real_t})2.0 * theta) - pi_sin_phi)
                         / (({real_t})2.0 + ({real_t})2.0 * cos(({real_t})2.0 * theta));
        theta += dtheta;
        if (fabs(dtheta) < ({real_t})1e-14) break;
    }}
    {real_t} easting  = (lam * ({real_t})2.0 * SQRT2 / {pi} * cos(theta)) * a + x0;
    {real_t} northing = (SQRT2 * sin(theta)) * a + y0;
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
    {real_t} sin_chi = sin(chi), cos_chi = cos(chi);
    {real_t} cos_lam_s = cos(lam_s), sin_lam_s = sin(lam_s);
    {real_t} k_den = ({real_t})1.0 + sin_chi0*sin_chi + cos_chi0*cos_chi*cos_lam_s;
    {real_t} easting  = (({real_t})2.0 * R * k0 * cos_chi * sin_lam_s / k_den) * a + x0;
    {real_t} northing = (({real_t})2.0 * R * k0 * (cos_chi0*sin_chi - sin_chi0*cos_chi*cos_lam_s) / k_den) * a + y0;
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
    {real_t} sin_ce = sin(ce), cos_ce = cos(ce);
    {real_t} sin_chi = cos_ce*sin_chi0 + ys*sin_ce*cos_chi0 / fmax(rho, ({real_t})1e-30);
    {real_t} chi = asin(fmin(fmax(sin_chi, ({real_t})-1.0), ({real_t})1.0));
    {real_t} lam_s = atan2(xs*sin_ce, rho*cos_chi0*cos_ce - ys*sin_chi0*sin_ce);
    {real_t} lam = lam_s / nn;
    // Conformal sphere -> geodetic (iterative)
    {real_t} psi = ({real_t})0.5 * (log((({real_t})1.0 + sin_chi) / fmax(({real_t})1.0 - sin_chi, ({real_t})1e-30)) - log(c)) / nn;
    {real_t} phi = ({real_t})2.0 * atan(exp(psi)) - ({real_t})0.5 * {pi};
    for (int i = 0; i < 15; i++) {{
        {real_t} sp = sin(phi);
        {real_t} es = e * sp;
        {real_t} psi_c = log(tan(({real_t})0.25*{pi} + ({real_t})0.5*phi) * pow((({real_t})1.0-es)/(({real_t})1.0+es), ({real_t})0.5*e));
        {real_t} dphi = (psi - psi_c) * cos(phi) * (({real_t})1.0 - e*e*sp*sp) / (({real_t})1.0 - e*e);
        phi += dphi;
        if (fabs(dphi) < ({real_t})1e-14) break;
    }}
"""
    + _INV_POSTAMBLE
    + "}}"
)

# ===================================================================
# Geostationary Satellite kernels
# ===================================================================

_GEOS_FORWARD_SOURCE = (
    _FWD_SIGNATURE.format(func="geos_forward", real_t="{real_t}")
    + """
    {real_t} H, {real_t} r_eq2, {real_t} r_pol2,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _FWD_PREAMBLE
    + """
    {real_t} phi_gc = atan(r_pol2 / r_eq2 * tan(phi));
    {real_t} cos_pgc = cos(phi_gc), sin_pgc = sin(phi_gc);
    // Geocentric earth radius (CGMS standard)
    {real_t} r_pol = sqrt(r_pol2);
    {real_t} r_earth = r_pol / sqrt(({real_t})1.0 - (r_eq2 - r_pol2) / r_eq2 * cos_pgc * cos_pgc);
    {real_t} cos_l = cos(lam);
    {real_t} Sx = H - r_earth * cos_pgc * cos_l;
    {real_t} Sy = -r_earth * cos_pgc * sin(lam);
    {real_t} Sz = r_earth * sin_pgc;
    // Sweep Y (GOES-R PUG): x = arcsin(-s_y/|s|), y = arctan(s_z/s_x)
    {real_t} sn = sqrt(Sx*Sx + Sy*Sy + Sz*Sz);
    {real_t} easting  = asin(fmin(fmax(-Sy / sn, ({real_t})-1.0), ({real_t})1.0)) + x0;
    {real_t} northing = atan2(Sz, Sx) + y0;
"""
    + _FWD_POSTAMBLE
    + "}}"
)

_GEOS_INVERSE_SOURCE = (
    _INV_SIGNATURE.format(func="geos_inverse", real_t="{real_t}")
    + """
    {real_t} H, {real_t} r_eq2, {real_t} r_pol2,
    {real_t} lam0, {real_t} a, {real_t} x0, {real_t} y0,
    int src_north_first, int dst_north_first, int n
) {{"""
    + _INV_PREAMBLE
    + """
    {real_t} xp = cx * a, yp = cy * a;
    {real_t} sx = sin(xp), cx2 = cos(xp), sy = sin(yp), cy2 = cos(yp);
    {real_t} ac = sx*sx + cx2*cx2*(cy2*cy2 + sy*sy*r_eq2/r_pol2);
    {real_t} bc = ({real_t})-2.0*H*cx2*cy2;
    {real_t} cc = H*H - a*a;
    {real_t} disc = bc*bc - ({real_t})4.0*ac*cc;
    disc = fmax(disc, ({real_t})0.0);
    {real_t} rs = (-bc - sqrt(disc)) / (({real_t})2.0*ac);
    {real_t} Sx2 = rs*cx2*cy2, Sy2 = -rs*sx, Sz2 = rs*cx2*sy;
    {real_t} lam = atan2(-Sy2, H - Sx2);
    {real_t} phi = atan(Sz2 * r_eq2 / (sqrt((H-Sx2)*(H-Sx2)+Sy2*Sy2) * r_pol2));
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
    const {real_t} TX[19] = {{1.0,0.9986,0.9954,0.99,0.9822,0.973,0.96,0.9427,0.9216,0.8962,0.8679,0.835,0.7986,0.7597,0.7186,0.6732,0.6213,0.5722,0.5322}};
    const {real_t} TY[19] = {{0.0,0.062,0.124,0.186,0.248,0.31,0.372,0.434,0.4958,0.5571,0.6176,0.6769,0.7346,0.7903,0.8435,0.8936,0.9394,0.9761,1.0}};
    const {real_t} FXC = ({real_t})0.8487, FYC = ({real_t})1.3523;
"""
    + _FWD_PREAMBLE
    + """
    {real_t} abs_phi = fabs(phi);
    {real_t} phi_deg = abs_phi * ({real_t})180.0 / {pi};
    int ti = (int)fmin(phi_deg / ({real_t})5.0, ({real_t})17.0);
    {real_t} frac = phi_deg / ({real_t})5.0 - ({real_t})ti;
    int ti1 = ti < 18 ? ti+1 : 18;
    {real_t} X = TX[ti] + frac * (TX[ti1] - TX[ti]);
    {real_t} Y = TY[ti] + frac * (TY[ti1] - TY[ti]);
    {real_t} easting  = (FXC * X * lam / {pi}) * a + x0;
    {real_t} sgn = phi < ({real_t})0.0 ? ({real_t})-1.0 : ({real_t})1.0;
    {real_t} northing = (FYC * Y * sgn) * a + y0;
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
    const {real_t} TX[19] = {{1.0,0.9986,0.9954,0.99,0.9822,0.973,0.96,0.9427,0.9216,0.8962,0.8679,0.835,0.7986,0.7597,0.7186,0.6732,0.6213,0.5722,0.5322}};
    const {real_t} TY[19] = {{0.0,0.062,0.124,0.186,0.248,0.31,0.372,0.434,0.4958,0.5571,0.6176,0.6769,0.7346,0.7903,0.8435,0.8936,0.9394,0.9761,1.0}};
    const {real_t} FXC = ({real_t})0.8487, FYC = ({real_t})1.3523;
"""
    + _INV_PREAMBLE
    + """
    {real_t} abs_y = fabs(cy) / FYC;
    int ti = 0;
    for (int i = 0; i < 18; i++) {{ if (TY[i+1] >= abs_y) {{ ti = i; break; }} if (i == 17) ti = 17; }}
    int ti1 = ti < 18 ? ti+1 : 18;
    {real_t} frac = (abs_y - TY[ti]) / fmax(TY[ti1] - TY[ti], ({real_t})1e-30);
    {real_t} phi_deg = (({real_t})ti + frac) * ({real_t})5.0;
    {real_t} X = TX[ti] + frac * (TX[ti1] - TX[ti]);
    {real_t} phi = phi_deg * {pi} / ({real_t})180.0;
    if (cy < ({real_t})0.0) phi = -phi;
    {real_t} lam = cx * {pi} / (FXC * fmax(X, ({real_t})1e-30));
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
    {real_t} cos_phi = cos(phi);
    {real_t} alpha = acos(fmin(fmax(cos_phi * cos(lam * ({real_t})0.5), ({real_t})-1.0), ({real_t})1.0));
    {real_t} sinc_a = fabs(alpha) < ({real_t})1e-10 ? ({real_t})1.0 : sin(alpha) / alpha;
    {real_t} easting  = ((({real_t})2.0 * cos_phi * sin(lam*({real_t})0.5) / sinc_a + lam*cos_phi1) * ({real_t})0.5) * a + x0;
    {real_t} northing = ((sin(phi) / sinc_a + phi) * ({real_t})0.5) * a + y0;
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
    for (int i = 0; i < 20; i++) {{
        {real_t} cp = cos(phi), sp = sin(phi);
        {real_t} chl = cos(lam * ({real_t})0.5), shl = sin(lam * ({real_t})0.5);
        {real_t} al = acos(fmin(fmax(cp * chl, ({real_t})-1.0), ({real_t})1.0));
        {real_t} sc = fabs(al) < ({real_t})1e-10 ? ({real_t})1.0 : sin(al) / al;
        {real_t} fx = (({real_t})2.0*cp*shl/sc + lam*cos_phi1)*({real_t})0.5 - cx;
        {real_t} fy = (sp/sc + phi)*({real_t})0.5 - cy;
        lam -= fx * ({real_t})0.5;
        phi -= fy * ({real_t})0.5;
        if (fabs(fx) < ({real_t})1e-10 && fabs(fy) < ({real_t})1e-10) break;
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
    {real_t} easting  = (lam * (A0 + p2*(A1 + p2*(A2 + p4*(A3 + p2*A4))))) * a + x0;
    {real_t} northing = (phi * (B0 + p2*(B1 + p4*(B2 + p2*(B3 + p2*B4))))) * a + y0;
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
        if (fabs(fy) < ({real_t})1e-14) break;
    }}
    {real_t} p2 = phi*phi, p4 = p2*p2;
    {real_t} lam = cx / (A0 + p2*(A1 + p2*(A2 + p4*(A3 + p2*A4))));
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
    {real_t} sp = sin(phi), cp = cos(phi), cl = cos(lam), sl = sin(lam);
    {real_t} cos_c = fmin(fmax(sin_phi0*sp + cos_phi0*cp*cl, ({real_t})-1.0), ({real_t})1.0);
    {real_t} c2 = acos(cos_c);
    {real_t} k = fabs(c2) < ({real_t})1e-10 ? ({real_t})1.0 : c2 / sin(c2);
    {real_t} easting  = (k * cp * sl) * a + x0;
    {real_t} northing = (k * (cos_phi0*sp - sin_phi0*cp*cl)) * a + y0;
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
    {real_t} sin_c = sin(c2), cos_c = cos(c2);
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
extern "C" __global__ void tm_forward_ds(
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
    ds_t Cn = ds_gatg(c0,c1,c2,c3,c4,c5, phi, ds_cos(two_phi), ds_sin(two_phi));

    // Gaussian -> complex spherical
    ds_t sin_Cn = ds_sin(Cn), cos_Cn = ds_cos(Cn);
    ds_t sin_Ce = ds_sin(lam), cos_Ce = ds_cos(lam);
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

    if (dst_north_first) {{ out_x[idx] = northing; out_y[idx] = easting; }}
    else                 {{ out_x[idx] = easting;  out_y[idx] = northing; }}
}}
"""
)

_TM_INVERSE_DS_SOURCE = (
    _DS_TM_DEVICE_FNS
    + """
extern "C" __global__ void tm_inverse_ds(
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
    double d_n, d_e;
    if (src_north_first) {{ d_n = d1; d_e = d2; }} else {{ d_e = d1; d_n = d2; }}

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
    ds_t sin_ar = ds_sin(two_Cn), cos_ar = ds_cos(two_Cn);
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
    ds_t sin_Cn = ds_sin(Cn), cos_Cn = ds_cos(Cn);
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

_TYPE_MAP = {
    "float64": "double",
    "float32": "float",
}


def _get_kernel(projection_name: str, direction: str, compute_dtype: str):
    """Get or compile a fused kernel (thread-safe).

    Uses double-checked locking: the fast path (cache hit) is lock-free.
    The lock is only acquired on cache miss to serialize NVRTC compilation.

    compute_dtype: "float64", "float32", or "ds" (double-single fp32).
    I/O arrays are always double* regardless of compute precision.
    """
    import cupy as cp

    key = (projection_name, direction, compute_dtype)
    # Fast path: lock-free read (dict reads are thread-safe in CPython)
    if key in _kernel_cache:
        return _kernel_cache[key]

    # Slow path: compile under lock
    with _kernel_cache_lock:
        # Re-check after acquiring lock (another thread may have compiled)
        if key in _kernel_cache:
            return _kernel_cache[key]

        if compute_dtype == "ds":
            ds_key = (projection_name, direction)
            if ds_key in _DS_SOURCE_MAP:
                source, func_name = _DS_SOURCE_MAP[ds_key]
                source = source.format()
            else:
                # Fallback to fp64 (RLock allows re-entrant acquisition)
                return _get_kernel(projection_name, direction, "float64")
        else:
            template, func_name = _SOURCE_MAP[(projection_name, direction)]
            source = template.format(
                real_t=_TYPE_MAP[compute_dtype], pi=_PI_LITERALS[compute_dtype]
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


def compile_kernels(projections=None, *, precision="auto"):
    """Pre-compile fused NVRTC kernels to eliminate first-call latency.

    Parameters
    ----------
    projections : list of str, optional
        Projection names to compile (e.g. ["tmerc", "webmerc"]).
        If None, compiles all supported projections.
    precision : str
        Compute precision: "auto"/"fp64"/"fp32"/"ds".
    """
    compute_dtype = {"auto": "float64", "fp64": "float64", "fp32": "float32", "ds": "ds"}.get(
        precision, "float64"
    )
    if projections is None:
        targets = list(_SUPPORTED)
    else:
        targets = [
            (p, d) for p in projections for d in ("forward", "inverse") if (p, d) in _SUPPORTED
        ]
    for proj_name, direction in targets:
        _get_kernel(proj_name, direction, compute_dtype)


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
) -> tuple | None:
    """Execute a fused GPU kernel for the full transform pipeline.

    Parameters
    ----------
    out_x, out_y : cupy.ndarray, optional
        Pre-allocated output arrays. Pass these to avoid allocation.
    precision : str
        "fp64" = full double precision (default for fp64 input).
        "fp32" = fp32 compute with fp64 I/O (ADR-0002 mixed precision).
        "auto" = fp64 (projection math is trig-dominated / SFU-bound).

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
        compute_dtype = "float64"  # fp64 always — trig-dominated; see gpu_detect.py
    elif precision == "fp32":
        compute_dtype = "float32"  # raw fp32 (lossy — expert opt-in)
    elif precision == "ds":
        compute_dtype = "ds"  # double-single fp32 (fp64-equivalent accuracy)
    elif precision == "fp64":
        compute_dtype = "float64"
    else:
        compute_dtype = "float64"

    # I/O is always fp64 (ADR-0002: storage is always fp64)
    # Kernel reads double*, computes in real_t, writes double*
    io_dtype = np.float64
    if arg1.dtype != np.float64:
        arg1 = arg1.astype(np.float64)
    if arg2.dtype != np.float64:
        arg2 = arg2.astype(np.float64)

    # ds kernels take double params (the ds arithmetic is internal)
    real_t = np.float64 if compute_dtype in ("float64", "ds") else np.float32
    kernel = _get_kernel(projection_name, direction, compute_dtype)
    if out_x is None:
        out_x = cp.empty(n, dtype=io_dtype)
    if out_y is None:
        out_y = cp.empty(n, dtype=io_dtype)
    block = 256
    grid = max(1, (n + block - 1) // block)
    snf = np.int32(src_north_first)
    dnf = np.int32(dst_north_first)
    nn = np.int32(n)

    # Build args per projection
    base = (arg1, arg2, out_x, out_y)

    if projection_name in ("webmerc", "sinu"):
        args = base + (
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "eqc":
        args = base + (
            real_t(computed["cos_lat_ts"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "merc":
        args = base + (
            real_t(computed["e"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "tmerc":
        if direction == "forward":
            c6 = [real_t(c) for c in computed["cbg"]]
            g6 = [real_t(c) for c in computed["gtu"]]
        else:
            c6 = [real_t(c) for c in computed["cgb"]]
            g6 = [real_t(c) for c in computed["utg"]]
        args = base + (
            *c6,
            *g6,
            real_t(computed["Qn"]),
            real_t(computed["Zb"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "lcc":
        args = base + (
            real_t(computed["n"]),
            real_t(computed["F"]),
            real_t(computed["rho0"]),
            real_t(computed["e"]),
            real_t(computed["k0"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "stere":
        args = base + (
            real_t(computed["akm1"]),
            real_t(computed["sign"]),
            real_t(computed["e"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "aea":
        args = base + (
            real_t(computed["n"]),
            real_t(computed["C"]),
            real_t(computed["rho0"]),
            real_t(computed["e"]),
            real_t(computed["es"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "laea":
        mode_map = {"oblique": 0, "equatorial": 1, "north_pole": 2, "south_pole": 3}
        args = base + (
            np.int32(mode_map[computed["mode"]]),
            real_t(computed["Rq"]),
            real_t(computed["D"]),
            real_t(computed["qp"]),
            real_t(computed["sin_beta0"]),
            real_t(computed["cos_beta0"]),
            real_t(computed["e"]),
            real_t(computed["es"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "eqearth":
        if direction == "forward":
            args = base + (
                real_t(computed["e"]),
                real_t(computed["qp"]),
                real_t(computed["rqda"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
                snf,
                dnf,
                nn,
            )
        else:
            args = base + (
                real_t(computed["e"]),
                real_t(computed["es"]),
                real_t(computed["qp"]),
                real_t(computed["rqda"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
                snf,
                dnf,
                nn,
            )

    elif projection_name == "moll":
        args = base + (
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "cea":
        if direction == "forward":
            args = base + (
                real_t(computed["e"]),
                real_t(computed["k0"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
                snf,
                dnf,
                nn,
            )
        else:
            args = base + (
                real_t(computed["e"]),
                real_t(computed["es"]),
                real_t(computed["k0"]),
                real_t(computed["lam0"]),
                real_t(computed["a"]),
                real_t(computed["x0"]),
                real_t(computed["y0"]),
                snf,
                dnf,
                nn,
            )

    elif projection_name in ("ortho", "gnom", "aeqd"):
        args = base + (
            real_t(computed["sin_phi0"]),
            real_t(computed["cos_phi0"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "sterea":
        args = base + (
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
            snf,
            dnf,
            nn,
        )

    elif projection_name == "geos":
        args = base + (
            real_t(computed["H"]),
            real_t(computed["r_eq2"]),
            real_t(computed["r_pol2"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name in ("robin", "natearth"):
        args = base + (
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    elif projection_name == "wintri":
        args = base + (
            real_t(computed["cos_phi1"]),
            real_t(computed["lam0"]),
            real_t(computed["a"]),
            real_t(computed["x0"]),
            real_t(computed["y0"]),
            snf,
            dnf,
            nn,
        )

    else:
        return None

    if stream is not None:
        with stream:
            kernel((grid,), (block,), args)
    else:
        kernel((grid,), (block,), args)
    return out_x, out_y
