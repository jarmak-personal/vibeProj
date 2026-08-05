"""Reusable CUDA source fragments for transcendental implementations.

The fragments in this module are compile-time building blocks for fused NVRTC
kernels.  They contain no device selection or public policy logic: callers
choose an implementation on the host and compile exactly one source variant.
"""

from vibeproj._fixed_trig_device_fns import FIXED_TRIG_DEVICE_FNS

# fmt: off

NATIVE_PAIRED_SINCOS_DEVICE_FNS = r"""
// ---- Native paired sine/cosine ----
// Overloads keep fp32 computation in sincosf and fp64 computation in sincos.
__device__ __forceinline__ void vp_native_sincos(
    float angle, float* sin_out, float* cos_out
) {
    sincosf(angle, sin_out, cos_out);
}

__device__ __forceinline__ void vp_native_sincos(
    double angle, double* sin_out, double* cos_out
) {
    sincos(angle, sin_out, cos_out);
}
"""


TM_UTM_QUALIFIED_DEVICE_FNS = (
    FIXED_TRIG_DEVICE_FNS
    + r"""
// ---- Accuracy-qualified forward UTM transcendentals ----
// The host only selects this source for fp64 forward UTM. Per-element guards
// preserve native behavior for values outside the proven approximation domain.
__device__ __forceinline__ void vp_tm_utm_sincos(
    double angle, double longitude_offset, double* sin_out, double* cos_out
) {
    if (fabs(longitude_offset) <= 0.06) {
        vp_fixed_sincos_bounded(angle, sin_out, cos_out);
    } else {
        sincos(angle, sin_out, cos_out);
    }
}

__device__ __forceinline__ double vp_refined_reciprocal(double value) {
    double reciprocal = (double)__frcp_rn((float)value);
    reciprocal *= fma(-value, reciprocal, 2.0);
    return reciprocal;
}

__device__ __forceinline__ double vp_tm_utm_latitude(
    double gaussian_lat,
    double sin_lat,
    double cos_lat,
    double cos_lon,
    double native_denominator,
    double longitude_offset
) {
    // The correction identity returns the principal atan2 branch only while
    // the Gaussian latitude B is in [-pi/2, pi/2] and cos(L) is positive.
    // The longitude bound implies cos(L) > 0. Inputs outside either bound,
    // including non-finite values, must retain native atan2 branch behavior.
    if (!(fabs(longitude_offset) <= 0.06)) {
        return atan2(sin_lat, native_denominator);
    }
    if (!(fabs(gaussian_lat) <= 1.5707963267948966192313216916397514)) {
        // The longitude guard selected Q1.62 trig before this principal-branch
        // check. Recompute both native pairs so signed-zero behavior at odd
        // multiples of pi exactly matches the native implementation.
        double native_sin_lat, native_cos_lat;
        double native_sin_lon, native_cos_lon;
        sincos(gaussian_lat, &native_sin_lat, &native_cos_lat);
        sincos(longitude_offset, &native_sin_lon, &native_cos_lon);
        return atan2(native_sin_lat, native_cos_lat * native_cos_lon);
    }

    // atan2(sin(B), cos(B)cos(L)) = B + atan(delta). Over the guarded
    // |L| <= 0.06 domain, |delta| <= 9.01e-4; the odd degree-5 correction
    // remains fp64-accurate (validated max error 2.22e-16 radians).
    const double denominator = fma(
        cos_lon * cos_lat, cos_lat, sin_lat * sin_lat
    );
    const double correction = sin_lat * cos_lat * (1.0 - cos_lon)
        * vp_refined_reciprocal(denominator);
    const double correction_sq = correction * correction;
    return gaussian_lat + correction * fma(
        correction_sq, fma(correction_sq, 0.2, -1.0 / 3.0), 1.0
    );
}

__device__ __forceinline__ double vp_tm_utm_asinh(double value) {
    if (!(fabs(value) <= 0.06)) {
        return asinh(value);
    }

    // Odd asinh series through x^11. The next term is below one ulp over the
    // guarded domain; normal UTM inputs satisfy tan(3 degrees) < 0.053.
    const double x2 = value * value;
    double polynomial = -63.0 / 2816.0;
    polynomial = fma(polynomial, x2, 35.0 / 1152.0);
    polynomial = fma(polynomial, x2, -5.0 / 112.0);
    polynomial = fma(polynomial, x2, 3.0 / 40.0);
    polynomial = fma(polynomial, x2, -1.0 / 6.0);
    return value * fma(polynomial, x2, 1.0);
}
"""
)


HELMERT_FIXED_Q62_DEVICE_FNS = (
    FIXED_TRIG_DEVICE_FNS
    + r"""
// Height recovery is ill-conditioned near a pole: p/cos(lat) amplifies even
// sub-ULP cosine differences. Keep that uncommon latitude band native.
__device__ __forceinline__ void vp_helmert_fixed_q62_sincos(
    double angle, double* sin_out, double* cos_out
) {
    if (fabs(fabs(angle) - 1.5707963267948966192313216916397514) < 0.02) {
        sincos(angle, sin_out, cos_out);
        return;
    }
    vp_fixed_sincos_bounded(angle, sin_out, cos_out);
}

__device__ __forceinline__ double vp_helmert_fixed_q62_sin(double angle) {
    if (fabs(fabs(angle) - 1.5707963267948966192313216916397514) < 0.02) {
        return sin(angle);
    }
    return vp_fixed_sin_bounded(angle);
}
"""
)

# fmt: on
