"""Reusable CUDA source fragments for transcendental implementations.

The fragments in this module are compile-time building blocks for fused NVRTC
kernels.  They contain no device selection or public policy logic: callers
choose an implementation on the host and compile exactly one source variant.
"""

import struct

from vibeproj._fixed_trig_device_fns import FIXED_TRIG_DEVICE_FNS
from vibeproj.transcendentals import PROJECTION_FIXED_Q62_MAX_SCALE_M

# fmt: off

# The shared registry owns the physical-scale domain. Its exact binary64 bits
# feed the device-side integer range check so the host and CUDA limits cannot
# drift independently.
_PROJECTION_FIXED_Q62_MAX_SCALE_BITS = struct.unpack(
    "=Q", struct.pack("=d", PROJECTION_FIXED_Q62_MAX_SCALE_M)
)[0]

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


PROJECTION_SCALE_GUARD_DEVICE_FNS = (
    f"\n#define VP_PROJECTION_FIXED_Q62_MAX_SCALE_M {PROJECTION_FIXED_Q62_MAX_SCALE_M:.1f}\n"
    + f"#define VP_PROJECTION_FIXED_Q62_MAX_SCALE_BITS 0x{_PROJECTION_FIXED_Q62_MAX_SCALE_BITS:016x}ULL\n"
    + r"""
// ---- Shared projection physical-scale guard ----
__device__ __forceinline__ bool vp_projection_fixed_scale_is_qualified(
    double physical_scale
) {
    // Positive IEEE-754 binary64 values are monotonically ordered by their
    // unsigned representation. One integer range comparison rejects +0,
    // negative values, infinities, NaNs, and values above the scale ceiling
    // without adding slow fp64 comparisons to every coordinate.
    const unsigned long long bits = (unsigned long long)__double_as_longlong(
        physical_scale
    );
    return bits - 1ULL < VP_PROJECTION_FIXED_Q62_MAX_SCALE_BITS;
}
"""
)


PROJECTION_BOUNDED_Q62_DEVICE_FNS = (
    FIXED_TRIG_DEVICE_FNS
    + PROJECTION_SCALE_GUARD_DEVICE_FNS
    + r"""
// ---- Shared projection-domain Q1.62 primitives ----
// Callers pass their proved angular bound and launch-uniform physical scale.
// Negated comparisons send non-finite and out-of-domain inputs through native
// libdevice. The scale predicate is uniform because one projection scale is a
// scalar kernel argument shared by every coordinate in the launch.

__device__ __forceinline__ void vp_projection_fixed_sincos(
    double angle,
    double max_abs_angle,
    double physical_scale,
    double* sin_out,
    double* cos_out
) {
    if (!(vp_projection_fixed_scale_is_qualified(physical_scale)
          && fabs(angle) <= max_abs_angle && fabs(angle) <= VP_PI_D)) {
        sincos(angle, sin_out, cos_out);
        return;
    }
    vp_fixed_sincos_bounded(angle, sin_out, cos_out);
}

// Two-angle form makes the projection-domain guard atomic: if either value
// is invalid, both pairs retain native behavior. This is used when both pairs
// feed one result, such as orthographic forward.
__device__ __forceinline__ void vp_projection_fixed_sincos_pair(
    double first,
    double first_max_abs,
    double* first_sin,
    double* first_cos,
    double second,
    double second_max_abs,
    double* second_sin,
    double* second_cos,
    double physical_scale
) {
    const bool in_domain =
        vp_projection_fixed_scale_is_qualified(physical_scale)
        && fabs(first) <= first_max_abs && fabs(first) <= VP_PI_D
        && fabs(second) <= second_max_abs && fabs(second) <= VP_PI_D;
    if (!in_domain) {
        sincos(first, first_sin, first_cos);
        sincos(second, second_sin, second_cos);
        return;
    }

    vp_q62_t sin_fixed;
    vp_q62_t cos_fixed;
    vp_fixed_sincos_reduced(first, &sin_fixed, &cos_fixed);
    *first_sin = vp_q62_to_double(sin_fixed);
    *first_cos = vp_q62_to_double(cos_fixed);
    vp_fixed_sincos_reduced(second, &sin_fixed, &cos_fixed);
    *second_sin = vp_q62_to_double(sin_fixed);
    *second_cos = vp_q62_to_double(cos_fixed);
}

// Cosine-only specialization avoids evaluating and converting the unused
// sine polynomial in projections such as sinusoidal forward.
__device__ __forceinline__ double vp_projection_fixed_cos(
    double angle, double max_abs_angle, double physical_scale
) {
    if (!(vp_projection_fixed_scale_is_qualified(physical_scale)
          && fabs(angle) <= max_abs_angle && fabs(angle) <= VP_PI_D)) {
        return cos(angle);
    }

    const int quadrant = __double2int_rn(
        angle * 0.6366197723675813430755350534900574
    );
    const vp_q62_t residual = vp_q62_from_double(
        angle - (double)quadrant * 1.5707963267948966192313216916397514
    );
    const vp_q62_t r2 = vp_q62_mul(residual, residual);
    const bool use_sine = quadrant & 1;

    // Select coefficients instead of branching by quadrant. Random geographic
    // inputs mix quadrants within a warp; a branch executes both full Horner
    // paths serially. Align sin(r)/r with a leading zero r^18 coefficient so
    // either result uses one degree-18 Horner chain.
    vp_q62_t horner = use_sine ? (vp_q62_t)0LL : (vp_q62_t)-720LL;
    horner = (use_sine ? (vp_q62_t)12966LL : (vp_q62_t)220414LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)-3526632LL : (vp_q62_t)-52899477LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)740592679LL : (vp_q62_t)9627704831LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)-115532457973LL : (vp_q62_t)-1270857037706LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)12708570377060LL : (vp_q62_t)114377133393536LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)-915017067148291LL : (vp_q62_t)-6405119470038039LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)38430716820228232LL : (vp_q62_t)192153584101141152LL)
        + vp_q62_mul(r2, horner);
    horner = (use_sine ? (vp_q62_t)-768614336404564608LL : (vp_q62_t)-2305843009213693952LL)
        + vp_q62_mul(r2, horner);
    horner = (vp_q62_t)4611686018427387904LL + vp_q62_mul(r2, horner);
    const vp_q62_t scale = use_sine
        ? residual : (vp_q62_t)4611686018427387904LL;
    vp_q62_t result = vp_q62_mul(scale, horner);

    // Quadrant signs for cosine are +, -, -, +.
    if ((quadrant + 1) & 2) result = -result;
    return vp_q62_to_double(result);
}
"""
)


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
