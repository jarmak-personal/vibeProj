"""Bounded fixed-point trigonometric CUDA device functions.

The Q1.62 implementation is specialized for angles in [-pi, pi], which covers
validated geographic latitude/longitude and Helmert's recovered latitude.
Inputs outside that range, including non-finite values, fall back to CUDA's
native fp64 functions so this optimization cannot narrow public behavior.

The algorithm reduces to the nearest quadrant, leaving a residual in
[-pi/4, pi/4], then evaluates degree-17 sine and degree-18 cosine Taylor
polynomials in signed Q1.62 arithmetic. Full-width products use __umul64hi.
"""

# fmt: off

FIXED_TRIG_DEVICE_FNS = r"""
// ---- Bounded Q1.62 sine/cosine ----
// Fast path domain: [-pi, pi]. Outside it, use native fp64 libdevice calls.

typedef long long vp_q62_t;
typedef unsigned long long vp_uq64_t;

#define VP_Q62_SCALE_D 4611686018427387904.0
#define VP_Q62_INV_SCALE_D 2.1684043449710088680149056017398834228515625e-19
#define VP_PI_D 3.141592653589793238462643383279502884

__device__ __forceinline__ vp_q62_t vp_q62_from_double(double value) {
    return __double2ll_rn(value * VP_Q62_SCALE_D);
}

__device__ __forceinline__ double vp_q62_to_double(vp_q62_t value) {
    return (double)value * VP_Q62_INV_SCALE_D;
}

__device__ __forceinline__ vp_q62_t vp_q62_mul(vp_q62_t a, vp_q62_t b) {
    const bool negative = (a < 0) != (b < 0);
    const vp_uq64_t ua = a < 0 ? 0ULL - (vp_uq64_t)a : (vp_uq64_t)a;
    const vp_uq64_t ub = b < 0 ? 0ULL - (vp_uq64_t)b : (vp_uq64_t)b;
    const vp_uq64_t lo = ua * ub;
    vp_uq64_t hi = __umul64hi(ua, ub);
    const vp_uq64_t rounded_lo = lo + (1ULL << 61);
    hi += rounded_lo < lo;
    const vp_uq64_t magnitude = (hi << 2) | (rounded_lo >> 62);
    const vp_q62_t result = (vp_q62_t)magnitude;
    return negative ? -result : result;
}

__device__ __forceinline__ void vp_fixed_sincos_reduced(
    double angle, vp_q62_t* sin_out, vp_q62_t* cos_out
) {
    // Nearest-quadrant reduction gives a residual in [-pi/4, pi/4].
    const int quadrant = __double2int_rn(
        angle * 0.6366197723675813430755350534900574
    );
    const vp_q62_t residual = vp_q62_from_double(
        angle - (double)quadrant * 1.5707963267948966192313216916397514
    );
    const vp_q62_t r2 = vp_q62_mul(residual, residual);

    // sin(r) / r, through r^16 (degree-17 sine).
    vp_q62_t sin_horner = (vp_q62_t)12966LL;
    sin_horner = (vp_q62_t)-3526632LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)740592679LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)-115532457973LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)12708570377060LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)-915017067148291LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)38430716820228232LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)-768614336404564608LL + vp_q62_mul(r2, sin_horner);
    sin_horner = (vp_q62_t)4611686018427387904LL + vp_q62_mul(r2, sin_horner);

    // cos(r), through r^18.
    vp_q62_t cos_horner = (vp_q62_t)-720LL;
    cos_horner = (vp_q62_t)220414LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)-52899477LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)9627704831LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)-1270857037706LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)114377133393536LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)-6405119470038039LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)192153584101141152LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)-2305843009213693952LL + vp_q62_mul(r2, cos_horner);
    cos_horner = (vp_q62_t)4611686018427387904LL + vp_q62_mul(r2, cos_horner);

    const vp_q62_t sin_residual = vp_q62_mul(residual, sin_horner);
    const vp_q62_t cos_residual = cos_horner;
    switch (quadrant & 3) {
        case 0: *sin_out = sin_residual;  *cos_out = cos_residual;  break;
        case 1: *sin_out = cos_residual;  *cos_out = -sin_residual; break;
        case 2: *sin_out = -sin_residual; *cos_out = -cos_residual; break;
        default:*sin_out = -cos_residual; *cos_out = sin_residual;  break;
    }
}

__device__ __forceinline__ void vp_fixed_sincos_bounded(
    double angle, double* sin_out, double* cos_out
) {
    // The negated <= form also routes NaN to native libdevice behavior.
    if (!(fabs(angle) <= VP_PI_D)) {
        sincos(angle, sin_out, cos_out);
        return;
    }
    vp_q62_t sin_fixed;
    vp_q62_t cos_fixed;
    vp_fixed_sincos_reduced(angle, &sin_fixed, &cos_fixed);
    *sin_out = vp_q62_to_double(sin_fixed);
    *cos_out = vp_q62_to_double(cos_fixed);
}

__device__ __forceinline__ double vp_fixed_sin_bounded(double angle) {
    if (!(fabs(angle) <= VP_PI_D)) {
        return sin(angle);
    }
    const int quadrant = __double2int_rn(
        angle * 0.6366197723675813430755350534900574
    );
    const vp_q62_t residual = vp_q62_from_double(
        angle - (double)quadrant * 1.5707963267948966192313216916397514
    );
    const vp_q62_t r2 = vp_q62_mul(residual, residual);
    vp_q62_t result;

    if (quadrant & 1) {
        // Odd quadrants map sine to cos(residual).
        vp_q62_t horner = (vp_q62_t)-720LL;
        horner = (vp_q62_t)220414LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-52899477LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)9627704831LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-1270857037706LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)114377133393536LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-6405119470038039LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)192153584101141152LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-2305843009213693952LL + vp_q62_mul(r2, horner);
        result = (vp_q62_t)4611686018427387904LL + vp_q62_mul(r2, horner);
    } else {
        // Even quadrants use sin(residual).
        vp_q62_t horner = (vp_q62_t)12966LL;
        horner = (vp_q62_t)-3526632LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)740592679LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-115532457973LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)12708570377060LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-915017067148291LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)38430716820228232LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)-768614336404564608LL + vp_q62_mul(r2, horner);
        horner = (vp_q62_t)4611686018427387904LL + vp_q62_mul(r2, horner);
        result = vp_q62_mul(residual, horner);
    }
    return vp_q62_to_double((quadrant & 2) ? -result : result);
}
"""

# fmt: on
