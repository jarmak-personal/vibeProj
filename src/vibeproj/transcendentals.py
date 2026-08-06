"""Hardware-aware transcendental implementation registry and resolver.

The registry describes CUDA implementation variants; it does not contain CUDA
source and it does not choose numeric compute precision.  Device discovery is
lazy so importing :mod:`vibeproj` never imports CuPy or queries a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal, cast

TranscendentalPolicy = Literal["auto", "native", "accelerated"]
DeviceBackend = Literal["cpu", "cuda", "unknown"]
ComputePrecision = Literal["auto", "fp64", "fp32", "ds"]

NATIVE_LIBDEVICE = "native.libdevice"
HELMERT_FIXED_Q62 = "helmert.fixed_q62"
TMERC_FIXED_Q62 = "tmerc.forward.fixed_q62"
SINU_FORWARD_FIXED_Q62 = "sinu.forward.fixed_q62"
ORTHO_FORWARD_FIXED_Q62 = "ortho.forward.fixed_q62"
ORTHO_INVERSE_GUARDED_REFRAME = "ortho.inverse.guarded_reframe"
GNOM_INVERSE_GUARDED_RSQRT_REFRAME = "gnom.inverse.guarded_rsqrt_reframe"
STERE_INVERSE_FIXED_Q62 = "stere.inverse.fixed_q62"
GEOS_FORWARD_FIXED_Q62 = "geos.forward.fixed_q62"
LAEA_FORWARD_POLAR_FIXED_Q62 = "laea.forward.polar.fixed_q62"
LCC_FORWARD_CONFORMAL_REFRAME = "lcc.forward.conformal_reframe"
LCC_INVERSE_CONFORMAL_REFRAME = "lcc.inverse.conformal_reframe"
MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY = "merc.forward.ellipsoidal.product_poly"
MERC_FORWARD_SPHERICAL_PRODUCT_POLY = "merc.forward.spherical.product_poly"
MERC_INVERSE_EXP_SERIES = "merc.inverse.exp_series"
PROJECTION_FIXED_Q62_MAX_SCALE_M = 6_400_000.0
TMERC_FIXED_Q62_MIN_ELEMENTS = 256
HELMERT_FIXED_Q62_MIN_ELEMENTS = 131_072
SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS = 524_288
ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS = 262_144
ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS = 524_288
STERE_INVERSE_FIXED_Q62_MIN_ELEMENTS = 1_000_000
GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS = 2_097_152
LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS = 1_048_576
LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS = 65_536
LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS = 128
MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS = 262_144
MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS = 65_536

_EXACT_DOMAIN_FAMILIES = frozenset(
    {
        "aeqd",
        "geos",
        "gnom",
        "laea",
        "lcc",
        "merc",
        "ortho",
        "sinu",
        "stere",
        "sterea",
        "webmerc",
    }
)


def attach_projection_strategy_metadata(
    computed: dict,
    *,
    operation_method: str | None,
    eccentricity_squared: float,
    latitude_origin_degrees: float,
) -> dict:
    """Attach stable CRS/setup facts used by exact strategy-domain planning."""
    computed["_strategy_geometry"] = "spherical" if eccentricity_squared == 0.0 else "ellipsoidal"
    computed["_strategy_operation_method"] = operation_method
    computed["_strategy_latitude_origin"] = latitude_origin_degrees
    return computed


def _origin_mode(computed: dict) -> str:
    latitude = float(computed.get("_strategy_latitude_origin", 0.0))
    if math.isclose(latitude, 90.0, rel_tol=0.0, abs_tol=1e-10):
        return "north_pole"
    if math.isclose(latitude, -90.0, rel_tol=0.0, abs_tol=1e-10):
        return "south_pole"
    if math.isclose(latitude, 0.0, rel_tol=0.0, abs_tol=1e-10):
        return "equatorial"
    return "oblique"


def projection_strategy_domain(projection: str, direction: str, computed: dict) -> str:
    """Return the canonical dispatch domain for one concrete projection stage."""
    if direction not in ("forward", "inverse"):
        raise ValueError(f"Invalid projection strategy direction: {direction!r}")
    if projection == "tmerc" and direction == "forward":
        return "utm" if computed.get("is_utm", False) else "global"
    if projection not in _EXACT_DOMAIN_FAMILIES:
        return f"{projection}.{direction}"

    geometry = str(computed.get("_strategy_geometry", "unspecified"))
    method = computed.get("_strategy_operation_method")
    if projection == "sinu":
        return f"sinu.{direction}.{geometry}"
    if projection == "merc":
        variants = {
            "Mercator (variant A)": "variant_a",
            "Mercator (1SP)": "variant_a",
            "Mercator (variant B)": "variant_b",
            "Mercator (2SP)": "variant_b",
        }
        variant = variants.get(method, "custom")
        return f"merc.{direction}.{geometry}.{variant}"
    if projection == "lcc":
        variant = str(computed.get("lcc_variant", "unknown"))
        setup_names = (
            "n",
            "F",
            "rho0",
            "e",
            "k0",
            "lam0",
            "a",
            "x0",
            "y0",
            "x_unit_to_m",
            "y_unit_to_m",
        )
        setup = tuple(computed.get(name) for name in setup_names)
        setup_finite = all(value is not None and math.isfinite(float(value)) for value in setup)
        if not (
            setup_finite
            and geometry in {"spherical", "ellipsoidal"}
            and variant in {"1sp", "2sp"}
            and float(computed["n"]) != 0.0
            and float(computed["F"]) != 0.0
            and 0.0 <= float(computed["e"]) <= 0.1
            and float(computed["k0"]) == 1.0
            and 0.0 < float(computed["a"]) <= PROJECTION_FIXED_Q62_MAX_SCALE_M
            and float(computed["x_unit_to_m"]) != 0.0
            and float(computed["y_unit_to_m"]) != 0.0
        ):
            return f"lcc.{direction}.{geometry}.{variant}.invalid_setup"
        if direction == "forward":
            cone = "regular_cone" if abs(float(computed["n"])) >= 0.2 else "near_equator"
            return f"lcc.forward.{geometry}.{variant}.{cone}"
        return f"lcc.inverse.{geometry}.{variant}"
    if projection == "webmerc":
        return f"webmerc.{direction}.{geometry}.pseudo"
    if projection == "laea":
        mode = str(computed.get("mode", _origin_mode(computed)))
        return f"laea.{direction}.{geometry}.{mode}"
    if projection == "ortho":
        return f"ortho.{direction}.{geometry}.{_origin_mode(computed)}"
    if projection == "gnom":
        origin = _origin_mode(computed)
        if direction == "inverse":
            setup_values = (
                computed.get("sin_phi0"),
                computed.get("cos_phi0"),
                computed.get("a"),
                computed.get("lam0"),
                computed.get("x0"),
                computed.get("y0"),
            )
            if not all(value is not None and math.isfinite(float(value)) for value in setup_values):
                origin = "invalid_setup"
            elif origin == "oblique":
                origin = (
                    "oblique_bounded" if abs(float(computed["cos_phi0"])) >= 0.5 else "oblique_high"
                )
        return f"gnom.{direction}.{geometry}.{origin}"
    if projection == "aeqd":
        if method == "Guam Projection":
            semantics = "guam"
        elif method == "Modified Azimuthal Equidistant":
            semantics = "modified"
        elif method == "Azimuthal Equidistant (Spherical)" or geometry == "spherical":
            semantics = "spherical"
        else:
            semantics = "ellipsoidal"
        return f"aeqd.{direction}.{semantics}.{_origin_mode(computed)}"
    if projection == "stere":
        variants = {
            "Polar Stereographic (variant A)": "variant_a",
            "Polar Stereographic (variant B)": "variant_b",
            "Polar Stereographic (variant C)": "variant_c",
        }
        variant = variants.get(method, "custom")
        hemisphere = "south" if computed.get("is_south", False) else "north"
        return f"stere.{direction}.{geometry}.{variant}.{hemisphere}"
    if projection == "sterea":
        return f"sterea.{direction}.{geometry}.oblique"
    sweep_axis = str(computed.get("sweep_axis", "unknown"))
    return f"geos.{direction}.{geometry}.sweep_{sweep_axis}"


def projection_strategy_domains(projection: str, direction: str) -> tuple[str, ...]:
    """Return registry domains module-level warm-up must resolve for a target."""
    if projection == "tmerc" and direction == "forward":
        return ("global", "utm")
    if projection == "gnom":
        modes = (
            (
                "equatorial",
                "north_pole",
                "oblique_bounded",
                "oblique_high",
                "south_pole",
                "invalid_setup",
            )
            if direction == "inverse"
            else ("equatorial", "north_pole", "oblique", "south_pole")
        )
        return tuple(f"gnom.{direction}.spherical.{mode}" for mode in modes)
    if projection == "lcc":
        domains = []
        for geometry in ("spherical", "ellipsoidal"):
            for variant in ("1sp", "2sp"):
                prefix = f"lcc.{direction}.{geometry}.{variant}"
                if direction == "forward":
                    domains.extend(
                        (
                            f"{prefix}.regular_cone",
                            f"{prefix}.near_equator",
                            f"{prefix}.invalid_setup",
                        )
                    )
                else:
                    domains.extend((prefix, f"{prefix}.invalid_setup"))
        return tuple(domains)
    prefix = f"{projection}.{direction}"
    registered = {
        domain
        for implementation in _REGISTRY
        if implementation.operation is TranscendentalOperation.PROJECTION
        for domain in implementation.domains
        if domain == prefix or domain.startswith(f"{prefix}.")
    }
    return tuple(sorted(registered or {prefix}))


class TranscendentalOperation(str, Enum):
    """Operations with independently selectable transcendental strategies."""

    HELMERT = "helmert"
    PROJECTION = "projection"
    TMERC_FORWARD = "tmerc.forward"


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """Device facts used by the strategy resolver."""

    backend: DeviceBackend
    compute_capability: tuple[int, int] | None = None
    fp32_to_fp64_ratio: int | None = None
    name: str | None = None
    device_id: int | None = None


@dataclass(frozen=True, slots=True)
class AccuracyContract:
    """Validated error bounds relative to native fp64 execution."""

    reference: str
    max_horizontal_error_m: float
    max_vertical_error_m: float | None = None
    notes: str = ""
    max_physical_scale_m: float | None = None


@dataclass(frozen=True, slots=True)
class StrategyImplementation:
    """Immutable implementation metadata exposed by registry introspection.

    ``min_elements`` is the runtime crossover used by ``"auto"`` after device,
    domain, and precision qualification. Explicit ``"accelerated"`` requests
    do not apply that size threshold.
    """

    implementation_id: str
    operation: TranscendentalOperation
    family: str
    supported_policies: tuple[TranscendentalPolicy, ...]
    supported_backends: tuple[DeviceBackend, ...]
    supported_compute_capabilities: tuple[tuple[int, int], ...]
    min_fp32_to_fp64_ratio: int | None
    supported_compute_precisions: tuple[ComputePrecision, ...]
    min_elements: int
    domains: tuple[str, ...]
    accuracy: AccuracyContract
    native_fallback: bool
    priority: int = 0


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Immutable resolved decision, including fallback reason and workload size."""

    operation: TranscendentalOperation
    requested_policy: TranscendentalPolicy
    implementation_id: str
    family: str
    reason: str
    fallback: bool
    accuracy: AccuracyContract
    device: DeviceCapability
    domain: str
    workload_size: int | None


@dataclass(frozen=True, slots=True)
class ProjectionImplementation:
    """Resolved implementation for one projection stage."""

    projection: str
    direction: str
    domain: str
    implementation_id: str


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Deeply immutable dispatch plan shared by every stage in one call."""

    precision: ComputePrecision
    transcendentals: TranscendentalPolicy
    device: DeviceCapability
    workload_size: int | None
    projection_implementations: tuple[ProjectionImplementation, ...]
    helmert_implementation: str
    decisions: tuple[StrategyDecision, ...]

    def projection_implementation(self, projection: str, direction: str, domain: str) -> str:
        for implementation in self.projection_implementations:
            if (
                implementation.projection == projection
                and implementation.direction == direction
                and implementation.domain == domain
            ):
                return implementation.implementation_id
        return NATIVE_LIBDEVICE


@dataclass(frozen=True, slots=True)
class StrategyExplanation:
    """Immutable public explanation for every stage in one transform direction."""

    requested_policy: TranscendentalPolicy
    direction: Literal["FORWARD", "INVERSE"]
    device: DeviceCapability
    workload_size: int | None
    decisions: tuple[StrategyDecision, ...]


_NATIVE_ACCURACY = AccuracyContract(
    reference="native fp64 libdevice",
    max_horizontal_error_m=0.0,
    max_vertical_error_m=0.0,
    notes="Reference CUDA implementation; CPU/xp fallback uses native host functions.",
)

_REGISTRY = (
    StrategyImplementation(
        implementation_id=NATIVE_LIBDEVICE,
        operation=TranscendentalOperation.PROJECTION,
        family="native_libdevice",
        supported_policies=("auto", "native", "accelerated"),
        supported_backends=("cpu", "cuda", "unknown"),
        supported_compute_capabilities=(),
        min_fp32_to_fp64_ratio=None,
        supported_compute_precisions=("auto", "fp64", "fp32", "ds"),
        min_elements=0,
        domains=("*",),
        accuracy=_NATIVE_ACCURACY,
        native_fallback=False,
    ),
    StrategyImplementation(
        implementation_id=NATIVE_LIBDEVICE,
        operation=TranscendentalOperation.HELMERT,
        family="native_libdevice",
        supported_policies=("auto", "native", "accelerated"),
        supported_backends=("cpu", "cuda", "unknown"),
        supported_compute_capabilities=(),
        min_fp32_to_fp64_ratio=None,
        supported_compute_precisions=("auto", "fp64", "fp32", "ds"),
        min_elements=0,
        domains=("global",),
        accuracy=_NATIVE_ACCURACY,
        native_fallback=False,
    ),
    StrategyImplementation(
        implementation_id=SINU_FORWARD_FIXED_Q62,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_sinu_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        domains=("sinu.forward.spherical",),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Qualified spherical sinusoidal-forward Q1.62 cosine over finite latitude "
                "[-pi/2, pi/2] and wrapped longitude [-pi, pi], with native fallback "
                "outside the guarded domain or above 6,400,000 m physical scale. "
                "Final public WGS84 maximum/p99 horizontal error: 3.725/1.863 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=ORTHO_FORWARD_FIXED_Q62,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_ortho_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        domains=("ortho.forward.spherical.oblique",),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Qualified orthographic-forward Q1.62 paired trig over finite latitude "
                "[-pi/2, pi/2] and wrapped longitude [-pi, pi], with atomic native "
                "fallback outside the guarded domain or above 6,400,000 m physical "
                "scale. Final public WGS84 maximum/p99 horizontal error: "
                "2.033106/1.396984 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=ORTHO_INVERSE_GUARDED_REFRAME,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_ortho_inverse_reframe",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS,
        domains=("ortho.inverse.spherical.equatorial",),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Qualified spherical equatorial orthographic inverse algebraic "
                "reframe for finite non-axis points with normalized "
                "1e-16 < rho^2 <= 0.99. Center and the rho^2 <= 1e-16 near-center "
                "band, axes, horizon/outside-disk, non-finite, larger-scale, and "
                "ill-conditioned output-latitude inputs use exact native fallback. "
                "Final RTX 4090 full-valid-disk maximum/p99 horizontal error: "
                "6.328/1.584 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=GNOM_INVERSE_GUARDED_RSQRT_REFRAME,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_gnom_inverse_reframe",
        supported_policies=("accelerated",),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=0,
        domains=(
            "gnom.inverse.spherical.equatorial",
            "gnom.inverse.spherical.oblique_bounded",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Spherical Gnomonic inverse algebraically reframes the current "
                "hypot-based central-angle normalization with a guarded reciprocal "
                "square root. Exact host domains require finite setup and "
                "abs(cos(phi0)) >= 0.5. The fused kernel additionally requires "
                "1e-24 < rho^2 <= 0.02, non-axis finite coordinates, and physical "
                "scale a <= 6,400,000 m; every guard failure recomputes the exact "
                "native expression. RTX 4090 maximum/p99 native-relative horizontal "
                "error: 7.910/3.955 nm. Expert accelerated opt-in only: auto remains "
                "native because host dispatch cannot observe mixed per-coordinate rho "
                "domains and a 10% cold mixture regresses on Ada."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=STERE_INVERSE_FIXED_Q62,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_stere_inverse_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=STERE_INVERSE_FIXED_Q62_MIN_ELEMENTS,
        domains=(
            "stere.inverse.ellipsoidal.variant_a.north",
            "stere.inverse.ellipsoidal.variant_a.south",
            "stere.inverse.ellipsoidal.variant_b.north",
            "stere.inverse.ellipsoidal.variant_b.south",
            "stere.inverse.ellipsoidal.variant_c.south",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Polar Stereographic inverse uses Q1.62 sine inside the "
                "ellipsoidal phi2 iteration. CRS setup proves finite positive "
                "akm1 and exact sign +/-1 for the five public method domains. "
                "The uniform device guard additionally requires "
                "0.05 <= eccentricity <= 0.2 and physical scale "
                "a <= 6,400,000 m. The shared bounded helper routes non-finite "
                "or out-of-range iterative angles to native sine. Representative "
                "and adversarial maximum native-relative horizontal errors: "
                "1.582 and 3.164 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=GEOS_FORWARD_FIXED_Q62,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_geos_forward_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS,
        domains=(
            "geos.forward.spherical.sweep_x",
            "geos.forward.spherical.sweep_y",
            "geos.forward.ellipsoidal.sweep_x",
            "geos.forward.ellipsoidal.sweep_y",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Q1.62 paired geocentric-latitude and longitude trig. The "
                "launch-uniform native fallback requires a finite positive "
                "satellite height, exact finite H=h+a with H>h, and physical "
                "Earth radius a <= 6,400,000 m. The scan-angle perturbation's "
                ">=h line-of-sight denominator cancels the final h output scale. "
                "A bounded visibility-uncertainty band recomputes the complete "
                "native trig, classification, and projected output at the limb. "
                "Formal four-domain maximum/p99 projected error: 3.979/1.863 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=LAEA_FORWARD_POLAR_FIXED_Q62,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_laea_polar_forward_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS,
        domains=(
            "laea.forward.spherical.north_pole",
            "laea.forward.spherical.south_pole",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Polar LAEA forward uses Q1.62 paired longitude trig with "
                "launch-uniform native fallback above 6,400,000 m physical scale. "
                "Formal spherical-polar maximum/p99 error: 4.165/2.634 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=LCC_FORWARD_CONFORMAL_REFRAME,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_lcc_forward_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS,
        domains=tuple(
            f"lcc.forward.{geometry}.{variant}.regular_cone"
            for geometry in ("spherical", "ellipsoidal")
            for variant in ("1sp", "2sp")
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "LCC forward uses a setup-uniform geometry branch: spherical setups "
                "reassociate the outer power as log/exp, while ellipsoidal setups "
                "retain the native outer power and replace only the inner eccentricity "
                "power with exp(e*atanh(e*sin(phi))). Exact 1SP/2SP setup requires "
                "finite nonzero signed units, k0=1, abs(n)>=0.2, 0<=e<=0.1, and "
                "0<a<=6,400,000 m. Near-equator cones stay native. Poles, nonfinite "
                "coordinates, invalid derived values, or setup failures use the complete "
                "native expression. RTX 4090 maximum native-relative error: 6.780 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=LCC_INVERSE_CONFORMAL_REFRAME,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_lcc_inverse_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS,
        domains=tuple(
            f"lcc.inverse.{geometry}.{variant}"
            for geometry in ("spherical", "ellipsoidal")
            for variant in ("1sp", "2sp")
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "LCC inverse uses a setup-uniform geometry branch. Spherical setups "
                "recover latitude through the log-domain conformal value; ellipsoidal "
                "setups retain the native outer power and replace the phi2 inner power "
                "with exp/atanh for at most six iterations using the native 1e-14 "
                "convergence rule. If the sixth step has not reached that rule, one "
                "fixed-point contraction correction is applied to that step without a "
                "seventh transcendental evaluation; this closes the e=0.1 accuracy "
                "boundary. Exact 1SP/2SP setup requires finite nonzero signed "
                "units, k0=1, 0<=e<=0.1, and 0<a<=6,400,000 m. Apex, nonpositive "
                "radius ratio, nonfinite coordinates/results, or setup failures use "
                "the complete native expression. RTX 4090 maximum native-relative "
                "horizontal error: 6.328 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=MERC_FORWARD_SPHERICAL_PRODUCT_POLY,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_spherical_merc_forward_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS,
        domains=(
            "merc.forward.spherical.variant_a",
            "merc.forward.spherical.variant_b",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Spherical Mercator forward removes the zero-exponent pow while "
                "preserving the native latitude clamp and log-tan expression. "
                "Finite raw/derived coordinates and finite nonzero-unit setup with "
                "e=0, 0<k0<=1, and 0<a<=6,400,000 m are qualified. Formal "
                "native-relative coordinate error is bitwise zero."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_ellipsoidal_merc_forward_transcendentals",
        supported_policies=("accelerated",),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=0,
        domains=(
            "merc.forward.ellipsoidal.variant_a",
            "merc.forward.ellipsoidal.variant_b",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Ellipsoidal Mercator forward preserves the native latitude clamp and log-tan "
                "form while replacing the ellipsoidal pow correction with "
                "e*atanh(e*sin(phi)) and a degree-eight polynomial for its small "
                "exponential. The launch setup guard requires finite parameters "
                "and nonzero units, 0 < e <= 0.1, 0 < k0 <= 1, and "
                "0 < a <= 6,400,000 m. Both raw and derived coordinates must be "
                "finite and the forward hot latitude is bounded to +/-89.9 degrees; "
                "any setup or coordinate failure sends the complete warp "
                "through the exact native expression. Formal maximum/p99 "
                "native-relative projected error: 7.451/0.931 nm. Automatic "
                "selection is disabled because 10-50% random polar-cap mixtures "
                "make warp-wide fallback slower than native."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=MERC_INVERSE_EXP_SERIES,
        operation=TranscendentalOperation.PROJECTION,
        family="qualified_merc_inverse_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS,
        domains=(
            "merc.inverse.spherical.variant_a",
            "merc.inverse.spherical.variant_b",
            "merc.inverse.ellipsoidal.variant_a",
            "merc.inverse.ellipsoidal.variant_b",
        ),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_physical_scale_m=PROJECTION_FIXED_Q62_MAX_SCALE_M,
            notes=(
                "Mercator inverse retains the native exp/atan conformal seed and "
                "replaces the iterative ellipsoidal recovery with a reusable "
                "sixth-order Poder/Engsager conformal-to-geodetic series. The "
                "launch setup guard requires finite parameters and nonzero units, "
                "0 <= e <= 0.1, finite k0 > 0, and 0 < a <= 6,400,000 m. All "
                "finite raw coordinates whose normalized coordinates remain "
                "finite, including signed zero and well-conditioned huge finite "
                "northings, are qualified; any failure sends the complete warp "
                "through the exact native expression. Formal maximum/p99 "
                "native-relative horizontal error: 6.328/3.164 nm."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=HELMERT_FIXED_Q62,
        operation=TranscendentalOperation.HELMERT,
        family="bounded_q1_62",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64", "fp32", "ds"),
        min_elements=HELMERT_FIXED_Q62_MIN_ELEMENTS,
        domains=("global",),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            max_vertical_error_m=2e-7,
            notes=(
                "Q1.62 paired trig on bounded angles with native fallback for "
                "unbounded/non-finite inputs and the near-pole height-recovery band."
            ),
        ),
        native_fallback=True,
    ),
    StrategyImplementation(
        implementation_id=NATIVE_LIBDEVICE,
        operation=TranscendentalOperation.TMERC_FORWARD,
        family="native_libdevice",
        supported_policies=("auto", "native", "accelerated"),
        supported_backends=("cpu", "cuda", "unknown"),
        supported_compute_capabilities=(),
        min_fp32_to_fp64_ratio=None,
        supported_compute_precisions=("auto", "fp64", "fp32", "ds"),
        min_elements=0,
        domains=("global", "utm"),
        accuracy=_NATIVE_ACCURACY,
        native_fallback=False,
    ),
    StrategyImplementation(
        implementation_id=TMERC_FIXED_Q62,
        operation=TranscendentalOperation.TMERC_FORWARD,
        family="qualified_utm_transcendentals",
        supported_policies=("auto", "accelerated"),
        supported_backends=("cuda",),
        supported_compute_capabilities=((8, 9),),
        min_fp32_to_fp64_ratio=16,
        supported_compute_precisions=("auto", "fp64"),
        min_elements=TMERC_FIXED_Q62_MIN_ELEMENTS,
        domains=("utm",),
        accuracy=AccuracyContract(
            reference=NATIVE_LIBDEVICE,
            max_horizontal_error_m=1e-8,
            notes=(
                "Qualified forward-UTM bundle: Q1.62 paired trig, refined reciprocal "
                "atan2 identity, and degree-11 asinh; native fallback outside |lam| <= 0.06."
            ),
        ),
        native_fallback=True,
    ),
)


def normalize_transcendental_policy(value: str) -> TranscendentalPolicy:
    """Validate and normalize a public transcendental policy."""
    if value not in ("auto", "native", "accelerated"):
        raise ValueError(
            f"Invalid transcendentals policy: {value!r}. "
            "Must be 'auto', 'native', or 'accelerated'."
        )
    return cast(TranscendentalPolicy, value)


def normalize_compute_precision(value: str) -> ComputePrecision:
    """Validate numeric compute precision independently of strategy policy."""
    if value not in ("auto", "fp64", "fp32", "ds"):
        raise ValueError(f"Invalid precision: {value!r}. Must be 'fp64', 'fp32', 'ds', or 'auto'.")
    return cast(ComputePrecision, value)


def list_transcendental_strategies() -> tuple[StrategyImplementation, ...]:
    """Return the immutable built-in transcendental strategy registry.

    Entries expose stable IDs, hardware/domain/precision qualifications,
    ``min_elements`` crossover thresholds, accuracy contracts, and native
    fallback behavior. Returning a tuple of frozen dataclasses prevents callers
    from changing global dispatch policy.
    """
    return _REGISTRY


def detect_device_capability(xp=None, *, device_id: int | None = None) -> DeviceCapability:
    """Lazily describe the current execution device.

    Passing NumPy (or another host array module) returns a CPU capability
    without importing CuPy.  With no argument, CuPy is imported lazily and the
    current CUDA device is queried when available.
    """
    if xp is not None and getattr(xp, "__name__", "").split(".", 1)[0] != "cupy":
        return DeviceCapability(backend="cpu", name="CPU")

    if xp is None:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError):
            return DeviceCapability(backend="cpu", name="CPU")
    else:
        cp = xp

    try:
        resolved_device_id = (
            int(cp.cuda.runtime.getDevice()) if device_id is None else int(device_id)
        )
        return _detect_cuda_device_capability(cp, resolved_device_id)
    except (AttributeError, RuntimeError, OSError):
        return DeviceCapability(backend="unknown")


@lru_cache(maxsize=None)
def _detect_cuda_device_capability(cp, device_id: int) -> DeviceCapability:
    """Return cached facts for a CuPy module/device pair."""
    try:
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        device = cp.cuda.Device(device_id)
        major = int(properties.get("major", properties.get(b"major", 0)))
        minor = int(properties.get("minor", properties.get(b"minor", 0)))
        ratio = int(device.attributes.get("SingleToDoublePrecisionPerfRatio", 0))
        raw_name = properties.get("name", properties.get(b"name"))
        if isinstance(raw_name, bytes):
            name = raw_name.decode(errors="replace")
        else:
            name = str(raw_name) if raw_name is not None else None
        return DeviceCapability(
            backend="cuda",
            compute_capability=(major, minor),
            fp32_to_fp64_ratio=ratio or None,
            name=name,
            device_id=device_id,
        )
    except (AttributeError, RuntimeError, OSError):
        return DeviceCapability(backend="unknown")


def _implementation(
    operation: TranscendentalOperation, implementation_id: str
) -> StrategyImplementation:
    for implementation in _REGISTRY:
        if (
            implementation.operation is operation
            and implementation.implementation_id == implementation_id
        ):
            return implementation
    raise ValueError(
        f"Unknown transcendental implementation {implementation_id!r} "
        f"for operation {operation.value!r}."
    )


def _supports(
    implementation: StrategyImplementation,
    policy: TranscendentalPolicy,
    device: DeviceCapability,
    domain: str,
    precision: str,
) -> bool:
    if policy not in implementation.supported_policies:
        return False
    if device.backend not in implementation.supported_backends:
        return False
    if (
        implementation.domains
        and "*" not in implementation.domains
        and domain not in implementation.domains
    ):
        return False
    if precision not in implementation.supported_compute_precisions:
        return False
    if (
        implementation.supported_compute_capabilities
        and device.compute_capability not in implementation.supported_compute_capabilities
    ):
        return False
    minimum = implementation.min_fp32_to_fp64_ratio
    if minimum is not None and (device.fp32_to_fp64_ratio or 0) < minimum:
        return False
    return True


def _accelerated_candidate(
    operation: TranscendentalOperation,
    policy: TranscendentalPolicy,
    device: DeviceCapability,
    domain: str,
    precision: ComputePrecision,
) -> tuple[StrategyImplementation | None, tuple[StrategyImplementation, ...], bool]:
    """Return the highest-priority eligible accelerated implementation.

    Hardware, policy, domain, and precision eligibility is applied before
    priority and ambiguity handling. The candidate tuple and boolean retain
    enough information to explain exact native fallback causes.
    """
    operation_candidates = tuple(
        implementation
        for implementation in _REGISTRY
        if implementation.operation is operation
        and implementation.implementation_id != NATIVE_LIBDEVICE
    )
    domain_candidates = tuple(
        implementation
        for implementation in operation_candidates
        if "*" in implementation.domains or domain in implementation.domains
    )
    eligible_candidates = tuple(
        implementation
        for implementation in domain_candidates
        if _supports(implementation, policy, device, domain, precision)
    )
    if not eligible_candidates:
        return None, domain_candidates, bool(operation_candidates)
    highest_priority = max(implementation.priority for implementation in eligible_candidates)
    preferred = tuple(
        implementation
        for implementation in eligible_candidates
        if implementation.priority == highest_priority
    )
    if len(preferred) > 1:
        implementation_ids = ", ".join(
            sorted(implementation.implementation_id for implementation in preferred)
        )
        raise RuntimeError(
            f"Ambiguous accelerated transcendental implementations for "
            f"{operation.value!r} in {domain!r} at priority {highest_priority}: "
            f"{implementation_ids}"
        )
    return preferred[0], domain_candidates, bool(operation_candidates)


def resolve_transcendental_strategy(
    operation: TranscendentalOperation | str,
    policy: TranscendentalPolicy = "auto",
    *,
    device: DeviceCapability | None = None,
    domain: str = "global",
    precision: str = "fp64",
    workload_size: int | None = None,
    _normalized: bool = False,
) -> StrategyDecision:
    """Resolve a public policy to an accuracy-qualified implementation.

    ``precision`` accepts ``"auto"``, ``"fp64"``, ``"fp32"``, or ``"ds"`` and
    remains independent of ``policy``. For ``policy="auto"``, a concrete
    ``workload_size`` must meet the selected implementation's ``min_elements``;
    ``None`` represents compile/explain planning and selects an otherwise
    qualified implementation. Explicit ``"accelerated"`` ignores the workload
    threshold but returns an observable native fallback when other
    qualifications are not met.
    """
    if _normalized:
        requested = cast(TranscendentalPolicy, policy)
        resolved_precision = cast(ComputePrecision, precision)
    else:
        requested = normalize_transcendental_policy(policy)
        resolved_precision = normalize_compute_precision(precision)
    try:
        resolved_operation = TranscendentalOperation(operation)
    except ValueError as exc:
        raise ValueError(f"Unknown transcendental operation: {operation!r}.") from exc
    if device is None:
        device = detect_device_capability()

    return _resolve_transcendental_strategy_cached(
        resolved_operation,
        requested,
        device,
        domain,
        resolved_precision,
        workload_size,
    )


@lru_cache(maxsize=256)
def _resolve_transcendental_strategy_cached(
    resolved_operation: TranscendentalOperation,
    requested: TranscendentalPolicy,
    device: DeviceCapability,
    domain: str,
    precision: ComputePrecision,
    workload_size: int | None,
) -> StrategyDecision:
    """Build and cache an immutable decision for an exact execution context."""

    native = _implementation(resolved_operation, NATIVE_LIBDEVICE)
    accelerated = None
    domain_candidates: tuple[StrategyImplementation, ...] = ()
    operation_has_accelerated = False
    if requested != "native":
        accelerated, domain_candidates, operation_has_accelerated = _accelerated_candidate(
            resolved_operation,
            requested,
            device,
            domain,
            precision,
        )

    if requested == "native":
        chosen = native
        reason = "native policy requested"
        fallback = False
    elif accelerated is not None and (
        requested == "accelerated"
        or workload_size is None
        or workload_size >= accelerated.min_elements
    ):
        chosen = accelerated
        reason = (
            "accuracy-qualified accelerated implementation selected for "
            f"{device.backend} {device.compute_capability} in {domain!r} domain"
        )
        fallback = False
    else:
        chosen = native
        fallback = requested == "accelerated"
        explanation_candidate = accelerated
        if explanation_candidate is None and len(domain_candidates) == 1:
            explanation_candidate = domain_candidates[0]
        if (
            accelerated is not None
            and requested == "auto"
            and workload_size is not None
            and workload_size < accelerated.min_elements
        ):
            unsupported_reason = (
                f"workload size {workload_size} is below the accelerated crossover "
                f"{accelerated.min_elements}"
            )
        elif not domain_candidates and operation_has_accelerated:
            unsupported_reason = f"domain {domain!r} is not accuracy-qualified"
        elif not operation_has_accelerated:
            unsupported_reason = "no accuracy-qualified accelerated implementation is registered"
        elif explanation_candidate is None:
            unsupported_reason = (
                "no accelerated implementation is eligible for the complete "
                "policy/device/precision context"
            )
        elif requested not in explanation_candidate.supported_policies:
            unsupported_reason = f"policy {requested!r} is not supported"
        elif device.backend != "cuda":
            unsupported_reason = f"{device.backend} backend has no CUDA accelerated implementation"
        elif precision not in explanation_candidate.supported_compute_precisions:
            unsupported_reason = f"compute precision {precision!r} is not supported"
        elif device.compute_capability not in explanation_candidate.supported_compute_capabilities:
            unsupported_reason = (
                f"compute capability {device.compute_capability!r} is not accuracy-qualified"
            )
        else:
            unsupported_reason = (
                f"fp32:fp64 ratio {device.fp32_to_fp64_ratio!r} does not meet "
                f"the required {explanation_candidate.min_fp32_to_fp64_ratio}"
            )
        if requested == "accelerated":
            reason = f"accelerated policy fell back to native: {unsupported_reason}"
        else:
            reason = f"auto policy selected native: {unsupported_reason}"

    return StrategyDecision(
        operation=resolved_operation,
        requested_policy=requested,
        implementation_id=chosen.implementation_id,
        family=chosen.family,
        reason=reason,
        fallback=fallback,
        accuracy=chosen.accuracy,
        device=device,
        domain=domain,
        workload_size=workload_size,
    )


def _resolve_exact_strategy(
    operation: TranscendentalOperation | str,
    implementation_id: str,
    *,
    device: DeviceCapability | None = None,
    domain: str = "global",
) -> StrategyDecision:
    """Resolve an exact internal variant for deterministic tests/benchmarks."""
    resolved_operation = TranscendentalOperation(operation)
    chosen = _implementation(resolved_operation, implementation_id)
    if device is None:
        device = detect_device_capability()
    return StrategyDecision(
        operation=resolved_operation,
        requested_policy="accelerated" if implementation_id != NATIVE_LIBDEVICE else "native",
        implementation_id=implementation_id,
        family=chosen.family,
        reason="exact internal implementation requested",
        fallback=False,
        accuracy=chosen.accuracy,
        device=device,
        domain=domain,
        workload_size=None,
    )


__all__ = [
    "AccuracyContract",
    "ComputePrecision",
    "DeviceCapability",
    "ExecutionContext",
    "GEOS_FORWARD_FIXED_Q62",
    "GEOS_FORWARD_FIXED_Q62_MIN_ELEMENTS",
    "GNOM_INVERSE_GUARDED_RSQRT_REFRAME",
    "HELMERT_FIXED_Q62",
    "HELMERT_FIXED_Q62_MIN_ELEMENTS",
    "LAEA_FORWARD_POLAR_FIXED_Q62",
    "LAEA_FORWARD_POLAR_FIXED_Q62_MIN_ELEMENTS",
    "LCC_FORWARD_CONFORMAL_REFRAME",
    "LCC_FORWARD_CONFORMAL_REFRAME_MIN_ELEMENTS",
    "LCC_INVERSE_CONFORMAL_REFRAME",
    "LCC_INVERSE_CONFORMAL_REFRAME_MIN_ELEMENTS",
    "MERC_FORWARD_ELLIPSOIDAL_PRODUCT_POLY",
    "MERC_FORWARD_SPHERICAL_PRODUCT_POLY",
    "MERC_FORWARD_SPHERICAL_PRODUCT_POLY_MIN_ELEMENTS",
    "MERC_INVERSE_EXP_SERIES",
    "MERC_INVERSE_EXP_SERIES_MIN_ELEMENTS",
    "NATIVE_LIBDEVICE",
    "ORTHO_FORWARD_FIXED_Q62",
    "ORTHO_FORWARD_FIXED_Q62_MIN_ELEMENTS",
    "ORTHO_INVERSE_GUARDED_REFRAME",
    "ORTHO_INVERSE_GUARDED_REFRAME_MIN_ELEMENTS",
    "PROJECTION_FIXED_Q62_MAX_SCALE_M",
    "ProjectionImplementation",
    "SINU_FORWARD_FIXED_Q62",
    "SINU_FORWARD_FIXED_Q62_MIN_ELEMENTS",
    "STERE_INVERSE_FIXED_Q62",
    "STERE_INVERSE_FIXED_Q62_MIN_ELEMENTS",
    "StrategyDecision",
    "StrategyExplanation",
    "StrategyImplementation",
    "TMERC_FIXED_Q62",
    "TMERC_FIXED_Q62_MIN_ELEMENTS",
    "TranscendentalOperation",
    "TranscendentalPolicy",
    "detect_device_capability",
    "list_transcendental_strategies",
    "normalize_compute_precision",
    "normalize_transcendental_policy",
    "projection_strategy_domain",
    "projection_strategy_domains",
    "resolve_transcendental_strategy",
]
