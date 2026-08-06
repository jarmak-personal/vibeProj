"""Shared coefficient builders for conformal-latitude series."""

from __future__ import annotations


def conformal_to_geodetic_coefficients(third_flattening: float) -> tuple[float, ...]:
    """Return the sixth-order Poder/Engsager conformal-to-geodetic series."""
    n = third_flattening
    power = n
    c0 = n * (
        2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0)))))
    )
    power *= n
    c1 = power * (
        7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0))))
    )
    power *= n
    c2 = power * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))))
    power *= n
    c3 = power * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0)))
    power *= n
    c4 = power * (4174 / 315.0 + n * (-144838 / 6237.0))
    power *= n
    c5 = power * (601676 / 22275.0)
    return (
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
    )


__all__ = ["conformal_to_geodetic_coefficients"]
