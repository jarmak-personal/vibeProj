"""Custom exceptions for vibeProj."""


class VibeProjectionError(Exception):
    """Base exception for vibeProj errors."""


class UnsupportedProjectionError(VibeProjectionError):
    """Raised when a CRS uses a projection method vibeProj doesn't support."""


class CRSResolutionError(VibeProjectionError):
    """Raised when a CRS input cannot be parsed or resolved."""


class CoordinateValidationError(VibeProjectionError):
    """Raised when input coordinates are invalid (wrong shape, dtype, etc.)."""
