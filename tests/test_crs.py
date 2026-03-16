"""Tests for CRS parsing and resolution."""

from vibeproj.crs import parse_crs_input, resolve_projection_params


def test_parse_epsg_int():
    crs = parse_crs_input(4326)
    assert crs.to_epsg() == 4326


def test_parse_epsg_string():
    crs = parse_crs_input("EPSG:4326")
    assert crs.to_epsg() == 4326


def test_parse_epsg_string_lowercase():
    crs = parse_crs_input("epsg:4326")
    assert crs.to_epsg() == 4326


def test_parse_epsg_tuple():
    crs = parse_crs_input(("EPSG", 4326))
    assert crs.to_epsg() == 4326


def test_parse_plain_int_string():
    crs = parse_crs_input("4326")
    assert crs.to_epsg() == 4326


def test_resolve_geographic():
    crs = parse_crs_input(4326)
    params = resolve_projection_params(crs)
    assert params.projection_name == "longlat"


def test_resolve_utm():
    crs = parse_crs_input(32631)
    params = resolve_projection_params(crs)
    assert params.projection_name == "tmerc"
    assert params.utm_zone == 31
    assert not params.south


def test_resolve_utm_south():
    crs = parse_crs_input(32756)
    params = resolve_projection_params(crs)
    assert params.projection_name == "tmerc"
    assert params.utm_zone == 56
    assert params.south


def test_resolve_web_mercator():
    crs = parse_crs_input(3857)
    params = resolve_projection_params(crs)
    assert params.projection_name == "webmerc"


def test_resolve_lcc():
    crs = parse_crs_input(2154)  # France Lambert 93
    params = resolve_projection_params(crs)
    assert params.projection_name == "lcc"


def test_resolve_albers():
    crs = parse_crs_input(5070)  # NAD83 / Conus Albers
    params = resolve_projection_params(crs)
    assert params.projection_name == "aea"


def test_resolve_laea():
    crs = parse_crs_input(3035)  # ETRS89-LAEA Europe
    params = resolve_projection_params(crs)
    assert params.projection_name == "laea"


def test_resolve_polar_stereo():
    crs = parse_crs_input(3031)  # Antarctic
    params = resolve_projection_params(crs)
    assert params.projection_name == "stere"
