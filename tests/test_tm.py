"""Tests for `dda.pipeline.tm`: TM v2 project fetch + AOI union."""

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

from dda.pipeline.tm import TMProject, fetch_tm_project, union_aois


def _canned_response(payload: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


class _FakeCtx:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _patch_urlopen(monkeypatch, payload_or_error):
    def _fake(req, timeout):
        if isinstance(payload_or_error, Exception):
            raise payload_or_error
        return _FakeCtx(json.dumps(payload_or_error).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake)


def test_fetch_tm_project_happy_path(monkeypatch):
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                "properties": {},
            }
        ],
    }
    _patch_urlopen(
        monkeypatch,
        {
            "projectId": 1234,
            "areaOfInterest": aoi,
            "imagery": "https://tiles.example/{z}/{x}/{y}.png",
            "projectInfo": {"name": "Example project"},
        },
    )
    proj = fetch_tm_project(1234)
    assert isinstance(proj, TMProject)
    assert proj.project_id == 1234
    assert proj.name == "Example project"
    assert proj.aoi == aoi
    assert proj.imagery_tms == "https://tiles.example/{z}/{x}/{y}.png"


def test_fetch_tm_project_null_imagery(monkeypatch):
    _patch_urlopen(
        monkeypatch,
        {
            "areaOfInterest": {"type": "FeatureCollection", "features": []},
            "imagery": None,
            "projectInfo": {"name": "No imagery"},
        },
    )
    proj = fetch_tm_project(1)
    assert proj.imagery_tms is None


def test_fetch_tm_project_empty_string_imagery_becomes_none(monkeypatch):
    _patch_urlopen(
        monkeypatch,
        {
            "areaOfInterest": {"type": "FeatureCollection", "features": []},
            "imagery": "   ",
            "projectInfo": {"name": "Blank imagery"},
        },
    )
    assert fetch_tm_project(1).imagery_tms is None


def test_fetch_tm_project_missing_aoi_fails_loud(monkeypatch):
    _patch_urlopen(monkeypatch, {"imagery": "x", "projectInfo": {"name": "x"}})
    with pytest.raises(RuntimeError, match="areaOfInterest"):
        fetch_tm_project(99)


def test_fetch_tm_project_http_error_wrapped(monkeypatch):
    err = urllib.error.HTTPError(
        url="https://example",
        code=404,
        msg="Not Found",
        hdrs=email.message.Message(),
        fp=None,
    )
    _patch_urlopen(monkeypatch, err)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch_tm_project(999999)


def test_union_aois_preserves_tm_project_id():
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    projects = [
        TMProject(
            1,
            "a",
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": poly, "properties": {}}],
            },
            None,
        ),
        TMProject(
            2,
            "b",
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": poly, "properties": {}}],
            },
            None,
        ),
    ]
    fc = union_aois(projects)
    assert fc["type"] == "FeatureCollection"
    assert [f["properties"]["tm_project_id"] for f in fc["features"]] == [1, 2]


def test_union_aois_handles_bare_geometry():
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    proj = TMProject(7, "raw", poly, None)
    fc = union_aois([proj])
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["tm_project_id"] == 7
    assert fc["features"][0]["geometry"] == poly
