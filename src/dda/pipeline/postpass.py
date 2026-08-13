"""Fetch OSM building footprints via Geofabrik's PostPass (SQL-backed, GeoJSON output).

Distinct from Overpass. Output schema matches `buildings.py` so downstream `damage` is agnostic
to whether footprints came from fAIr detection or from OSM.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape

log = logging.getLogger(__name__)

POSTPASS_URL = "https://postpass.geofabrik.de/api/interpreter"
USER_AGENT = "dda-pipeline/0.1 (contact: krschap@duck.com)"


def fetch_postpass_buildings(aoi_geojson: Path, out_geojson: Path, timeout: int = 180) -> Path:
    aoi = gpd.read_file(aoi_geojson).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = aoi.total_bounds
    sql = (
        "SELECT osm_id, tags, geom FROM postpass_polygon "
        f"WHERE ST_Intersects(geom, ST_MakeEnvelope({minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f},4326)) "
        "AND tags ? 'building'"
    )
    log.info("querying PostPass: bbox %.4f,%.4f,%.4f,%.4f", minx, miny, maxx, maxy)
    body = urllib.parse.urlencode({"data": sql}).encode("utf-8")
    req = urllib.request.Request(
        POSTPASS_URL,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"PostPass HTTP {err.code}: {err.reason}") from err

    raw = payload.get("features", [])
    log.info("PostPass returned %d building features", len(raw))

    features = [
        {
            "type": "Feature",
            "geometry": shape(feat["geometry"]).__geo_interface__,
            "properties": {
                "id": i,
                "class": 1,
                "score": 1.0,
                "source": "osm",
                "osm_id": (feat.get("properties") or {}).get("osm_id"),
            },
        }
        for i, feat in enumerate(raw)
        if feat.get("geometry")
    ]
    fc = {"type": "FeatureCollection", "features": features}
    gdf = gpd.GeoDataFrame.from_features(fc, crs="EPSG:4326")
    inside = gdf[gdf.intersects(aoi.union_all())].reset_index(drop=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    inside[["id", "class", "score", "source", "osm_id", "geometry"]].to_file(out_geojson, driver="GeoJSON")
    log.info("wrote %d OSM buildings -> %s", len(inside), out_geojson)
    return out_geojson
