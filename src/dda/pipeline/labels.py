"""Label Studio round-trip for the damage fewshot loop.

Export chunks pre_aligned + post_aligned into tile pairs, ships them with a Label Studio
project (side-by-side pre/post, PolygonLabels attached to pre so destroyed-in-post buildings
are still captured, existing footprints pre-annotated so the labeler classifies rather than
draws). Import reverses tile pixel coords to geo and writes a `damage` GeoJSON that plugs
straight into `dda fewshot damage`.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon, box, mapping

log = logging.getLogger(__name__)

DEFAULT_TILE = 1024
DEFAULT_MIN_BUILDINGS = 1
LOCAL_FILES_URL = "/data/local-files/?d={rel}"

DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed", "un-classified")
CLASS_COLORS = {
    "no-damage": "#3ecc3e",
    "minor-damage": "#ffdc1e",
    "major-damage": "#ff8200",
    "destroyed": "#e61e1e",
    "un-classified": "#7878c8",
}


@dataclass(frozen=True)
class TileMeta:
    tile_id: str
    col_off: int
    row_off: int
    width: int
    height: int
    transform: list[float]
    crs: str
    n_buildings: int


def export_for_labelstudio(
    pre_aligned: Path,
    post_aligned: Path,
    buildings_geojson: Path,
    out_dir: Path,
    *,
    docroot_key: str,
    tile_size: int = DEFAULT_TILE,
    min_buildings: int = DEFAULT_MIN_BUILDINGS,
) -> Path:
    """Write tile pairs + Label Studio import JSON + interface XML under `out_dir`.

    `docroot_key` is the path from Label Studio's LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT to
    `out_dir` (e.g. `colombia_eq_pereira/label_export` when the container mounts `./outputs`
    at `/data`). Returned path is the tasks JSON to import into Label Studio.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = out_dir / "tiles" / "pre"
    post_dir = out_dir / "tiles" / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_file(buildings_geojson)
    with rasterio.open(pre_aligned) as pre, rasterio.open(post_aligned) as post:
        _check_grids_match(pre, post)
        raster_crs = post.crs
        buildings_r = buildings.to_crs(raster_crs)
        metas: list[TileMeta] = []
        tasks: list[dict] = []
        for row_off in range(0, post.height, tile_size):
            for col_off in range(0, post.width, tile_size):
                w = min(tile_size, post.width - col_off)
                h = min(tile_size, post.height - row_off)
                win = Window(col_off, row_off, w, h)  # ty: ignore[too-many-positional-arguments]
                tile_transform = post.window_transform(win)
                left, top = tile_transform * (0, 0)
                right, bottom = tile_transform * (w, h)
                bbox = (
                    min(left, right),
                    min(top, bottom),
                    max(left, right),
                    max(top, bottom),
                )
                tile_buildings = buildings_r.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
                if len(tile_buildings) < min_buildings:
                    continue

                post_arr = post.read([1, 2, 3], window=win)
                if _mostly_nodata(post_arr):
                    continue
                pre_arr = pre.read([1, 2, 3], window=win)

                tile_id = f"{row_off:06d}_{col_off:06d}"
                Image.fromarray(post_arr.transpose(1, 2, 0)).save(post_dir / f"{tile_id}.png")
                Image.fromarray(pre_arr.transpose(1, 2, 0)).save(pre_dir / f"{tile_id}.png")

                meta = TileMeta(
                    tile_id=tile_id,
                    col_off=col_off,
                    row_off=row_off,
                    width=w,
                    height=h,
                    transform=list(tile_transform)[:6],
                    crs=str(raster_crs),
                    n_buildings=len(tile_buildings),
                )
                metas.append(meta)
                tasks.append(
                    _build_task(
                        docroot_key=docroot_key,
                        tile_id=tile_id,
                        tile_meta=meta,
                        tile_buildings=tile_buildings,
                    )
                )

    (out_dir / "tiles" / "meta.json").write_text(json.dumps({m.tile_id: asdict(m) for m in metas}, indent=2))
    tasks_path = out_dir / "tasks.json"
    tasks_path.write_text(json.dumps(tasks, indent=2))
    (out_dir / "config.xml").write_text(_render_config_xml())
    log.info(
        "label-export: %d tiles, %d buildings kept, tile=%d, min_buildings=%d -> %s",
        len(metas),
        sum(m.n_buildings for m in metas),
        tile_size,
        min_buildings,
        out_dir,
    )
    return tasks_path


def import_from_labelstudio(
    ls_export: Path,
    meta_json: Path,
    out_geojson: Path,
) -> Path:
    """Convert a Label Studio JSON export into a damage-labeled GeoJSON.

    Reads polygons in tile pixel space (LS exports normalised 0-100 percentages), reverse-maps
    via each tile's affine to geo coords in the raster CRS, writes a FeatureCollection with a
    `damage` string column that `dda fewshot damage` understands.
    """
    metas = {tid: TileMeta(**m) for tid, m in json.loads(meta_json.read_text()).items()}
    export = json.loads(ls_export.read_text())
    if not isinstance(export, list):
        raise ValueError(
            f"label-import: expected a JSON array from Label Studio, got {type(export).__name__}"
        )

    feats: list[dict] = []
    tile_crs: str | None = None
    for task in export:
        tile_id = _tile_id_from_task(task)
        if tile_id not in metas:
            log.warning("label-import: tile_id %s not in meta.json, skipping task", tile_id)
            continue
        meta = metas[tile_id]
        tile_crs = tile_crs or meta.crs
        aff = Affine(*meta.transform)
        for ann in task.get("annotations", []) or task.get("completions", []):
            for r in ann.get("result", []):
                if r.get("type") != "polygonlabels":
                    continue
                value = r["value"]
                labels = value.get("polygonlabels") or value.get("labels") or []
                if not labels:
                    continue
                label = labels[0]
                if label not in DAMAGE_CLASSES:
                    log.warning("label-import: unknown label %r on tile %s, skipping", label, tile_id)
                    continue
                w_ls = float(r.get("original_width", meta.width))
                h_ls = float(r.get("original_height", meta.height))
                pts_pct = value["points"]
                geo_pts = [aff * (px * w_ls / 100.0, py * h_ls / 100.0) for px, py in pts_pct]
                if len(geo_pts) < 3:
                    continue
                poly = Polygon(geo_pts)
                if not poly.is_valid or poly.area == 0:
                    continue
                feats.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(poly),
                        "properties": {"damage": label, "tile_id": tile_id},
                    }
                )

    if not feats:
        raise RuntimeError(f"label-import: no polygon annotations found in {ls_export}")

    gdf = gpd.GeoDataFrame.from_features(feats, crs=tile_crs)
    gdf.to_file(out_geojson, driver="GeoJSON")
    counts = gdf["damage"].value_counts().to_dict()
    log.info("label-import: wrote %d labels -> %s (%s)", len(gdf), out_geojson, counts)
    return out_geojson


def _check_grids_match(pre, post) -> None:
    if pre.crs != post.crs or pre.transform != post.transform or pre.shape != post.shape:
        raise ValueError(
            "label-export: pre_aligned and post_aligned must share crs+transform+shape (run coregister first)"
        )


def _mostly_nodata(arr: np.ndarray, threshold: float = 0.5) -> bool:
    valid = (arr > 0).any(axis=0)
    return float(valid.mean()) < threshold


def _tile_id_from_task(task: dict) -> str:
    for key in ("pre", "post"):
        url = task.get("data", {}).get(key)
        if url:
            return Path(url.split("?")[-1].rsplit("=", 1)[-1]).stem
    raise ValueError(f"label-import: task has no pre/post URL: {task.get('id')}")


def _build_task(
    *,
    docroot_key: str,
    tile_id: str,
    tile_meta: TileMeta,
    tile_buildings: gpd.GeoDataFrame,
) -> dict:
    pre_rel = f"{docroot_key}/tiles/pre/{tile_id}.png"
    post_rel = f"{docroot_key}/tiles/post/{tile_id}.png"
    predictions = _footprints_to_predictions(tile_buildings, tile_meta) if len(tile_buildings) else []
    return {
        "data": {
            "pre": LOCAL_FILES_URL.format(rel=pre_rel),
            "post": LOCAL_FILES_URL.format(rel=post_rel),
            "tile_id": tile_id,
        },
        "predictions": predictions,
        "meta": {"tile_id": tile_id, "n_buildings": tile_meta.n_buildings},
    }


def _polygons_only(geom) -> list:
    """Flatten a geometry into its Polygon components, dropping Points and Lines from clipping."""
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if g.geom_type == "Polygon"]
    return []


def _footprints_to_predictions(tile_buildings: gpd.GeoDataFrame, tile_meta: TileMeta) -> list[dict]:
    """Emit one Label Studio prediction per building so the labeler classifies, not draws.

    Building geometry is clipped to the tile bbox in geo space so predictions never render
    outside the canvas.
    """
    aff = Affine(*tile_meta.transform)
    inv = ~aff
    left, top = aff * (0, 0)
    right, bottom = aff * (tile_meta.width, tile_meta.height)
    tile_bbox = box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
    results = []
    for geom in tile_buildings.geometry:
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(tile_bbox)
        if clipped.is_empty:
            continue
        polys = _polygons_only(clipped)
        for poly in polys:
            xs, ys = poly.exterior.coords.xy
            pts_pct = []
            for x, y in zip(xs, ys, strict=False):
                px, py = inv * (x, y)
                pts_pct.append(
                    [
                        max(0.0, min(100.0, px / tile_meta.width * 100.0)),
                        max(0.0, min(100.0, py / tile_meta.height * 100.0)),
                    ]
                )
            if len(pts_pct) < 3:
                continue
            results.append(
                {
                    "value": {"points": pts_pct, "polygonlabels": ["un-classified"]},
                    "from_name": "dmg",
                    "to_name": "pre",
                    "type": "polygonlabels",
                    "original_width": tile_meta.width,
                    "original_height": tile_meta.height,
                }
            )
    return [{"result": results}] if results else []


def _render_config_xml() -> str:
    labels = "\n    ".join(
        f'<Label value="{name}" background="{CLASS_COLORS[name]}"/>' for name in DAMAGE_CLASSES
    )
    return f"""<View style="display:flex">
  <View style="flex:1;padding:5px">
    <Header value="PRE"/>
    <Image name="pre" value="$pre" zoom="true" zoomControl="true" rotateControl="false"/>
  </View>
  <View style="flex:1;padding:5px">
    <Header value="POST"/>
    <Image name="post" value="$post" zoom="true" zoomControl="true" rotateControl="false"/>
  </View>
  <PolygonLabels name="dmg" toName="pre">
    {labels}
  </PolygonLabels>
</View>
"""
