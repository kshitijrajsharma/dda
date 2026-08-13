"""Block-tiled damage assessment. Buildings with under 90% pre OR post coverage are marked
`no-data` (class -1) so tile-mosaic gaps don't score as confident 'destroyed'."""

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import torch
from omegaconf import DictConfig
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window

from dda.config import DAMAGE_CLASSES, N_DAMAGE_CLASSES
from dda.infer import load_model, radiometric_normalize, sliding_window_prob

log = logging.getLogger(__name__)

NO_DATA_CLASS = -1
NO_PRE_DATA_LABEL = "no-data (no pre)"
NO_POST_DATA_LABEL = "no-data (no post)"
NO_DATA_LABEL = NO_PRE_DATA_LABEL


@dataclass
class DamageConfig:
    core_px: int = 4096
    halo_px: int = 512
    pre_coverage_threshold: float = 0.9
    post_coverage_threshold: float = 0.9


def run_damage_blocked(
    cfg: DictConfig,
    ckpt_path: str | Path,
    post_raster: Path,
    pre_aligned: Path,
    buildings_geojson: Path,
    out_geojson: Path,
    damage_cfg: DamageConfig | None = None,
    device: str | None = None,
) -> Path:
    """Run the model in core + halo blocks and pool damage per building. Returns `out_geojson`."""
    damage_cfg = damage_cfg or DamageConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, cfg, device=device)

    buildings = gpd.read_file(buildings_geojson)
    with rasterio.open(post_raster) as post_src:
        post_crs = post_src.crs
        post_transform = post_src.transform
        post_height, post_width = post_src.height, post_src.width

    buildings_in_post = buildings.to_crs(post_crs)
    buildings_in_post["centroid_px"] = buildings_in_post.geometry.centroid.map(
        lambda p: (~post_transform) * (p.x, p.y)
    )

    core = damage_cfg.core_px
    halo = damage_cfg.halo_px
    blocks = [(cx, cy) for cy in range(0, post_height, core) for cx in range(0, post_width, core)]
    log.info(
        "damage %dx%d px, %d blocks (core=%d, halo=%d), %d buildings",
        post_width,
        post_height,
        len(blocks),
        core,
        halo,
        len(buildings_in_post),
    )

    accum: list[gpd.GeoDataFrame] = []
    for i, (cx, cy) in enumerate(blocks, start=1):
        block_gdf = _process_block(
            cfg=cfg,
            model=model,
            post_raster=post_raster,
            pre_aligned=pre_aligned,
            buildings=buildings_in_post,
            block_x=cx,
            block_y=cy,
            core=core,
            halo=halo,
            post_width=post_width,
            post_height=post_height,
            damage_cfg=damage_cfg,
            device=device,
        )
        log.info("block %d/%d -> %d buildings", i, len(blocks), len(block_gdf))
        if len(block_gdf) > 0:
            accum.append(block_gdf)

    out = pd.concat(accum, ignore_index=True) if accum else buildings_in_post.iloc[0:0].copy()
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=post_crs).to_crs(buildings.crs)
    out = out.drop(columns=[c for c in ["centroid_px"] if c in out.columns])
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_geojson, driver="GeoJSON")
    log.info("wrote %d buildings with damage -> %s", len(out), out_geojson)
    return out_geojson


def _process_block(
    *,
    cfg: DictConfig,
    model,
    post_raster: Path,
    pre_aligned: Path,
    buildings: gpd.GeoDataFrame,
    block_x: int,
    block_y: int,
    core: int,
    halo: int,
    post_width: int,
    post_height: int,
    damage_cfg: DamageConfig,
    device: str,
) -> gpd.GeoDataFrame:
    """One block: assign each contained-in-core building a class + confidence."""
    core_x0, core_y0 = block_x, block_y
    core_x1, core_y1 = min(block_x + core, post_width), min(block_y + core, post_height)
    in_core = buildings["centroid_px"].map(
        lambda cp: core_x0 <= cp[0] < core_x1 and core_y0 <= cp[1] < core_y1
    )
    block_buildings = buildings.loc[in_core].copy()
    if len(block_buildings) == 0:
        return _empty_result(buildings.crs)

    read_x0 = max(0, core_x0 - halo)
    read_y0 = max(0, core_y0 - halo)
    read_x1 = min(post_width, core_x1 + halo)
    read_y1 = min(post_height, core_y1 + halo)
    window = Window(read_x0, read_y0, read_x1 - read_x0, read_y1 - read_y0)  # ty: ignore[too-many-positional-arguments]

    with rasterio.open(post_raster) as post_src:
        post_arr = post_src.read([1, 2, 3], window=window).transpose(1, 2, 0).astype(np.uint8)
        block_transform = post_src.window_transform(window)
        block_crs = post_src.crs
    with rasterio.open(pre_aligned) as pre_src:
        pre_arr = pre_src.read([1, 2, 3], window=window).transpose(1, 2, 0).astype(np.uint8)

    if cfg.radiometric_normalize:
        if post_arr.any():
            post_arr = radiometric_normalize(post_arr)
        if pre_arr.any():
            pre_arr = radiometric_normalize(pre_arr)

    pre_valid = pre_arr.any(axis=2)
    post_valid = post_arr.any(axis=2)
    pre_frac = _coverage_fraction(block_buildings, pre_valid, block_transform)
    post_frac = _coverage_fraction(block_buildings, post_valid, block_transform)
    pre_ok = pre_frac >= damage_cfg.pre_coverage_threshold
    post_ok = post_frac >= damage_cfg.post_coverage_threshold
    valid_mask = pre_ok & post_ok

    covered_buildings = block_buildings.loc[valid_mask].copy()
    # Pre-failure dominates; separate label for post-only gaps lets operators spot mosaic 404s.
    no_pre_buildings = _label_no_data(block_buildings.loc[~pre_ok].copy(), NO_PRE_DATA_LABEL)
    no_post_buildings = _label_no_data(block_buildings.loc[pre_ok & ~post_ok].copy(), NO_POST_DATA_LABEL)

    scored = _empty_result(block_buildings.crs)
    if len(covered_buildings) > 0:
        prob = sliding_window_prob(
            model,
            post_arr,
            pre_arr,
            cfg.tile_window,
            cfg.tile_stride,
            cfg.temperature,
            device,
        )
        scored = _assign_from_prob(
            prob=prob,
            transform=block_transform,
            crs=block_crs,
            buildings=covered_buildings,
            pool_op=cfg.pool_op,
            percentile=cfg.pool_percentile,
            confidence_threshold=cfg.confidence_threshold,
        )

    merged = pd.concat([scored, no_pre_buildings, no_post_buildings], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=scored.crs)


def _coverage_fraction(buildings: gpd.GeoDataFrame, valid: np.ndarray, transform) -> np.ndarray:
    """Rasterize buildings on the block grid, return per-building fraction of valid pixels."""
    height, width = valid.shape
    ids = np.arange(1, len(buildings) + 1, dtype=np.int32)
    id_raster = rasterize(
        [(geom, i) for geom, i in zip(buildings.geometry, ids, strict=True)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    coverage = np.zeros(len(buildings), dtype=np.float32)
    for i, bid in enumerate(ids, start=0):
        mask = id_raster == bid
        pixels = int(mask.sum())
        if pixels == 0:
            coverage[i] = 0.0
            continue
        coverage[i] = float((valid[mask]).sum()) / pixels
    return coverage


def _label_no_data(buildings: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Attach no-data columns. `label` distinguishes 'no pre' from 'no post'."""
    out = buildings.copy()
    out["damage_class"] = NO_DATA_CLASS
    out["damage"] = label
    out["confidence"] = 0.0
    out["review"] = True
    for name in DAMAGE_CLASSES:
        out[f"p_{name.replace('-', '_')}"] = np.nan
    return out


def _empty_result(crs) -> gpd.GeoDataFrame:
    cols = ["damage_class", "damage", "confidence", "review", "geometry"] + [
        f"p_{n.replace('-', '_')}" for n in DAMAGE_CLASSES
    ]
    return gpd.GeoDataFrame({c: [] for c in cols}, geometry="geometry", crs=crs)


def _assign_from_prob(
    *,
    prob: np.ndarray,
    transform,
    crs,
    buildings: gpd.GeoDataFrame,
    pool_op: str,
    percentile: float,
    confidence_threshold: float,
) -> gpd.GeoDataFrame:
    """Percentile-pool probs inside each footprint, write class + confidence + per-class probs."""
    height, width = prob.shape[1:]
    inv = ~transform
    out = buildings.copy()
    ordinal = np.arange(N_DAMAGE_CLASSES)
    classes, labels, confs, reviews, per_class = [], [], [], [], []

    for geom in out.geometry:
        minx, miny, maxx, maxy = geom.bounds
        c_tl, r_tl = inv * (minx, maxy)
        c_br, r_br = inv * (maxx, miny)
        c0, c1 = max(0, int(c_tl)), min(width, int(c_br) + 1)
        r0, r1 = max(0, int(r_tl)), min(height, int(r_br) + 1)
        if c1 <= c0 or r1 <= r0:
            classes.append(NO_DATA_CLASS)
            labels.append(NO_DATA_LABEL)
            confs.append(0.0)
            reviews.append(True)
            per_class.append(np.full(N_DAMAGE_CLASSES, np.nan))
            continue
        window = Window(c0, r0, c1 - c0, r1 - r0)  # ty: ignore[too-many-positional-arguments]
        win_t = rasterio.windows.transform(window, transform)
        inside = geometry_mask([geom], out_shape=(r1 - r0, c1 - c0), transform=win_t, invert=True)
        if not inside.any():
            classes.append(NO_DATA_CLASS)
            labels.append(NO_DATA_LABEL)
            confs.append(0.0)
            reviews.append(True)
            per_class.append(np.full(N_DAMAGE_CLASSES, np.nan))
            continue
        pix = prob[:, r0:r1, c0:c1][:, inside]
        cls, agg = _pool_pixels(pix, pool_op, percentile, ordinal)
        conf = float(pix[cls].mean())
        classes.append(cls)
        labels.append(DAMAGE_CLASSES[cls])
        confs.append(conf)
        reviews.append(conf < confidence_threshold)
        per_class.append(agg)

    out["damage_class"] = classes
    out["damage"] = labels
    out["confidence"] = confs
    out["review"] = reviews
    stacked = np.vstack(per_class) if per_class else np.empty((0, N_DAMAGE_CLASSES))
    for k, name in enumerate(DAMAGE_CLASSES):
        out[f"p_{name.replace('-', '_')}"] = stacked[:, k]
    _ = crs
    return out


def _pool_pixels(
    pix: np.ndarray, pool_op: str, percentile: float, ordinal: np.ndarray
) -> tuple[int, np.ndarray]:
    agg = pix.mean(axis=1)
    if pool_op == "mean":
        return int(agg.argmax()), agg
    if pool_op == "max":
        return int(pix.argmax(axis=0).max()), agg
    severity = (ordinal[:, None] * pix).sum(axis=0)
    sev = float(np.percentile(severity, percentile))
    return int(np.clip(round(sev), 0, N_DAMAGE_CLASSES - 1)), agg
