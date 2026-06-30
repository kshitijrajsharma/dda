"""Assign each building footprint a damage class and confidence from a dense probability map.

Torch-free (serving needs only geopandas + rasterio + numpy). Severity is pooled at a high
percentile, not the mean, since damage is often local; confidence is the assigned class's mean
probability over the footprint.
"""

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.features import geometry_mask

from dda.config import DAMAGE_CLASSES, N_DAMAGE_CLASSES

log = logging.getLogger(__name__)

NO_DATA_CLASS = -1
_ORDINAL = np.arange(N_DAMAGE_CLASSES)


def _pool_one(pix: np.ndarray, pool_op: str, percentile: float) -> tuple[int, np.ndarray]:
    """pix is (C, n_pixels) of per-class probabilities. Returns (class, mean per-class prob)."""
    agg = pix.mean(axis=1)
    if pool_op == "mean":
        return int(agg.argmax()), agg
    if pool_op == "max":
        return int(pix.argmax(axis=0).max()), agg
    severity = (_ORDINAL[:, None] * pix).sum(axis=0)  # expected ordinal value per pixel
    sev = float(np.percentile(severity, percentile))
    return int(np.clip(round(sev), 0, N_DAMAGE_CLASSES - 1)), agg


def assign_damage(
    prob: np.ndarray,
    transform: "rasterio.Affine",
    crs: Any,
    buildings: gpd.GeoDataFrame,
    pool_op: str = "percentile",
    percentile: float = 80.0,
    confidence_threshold: float = 0.5,
) -> gpd.GeoDataFrame:
    """prob is (C, H, W) softmax probabilities in raster space. Returns the input footprints with
    `damage_class`, `damage`, `confidence`, `review`, and per-class probability columns."""
    height, width = prob.shape[1:]
    inv = ~transform
    out = buildings.to_crs(crs).copy()
    classes, labels, confs, reviews, per_class = [], [], [], [], []

    for geom in out.geometry:
        # Mask within the footprint's pixel window only; rasterising over the full grid per
        # building does not scale to large scenes.
        minx, miny, maxx, maxy = geom.bounds
        c_tl, r_tl = inv * (minx, maxy)
        c_br, r_br = inv * (maxx, miny)
        c0, c1 = max(0, int(c_tl)), min(width, int(c_br) + 1)
        r0, r1 = max(0, int(r_tl)), min(height, int(r_br) + 1)
        if c1 <= c0 or r1 <= r0:
            classes.append(NO_DATA_CLASS)
            labels.append("no-data")
            confs.append(0.0)
            reviews.append(True)
            per_class.append(np.full(N_DAMAGE_CLASSES, np.nan))
            continue
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)  # ty: ignore[too-many-positional-arguments]
        win_t = rasterio.windows.transform(window, transform)
        inside = geometry_mask([geom], out_shape=(r1 - r0, c1 - c0), transform=win_t, invert=True)
        if not inside.any():
            classes.append(NO_DATA_CLASS)
            labels.append("no-data")
            confs.append(0.0)
            reviews.append(True)
            per_class.append(np.full(N_DAMAGE_CLASSES, np.nan))
            continue
        pix = prob[:, r0:r1, c0:c1][:, inside]
        cls, agg = _pool_one(pix, pool_op, percentile)
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
    log.info("Assigned damage to %d buildings (%d flagged for review)", len(out), sum(reviews))
    return out
