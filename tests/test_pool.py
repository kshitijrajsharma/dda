import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from dda.config import N_DAMAGE_CLASSES
from dda.pool import _pool_one, assign_damage


def _onehot_pixels(class_counts: dict[int, int]) -> np.ndarray:
    cols = []
    for cls, n in class_counts.items():
        col = np.zeros((N_DAMAGE_CLASSES, n))
        col[cls] = 1.0
        cols.append(col)
    return np.concatenate(cols, axis=1)


def test_percentile_keeps_local_damage_that_mean_dilutes():
    pix = _onehot_pixels({0: 5, 3: 5})  # half intact, half destroyed
    cls_pct, _ = _pool_one(pix, "percentile", 80.0)
    cls_mean, _ = _pool_one(pix, "mean", 80.0)
    assert cls_pct > cls_mean


def test_uniform_region_class():
    pix = _onehot_pixels({2: 20})
    cls, agg = _pool_one(pix, "percentile", 80.0)
    assert cls == 2
    assert agg.argmax() == 2


def test_assign_damage_uniform_and_nodata():
    height = width = 8
    transform = from_origin(0, height, 1, 1)
    prob = np.zeros((N_DAMAGE_CLASSES, height, width), dtype=np.float32)
    prob[3, 1:4, 1:4] = 1.0  # destroyed block
    prob[0] = np.where(prob.sum(axis=0) == 0, 1.0, 0.0)  # rest no-damage

    inside = box(1.2, height - 3.8, 3.8, height - 1.2)  # over the destroyed block
    outside = box(100, 100, 101, 101)  # off-raster -> no-data
    buildings = gpd.GeoDataFrame(geometry=[inside, outside], crs="EPSG:3857")

    out = assign_damage(prob, transform, "EPSG:3857", buildings, pool_op="percentile", percentile=80.0)
    assert out.iloc[0]["damage"] == "destroyed"
    assert out.iloc[0]["confidence"] > 0.9
    assert out.iloc[1]["damage_class"] == -1
    assert bool(out.iloc[1]["review"]) is True
    assert {"p_no_damage", "p_destroyed"}.issubset(out.columns)
