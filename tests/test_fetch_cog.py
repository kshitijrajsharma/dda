"""Regression: `_fetch_cog` must intersect the requested window with the COG's own extent so
the written raster's transform still matches its pixel content. The Kumamoto v0.2 run wrote a
post.tif whose transform was shifted ~4 km north of the actual imagery because the AOI reached
above the source COG; pre_aligned then reprojected onto that bad grid, and the damage model saw
massive fake change everywhere.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from dda.pipeline.fetch import _fetch_cog


def _write_synthetic_cog(path: Path) -> None:
    """32x32 3-band raster in EPSG:3857, top-left at (0, 32), 1 m/px. Content = row index in R."""
    transform = from_origin(0.0, 32.0, 1.0, 1.0)
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    arr[0, :, :] = np.arange(32, dtype=np.uint8)[:, None]  # R band = row idx (0..31)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(arr)


def _write_aoi(path: Path, west: float, south: float, east: float, north: float) -> None:
    poly = Polygon([(west, south), (east, south), (east, north), (west, north)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:3857").to_crs("EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


def test_fetch_cog_transform_matches_data_when_aoi_extends_beyond(tmp_path: Path) -> None:
    """AOI top is ABOVE the COG top. The written raster's transform must match its actual data
    (start at the COG's real top row), not the requested-but-unavailable extent."""
    src = tmp_path / "src.tif"
    aoi = tmp_path / "aoi.geojson"
    out = tmp_path / "out.tif"
    _write_synthetic_cog(src)
    # AOI extends 10 m above the COG (top y = 42 vs COG top = 32) and 10 m below.
    _write_aoi(aoi, west=0, south=-10, east=32, north=42)

    _fetch_cog(str(src), aoi, out)

    with rasterio.open(out) as dst:
        # Row 0 of the written raster must be the COG's row 0 (whose R value = 0), not padded
        # nodata that would land at a phantom "row 0" corresponding to the AOI's ghost top.
        row0 = dst.read(1)[0]
        assert row0[0] == 0, f"row 0 R band should be 0 (COG top), got {row0[0]}"
        # And the transform's top-left y must equal the COG's top-left y (32.0), not the AOI's.
        assert abs(dst.transform.f - 32.0) < 1e-6, (
            f"transform.f should stay at COG top 32.0, got {dst.transform.f}"
        )


def test_fetch_cog_transform_matches_data_when_aoi_inside(tmp_path: Path) -> None:
    """AOI is fully inside the COG. Nothing exotic; verify baseline still works."""
    src = tmp_path / "src.tif"
    aoi = tmp_path / "aoi.geojson"
    out = tmp_path / "out.tif"
    _write_synthetic_cog(src)
    _write_aoi(aoi, west=4, south=4, east=20, north=20)

    _fetch_cog(str(src), aoi, out)

    with rasterio.open(out) as dst:
        # Row 0 should be COG row 12 (top of raster is y=32, AOI top=20, so 32-20=12 rows down).
        assert dst.read(1)[0][0] == 12
        assert abs(dst.transform.f - 20.0) < 1e-6
