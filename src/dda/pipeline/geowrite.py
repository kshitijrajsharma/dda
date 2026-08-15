"""Twin-write a GeoDataFrame as GeoJSON + GeoParquet-1.1 WKB with the spec-covering bbox column."""

from pathlib import Path

import geopandas as gpd


def write_dual(gdf: gpd.GeoDataFrame, geojson_path: Path) -> Path:
    """Write `<name>.geojson` and `<name>.parquet` from the same gdf; returns the parquet path."""
    geojson_path = Path(geojson_path)
    parquet_path = geojson_path.with_suffix(".parquet")
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_parquet(
        parquet_path,
        compression="zstd",
        index=False,
        schema_version="1.1.0",
        write_covering_bbox=True,
    )
    return parquet_path
