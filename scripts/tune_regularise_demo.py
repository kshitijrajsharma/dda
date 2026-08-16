"""Smoke-test the RegulariseParams TPE tuner on Pereira against OSM buildings."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd

from dda.pipeline.postpass import fetch_postpass_buildings
from dda.pipeline.tune_regularise import tune_regularise

log = logging.getLogger(__name__)

AOI_DIR = Path("/home/krschap/code/hotosm/dda/outputs/colombia_eq_pereira")
RAW_PATH = AOI_DIR / "buildings.geojson"
AOI_PATH = AOI_DIR / "aoi.geojson"
PRE_RASTER = AOI_DIR / "pre_aligned.tif"
GT_PATH = AOI_DIR / "osm_buildings_gt.geojson"
OUT_JSON = AOI_DIR / "regularise_tuned.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"missing raw predictions: {RAW_PATH}")
    if not AOI_PATH.exists():
        raise FileNotFoundError(f"missing AOI: {AOI_PATH}")
    if not GT_PATH.exists():
        log.info("no cached OSM GT; fetching via PostPass -> %s", GT_PATH)
        fetch_postpass_buildings(AOI_PATH, GT_PATH)

    raw = gpd.read_file(RAW_PATH).to_crs("EPSG:4326")
    gt = gpd.read_file(GT_PATH).to_crs("EPSG:4326")
    log.info("loaded raw=%d gt=%d polygons", len(raw), len(gt))

    log.info("starting Optuna TPE: n_trials=%d seed=%d", args.n_trials, args.seed)
    best_params, report = tune_regularise(
        raw_gdf=raw,
        gt_gdf=gt,
        raster_path=str(PRE_RASTER) if PRE_RASTER.exists() else None,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    log.info("best_value=%.4f trials=%d elapsed=%.1fs",
             report["best_value"], report["n_trials"], report["elapsed_s"])

    OUT_JSON.write_text(json.dumps({
        "best_value": report["best_value"],
        "n_trials": report["n_trials"],
        "elapsed_s": report["elapsed_s"],
        "best_params": asdict(best_params),
    }, indent=2))
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
