"""End-to-end orchestrator for `dda run --config event.yaml`. Stages run in fixed order and
call the same functions the standalone CLI subcommands use, so behaviour matches per-stage
runs. Outputs follow `PipelinePaths` under `outputs/<area>/`."""

import json
import logging
from pathlib import Path

from omegaconf import DictConfig

from dda.pipeline.paths import PipelinePaths

log = logging.getLogger(__name__)

STAGES = ("aoi", "prepare", "fewshot", "buildings", "damage", "publish")


def run_event(
    cfg: DictConfig,
    start: str | None = None,
    only: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run stages of the event pipeline. `start` runs from that stage through the end.
    `only` runs exactly that one stage. `dry_run` prints the plan without executing."""
    if start and only:
        raise ValueError("--start and --only are mutually exclusive")
    paths = PipelinePaths.for_area(cfg.area, outputs_root=cfg.outputs_root)
    paths.ensure_dirs()
    if only:
        planned = (only,)
    else:
        start_idx = STAGES.index(start) if start else 0
        planned = STAGES[start_idx:]
    log.info("dda run: area=%s outputs=%s stages=%s", cfg.area, paths.root, planned)

    for stage in planned:
        handler = _STAGES[stage]
        if dry_run:
            log.info("[dry-run] would execute stage: %s", stage)
            continue
        log.info("=== stage: %s ===", stage)
        handler(cfg, paths)


def _stage_aoi(cfg: DictConfig, paths: PipelinePaths) -> None:
    """Ensure `paths.aoi` exists. Fetch from TM if `tm_aoi_project` is given, else expect a file."""
    if paths.aoi.exists():
        log.info("aoi: reuse existing %s", paths.aoi)
        return
    if cfg.tm_aoi_project is not None:
        from dda.pipeline.tm import fetch_tm_project

        proj = fetch_tm_project(int(cfg.tm_aoi_project))
        paths.aoi.write_text(json.dumps(proj.aoi))
        log.info("aoi: wrote TM %d AOI -> %s (imagery=%s)", proj.project_id, paths.aoi, proj.imagery_tms)
        return
    if cfg.aoi:
        src = Path(cfg.aoi)
        if not src.exists():
            raise FileNotFoundError(f"aoi file missing: {src}")
        paths.aoi.write_bytes(src.read_bytes())
        log.info("aoi: copied %s -> %s", src, paths.aoi)
        return
    raise RuntimeError("aoi: either `aoi:` (file path) or `tm_aoi_project:` must be set in the config")


def _stage_prepare(cfg: DictConfig, paths: PipelinePaths) -> None:
    from dda.pipeline.coreg import coregister
    from dda.pipeline.fetch import fetch_raster

    fetch_raster(cfg.pre_img, paths.aoi, paths.pre_raw, zoom=cfg.zoom)
    fetch_raster(cfg.post_img, paths.aoi, paths.post_raw, zoom=cfg.zoom)
    coregister(
        paths.pre_raw,
        paths.post_raw,
        paths.pre_aligned,
        paths.post_aligned,
        paths.drift_json,
        paths.coreg_check_png,
        calibrate_photometry=cfg.photometric_calibration,
        stretch_percentiles=cfg.stretch_percentiles,
        keep_raw=cfg.keep_raw,
    )


def _stage_fewshot(cfg: DictConfig, paths: PipelinePaths) -> None:
    if not cfg.buildings.fewshot.tm_projects:
        log.info("fewshot: no tm_projects configured, skipping")
        return
    if cfg.buildings.ckpt:
        log.info("fewshot: buildings.ckpt already set (%s), skipping fewshot", cfg.buildings.ckpt)
        return
    from dda.pipeline.fewshot import fit_buildings_from_tm

    fs_dir = paths.root / "fs_buildings"
    fs = cfg.buildings.fewshot
    result = fit_buildings_from_tm(
        project_ids=list(fs.tm_projects),
        out_dir=fs_dir,
        zoom=fs.zoom,
        imagery_tms_override=fs.imagery_tms_override,
        epochs=fs.epochs,
        lr=fs.lr,
        val_frac=fs.val_frac,
        patience=fs.patience,
        hpo_trials=fs.hpo_trials,
        hpo_seeds=fs.hpo_seeds,
    )
    log.info(
        "fewshot: %d train / %d val, val IoU %.4f -> %.4f (delta %+.4f), ckpt=%s",
        result.n_train,
        result.n_val,
        result.val_iou_pretrained,
        result.val_iou_finetuned,
        result.delta,
        result.best_ckpt,
    )
    cfg.buildings.ckpt = str(result.best_ckpt)


def _stage_buildings(cfg: DictConfig, paths: PipelinePaths) -> None:
    if cfg.buildings.input:
        import geopandas as gpd

        gdf = gpd.read_file(cfg.buildings.input).to_crs("EPSG:4326").reset_index(drop=True)
        for col, default in [("class", 1), ("score", 1.0), ("source", "user")]:
            if col not in gdf.columns:
                gdf[col] = default
        if "id" not in gdf.columns:
            gdf["id"] = range(len(gdf))
        gdf[["id", "class", "score", "source", "geometry"]].to_file(paths.buildings, driver="GeoJSON")
        log.info("buildings: copied %d user footprints -> %s", len(gdf), paths.buildings)
        return
    if cfg.buildings.source == "osm":
        from dda.pipeline.postpass import fetch_postpass_buildings

        fetch_postpass_buildings(paths.aoi, paths.buildings)
        return
    from dda.pipeline.buildings import run_fair_buildings

    run_fair_buildings(
        paths.pre_aligned,
        paths.aoi,
        paths.buildings,
        ckpt=Path(cfg.buildings.ckpt) if cfg.buildings.ckpt else None,
    )


def _stage_damage(cfg: DictConfig, paths: PipelinePaths) -> None:
    from dda.config import load_config
    from dda.infer import resolve_ckpt
    from dda.pipeline.damage import run_damage_blocked

    train_cfg = load_config(None)
    run_damage_blocked(
        cfg=train_cfg,
        ckpt_path=resolve_ckpt(train_cfg, cfg.damage.ckpt),
        post_raster=paths.post_aligned,
        pre_aligned=paths.pre_aligned,
        buildings_geojson=paths.buildings,
        out_geojson=paths.damage,
    )


def _stage_publish(cfg: DictConfig, paths: PipelinePaths) -> None:
    if not cfg.publish.enabled:
        log.info("publish: disabled in config, skipping")
        return
    if not cfg.publish.repo_id:
        raise RuntimeError("publish.enabled=true but publish.repo_id is empty")
    from dda.pipeline.publish import push_area_to_hf

    if not paths.damage.exists():
        raise FileNotFoundError(f"damage.geojson missing at {paths.damage}. Run damage stage first.")
    push_area_to_hf(area=cfg.area, area_dir=paths.root, repo_id=cfg.publish.repo_id)


_STAGES = {
    "aoi": _stage_aoi,
    "prepare": _stage_prepare,
    "fewshot": _stage_fewshot,
    "buildings": _stage_buildings,
    "damage": _stage_damage,
    "publish": _stage_publish,
}
