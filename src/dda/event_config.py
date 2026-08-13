"""OmegaConf schema for `dda run --config event.yaml`.

One YAML declares an event end-to-end: AOI source, pre + post imagery, optional fewshot
adaptation, buildings source, damage config, optional publish target. Loaded via
`load_event_config(path, overrides=[...])`; overrides use OmegaConf dotlist syntax
(`buildings.fewshot.hpo_trials=4`).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class FewshotBuildingsConfig:
    tm_projects: list[int] = field(default_factory=list)
    imagery_tms_override: str | None = None
    hpo_trials: int = 8
    hpo_seeds: int = 1
    epochs: int = 10
    val_frac: float = 0.3
    patience: int = 3
    lr: float = 5e-5
    zoom: int = 19


@dataclass
class BuildingsConfig:
    source: str = "fair"
    input: str | None = None
    ckpt: str | None = None
    fewshot: FewshotBuildingsConfig = field(default_factory=FewshotBuildingsConfig)


@dataclass
class DamageConfig:
    ckpt: str | None = None


@dataclass
class PublishConfig:
    enabled: bool = False
    repo_id: str | None = None


@dataclass
class EventConfig:
    area: str = MISSING
    outputs_root: str = "outputs"
    aoi: str | None = None
    tm_aoi_project: int | None = None
    pre_img: str = MISSING
    post_img: str = MISSING
    zoom: int = 19
    photometric_calibration: bool = True
    stretch_percentiles: bool = True
    keep_raw: bool = False
    buildings: BuildingsConfig = field(default_factory=BuildingsConfig)
    damage: DamageConfig = field(default_factory=DamageConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)


def load_event_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML file into the EventConfig schema. Overrides are OmegaConf dotlist."""
    base = OmegaConf.structured(EventConfig)
    loaded = OmegaConf.load(Path(path))
    merged = OmegaConf.merge(base, loaded)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    return cast(DictConfig, merged)
