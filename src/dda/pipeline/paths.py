"""Canonical file paths under `outputs/<area>/`. `pre.tif` + `post.tif` are only kept when
`keep_raw` is set at prepare time; everything downstream reads the `_aligned` versions."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    """All paths for one area. Under `root/<area>/`."""

    area: str
    root: Path

    @classmethod
    def for_area(cls, area: str, outputs_root: Path | str = "outputs") -> "PipelinePaths":
        return cls(area=area, root=Path(outputs_root) / area)

    @property
    def aoi(self) -> Path:
        return self.root / "aoi.geojson"

    @property
    def pre_raw(self) -> Path:
        return self.root / "pre.tif"

    @property
    def post_raw(self) -> Path:
        return self.root / "post.tif"

    @property
    def pre_aligned(self) -> Path:
        return self.root / "pre_aligned.tif"

    @property
    def post_aligned(self) -> Path:
        return self.root / "post_aligned.tif"

    @property
    def coreg_dir(self) -> Path:
        return self.root / "coreg"

    @property
    def drift_json(self) -> Path:
        return self.coreg_dir / "drift.json"

    @property
    def coreg_check_png(self) -> Path:
        return self.coreg_dir / "checkerboard.png"

    @property
    def buildings(self) -> Path:
        return self.root / "buildings.geojson"

    @property
    def buildings_parquet(self) -> Path:
        return self.root / "buildings.parquet"

    @property
    def damage(self) -> Path:
        return self.root / "damage.geojson"

    @property
    def damage_parquet(self) -> Path:
        return self.root / "damage.parquet"

    @property
    def review_dir(self) -> Path:
        return self.root / "review_crops"

    @property
    def review_manifest(self) -> Path:
        return self.review_dir / "manifest.json"

    @property
    def wf_args(self) -> Path:
        return self.review_dir / "wf_args.json"

    @property
    def verdicts(self) -> Path:
        return self.review_dir / "verdicts.json"

    @property
    def hf_dir(self) -> Path:
        return self.root / "hf"

    @property
    def meta_yaml(self) -> Path:
        return self.root / "meta.yaml"

    @property
    def hf_area_dir(self) -> Path:
        return self.hf_dir / self.area

    @property
    def hf_viz_dir(self) -> Path:
        return self.hf_area_dir / "viz"

    @property
    def hf_readme(self) -> Path:
        return self.hf_area_dir / "README.md"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.coreg_dir, self.review_dir, self.hf_dir):
            d.mkdir(parents=True, exist_ok=True)
