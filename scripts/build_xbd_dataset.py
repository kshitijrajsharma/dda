"""Build an HF dataset of pre/post tiles + damage rasters + polygons from a raw xBD download.

Reads the canonical xBD layout (any mirror), rasterises post-disaster subtypes into a uint8 raster,
and keeps post polygons for object-level evaluation. Usage:
    uv run python scripts/build_xbd_dataset.py --xbd-root /path/to/xBD --repo-id <user>/xbd-damage
"""

import argparse
import json
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
import shapely.wkt
from datasets import Dataset, Features, Image, Value
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image as PILImage
from rasterio.features import rasterize

# Non-gated full-xBD mirror: sharded parts that concatenate into one zip (train/test/tier/hold).
MIRROR_REPO = "iamzihan/xView2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Raster codes: 0 background, 1..4 the Joint Damage Scale, 5 un-classified (ignored in training).
DAMAGE_CODE = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}
TILE = 1024
SPLIT_DIRS = ("train", "tier3", "tier", "test", "hold")


def _read_post_label(label_path: Path) -> list[tuple[str, str]]:
    """Return [(wkt, subtype), ...] for building features in a post label JSON."""
    data = json.loads(label_path.read_text())
    out: list[tuple[str, str]] = []
    for feat in data.get("features", {}).get("xy", []):
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        out.append((feat["wkt"], props.get("subtype", "un-classified")))
    return out


def _rasterize(buildings: list[tuple[str, str]]) -> np.ndarray:
    shapes = [(shapely.wkt.loads(wkt), DAMAGE_CODE.get(subtype, 5)) for wkt, subtype in buildings if wkt]
    if not shapes:
        return np.zeros((TILE, TILE), dtype=np.uint8)
    return rasterize(shapes, out_shape=(TILE, TILE), fill=0, dtype=np.uint8)


def _resolve_root(base: Path) -> Path:
    """Find the directory that actually holds the split folders; mirror zips often nest one level."""
    candidates = [base, *(p for p in base.rglob("*") if p.is_dir())]
    for cand in candidates:
        if any((cand / split / "labels").is_dir() for split in SPLIT_DIRS):
            return cand
    raise FileNotFoundError(f"No xBD split dirs ({SPLIT_DIRS}) found under {base}")


def _examples(xbd_root: Path):
    for split in SPLIT_DIRS:
        labels_dir = xbd_root / split / "labels"
        images_dir = xbd_root / split / "images"
        if not labels_dir.is_dir():
            log.warning("Missing split dir %s, skipping", labels_dir)
            continue
        post_labels = sorted(labels_dir.glob("*_post_disaster.json"))
        log.info("%s: %d post tiles", split, len(post_labels))
        for post_label in post_labels:
            stem = post_label.name.replace("_post_disaster.json", "")
            disaster = stem.rsplit("_", 1)[0]
            post_img = images_dir / f"{stem}_post_disaster.png"
            pre_img = images_dir / f"{stem}_pre_disaster.png"
            if not post_img.exists() or not pre_img.exists():
                continue
            buildings = _read_post_label(post_label)
            raster = _rasterize(buildings)
            polygons = json.dumps([{"wkt": w, "subtype": s} for w, s in buildings])
            yield {
                "disaster": disaster,
                "image_id": stem,
                "pre": str(pre_img),
                "post": str(post_img),
                "damage": PILImage.fromarray(raster, mode="L"),
                "polygons": polygons,
            }


def download_mirror(dest: Path, mirror_repo: str = MIRROR_REPO) -> Path:
    """Download the sharded mirror, concatenate to a zip, extract, return the xBD root.

    The mirror ships `images_part_NN` files that are a split zip; concatenating them in order
    reconstructs the archive holding the canonical train/test/tier/hold layout.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted = dest / "xbd"
    if extracted.is_dir():
        log.info("Reusing existing extraction at %s", extracted)
        return extracted

    parts = sorted(f for f in list_repo_files(mirror_repo, repo_type="dataset") if "images_part_" in f)
    if not parts:
        raise FileNotFoundError(f"No image shards found in mirror {mirror_repo}")
    log.info("Downloading %d shards from %s", len(parts), mirror_repo)
    local_parts = [
        Path(hf_hub_download(mirror_repo, p, repo_type="dataset", local_dir=str(dest / "shards")))
        for p in parts
    ]

    archive = dest / "xbd.zip"
    log.info("Concatenating shards into %s", archive)
    with archive.open("wb") as out:
        for part in local_parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out)

    log.info("Extracting %s", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    return extracted


def build(xbd_root: Path, repo_id: str, push: bool) -> None:
    features = Features(
        {
            "disaster": Value("string"),
            "image_id": Value("string"),
            "pre": Image(),
            "post": Image(),
            "damage": Image(),
            "polygons": Value("string"),
        }
    )
    xbd_root = _resolve_root(xbd_root)
    log.info("Using xBD root %s", xbd_root)
    ds = Dataset.from_generator(lambda: _examples(xbd_root), features=features)
    log.info("Built dataset with %d examples", len(ds))
    if push:
        ds.push_to_hub(repo_id, split="train")
        log.info("Pushed to %s", repo_id)
    else:
        log.info("Dry run; pass --push to upload to %s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the xBD damage HF dataset")
    parser.add_argument("--xbd-root", type=Path, help="Path to an extracted xBD download")
    parser.add_argument(
        "--mirror", type=Path, help="Download the non-gated full-xBD mirror into this dir, then build"
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if not args.xbd_root and not args.mirror:
        parser.error("Provide --xbd-root (local extraction) or --mirror (download the mirror)")
    xbd_root = download_mirror(args.mirror) if args.mirror else args.xbd_root
    build(xbd_root, args.repo_id, args.push)


if __name__ == "__main__":
    main()
