"""Reproject pre onto the post grid, correct sub-pixel drift, harmonise photometry, stretch
both to full 0-255 range, render a checkerboard PNG. Parameters are estimated on decimated
reads and applied at full resolution."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import shift as ndi_shift
from skimage.filters import threshold_triangle
from skimage.registration import phase_cross_correlation

log = logging.getLogger(__name__)

DRIFT_DECIMATION = 8
CHECKERBOARD_BLOCKS = 12
# Anything above ~20 m on VHR optical pairs is almost always a spurious phase_cross_correlation
# lock on cloud, haze, or texture-poor terrain; clamp to zero rather than shifting by that.
MAX_PLAUSIBLE_DRIFT_M = 30.0
STRETCH_PCT_LOW = 2.0
STRETCH_PCT_HIGH = 98.0
# Below this ground fraction the derived cutoff is not a real tail split; use plain 2-98.
STRETCH_GROUND_MIN_FRAC = 0.5


@dataclass
class DriftResult:
    dy_px: float
    dx_px: float
    pixel_size_m: float
    magnitude_m: float


def coregister(
    pre_raw: Path,
    post_raw: Path,
    pre_aligned: Path,
    post_aligned: Path,
    drift_json: Path,
    check_png: Path,
    *,
    calibrate_photometry: bool = True,
    stretch_percentiles: bool = True,
    keep_raw: bool = False,
) -> DriftResult:
    """Reproject, drift-correct, photometrically calibrate pre onto post, stretch both.

    Writes `pre_aligned.tif` (fully processed pre on the post grid) and `post_aligned.tif`
    (stretched copy of post_raw). Raw inputs are deleted unless `keep_raw=True`.
    """
    pre_aligned.parent.mkdir(parents=True, exist_ok=True)
    drift_json.parent.mkdir(parents=True, exist_ok=True)

    _reproject_onto_post(pre_raw, post_raw, pre_aligned)
    drift = _measure_drift(pre_aligned, post_raw)
    _apply_drift(pre_aligned, drift)
    if calibrate_photometry:
        _apply_photometric_calibration(pre_aligned, post_raw)
    post_aligned.write_bytes(post_raw.read_bytes())
    if stretch_percentiles:
        _apply_percentile_stretch(pre_aligned, "pre_aligned")
        _apply_percentile_stretch(post_aligned, "post_aligned")
    _render_checkerboard(pre_aligned, post_aligned, check_png)
    if not keep_raw:
        for p in (pre_raw, post_raw):
            if p.exists():
                p.unlink()
        log.info("coregister: deleted raw pre.tif + post.tif (set keep_raw=True to preserve)")

    drift_json.write_text(json.dumps(asdict(drift), indent=2))
    log.info(
        "drift dy=%.2f px dx=%.2f px (%.1f m at %.2f m/px)",
        drift.dy_px,
        drift.dx_px,
        drift.magnitude_m,
        drift.pixel_size_m,
    )
    return drift


def _gtiff_profile(height: int, width: int, crs, transform) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "yes",
    }


def _reproject_onto_post(pre_raw: Path, post: Path, out_aligned: Path) -> None:
    with rasterio.open(post) as post_src, rasterio.open(pre_raw) as pre_src:
        profile = _gtiff_profile(post_src.height, post_src.width, post_src.crs, post_src.transform)
        with rasterio.open(out_aligned, "w", **profile) as dst:
            for band in range(3):
                dest = np.zeros((post_src.height, post_src.width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(pre_src, band + 1),
                    destination=dest,
                    dst_transform=post_src.transform,
                    dst_crs=post_src.crs,
                    resampling=Resampling.cubic,
                )
                dst.write(dest, band + 1)


def _measure_drift(pre_aligned: Path, post: Path) -> DriftResult:
    with rasterio.open(post) as ps, rasterio.open(pre_aligned) as pr:
        step = DRIFT_DECIMATION
        post_gray = _decimated_grayscale(ps, step)
        pre_gray = _decimated_grayscale(pr, step)
        pixel_size_m = _mean_pixel_size_m(ps)
    common = (min(post_gray.shape[0], pre_gray.shape[0]), min(post_gray.shape[1], pre_gray.shape[1]))
    post_gray = post_gray[: common[0], : common[1]]
    pre_gray = pre_gray[: common[0], : common[1]]
    shift, _error, _diffphase = phase_cross_correlation(post_gray, pre_gray, upsample_factor=10)
    dy_px = float(shift[0]) * step
    dx_px = float(shift[1]) * step
    magnitude_m = float(np.hypot(dy_px, dx_px) * pixel_size_m)
    if magnitude_m > MAX_PLAUSIBLE_DRIFT_M:
        log.warning(
            "measured drift %.1f m exceeds plausible ceiling %.1f m; clamping to zero",
            magnitude_m,
            MAX_PLAUSIBLE_DRIFT_M,
        )
        return DriftResult(dy_px=0.0, dx_px=0.0, pixel_size_m=pixel_size_m, magnitude_m=0.0)
    return DriftResult(dy_px=dy_px, dx_px=dx_px, pixel_size_m=pixel_size_m, magnitude_m=magnitude_m)


def _decimated_grayscale(src: "rasterio.io.DatasetReader", step: int) -> np.ndarray:
    arr = src.read(
        [1, 2, 3],
        out_shape=(3, src.height // step, src.width // step),
        resampling=Resampling.average,
    ).astype(np.float32)
    return arr.mean(axis=0)


def _mean_pixel_size_m(src: "rasterio.io.DatasetReader") -> float:
    if src.crs.is_projected:
        return float(abs(src.transform.a) + abs(src.transform.e)) / 2.0
    centre_lat = (src.bounds.top + src.bounds.bottom) / 2.0
    metres_per_deg = 111_320.0 * np.cos(np.deg2rad(centre_lat))
    return float((abs(src.transform.a) + abs(src.transform.e)) / 2.0 * metres_per_deg)


def _apply_drift(pre_aligned: Path, drift: DriftResult) -> None:
    """Shift each band and overwrite `pre_aligned` via a sibling temp + atomic rename.

    Reads one band at a time so peak RAM stays at ~one band, not the full raster.
    """
    if drift.dy_px == 0.0 and drift.dx_px == 0.0:
        log.info("drift is zero, skipping shift; pre_aligned = reprojection only")
        return
    tmp = pre_aligned.with_suffix(pre_aligned.suffix + ".shifted.tmp")
    with rasterio.open(pre_aligned) as src:
        profile = _gtiff_profile(src.height, src.width, src.crs, src.transform)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band in range(1, 4):
                arr = src.read(band)
                shifted = ndi_shift(
                    arr, shift=(drift.dy_px, drift.dx_px), order=1, mode="constant", cval=0
                ).astype(np.uint8)
                dst.write(shifted, band)
                del arr, shifted
    tmp.replace(pre_aligned)


def _apply_photometric_calibration(pre_aligned: Path, post: Path) -> None:
    """Match pre's per-band mean and std to post's via a linear affine (no gamma, no tone map).

    Strips the trivial cross-sensor offset from exposure or white balance without warping damage
    signal. Nodata (any-channel == 0) is excluded from stats and preserved as 0 in the output.
    """
    with rasterio.open(pre_aligned) as pre, rasterio.open(post) as ps:
        step = DRIFT_DECIMATION
        pre_dec = pre.read(
            [1, 2, 3],
            out_shape=(3, pre.height // step, pre.width // step),
            resampling=Resampling.average,
        )
        post_dec = ps.read(
            [1, 2, 3],
            out_shape=(3, ps.height // step, ps.width // step),
            resampling=Resampling.average,
        )
    common = (
        min(pre_dec.shape[1], post_dec.shape[1]),
        min(pre_dec.shape[2], post_dec.shape[2]),
    )
    pre_dec = pre_dec[:, : common[0], : common[1]]
    post_dec = post_dec[:, : common[0], : common[1]]
    valid = (pre_dec.sum(axis=0) > 0) & (post_dec.sum(axis=0) > 0)
    if valid.sum() < 1000:
        log.warning(
            "photometric calibration skipped: only %d overlapping valid pixels (need >=1000)",
            int(valid.sum()),
        )
        return

    means_pre, stds_pre, means_post, stds_post = [], [], [], []
    for b in range(3):
        p = pre_dec[b][valid].astype(np.float32)
        q = post_dec[b][valid].astype(np.float32)
        means_pre.append(float(p.mean()))
        stds_pre.append(float(p.std()) or 1.0)
        means_post.append(float(q.mean()))
        stds_post.append(float(q.std()) or 1.0)
    log.info(
        "photometric calibration: pre RGB mean=%s std=%s -> post mean=%s std=%s",
        [round(m, 1) for m in means_pre],
        [round(s, 1) for s in stds_pre],
        [round(m, 1) for m in means_post],
        [round(s, 1) for s in stds_post],
    )

    tmp = pre_aligned.with_suffix(pre_aligned.suffix + ".calib.tmp")
    with rasterio.open(pre_aligned) as src:
        profile = _gtiff_profile(src.height, src.width, src.crs, src.transform)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band in range(1, 4):
                arr = src.read(band).astype(np.float32)
                nodata = arr == 0
                scale = stds_post[band - 1] / stds_pre[band - 1]
                out = (arr - means_pre[band - 1]) * scale + means_post[band - 1]
                out = np.clip(out, 0, 255).astype(np.uint8)
                out[nodata] = 0
                dst.write(out, band)
                del arr, out, nodata
    tmp.replace(pre_aligned)


def _apply_percentile_stretch(path: Path, label: str) -> None:
    """Rescale each band to fill 0-255 using 2-98 percentiles computed on ground pixels.

    Ground is `brightness <= threshold_triangle(brightness)` (Zack, Rogers, Latt 1977) so bright
    outliers (cloud, snow, specular) cannot pull the 98th percentile up and compress ground into
    the dark half. Falls back to plain percentiles when the derived ground fraction is under the
    floor, which is the expected outcome on scenes with no bright tail.
    """
    with rasterio.open(path) as src:
        step = DRIFT_DECIMATION
        dec = src.read(
            [1, 2, 3],
            out_shape=(3, src.height // step, src.width // step),
            resampling=Resampling.average,
        )
    valid = dec.sum(axis=0) > 0
    if valid.sum() < 1000:
        log.warning(
            "%s stretch skipped: only %d valid pixels in decimated view (need >=1000)",
            label,
            int(valid.sum()),
        )
        return
    brightness = dec.astype(np.float32).mean(axis=0)
    cutoff = float(threshold_triangle(brightness[valid].astype(np.uint8)))
    ground = valid & (brightness <= cutoff)
    ground_frac = float(ground.sum()) / max(1, int(valid.sum()))
    if ground_frac < STRETCH_GROUND_MIN_FRAC:
        log.warning(
            "%s stretch: triangle cutoff=%.1f kept only %.0f%% of valid area (<%.0f%% floor), "
            "falling back to plain 2-98 percentiles",
            label,
            cutoff,
            ground_frac * 100.0,
            STRETCH_GROUND_MIN_FRAC * 100.0,
        )
        ground = valid
    lows, highs = [], []
    for b in range(3):
        pixels = dec[b][ground].astype(np.float32)
        lo, hi = np.percentile(pixels, [STRETCH_PCT_LOW, STRETCH_PCT_HIGH])
        if hi - lo < 1.0:
            hi = lo + 1.0
        lows.append(float(lo))
        highs.append(float(hi))
    log.info(
        "%s stretch: bright-cutoff=%.1f ground=%.1f%% lows=%s highs=%s (2-98 on ground)",
        label,
        cutoff,
        ground_frac * 100.0,
        [round(x, 1) for x in lows],
        [round(x, 1) for x in highs],
    )

    tmp = path.with_suffix(path.suffix + ".stretch.tmp")
    with rasterio.open(path) as src:
        profile = _gtiff_profile(src.height, src.width, src.crs, src.transform)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band in range(1, 4):
                arr = src.read(band).astype(np.float32)
                nodata = arr == 0
                scaled = (arr - lows[band - 1]) * 255.0 / (highs[band - 1] - lows[band - 1])
                out = np.clip(scaled, 0, 255).astype(np.uint8)
                out[nodata] = 0
                dst.write(out, band)
                del arr, scaled, out, nodata
    tmp.replace(path)


def _render_checkerboard(
    pre_aligned: Path, post: Path, out_png: Path, blocks: int = CHECKERBOARD_BLOCKS
) -> None:
    """Alternating pre/post cells; roads and coastlines look continuous when alignment is clean."""
    with rasterio.open(post) as ps, rasterio.open(pre_aligned) as pr:
        h = min(ps.height, pr.height)
        w = min(ps.width, pr.width)
        step = 8
        post_arr = ps.read([1, 2, 3], out_shape=(3, h // step, w // step), resampling=Resampling.average)
        pre_arr = pr.read([1, 2, 3], out_shape=(3, h // step, w // step), resampling=Resampling.average)
    ch, cw = post_arr.shape[1], post_arr.shape[2]
    block_h = max(1, ch // blocks)
    block_w = max(1, cw // blocks)
    out = post_arr.copy()
    for by in range(blocks):
        for bx in range(blocks):
            if (by + bx) % 2 == 0:
                y0, y1 = by * block_h, (by + 1) * block_h
                x0, x1 = bx * block_w, (bx + 1) * block_w
                out[:, y0:y1, x0:x1] = pre_arr[:, y0:y1, x0:x1]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.transpose(1, 2, 0)).save(out_png)
