import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from huggingface_hub import hf_hub_download
from rasterio.warp import Resampling, reproject
from scipy.signal.windows import gaussian as gauss1d

from dda.config import N_DAMAGE_CLASSES
from dda.data import DINOV3_MEAN, DINOV3_STD
from dda.model import DinoV3DamageLit
from dda.pool import assign_damage

log = logging.getLogger(__name__)


def resolve_ckpt(cfg, ckpt_path: str | Path | None = None) -> str:
    """Use the given checkpoint, else auto-download the trained model from the Hub."""
    if ckpt_path:
        return str(ckpt_path)
    return hf_hub_download(repo_id=cfg.model_repo, filename=cfg.model_file)


def load_model(ckpt_path: str | Path, cfg, device: str = "cuda") -> DinoV3DamageLit:
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    # weights_only=False: our own checkpoint, whose hparams carry non-tensor config objects.
    model = DinoV3DamageLit.load_from_checkpoint(
        str(ckpt_path), map_location=device, ckpt_path=encoder_ckpt, weights_only=False
    )
    model.eval()
    return model


def _gaussian_kernel(size: int, sigma_frac: float = 0.125) -> np.ndarray:
    w = gauss1d(size, std=sigma_frac * size)
    k = np.outer(w, w)
    return k / k.max()


def _normalize(tile_uint8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(tile_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(DINOV3_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(DINOV3_STD, dtype=torch.float32).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def radiometric_normalize(
    image: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> np.ndarray:
    """Per-channel percentile stretch of an RGB scene to the full 8-bit range; disaster imagery arrives
    uncalibrated with a compressed range that is out of distribution for the model's fixed normalization."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    valid = image.any(axis=2)
    if not valid.any():
        raise ValueError("image has no valid (non-zero) pixels to normalize")
    out = np.empty_like(image, dtype=np.uint8)
    for channel_index in range(3):
        channel = image[..., channel_index]
        low, high = np.percentile(channel[valid], [low_percentile, high_percentile])
        if high <= low:
            raise ValueError(
                f"degenerate dynamic range in channel {channel_index}: "
                f"p{low_percentile}={low:.1f} >= p{high_percentile}={high:.1f}"
            )
        scaled = (channel.astype(np.float32) - low) / (high - low) * 255.0
        out[..., channel_index] = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return out


def _coregister(pre_src, out_hw: tuple[int, int], dst_transform, dst_crs) -> np.ndarray:
    """Reproject the pre-event raster onto the post grid so pre and post share pixels for the change model."""
    height, width = out_hw
    out = np.zeros((3, height, width), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=rasterio.band(pre_src, band + 1),
            destination=out[band],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.cubic,
        )
    return out.transpose(1, 2, 0)


def _pad(tile: np.ndarray, window: int) -> np.ndarray:
    if tile.shape[0] == window and tile.shape[1] == window:
        return tile
    pad = np.zeros((window, window, 3), dtype=np.uint8)
    pad[: tile.shape[0], : tile.shape[1]] = tile
    return pad


def sliding_window_prob(
    model: DinoV3DamageLit,
    post: np.ndarray,
    pre: np.ndarray | None,
    window: int,
    stride: int,
    temperature: float,
    device: str,
) -> np.ndarray:
    """Gaussian-blended per-class damage probabilities, shape (C, H, W)."""
    height, width = post.shape[:2]
    prob_acc = np.zeros((N_DAMAGE_CLASSES, height, width), dtype=np.float32)
    weight_acc = np.zeros((height, width), dtype=np.float32)
    kernel = _gaussian_kernel(window)

    rows = list(range(0, max(1, height - window + 1), stride))
    cols = list(range(0, max(1, width - window + 1), stride))
    if rows[-1] + window < height:
        rows.append(height - window)
    if cols[-1] + window < width:
        cols.append(width - window)

    with torch.inference_mode():
        for r in rows:
            for c in cols:
                post_tile = _pad(post[r : r + window, c : c + window, :], window)
                x_post = _normalize(post_tile, device)
                x_pre = None
                if pre is not None:
                    x_pre = _normalize(_pad(pre[r : r + window, c : c + window, :], window), device)
                _, dmg_logits = model(x_post, x_pre)
                probs = torch.softmax(dmg_logits[0] / temperature, dim=0).cpu().numpy()
                h = min(window, height - r)
                w = min(window, width - c)
                prob_acc[:, r : r + h, c : c + w] += probs[:, :h, :w] * kernel[:h, :w]
                weight_acc[r : r + h, c : c + w] += kernel[:h, :w]

    weight_acc = np.maximum(weight_acc, 1e-6)
    return prob_acc / weight_acc


def predict_damage(
    cfg,
    ckpt_path: str | Path,
    post_raster: str | Path,
    buildings_geojson: str | Path,
    out_geojson: str | Path,
    pre_raster: str | Path | None = None,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, cfg, device=device)

    with rasterio.open(post_raster) as src:
        post = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        transform, crs = src.transform, src.crs

    pre = None
    if pre_raster is not None:
        with rasterio.open(pre_raster) as src:
            pre = _coregister(src, post.shape[:2], transform, crs)

    if cfg.radiometric_normalize:
        post = radiometric_normalize(post)
        if pre is not None:
            pre = radiometric_normalize(pre)

    prob = sliding_window_prob(model, post, pre, cfg.tile_window, cfg.tile_stride, cfg.temperature, device)

    buildings = gpd.read_file(buildings_geojson)
    gdf = assign_damage(
        prob,
        transform,
        crs,
        buildings,
        pool_op=cfg.pool_op,
        percentile=cfg.pool_percentile,
        confidence_threshold=cfg.confidence_threshold,
    )

    out_geojson = Path(out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    log.info("Wrote %d buildings with damage classes to %s", len(gdf), out_geojson)
    return out_geojson
