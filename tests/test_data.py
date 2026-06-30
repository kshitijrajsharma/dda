import numpy as np
import torch
from PIL import Image as PILImage

from dda.config import N_DAMAGE_CLASSES
from dda.data import _build_transforms
from dda.losses import IGNORE_INDEX


def _datamodule():
    from dda.data import XbdDamageDataModule

    dm = XbdDamageDataModule(
        repo_id="unused",
        dataset_splits=("train",),
        img_size=256,
        use_pre=True,
        augment_photometric=False,
        val_events=(),
        test_events=(),
        val_frac=0.0,
        data_pct=100.0,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=0,
    )
    dm._train_tf = _build_transforms(256, train=True)
    dm._eval_tf = _build_transforms(256, train=False)
    return dm


def test_damage_target_mapping():
    dm = _datamodule()
    raster = torch.tensor([[0, 1, 2], [3, 4, 5]])  # bg, classes 1..4, un-classified
    target = dm._damage_target(raster)
    assert target[0, 0] == IGNORE_INDEX  # background
    assert target[0, 1] == 0  # no-damage -> 0
    assert target[1, 1] == 3  # destroyed -> 3
    assert target[1, 2] == IGNORE_INDEX  # un-classified (code 5) -> ignore


def _example(size: int = 512):
    raster = np.zeros((size, size), np.uint8)
    raster[50:150, 50:150] = 1
    raster[200:260, 200:260] = 3
    raster[300:330, 300:330] = 5
    rng = np.random.default_rng(0)
    return {
        "t2_image": PILImage.fromarray(rng.integers(0, 255, (size, size, 3), np.uint8)),
        "t1_image": PILImage.fromarray(rng.integers(0, 255, (size, size, 3), np.uint8)),
        "t2_mask": PILImage.fromarray(raster, "L"),
    }


def test_collate_shapes_and_codes():
    dm = _datamodule()
    batch = dm._collate([_example(), _example()], train=False)
    assert batch["post"].shape == (2, 3, 256, 256)
    assert batch["pre"].shape == (2, 3, 256, 256)
    assert batch["build_mask"].shape == (2, 256, 256)
    assert batch["damage"].dtype == torch.int64
    valid = batch["damage"][batch["damage"] != IGNORE_INDEX]
    if valid.numel():
        assert int(valid.min()) >= 0
        assert int(valid.max()) < N_DAMAGE_CLASSES


def test_post_only_collate_has_no_pre():
    dm = _datamodule()
    dm.use_pre = False
    batch = dm._collate([_example()], train=False)
    assert "pre" not in batch
