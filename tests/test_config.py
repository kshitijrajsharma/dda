from dda.config import N_DAMAGE_CLASSES, load_config


def test_defaults():
    cfg = load_config(None)
    assert cfg.img_size == 512
    assert cfg.backbone == "terratorch_dinov3_vitl16"
    assert "sat493m" in cfg.hf_ckpt_file
    assert tuple(cfg.seg_out_indices) == (5, 11, 17, 23)
    assert N_DAMAGE_CLASSES == 4


def test_dotlist_override():
    cfg = load_config(None, overrides=["img_size=256", "use_pre=false", "pool_percentile=90"])
    assert cfg.img_size == 256
    assert cfg.use_pre is False
    assert cfg.pool_percentile == 90


def test_yaml_merge(tmp_path):
    yaml = tmp_path / "c.yaml"
    yaml.write_text("run_name: custom\nbatch_size: 8\n")
    cfg = load_config(yaml)
    assert cfg.run_name == "custom"
    assert cfg.batch_size == 8
