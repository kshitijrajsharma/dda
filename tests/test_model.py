import pytest
import torch
from huggingface_hub import hf_hub_download

from dda.model import DinoV3DamageNet


@pytest.mark.slow
def test_forward_shapes_siamese_and_post_only():
    ckpt = hf_hub_download("kshitijrajsharma/dinov3", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    net = DinoV3DamageNet(
        ckpt_path=ckpt,
        seg_out_indices=(5, 11, 17, 23),
        decoder_channels=256,
        use_pre=True,
        modality_dropout=0.0,
    ).eval()
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in net.backbone.parameters())
    assert n_train < 40e6

    post = torch.randn(1, 3, 512, 512)
    pre = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        loc, dmg = net(post, pre)
    assert loc.shape == (1, 512, 512)
    assert dmg.shape == (1, 4, 512, 512)

    with torch.no_grad():
        _, dmg2 = net(post)  # pre omitted -> graceful post-only
    assert dmg2.shape == (1, 4, 512, 512)
