import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from dda.infer import load_model

log = logging.getLogger(__name__)


class _DamageHeadOnly(nn.Module):
    """Exposes the damage logits for (post, pre); the localization head is supervision-only."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, post: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        _, dmg = self.net(post, pre)
        return dmg


def export_onnx(
    cfg,
    ckpt_path: str | Path,
    out_path: str | Path,
    opset: int = 17,
    parity_atol: float = 5e-2,
    parity_class_tol: float = 0.01,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, cfg, device=device)
    inner = _DamageHeadOnly(model.net).to(device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    post = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
    pre = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
    onnx_program = torch.onnx.export(
        inner,
        (post, pre),
        str(out_path),
        opset_version=opset,
        input_names=["post", "pre"],
        output_names=["damage_logits"],
        dynamo=True,
    )
    if onnx_program is not None:
        onnx_program.optimize()
        onnx_program.save(str(out_path))

    with torch.inference_mode():
        torch_out = inner(post, pre).cpu().numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = np.asarray(session.run(None, {"post": post.cpu().numpy(), "pre": pre.cpu().numpy()})[0])
    diff = float(np.abs(torch_out - ort_out).max())
    total = torch_out[:, 0].size
    class_disagree = int((torch_out.argmax(1) != ort_out.argmax(1)).sum())
    disagree_frac = class_disagree / total
    # A handful of boundary pixels flip class under fp drift; per-building pooling absorbs it.
    if disagree_frac > parity_class_tol:
        raise RuntimeError(
            f"ONNX class parity failed: {class_disagree}/{total} ({disagree_frac:.2%}) pixels "
            f"disagree > tolerance {parity_class_tol:.2%}"
        )
    if diff > parity_atol:
        raise RuntimeError(f"ONNX parity failed: max abs diff {diff:.4e} > {parity_atol:.4e}")
    log.info(
        "ONNX exported to %s | logit max-abs-diff=%.2e | class disagree=%d/%d (%.3f%%)",
        out_path,
        diff,
        class_disagree,
        total,
        disagree_frac * 100,
    )
    return out_path
