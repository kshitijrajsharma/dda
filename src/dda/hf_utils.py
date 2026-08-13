"""Shared Hugging Face + SHA helpers for pinned checkpoints."""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_sha_pinned_ckpt(
    repo: str,
    filename: str,
    expected_sha256: str,
    *,
    repo_type: str = "model",
    override_path: Path | str | None = None,
    label: str = "checkpoint",
) -> Path:
    """Return a local ckpt path. If `override_path` is given, use it verbatim (no SHA check).

    Otherwise download `repo/filename` from HF, verify against `expected_sha256`, fail loud on
    mismatch. `repo_type` is 'model' or 'dataset' matching HF hub's split.
    """
    if override_path is not None:
        return Path(override_path)
    path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type))
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch at {path}:\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            f"Upstream {repo}/{filename} may have changed. Verify + update the pinned SHA."
        )
    return path
