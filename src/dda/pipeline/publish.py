"""Upload an area's HF folder to a Hugging Face dataset repo via the API (not git)."""

import logging
from pathlib import Path

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def push_area_to_hf(
    area: str,
    area_dir: Path,
    repo_id: str,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload `area_dir` to `<repo_id>/<area>/` on the HF hub. Returns the commit URL as a string."""
    api = HfApi(token=token)
    message = commit_message or f"add/update {area} deliverables"
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(area_dir),
        path_in_repo=area,
        commit_message=message,
    )
    log.info("uploaded %s -> %s (%s)", area_dir, repo_id, result)
    return str(result)
