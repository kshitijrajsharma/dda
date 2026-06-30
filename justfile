set shell := ["bash", "-uc"]

default:
    @just --list

setup:
    uv sync --all-groups
    uv run pre-commit install

lint:
    uv run ruff check --fix .
    uv run ruff format .
    uv run ty check src tests

test:
    uv run pytest -q

# Build the xBD HF dataset from the non-gated mirror and push to your HF repo.
data repo_id workdir="data/xbd_mirror":
    uv run python scripts/build_xbd_dataset.py --mirror {{workdir}} --repo-id {{repo_id}} --push

train:
    uv run dda train --config conf/train.yaml

# Pre+post damage assessment; pass a pre GeoTIFF to enable the change signal (co-registered automatically).
predict raster buildings out pre="":
    uv run dda predict --raster {{raster}} --buildings {{buildings}} --out {{out}} $([ -n "{{pre}}" ] && echo "--pre-raster {{pre}}")

export ckpt out:
    uv run dda export --ckpt {{ckpt}} --out {{out}}

# Object-level (per-building) damage F1 on the val or test split.
evaluate ckpt split="val":
    uv run dda evaluate --ckpt {{ckpt}} --split {{split}}
