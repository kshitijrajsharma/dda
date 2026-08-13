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

run config:
    uv run dda run --config {{config}}

train:
    uv run dda train --config conf/train.yaml
