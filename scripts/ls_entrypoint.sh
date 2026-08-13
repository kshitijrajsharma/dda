#!/bin/bash
# One-container Label Studio entrypoint: run LS's own boot script AND, once /health is up,
# seed any outputs/<area>/label_export project that is not already in LS. Idempotent, safe
# to re-run. Requires LABEL_STUDIO_USER_TOKEN in the container environment.

set -e

/label-studio/deploy/docker-entrypoint.sh label-studio &
LS_PID=$!

(
  until curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 2; done
  echo "[bootstrap] LS healthy, enabling legacy API token on the auto-created org"
  cd /label-studio/label_studio && DJANGO_SETTINGS_MODULE=core.settings.label_studio python3 -c "
import django; django.setup()
from organizations.models import Organization
n = 0
for org in Organization.objects.all():
    org.jwt.legacy_api_tokens_enabled = True
    org.jwt.save()
    n += 1
print(f'[bootstrap] set legacy_api_tokens_enabled=True on {n} org(s)')
" 2>&1 | grep -E "\[bootstrap\]|Error" | tail -3
  cd - >/dev/null
  echo "[bootstrap] checking for label_export/ projects to seed"
  python3 - <<'PY'
import json, os, sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

LS = "http://127.0.0.1:8080"
TOKEN = os.environ.get("LABEL_STUDIO_USER_TOKEN") or ""
DATA = Path(os.environ.get("DATA_ROOT", "/data"))

if not TOKEN:
    print("[bootstrap] LABEL_STUDIO_USER_TOKEN not set, skipping auto-seed")
    sys.exit(0)

def call(path, data=None, headers=None, method=None):
    req = Request(
        f"{LS}{path}",
        data=data,
        method=method or ("GET" if data is None else "POST"),
        headers={"Authorization": f"Token {TOKEN}", **(headers or {})},
    )
    try:
        raw = urlopen(req, timeout=120).read()
    except HTTPError as e:
        body = e.read()[:200].decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {path}: {body}") from None
    return json.loads(raw) if raw else {}

listing = call("/api/projects/")
have = {p["title"] for p in listing.get("results", [])}
print(f"[bootstrap] existing projects: {sorted(have) or '(none)'}")

for exp in sorted(DATA.glob("*/label_export")):
    title = exp.parent.name
    if title in have:
        print(f"[bootstrap] skip {title}, project already exists")
        continue
    cfg = exp / "config.xml"
    tasks = exp / "tasks.json"
    if not cfg.exists() or not tasks.exists():
        print(f"[bootstrap] skip {title}, missing config.xml or tasks.json")
        continue
    proj = call(
        "/api/projects/",
        data=json.dumps({"title": title, "label_config": cfg.read_text()}).encode(),
        headers={"Content-Type": "application/json"},
    )
    pid = proj["id"]
    call(
        f"/api/projects/{pid}/import",
        data=tasks.read_bytes(),
        headers={"Content-Type": "application/json"},
    )
    print(f"[bootstrap] seeded {title} (project {pid})")
PY
) &

wait "$LS_PID"
