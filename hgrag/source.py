"""Select and check the original checkout without importing historical code."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

from .common.files import ROOT, project_path, read_json, sha256


def verify_source(config):
    source = project_path(config["source"]["checkout"])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"], text=True).strip()
    if commit != config["source"]["commit"] or dirty:
        raise ValueError("The upstream checkout must match the clean pinned commit")
    records = read_json(ROOT / "configs/source-files.json")
    for record in records:
        if sha256(project_path(record["path"]), record.get("normalization", "none")) != record["sha256"]:
            raise ValueError(f"Upstream file differs: {record['path']}")
    return commit


def select_source(config):
    source = project_path(config["source"]["checkout"])
    existing = sys.modules.get("hypergraphrag")
    if existing is not None and Path(existing.__file__).resolve().parent != source / "hypergraphrag":
        raise RuntimeError("Another hypergraphrag package is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.invalidate_caches()
