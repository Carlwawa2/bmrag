"""Capture the current baseline and source for an individual experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import uuid

from ..common.files import ROOT, now, sha256, write_json
from ..common.provenance import capture_software
from ..configuration import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory under artifacts/; defaults to a unique snapshot directory")
    args = parser.parse_args()
    config = load()
    directory = args.output or ROOT / "artifacts/snapshots" / uuid.uuid4().hex[:12]
    directory = directory.resolve()
    directory.relative_to((ROOT / "artifacts").resolve())
    if directory.exists():
        raise ValueError("Choose a new snapshot directory")
    records = capture_software(directory)
    files = ("configs/baseline.json", "configs/manifest.json", "configs/source-files.json", "configs/pilot.json", "environment/requirements-win-py311.lock.txt", "models/bge-m3/manifest.json")
    import shutil
    for name in files:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    manifest = {
        "created_at": now(), "baseline_id": config["baseline_id"], "source_commit": config["source"]["commit"],
        "software_files": records, "input_files": [{"path": name, "sha256": sha256(ROOT / name)} for name in files],
    }
    write_json(directory / "manifest.json", manifest)
    print(directory / "manifest.json")
