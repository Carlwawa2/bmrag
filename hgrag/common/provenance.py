"""Capture source files for future runs without freezing editable project code."""
from __future__ import annotations

import shutil

from .files import ROOT, read_json, sha256, write_json


def source_records():
    return [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in sorted((ROOT / "hgrag").rglob("*.py"))]


def capture_software(directory):
    records = source_records()
    manifest = directory / "software/manifest.json"
    if manifest.exists():
        if read_json(manifest)["files"] != records:
            raise ValueError("This run belongs to a different software snapshot; choose a new run name")
        return records
    for record in records:
        destination = directory / "software" / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["path"], destination)
    write_json(manifest, {"files": records})
    return records
