"""One implementation of project paths, hashing and atomic JSON output."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def now():
    return datetime.now(LOCAL_TIMEZONE).isoformat()


def project_path(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    return path


def sha256(path, normalization="none"):
    path = Path(path)
    if normalization == "lf":
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if normalization != "none":
        raise ValueError(f"Unsupported normalization: {normalization}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def redact(text, key=""):
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"oc_sk_[A-Za-z0-9_-]+", "[REDACTED]", text)


def write_json(path, value, key="", *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = redact(json.dumps(value, ensure_ascii=False, indent=2), key) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
