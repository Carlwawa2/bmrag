"""Prepare or verify the upstream tiktoken data with the SDK's pinned hash."""
from __future__ import annotations

import hashlib
import json
import os

from ..common.files import ROOT, sha256

SOURCE = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
EXPECTED = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def load_tokenizer(*, download=False):
    directory = ROOT / "artifacts/tokenizer-cache"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(directory)
    cached = directory / hashlib.sha1(SOURCE.encode()).hexdigest()
    if not download and not cached.is_file():
        raise RuntimeError("Prepare tokenizer data first: python -m hgrag prepare-tokenizer")
    if cached.is_file() and sha256(cached) != EXPECTED:
        raise ValueError("The pinned tiktoken data hash differs")
    import tiktoken
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    if encoder.name != "o200k_base" or sha256(cached) != EXPECTED:
        raise ValueError("Unexpected tokenizer encoding or data hash")
    return encoder


if __name__ == "__main__":
    tokenizer = load_tokenizer(download=True)
    print(json.dumps({"encoding": tokenizer.name, "source": SOURCE, "sha256": EXPECTED, "status": "passed"}, indent=2))
