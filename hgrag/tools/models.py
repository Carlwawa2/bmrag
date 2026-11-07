"""Download or verify the exact public BGE-M3 artifact without API credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

from ..common.files import ROOT, now, sha256, write_json
PROFILE = ROOT / "models/bge-m3"
PIN = PROFILE / "revision.json"
MANIFEST = PROFILE / "manifest.json"
VALIDATION = ROOT / "reports/models/bge-m3-validation.json"






def load_pin():
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    if pin["repository"] != "BAAI/bge-m3" or not re.fullmatch(r"[a-f0-9]{40}", pin["revision"]):
        raise RuntimeError("The repository and revision must be pinned explicitly")
    destination = (ROOT / pin["artifact_directory"]).resolve()
    destination.relative_to((ROOT / "artifacts/models").resolve())
    names = [entry["path"] for entry in pin["files"]]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate model file names")
    for name in names:
        (destination / name).resolve().relative_to(destination)
    return pin, destination


def verify_file(directory, entry):
    path = directory / entry["path"]
    if path.stat().st_size != entry["size"]:
        raise RuntimeError(f"Wrong file size: {entry['path']}")
    actual = sha256(path)
    if entry["lfs_sha256"]:
        if actual != entry["lfs_sha256"]:
            raise RuntimeError(f"LFS SHA-256 mismatch: {entry['path']}")
        method = "official_lfs_sha256"
    else:
        content = path.read_bytes()
        git_oid = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
        if git_oid != entry["git_blob_oid"]:
            raise RuntimeError(f"Git blob OID mismatch: {entry['path']}")
        method = "official_git_blob_oid"
    return {"path": entry["path"], "size": entry["size"], "sha256": actual, "verified_against": method}


def download(pin, directory):
    os.environ["HF_HOME"] = str(ROOT / "artifacts/huggingface")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    for entry in pin["files"]:
        if (directory / entry["path"]).is_file():
            verify_file(directory, entry)
            print(f"Already verified: {entry['path']}", flush=True)
            continue
        print(f"Downloading: {entry['path']} ({entry['size'] / 1024**2:.2f} MiB)", flush=True)
        for attempt in range(1, 4):
            try:
                hf_hub_download(
                    repo_id=pin["repository"], filename=entry["path"],
                    revision=pin["revision"], endpoint="https://huggingface.co",
                    local_dir=str(directory), token=False,
                )
                verify_file(directory, entry)
                print(f"Downloaded and verified: {entry['path']}", flush=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                print(f"Retrying {entry['path']} after attempt {attempt}", flush=True)
                time.sleep(2 * attempt)


def check_local_artifacts(pin, directory):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from transformers import AutoConfig, AutoTokenizer
    import torch

    config = AutoConfig.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
    assert config.model_type == "xlm-roberta"
    assert config.hidden_size == pin["embedding_dimensions"]
    sentence_config = json.loads((directory / "sentence_bert_config.json").read_text(encoding="utf-8"))
    assert sentence_config["max_seq_length"] == pin["max_input_tokens_including_special_tokens"]
    pooling = json.loads((directory / "1_Pooling/config.json").read_text(encoding="utf-8"))
    assert pooling["pooling_mode_cls_token"] is True
    encoded = tokenizer(["预算约束下的互补证据", "Complementary evidence under a budget"], truncation=False)
    assert len(encoded["input_ids"]) == 2 and all(len(ids) > 2 for ids in encoded["input_ids"])
    # Memory mapping validates deserialization without allocating a second full weight copy.
    weights = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(weights, dict) and len(weights) > 0
    word_embeddings = [value for key, value in weights.items() if key.endswith("embeddings.word_embeddings.weight")]
    assert len(word_embeddings) == 1
    assert tuple(word_embeddings[0].shape) == (config.vocab_size, config.hidden_size)
    for filename in ("colbert_linear.pt", "sparse_linear.pt"):
        head = torch.load(directory / filename, map_location="cpu", weights_only=True)
        assert isinstance(head, dict) and head
    return {
        "offline_config_load": True,
        "offline_fast_tokenizer_load": True,
        "chinese_and_english_tokenization": True,
        "weight_deserialization_weights_only": True,
        "optional_head_deserialization_weights_only": True,
        "parameter_tensor_count": len(weights),
        "word_embedding_shape": list(word_embeddings[0].shape),
        "dimensions": config.hidden_size,
        "max_input_tokens": sentence_config["max_seq_length"],
        "pooling": "CLS",
        "inference_executed": False,
        "rag_integration_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify existing artifacts entirely offline")
    args = parser.parse_args()
    pin, directory = load_pin()
    if not args.verify_only:
        download(pin, directory)
    records = [verify_file(directory, entry) for entry in pin["files"]]
    print(f"Verified {len(records)} files against the pinned official metadata", flush=True)
    checks = check_local_artifacts(pin, directory)
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "timezone": "Asia/Shanghai",
        "repository": pin["repository"],
        "revision": pin["revision"],
        "artifact_directory": pin["artifact_directory"],
        "revision_file_sha256": sha256(PIN),
        "total_bytes": sum(entry["size"] for entry in records),
        "files": records,
    }
    if MANIFEST.exists():
        frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in ("repository", "revision", "artifact_directory", "revision_file_sha256", "total_bytes", "files"):
            if frozen[key] != manifest[key]:
                raise RuntimeError(f"Model manifest differs from the existing frozen record: {key}")
    else:
        raise RuntimeError("Restore the tracked model manifest before downloading this pinned model")
    report = {
        "verified_at": now(), "status": "passed", "offline_verification": True,
        "repository": pin["repository"], "revision": pin["revision"],
        "files_verified": len(records), "total_bytes": manifest["total_bytes"],
        "manifest_sha256": sha256(MANIFEST),
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in ("huggingface-hub", "transformers", "torch", "tokenizers")},
        "checks": checks,
    }
    write_json(VALIDATION, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
