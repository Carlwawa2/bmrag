"""Load and verify the standalone model-replaced HyperGraphRAG baseline."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .common.files import ROOT, project_path, read_json, sha256
from .source import verify_source

CONFIG_PATH = ROOT / "configs/baseline.json"


def scientific_fingerprint(config):
    source = deepcopy(config["source"])
    source.pop("entrypoint", None)
    models = deepcopy(config["models"])
    for field in ("local_directory", "manifest", "manifest_sha256"):
        models["embedding"].pop(field, None)
    parameters = {
        "source": source, "api": config["api"], "models": models,
        "rag": config["rag"], "retrieval": config["retrieval"],
        "vector_search": config["vector_search"], "budget_semantics": config["budget_semantics"],
        "runtime": {field: config["runtime"][field] for field in ("python_target", "python_hash_seed", "local_random_seed")},
    }
    return hashlib.sha256(json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_model(config):
    embedding = config["models"]["embedding"]
    manifest_path = project_path(embedding["manifest"])
    if sha256(manifest_path) != embedding["manifest_sha256"]:
        raise ValueError("The embedding manifest has changed")
    manifest = read_json(manifest_path)
    if (manifest["repository"], manifest["revision"], manifest["artifact_directory"]) != (embedding["name"], embedding["revision"], embedding["local_directory"]):
        raise ValueError("Embedding model provenance differs")
    directory = project_path(embedding["local_directory"])
    directory.relative_to((ROOT / "artifacts/models").resolve())
    for record in manifest["files"]:
        path = (directory / record["path"]).resolve()
        path.relative_to(directory)
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Embedding artifact changed: {record['path']}")
    return directory


def load(*, verify_artifacts=False):
    inputs = read_json(ROOT / "configs/manifest.json")
    for record in inputs["files"]:
        if sha256(project_path(record["path"])) != record["sha256"]:
            raise ValueError(f"Frozen experiment input changed: {record['path']}")
    config = read_json(CONFIG_PATH)
    if config["source"]["entrypoint"] != "hgrag.runtime.make_rag":
        raise ValueError("Use the unified runtime entrypoint")
    if len({config["models"][role] for role in ("extraction", "summarization", "query_analysis", "generation")}) != 1:
        raise ValueError("The baseline uses one shared LLM callback")
    if config["models"]["generation"] != config["rag"]["llm_model_name"]:
        raise ValueError("The LLM callback and configured model must agree")
    verify_source(config)
    if verify_artifacts:
        verify_model(config)
    return config
