"""Single baseline factory shared by checks, pilots and later research methods."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import wraps
import hashlib
import os
from pathlib import Path
import random
import re
import sys

from .common.files import ROOT, read_json, sha256, write_json
from .common.provenance import capture_software
from .configuration import CONFIG_PATH, load
from .source import select_source


def limit_calls(function, maximum):
    semaphore = asyncio.Semaphore(maximum)

    @wraps(function)
    async def limited(*args, **kwargs):
        async with semaphore:
            return await function(*args, **kwargs)
    return limited


def make_query_param():
    config = load()
    select_source(config)
    from hypergraphrag import QueryParam
    return QueryParam(**deepcopy(config["retrieval"]))


def make_rag(run_name="official-example", *, api_key=None, offline=False, encoder=None, working_dir=None, read_only=False):
    config = load()
    if sys.version_info[:2] != (3, 11) or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("Use Python 3.11 with PYTHONHASHSEED=42 set before startup")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("Invalid run name")
    if read_only and not offline:
        raise ValueError("Read-only opening is for offline verification")
    key = None if offline else api_key or os.environ.get(config["api"]["api_key_environment_variable"])
    if not offline and not key:
        raise RuntimeError("Set OPENCODE_API_KEY, or use offline=True for local checks")
    select_source(config)
    from hypergraphrag import HyperGraphRAG
    from hypergraphrag.utils import EmbeddingFunc
    import hypergraphrag.utils as upstream_utils
    import numpy as np
    from .models.bge_m3 import BGEEncoder
    from .models.opencode import complete, save_audit
    from .models.tokenizer import load_tokenizer

    random.seed(config["runtime"]["local_random_seed"])
    np.random.seed(config["runtime"]["local_random_seed"])
    upstream_utils.ENCODER = load_tokenizer()
    encoder = encoder or BGEEncoder(config)
    if type(encoder) is not BGEEncoder or encoder.settings != config["models"]["embedding"]:
        raise ValueError("Encoder settings must match the active profile")
    config_hash = sha256(CONFIG_PATH)
    mode = "offline" if offline else "online"
    directory = (ROOT / config["runtime"]["artifact_root"] / config_hash[:12] / mode / run_name) if working_dir is None else Path(working_dir).resolve()
    directory.resolve().relative_to((ROOT / "artifacts").resolve())
    if read_only:
        existing = read_json(directory / "baseline_config.json")
        from .configuration import scientific_fingerprint
        if scientific_fingerprint(existing) != scientific_fingerprint(config):
            raise ValueError("Stored graph settings differ from the active baseline")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        capture_software(directory)
    session = "ses_" + hashlib.sha256(f"{config_hash}:{run_name}".encode()).hexdigest()[:32]

    async def fixed_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        if offline:
            raise RuntimeError("LLM requests are disabled in this offline instance")
        text, audit = await asyncio.to_thread(complete, config, key, session, prompt, system_prompt, history_messages, kwargs)
        audit["config_sha256"] = config_hash
        save_audit(directory / "model_calls", audit, key)
        return text

    async def fixed_embedding(texts):
        return await encoder(texts)

    embedding = EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=fixed_embedding, concurrent_limit=1)
    parameters = deepcopy(config["rag"])
    parameters["llm_model_kwargs"] = {}
    rag = HyperGraphRAG(working_dir=str(directory), llm_model_func=fixed_llm, embedding_func=embedding, **parameters)
    rag.embedding_func = embedding
    for storage in (rag.full_docs, rag.text_chunks, rag.chunk_entity_relation_graph, rag.entities_vdb, rag.hyperedges_vdb, rag.chunks_vdb):
        storage.embedding_func = embedding
    rag.llm_model_func = limit_calls(rag.llm_model_func.__wrapped__, config["rag"]["llm_model_max_async"])
    for storage in (rag.entities_vdb, rag.hyperedges_vdb, rag.chunks_vdb):
        if storage.cosine_better_than_threshold != config["vector_search"]["cosine_better_than_threshold"]:
            raise ValueError("Vector threshold differs from the active baseline")
    if not read_only:
        write_json(directory / "baseline_config.json", config)
    return rag
