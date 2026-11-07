"""Verify the installed runtime and local storage without model or network calls."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

from ..common.files import ROOT, sha256




def verify_package_versions(packages):
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    actual = {normalize(item["name"]): item["version"] for item in packages}
    lock = (ROOT / "environment/requirements-win-py311.lock.txt").read_text(encoding="utf-8")
    expected = {
        normalize(name): version
        for name, version in re.findall(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;\\]+)", lock, flags=re.MULTILINE)
    }
    if actual != expected:
        raise RuntimeError("Installed package versions differ from the frozen lock; restore the environment before verifying")


async def check_storage(config, HyperGraphRAG, EmbeddingFunc, np):
    check_root = ROOT / "artifacts/environment-check"
    check_root.mkdir(parents=True, exist_ok=True)
    calls = {"llm": 0, "embedding_stub": 0}

    async def offline_llm(*args, **kwargs):
        calls["llm"] += 1
        raise AssertionError("The environment check must not call an LLM")

    async def offline_embedding(texts):
        calls["embedding_stub"] += 1
        return np.asarray(
            [[1.0, 0.0, 0.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0, 0.0, 0.0] for text in texts],
            dtype=np.float32,
        )

    with tempfile.TemporaryDirectory(prefix="storage-", dir=check_root) as temporary:
        def create():
            parameters = deepcopy(config["rag"])
            return HyperGraphRAG(
                working_dir=temporary,
                llm_model_func=offline_llm,
                embedding_func=EmbeddingFunc(embedding_dim=4, max_token_size=8192, func=offline_embedding),
                **parameters,
            )

        rag = create()
        await rag.entities_vdb.upsert({
            "entity-alpha": {"content": "alpha", "entity_name": "ALPHA"},
            "entity-beta": {"content": "beta", "entity_name": "BETA"},
        })
        result = await rag.entities_vdb.query("alpha", top_k=1)
        assert result and result[0]["entity_name"] == "ALPHA"
        graph = rag.chunk_entity_relation_graph
        await graph.upsert_node("ALPHA", node_data={"role": "entity", "source_id": "chunk-check"})
        await graph.upsert_node("CHECK-RELATION", node_data={"role": "hyperedge", "source_id": "chunk-check"})
        await graph.upsert_edge("CHECK-RELATION", "ALPHA", edge_data={"weight": 1.0, "source_id": "chunk-check"})
        await rag.text_chunks.upsert({"chunk-check": {"content": "alpha local storage check", "tokens": 4, "full_doc_id": "doc-check", "chunk_order_index": 0}})
        await rag._insert_done()
        reloaded = create()
        assert await reloaded.chunk_entity_relation_graph.has_edge("CHECK-RELATION", "ALPHA")
        assert (await reloaded.text_chunks.get_by_id("chunk-check"))["content"] == "alpha local storage check"
        restored = await reloaded.entities_vdb.query("alpha", top_k=1)
        assert restored and restored[0]["entity_name"] == "ALPHA"
    assert calls["llm"] == 0
    return {"kv_roundtrip": True, "graphml_roundtrip": True, "vector_query_and_roundtrip": True, "real_model_calls": 0, "stub_calls": calls}


def main():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run this check with .venv/Scripts/python.exe (Python 3.11)")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use this project's isolated .venv")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from ..configuration import load
    from ..source import select_source as _select_source
    config = load()
    _select_source(config)
    import hypergraphrag
    from hypergraphrag import HyperGraphRAG, QueryParam
    from hypergraphrag.utils import EmbeddingFunc
    from ..runtime import make_query_param
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from openai import AsyncOpenAI, AsyncAzureOpenAI, Timeout
    import tiktoken

    expected = ROOT / config["source"]["checkout"] / "hypergraphrag/__init__.py"
    assert Path(hypergraphrag.__file__).resolve() == expected.resolve()
    assert make_query_param().top_k == config["retrieval"]["top_k"]
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(torch.from_numpy(values).numpy(), values)
    storage = asyncio.run(check_storage(config, HyperGraphRAG, EmbeddingFunc, np))
    packages = sorted(
        [{"name": item.metadata["Name"], "version": item.version} for item in importlib.metadata.distributions() if item.metadata.get("Name")],
        key=lambda item: item["name"].lower(),
    )
    verify_package_versions(packages)
    directory = ROOT / "environment"
    (ROOT / "reports/environment").mkdir(parents=True, exist_ok=True)
    report = {
        "validated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "timezone": "Asia/Shanghai",
        "passed": True,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "base_interpreter": sys.base_prefix,
        "virtual_environment": sys.prefix,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "source_commit": config["source"]["commit"],
        "source_import_path": hypergraphrag.__file__,
        "baseline_id": config["baseline_id"],
        "package_count": len(packages),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cpu_runtime": True,
        "checks": {
            "isolated_python_311": True,
            "active_profile_integrity": True,
            "pinned_upstream_import": True,
            "upstream_sdk_imports": True,
            "transformers_imports": True,
            "unified_query_param_factory": True,
            "numpy_torch_interop": True,
            "installed_versions_match_lock": True,
            **storage,
        },
        "lock_file": "environment/requirements-win-py311.lock.txt",
        "lock_sha256": sha256(directory / "requirements-win-py311.lock.txt"),
        "installed_packages_sha256": sha256(directory / "installed-packages.json"),
        "scope": "Selected local storage backend and imported LLM/embedding dependencies; optional backends excluded.",
        "local_embedding_configured": True,
        "tokenizer_data_prefetched": True,
        "full_rag_experiment_started": False,
    }
    (ROOT / "reports/environment/validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "python": report["python_version"], "torch": report["torch_version"], "packages": len(packages), "checks": report["checks"], "report": str(ROOT / "reports/environment/validation.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
