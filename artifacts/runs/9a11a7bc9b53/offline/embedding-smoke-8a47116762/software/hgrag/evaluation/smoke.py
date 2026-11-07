"""Measure real local embeddings and exercise the pinned RAG vector callbacks."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
import platform
import time
import uuid

from ..common.files import ROOT, sha256
from ..configuration import load
from ..models.bge_m3 import BGEEncoder, InputTooLong
from ..models.tokenizer import load_tokenizer

TEXTS = [
    "How do mitochondria generate cellular energy?",
    "Mitochondria produce ATP through oxidative phosphorylation to supply energy to cells.",
    "The city council approved a plan to renovate the public library.",
    "线粒体如何为细胞提供能量？",
    "线粒体通过氧化磷酸化产生 ATP，为细胞活动提供能量。",
    "市议会批准了公共图书馆的翻修计划。",
]


async def storage_checks(encoder, oversized, run_name):
    from ..runtime import make_query_param, make_rag
    rag = make_rag(run_name, offline=True, encoder=encoder)
    assert make_query_param().top_k == 60
    await rag.entities_vdb.upsert({
        "mitochondria": {"content": TEXTS[1], "entity_name": "MITOCHONDRIA"},
        "library": {"content": TEXTS[2], "entity_name": "LIBRARY"},
    })
    await rag.hyperedges_vdb.upsert({"energy-process": {"content": TEXTS[1], "hyperedge_name": "ATP-PRODUCTION"}})
    await rag.chunks_vdb.upsert({"energy-chunk": {"content": TEXTS[1]}})
    query = await rag.entities_vdb.query(TEXTS[0], top_k=2)
    assert query and query[0]["entity_name"] == "MITOCHONDRIA"
    assert (await rag.hyperedges_vdb.query(TEXTS[0], top_k=1))[0]["hyperedge_name"] == "ATP-PRODUCTION"
    assert (await rag.chunks_vdb.query(TEXTS[0], top_k=1))[0]["id"] == "energy-chunk"
    try:
        await rag.embedding_func([oversized])
    except InputTooLong:
        pass
    else:
        raise AssertionError("The RAG callback accepted an oversized input")
    recovered = await asyncio.wait_for(rag.entities_vdb.query(TEXTS[0], top_k=1), timeout=60)
    assert recovered[0]["entity_name"] == "MITOCHONDRIA"
    await rag._insert_done()
    restored = make_rag(run_name, offline=True, encoder=encoder)
    reloaded = await restored.entities_vdb.query(TEXTS[0], top_k=1)
    assert reloaded[0]["entity_name"] == "MITOCHONDRIA"
    # Two concurrent callers must finish through the same one-slot semaphore.
    values = await asyncio.wait_for(asyncio.gather(
        restored.embedding_func([TEXTS[0]]), restored.embedding_func([TEXTS[3]])
    ), timeout=60)
    assert all(value.shape == (1, 1024) for value in values)
    return {
        "rag_factory": True, "entity_vector_search": True,
        "hyperedge_vector_search": True, "chunk_vector_search": True,
        "vector_persistence_and_reload": True,
        "oversized_input_rejected_by_rag_callback": True,
        "semaphore_released_after_error": True,
        "concurrent_embedding_callers": True,
        "directory": str(rag.working_dir),
        "entity_query_result": [
            {"id": row["id"], "entity_name": row["entity_name"], "cosine_similarity": float(row["distance"])}
            for row in query
        ],
        "llm_api_calls": 0,
    }


def main():
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("Set PYTHONHASHSEED=42 before starting the smoke check")
    config = load()
    import numpy as np
    import psutil
    from ..source import select_source as _select_source
    _select_source(config)
    from hypergraphrag.operate import chunking_by_token_size
    import hypergraphrag.utils as upstream_utils

    upstream_utils.ENCODER = load_tokenizer()
    process = psutil.Process()
    started = time.perf_counter()
    encoder = BGEEncoder(config)
    print(f"Loaded BGE-M3 in {encoder.load_seconds:.3f} seconds", flush=True)
    short_started = time.perf_counter()
    vectors = encoder.encode(TEXTS)
    short_seconds = time.perf_counter() - short_started
    assert vectors.shape == (6, 1024) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    similarities = vectors @ vectors.T
    assert similarities[0, 1] > similarities[0, 2]
    assert similarities[3, 4] > similarities[3, 5]
    repeated = encoder.encode([TEXTS[0]])
    difference = float(np.abs(vectors[0] - repeated[0]).max())
    assert difference <= 1e-6
    assert encoder.encode([]).shape == (0, 1024)
    long_results = []
    for language, paragraph in (
        ("English", "A bridge evidence item connects two facts needed to answer a multi-hop question. Evidence selection must account for redundancy and a limited context budget. "),
        ("Chinese", "桥接证据连接回答多跳问题所需的两个事实。证据选择需要同时考虑证据冗余、互补关系和有限的上下文预算。"),
    ):
        chunk = chunking_by_token_size(paragraph * 160, overlap_token_size=100, max_token_size=1200, tiktoken_model="gpt-4o-mini")[0]
        text = chunk["content"]
        count = encoder.token_lengths([text])[0]
        tick = time.perf_counter()
        result = encoder.encode([text])
        elapsed = time.perf_counter() - tick
        assert result.shape == (1, 1024) and np.isfinite(result).all()
        long_results.append({"language": language, "upstream_declared_tokens": chunk["tokens"], "bge_tokens_including_special_tokens": count, "elapsed_seconds": round(elapsed, 4)})
        print(f"{language} chunk: upstream={chunk['tokens']}, BGE={count}, seconds={elapsed:.3f}", flush=True)
    oversized = "evidence " * 9000
    oversized_tokens = encoder.token_lengths([oversized])[0]
    try:
        encoder.encode([oversized])
    except InputTooLong:
        overflow_rejected = True
    else:
        raise AssertionError("The encoder silently truncated an oversized input")
    run_name = "embedding-smoke-" + uuid.uuid4().hex[:10]
    storage = asyncio.run(storage_checks(encoder, oversized, run_name))
    memory = process.memory_info()
    report = {
        "validated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "timezone": "Asia/Shanghai", "status": "passed", "baseline_id": config["baseline_id"],
        "config_sha256": sha256(ROOT / "configs/baseline.json"),
        "manifest_sha256": sha256(ROOT / "configs/manifest.json"),
        "model_revision": encoder.settings["revision"],
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tiktoken")},
        "encoding": encoder.settings,
        "hardware": {"cpu_logical": os.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False), "memory_total_gib": round(psutil.virtual_memory().total / 2**30, 3)},
        "timings": {"model_load_seconds": round(encoder.load_seconds, 4), "six_short_inputs_seconds": round(short_seconds, 4), "long_chunks": long_results, "total_check_seconds": round(time.perf_counter() - started, 4)},
        "memory": {"rss_gib_at_end": round(memory.rss / 2**30, 3), "process_peak_rss_gib": round(getattr(memory, "peak_wset", memory.rss) / 2**30, 3)},
        "checks": {"shape_6_by_1024": True, "dtype_float32": True, "all_values_finite": True, "unit_l2_norm": True, "empty_batch": True, "repeat_max_abs_difference": difference, "oversized_input_rejected": overflow_rejected, "oversized_test_tokens": oversized_tokens, "no_silent_truncation": True, **storage},
        "short_samples": TEXTS, "short_input_token_lengths": encoder.token_lengths(TEXTS),
        "similarities": {"english_positive": float(similarities[0, 1]), "english_negative": float(similarities[0, 2]), "chinese_positive": float(similarities[3, 4]), "chinese_negative": float(similarities[3, 5]), "scope": "Handwritten smoke examples; not a retrieval benchmark or quality estimate"},
        "embedding_calls": encoder.calls, "inputs_encoded": encoder.inputs_encoded,
        "full_rag_answering_executed": False,
        "scope": "Real offline embeddings and all three upstream vector stores; no extraction, graph construction, LLM answering or research evaluation",
    }
    path = ROOT / "reports/embedding/validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
