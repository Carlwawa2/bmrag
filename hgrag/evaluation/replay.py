"""Offline regression: real BGE-M3 retrieval, recorded LLM responses, original graph."""
from __future__ import annotations

from copy import deepcopy
import asyncio
import json

from ..common.files import ROOT, now, read_json, sha256, write_json
from ..configuration import CONFIG_PATH, load
from ..models.bge_m3 import BGEEncoder
from ..models.opencode import chat_parameters
from ..runtime import make_rag, make_query_param
from .pilot import graph_summary
from .records import load_run


async def replay():
    config = load()
    latest, original, directory, documents = load_run(config)
    summary_path = ROOT / latest["summary"]
    encoder = BGEEncoder(config)
    rag = make_rag("baseline-replay", offline=True, encoder=encoder, working_dir=directory, read_only=True)
    stats, _ = await graph_summary(rag, documents)
    serialized_stats = json.loads(json.dumps(stats))
    assert all(value == original["graph"][key] for key, value in serialized_stats.items())
    audits = [read_json(path) for path in (directory / "model_calls").glob("*.json")]
    used = []

    async def recorded_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(deepcopy(history_messages or []))
        messages.append({"role": "user", "content": prompt})
        candidates = [audit for audit in audits if audit["request"]["messages"] == messages]
        if len(candidates) != 1:
            raise AssertionError("Recomputed prompt/context differs from the saved model request")
        audit = candidates[0]
        payload = {"model": config["models"]["generation"], "messages": messages, **chat_parameters(config, kwargs)}
        assert payload == audit["request"], "Model request parameters changed"
        used.append({"kind": "generation" if system_prompt else "query_analysis", "messages_and_parameters_exact_match": True})
        return audit["response"]["choices"][0]["message"]["content"]

    rag.llm_model_func = recorded_llm
    answers = []
    # Any accidental network request fails instead of silently contacting an API.
    from unittest.mock import patch
    with patch("socket.socket.connect", side_effect=AssertionError("Offline replay attempted a network connection")) as connection:
        for previous in original["answers"]:
            answer = await rag.aquery(previous["question"], param=make_query_param())
            assert answer == previous["answer"]
            answers.append({"id": previous["id"], "answer_exact_match": True, "actual_generation_request_exact_match": True})
        assert connection.call_count == 0
    assert len(used) == 4
    # The replay never saves into the historical run directory.
    assert sha256(summary_path) == latest["sha256"]
    for record in original["evidence_files"]:
        assert sha256(directory / record["path"]) == record["sha256"]
    report = {
        "verified_at": now(), "status": "passed", "configuration": "configs/baseline.json",
        "config_sha256": sha256(CONFIG_PATH), "scientific_settings_unchanged": True,
        "baseline_run": latest["summary"], "evidence_files_unchanged": len(original["evidence_files"]),
        "graph": stats, "model_requests_replayed": used, "answers": answers,
        "live_llm_api_calls": 0, "network_connections": 0,
        "embedding_model": "Actual local BGE-M3, not a stub",
        "scope": "Recompute retrieval and compare exact prompts, parameters and recorded answers; not a new live-LLM experiment",
    }
    write_json(ROOT / "reports/baseline/replay.json", report)
    return report


def main():
    report = asyncio.run(replay())
    print(json.dumps({"status": report["status"], "queries": len(report["answers"]), "exact_model_requests": len(report["model_requests_replayed"]), "live_llm_api_calls": 0, "report": "reports/baseline/replay.json"}, indent=2))


if __name__ == "__main__":
    main()
