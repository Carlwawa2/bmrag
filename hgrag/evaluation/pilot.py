"""Run the real-corpus pipeline with exact context and model-call observations."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import getpass
import html
import json
import os
from pathlib import Path
import re
import time
import uuid

from ..common.files import ROOT, LOCAL_TIMEZONE, now, sha256, write_json
from ..configuration import load
from ..common.provenance import source_records
from ..models.bge_m3 import BGEEncoder
from ..runtime import make_query_param, make_rag
from ..models.tokenizer import load_tokenizer
from ..models.opencode import APIError
from ..common.files import redact

PILOT = ROOT / "pilots/hotpot-mini"




def normalized(text):
    return " ".join(html.unescape(text).casefold().split())


class Observer:
    def __init__(self, directory, key, config, policy):
        self.directory, self.key = directory, key
        self.policy = policy
        self.phase = "construction"
        self.question_id = None
        self.attempt_count = 0
        self.call_count = 0
        self.calls = []
        self.contexts = {}
        self.guard = asyncio.Semaphore(config["rag"]["llm_model_max_async"])

    def bind(self, rag):
        # Preserve the exact partial/kwargs from the frozen factory, replacing
        # only upstream's exception-unsafe counter with a semaphore of the same size.
        original = rag.llm_model_func.__wrapped__

        async def traced(prompt, *args, **kwargs):
            async with self.guard:
                self.call_count += 1
                call_id = self.call_count
                kind = "generation" if kwargs.get("system_prompt") else (
                    "summary" if "max_tokens" in kwargs else (
                        "query_analysis" if self.phase == "query" else (
                            "gleaning_or_loop_check" if kwargs.get("history_messages") else "extraction"
                        )
                    )
                )
                for attempt in range(1, self.policy["max_attempts_per_call"] + 1):
                    if self.attempt_count >= self.policy["max_total_api_attempts"]:
                        raise RuntimeError("Pilot API attempt limit reached")
                    self.attempt_count += 1
                    started = time.perf_counter()
                    row = {"call_id": call_id, "attempt": attempt, "phase": self.phase, "question_id": self.question_id, "kind": kind, "started_at": now()}
                    try:
                        response = await original(prompt, *args, **kwargs)
                    except Exception as error:
                        row.update(status="failed", error=redact(str(error), self.key), elapsed_seconds=round(time.perf_counter() - started, 4))
                        self.calls.append(row)
                        self.save_calls()
                        transient = isinstance(error, APIError) and (error.status in ("network", 429) or isinstance(error.status, int) and error.status >= 500)
                        if not transient or attempt == self.policy["max_attempts_per_call"]:
                            raise
                        await asyncio.sleep(attempt)
                        continue
                    row.update(status="passed", elapsed_seconds=round(time.perf_counter() - started, 4), response_characters=len(response))
                    if kind in ("extraction", "query_analysis", "gleaning_or_loop_check"):
                        row["hyperrelation_record_markers"] = response.count('"hyper-relation"<|>')
                        row["entity_record_markers"] = response.count('"entity"<|>')
                        row["completion_marker"] = "<|COMPLETE|>" in response
                    self.calls.append(row)
                    self.save_calls()
                    if kind in ("extraction", "generation"):
                        print(f"MODEL {kind}: call {call_id}, {row['elapsed_seconds']:.2f}s, {len(response)} characters", flush=True)
                    return response

        rag.llm_model_func = traced

    def save_calls(self):
        write_json(self.directory / "pilot-calls.json", self.calls, self.key)


async def graph_summary(rag, documents):
    from hypergraphrag.prompt import GRAPH_FIELD_SEP
    graph = rag.chunk_entity_relation_graph._graph
    entity_nodes = [name for name, attrs in graph.nodes(data=True) if attrs.get("role") == "entity"]
    hyperedges = [name for name, attrs in graph.nodes(data=True) if attrs.get("role") == "hyperedge"]
    chunks = {name: await rag.text_chunks.get_by_id(name) for name in await rag.text_chunks.all_keys()}
    assert len(await rag.full_docs.all_keys()) == len(documents)
    assert entity_nodes and hyperedges and graph.number_of_edges() > 0 and chunks
    assert all(attrs.get("role") in ("entity", "hyperedge") for _, attrs in graph.nodes(data=True))
    assert all(graph.nodes[left]["role"] != graph.nodes[right]["role"] for left, right in graph.edges())
    missing_sources = []
    for name, attrs in graph.nodes(data=True):
        for source in attrs.get("source_id", "").split(GRAPH_FIELD_SEP):
            if source not in chunks:
                missing_sources.append({"node": name, "source_id": source})
    assert not missing_sources, "Graph source links must point to indexed chunks"
    degrees = Counter(graph.degree(name) for name in hyperedges)
    assert any(degree >= 3 for degree in degrees), "Check that at least one hyperedge has 3+ members"
    return {
        "documents": len(await rag.full_docs.all_keys()), "chunks": len(chunks),
        "entity_nodes": len(entity_nodes), "hyperedge_nodes": len(hyperedges),
        "incidence_edges": graph.number_of_edges(),
        "hyperedge_cardinality_histogram": dict(sorted(degrees.items())),
        "hyperedges_with_at_least_three_entities": sum(count for degree, count in degrees.items() if degree >= 3),
        "bipartite_incidence_valid": True, "source_links_valid": True,
        "vector_counts": {name: len(getattr(rag, name).client_storage["data"]) for name in ("entities_vdb", "hyperedges_vdb", "chunks_vdb")},
    }, chunks


async def execute(args, key, config, policy, documents, questions):
    encoder = BGEEncoder(config)
    rag = make_rag(args.run_name, api_key=key, encoder=encoder)
    directory = Path(rag.working_dir)
    if (directory / "pilot-summary.json").exists() and not args.query_only:
        raise RuntimeError("This run already has observations; select a new run name")
    observer = Observer(directory, key, config, policy)
    encoder_for_counts = load_tokenizer()
    started = time.perf_counter()
    report = {
        "started_at": now(), "run_name": args.run_name, "status": "running",
        "baseline_id": config["baseline_id"], "config_sha256": sha256(ROOT / "configs/baseline.json"),
        "config_manifest_sha256": sha256(ROOT / "configs/manifest.json"),
        "corpus_manifest_sha256": sha256(PILOT / "manifest.json"),
        "policy": policy, "credential_persisted": False,
        "script_files": source_records(),
        "answers": [], "official_metrics_computed": False,
        "scope": "Two preselected development questions and 20 candidate paragraphs; engineering check, not formal benchmark performance",
    }
    write_json(directory / "pilot-summary.json", report, key)
    original_context_builder = None
    try:
        observer.bind(rag)
        if args.query_only:
            if not (directory / "build-summary.json").is_file():
                raise RuntimeError("Query-only mode requires a completed graph construction record")
            previous = json.loads((directory / "build-summary.json").read_text(encoding="utf-8"))
            if previous["config_sha256"] != report["config_sha256"] or previous["corpus_manifest_sha256"] != report["corpus_manifest_sha256"]:
                raise RuntimeError("Cannot reuse a graph built with different configuration or corpus")
            build_seconds = None
        else:
            if await rag.full_docs.all_keys():
                raise RuntimeError("Fresh pilot construction requires an empty run directory")
            tick = time.perf_counter()
            print(f"BUILD: inserting {len(documents)} original candidate paragraphs", flush=True)
            await rag.ainsert([doc["content"] for doc in documents])
            build_seconds = time.perf_counter() - tick
        stats, chunks = await graph_summary(rag, documents)
        build = {**stats, "build_seconds": build_seconds, "config_sha256": report["config_sha256"], "corpus_manifest_sha256": report["corpus_manifest_sha256"]}
        if not args.query_only:
            write_json(directory / "build-summary.json", build, key)
        report["graph"] = build
        print("BUILD PASSED: " + json.dumps(stats), flush=True)
        # Read the graph and all stores back from the files actually written by ainsert.
        rag = make_rag(args.run_name, api_key=key, encoder=encoder)
        restored, chunks = await graph_summary(rag, documents)
        assert restored == stats
        observer.bind(rag)
        report["graph_reloaded_and_verified"] = True
        observer.phase = "query"
        import hypergraphrag.operate as operation
        original_context_builder = operation._build_query_context

        async def observed_context(*call_args, **call_kwargs):
            context = await original_context_builder(*call_args, **call_kwargs)
            identifier = observer.question_id
            observer.contexts[identifier] = {"content": context, "keywords": call_args[0], "effective_mode": call_args[-1].mode}
            if context:
                (directory / f"context-{identifier}.txt").write_text(context, encoding="utf-8")
            return context

        operation._build_query_context = observed_context
        document_by_id = {doc["id"]: doc for doc in documents}
        for question in questions:
            identifier = observer.question_id = question["id"]
            tick = time.perf_counter()
            answer = await rag.aquery(question["question"], param=make_query_param())
            elapsed = time.perf_counter() - tick
            context_observation = observer.contexts.get(identifier)
            assert context_observation and isinstance(context_observation["content"], str) and context_observation["content"].strip(), "The actual answering call must use retrieved context"
            assert isinstance(answer, str) and answer.strip() and "Sorry, I'm not able" not in answer
            context = context_observation["content"]
            sources_match = re.search(r"-----Sources-----\s*```csv\s*([\s\S]*?)```", context)
            assert sources_match, "The upstream context must contain a Sources section"
            sources_text = sources_match.group(1)
            retained = [name for name, chunk in chunks.items() if normalized(chunk["content"]) in normalized(sources_text)]
            retained_titles = sorted({document_by_id[chunks[name]["full_doc_id"]]["title"] for name in retained})
            support_titles = sorted(set(question["supporting_facts"]["title"]))
            support_checks = []
            for title, sent_id in zip(question["supporting_facts"]["title"], question["supporting_facts"]["sent_id"]):
                document = next(doc for doc in documents if doc["title"] == title)
                sentence = document["sentences"][sent_id]
                support_checks.append({"title": title, "sentence_index": sent_id, "sentence": sentence, "present_in_actual_sources": normalized(sentence) in normalized(sources_text)})
            record = {
                "id": identifier, "question": question["question"], "type": question["type"],
                "reference_answer_for_posthoc_check_only": question["answer"], "answer": answer,
                "reference_phrase_present": bool(re.search(r"(?<!\w)" + re.escape(normalized(question["answer"])) + r"(?!\w)", normalized(answer))),
                "elapsed_seconds": round(elapsed, 4), "retrieval_mode": context_observation["effective_mode"],
                "retrieval_keywords": context_observation["keywords"], "serialized_context_tokens": len(encoder_for_counts.encode(context)),
                "retained_chunk_ids_by_full_text_match": retained, "retained_source_titles": retained_titles,
                "reference_support_titles": support_titles, "all_reference_titles_retrieved": set(support_titles).issubset(retained_titles),
                "supporting_sentence_checks": support_checks,
                "all_supporting_sentences_in_sources": all(item["present_in_actual_sources"] for item in support_checks),
                "context_file": f"context-{identifier}.txt",
                "reference_phrase_check_is_not_official_exact_match": True,
            }
            report["answers"].append(record)
            write_json(directory / f"answer-{identifier}.json", record, key)
            write_json(directory / "pilot-summary.json", report, key)
            print(f"ANSWER {identifier}: reference phrase={record['reference_phrase_present']}; all support titles={record['all_reference_titles_retrieved']}; seconds={elapsed:.2f}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": redact(str(error), key)}
        raise
    finally:
        if original_context_builder is not None:
            import hypergraphrag.operate as operation
            operation._build_query_context = original_context_builder
        report["finished_at"] = now()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        report["api_attempt_count"] = observer.attempt_count
        report["model_call_kinds"] = dict(Counter(row["kind"] for row in observer.calls if row["status"] == "passed"))
        audits = sorted((directory / "model_calls").glob("*.json"))
        usage_totals = Counter()
        for path in audits:
            audit = json.loads(path.read_text(encoding="utf-8"))
            usage = audit.get("response", {}).get("usage", {})
            usage_totals["input_tokens"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            usage_totals["output_tokens"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            usage_totals["total_tokens"] += usage.get("total_tokens", 0) or 0
        report["usage_from_successful_response_audits"] = dict(usage_totals)
        report["successful_response_audit_files"] = len(audits)
        report["evidence_files"] = [{"path": path.relative_to(directory).as_posix(), "sha256": sha256(path)} for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in ("pilot-summary.json", "pilot-summary.json.tmp", "evidence-manifest.json")]
        write_json(directory / "pilot-summary.json", report, key)
        print(f"PILOT {report['status']}: {directory / 'pilot-summary.json'}", flush=True)
    return report, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="hotpot-mini-" + datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8])
    parser.add_argument("--query-only", action="store_true")
    args = parser.parse_args()
    config = load()
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Pilot input changed: {item['path']}")
    policy = json.loads((ROOT / "configs/pilot.json").read_text(encoding="utf-8"))
    if policy["baseline_config_sha256"] != sha256(ROOT / "configs/baseline.json"):
        raise ValueError("Pilot requires the current baseline configuration")
    documents = json.loads((PILOT / "documents.json").read_text(encoding="utf-8"))
    questions = json.loads((PILOT / "questions.json").read_text(encoding="utf-8"))
    key = os.environ.get(config["api"]["api_key_environment_variable"])
    if not key:
        key = getpass.getpass("OpenCode API key (hidden, not saved): ").strip()
    if not key:
        raise RuntimeError("No API credential supplied")
    try:
        report, directory = asyncio.run(execute(args, key, config, policy, documents, questions))
        if report["status"] == "passed":
            write_json(ROOT / "reports/hotpot-mini/latest-run.json", {"summary": (directory / "pilot-summary.json").relative_to(ROOT).as_posix(), "sha256": sha256(directory / "pilot-summary.json"), "status": report["status"], "finished_at": report["finished_at"]}, key)
    finally:
        key = ""


if __name__ == "__main__":
    main()
