"""Validate a baseline run using its actual inputs and saved artifacts."""
from __future__ import annotations

from ..common.files import ROOT, project_path, read_json, sha256
from ..configuration import load, scientific_fingerprint

DATA = ROOT / "pilots/hotpot-mini"
LATEST = ROOT / "reports/hotpot-mini/latest-run.json"


def load_run(config=None):
    config = config or load()
    for record in read_json(DATA / "manifest.json")["files"]:
        if sha256(project_path(record["path"])) != record["sha256"]:
            raise ValueError(f"Pilot input differs: {record['path']}")
    latest = read_json(LATEST)
    summary_path = project_path(latest["summary"])
    directory = summary_path.parent
    directory.relative_to((ROOT / "artifacts").resolve())
    if sha256(summary_path) != latest["sha256"]:
        raise ValueError("The run summary checksum differs")
    report = read_json(summary_path)
    if report["status"] != "passed":
        raise ValueError("The saved run did not complete")
    for record in report["evidence_files"]:
        path = (directory / record["path"]).resolve()
        path.relative_to(directory)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Run artifact differs: {record['path']}")
    saved_config = read_json(directory / "baseline_config.json")
    if scientific_fingerprint(saved_config) != scientific_fingerprint(config):
        raise ValueError("Run model, source or retrieval parameters differ from the current baseline")
    documents = read_json(DATA / "documents.json")
    stored = read_json(directory / "kv_store_full_docs.json")
    if {doc["id"]: doc["content"] for doc in documents} != {key: value["content"] for key, value in stored.items()}:
        raise ValueError("Stored documents differ from the pilot corpus")
    questions = {item["id"]: item for item in read_json(DATA / "questions.json")}
    for answer in report["answers"]:
        reference = questions[answer["id"]]
        if answer["question"] != reference["question"] or answer["reference_answer_for_posthoc_check_only"] != reference["answer"]:
            raise ValueError("Saved questions differ from the pilot inputs")
    return latest, report, directory, documents
