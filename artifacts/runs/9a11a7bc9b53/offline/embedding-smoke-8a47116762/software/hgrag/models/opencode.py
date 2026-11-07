"""Small stdlib transport with explicit endpoints and redacted audit output."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

from ..common.files import redact, write_json


class APIError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail
        super().__init__(f"API request failed: HTTP {status}; {detail}")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None




def request_json(url: str, key: str | None, session_id: str, payload=None, timeout=30):
    target = urlsplit(url)
    if target.scheme != "https" or target.hostname != "opencode.ai" or target.username or target.password or target.query or target.fragment:
        raise ValueError("The OpenCode transport only accepts the configured HTTPS host")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "HyperGraphRAG-Research/2.0",
        "x-opencode-session": session_id,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    content = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=content, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            metadata = {
                "status": response.status,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "request_id": response.headers.get("x-request-id"),
            }
            return body, metadata
    except urllib.error.HTTPError as error:
        detail = redact(error.read(8192).decode("utf-8", errors="replace"), key or "")
        raise APIError(error.code, detail[:1500]) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise APIError("network", redact(str(error), key or "")) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise APIError("invalid-response", "The endpoint did not return valid JSON") from None


def chat_parameters(config: dict, overrides: dict | None = None) -> dict:
    parameters = deepcopy(config["rag"]["llm_model_kwargs"])
    overrides = dict(overrides or {})
    overrides.pop("hashing_kv", None)
    overrides.pop("keyword_extraction", None)
    # The pinned upstream summary call still uses max_tokens=500.
    if "max_tokens" in overrides:
        overrides["max_completion_tokens"] = overrides.pop("max_tokens")
    if overrides.get("stream"):
        raise ValueError("The baseline uses non-streaming responses")
    if set(overrides) & {"model", "base_url", "api_key", "messages"}:
        raise ValueError("Per-call overrides cannot change the frozen provider or model")
    parameters.update(overrides)
    return parameters


def complete(config, key, session_id, prompt, system_prompt=None, history_messages=None, overrides=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(deepcopy(history_messages or []))
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": config["models"]["generation"],
        "messages": messages,
        **chat_parameters(config, overrides),
    }
    body, metadata = request_json(
        config["api"]["base_url"].rstrip("/") + "/chat/completions",
        key, session_id, payload, config["api"]["timeout_seconds"],
    )
    if not isinstance(body, dict) or body.get("error"):
        raise APIError("provider-error", redact(json.dumps(body, ensure_ascii=False), key)[:1500])
    choices = body.get("choices") or []
    text = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(text, str) or not text.strip():
        raise APIError("empty-output", "The completion did not contain usable text")
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": config["api"]["provider"],
        "endpoint": config["api"]["base_url"].rstrip("/") + "/chat/completions",
        "session_id": session_id,
        "request": payload,
        "response": body,
        "transport": metadata,
    }
    return text, audit


def save_audit(directory: Path, record: dict, key: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}.json"
    write_json(path, record, key)
    return path
