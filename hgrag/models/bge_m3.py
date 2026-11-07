"""Local, length-checked BGE-M3 dense embeddings using the locked dependencies."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading
import time

from ..common.files import ROOT
from ..configuration import verify_model


class InputTooLong(ValueError):
    pass


class BGEEncoder:
    def __init__(self, config):
        self.settings = dict(config["models"]["embedding"])
        self.directory = verify_model(config)
        self._lock = threading.Lock()
        os.environ["HF_HOME"] = str(ROOT / "artifacts/huggingface")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.torch = torch
        torch.set_num_threads(self.settings["cpu_threads"])
        torch.manual_seed(config["runtime"]["local_random_seed"])
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.directory), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        self.model, loading = AutoModel.from_pretrained(
            str(self.directory), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=False,
            attn_implementation=self.settings["attention_implementation"], output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"BGE-M3 weights did not load exactly: {loading}")
        self.model.eval()
        self.model.requires_grad_(False)
        self.load_seconds = time.perf_counter() - started
        self.calls = 0
        self.inputs_encoded = 0

    def token_lengths(self, texts):
        return [len(ids) for ids in self.tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]]

    def encode(self, texts):
        import numpy as np
        if not isinstance(texts, (list, tuple)) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be a list of nonempty strings")
        if not texts:
            return np.empty((0, self.settings["dimensions"]), dtype=np.float32)
        with self._lock:
            tokenized = self.tokenizer(list(texts), add_special_tokens=True, truncation=False, verbose=False)
            lengths = [len(ids) for ids in tokenized["input_ids"]]
            for index, length in enumerate(lengths):
                if length > self.settings["max_input_tokens"]:
                    raise InputTooLong(f"Embedding input {index} has {length} BGE-M3 tokens, exceeding {self.settings['max_input_tokens']} (including special tokens)")
            embeddings = []
            # A single CPU forward pass per item prevents large attention padding allocations.
            for index in range(len(texts)):
                encoded = self.tokenizer.pad(
                    {key: [values[index]] for key, values in tokenized.items()},
                    padding=True, return_tensors="pt", verbose=False,
                )
                with self.torch.inference_mode():
                    result = self.model(**encoded).last_hidden_state[:, 0]
                    result = self.torch.nn.functional.normalize(result, p=2, dim=1)
                embeddings.append(result.cpu().numpy())
            values = np.concatenate(embeddings).astype(np.float32, copy=False)
            if values.shape != (len(texts), self.settings["dimensions"]) or not np.isfinite(values).all():
                raise ValueError("BGE-M3 returned invalid embedding values")
            self.calls += 1
            self.inputs_encoded += len(texts)
            return values

    async def __call__(self, texts):
        return await asyncio.to_thread(self.encode, texts)
