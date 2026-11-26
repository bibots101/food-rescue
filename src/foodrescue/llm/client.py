"""Small client for a local LLM served by Ollama.

* `chat_json` asks for JSON matching a pydantic model. Ollama constrains generation with the
  model's JSON schema, so the output is always syntactically valid JSON. We still validate it
  with pydantic, and on failure we retry with the validation error as feedback.
* Answers are cached on disk so the demo can run without the LLM:
    live   = always call the model, never use the cache;
    record = reuse a cached answer if there is one, otherwise call the model and save the answer;
    replay = cached answers only (fails if one is missing).
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from foodrescue.config import LLMConfig, project_path

T = TypeVar("T", bound=BaseModel)


class LLMMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class LLMError(RuntimeError):
    """The model is unreachable, has no recorded answer, or keeps returning invalid output."""


class LLMClient:
    def __init__(self, model: str, *, host: str | None = None, mode: LLMMode = LLMMode.RECORD,
                 cache_dir: Path | None = None, temperature: float = 0.0, seed: int = 42,
                 num_ctx: int = 8192, think: bool | None = False, max_retries: int = 2) -> None:
        self.model = model
        self.mode = LLMMode(mode)
        self.cache_dir = cache_dir
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self.think = think
        self.max_retries = max_retries
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.last_latency_s: float | None = None
        self.last_from_cache = False

    @classmethod
    def from_config(cls, config: LLMConfig, **overrides: Any) -> LLMClient:
        settings = {
            "host": config.host, "mode": config.mode, "cache_dir": project_path(config.cache_dir),
            "temperature": config.temperature, "seed": config.seed, "num_ctx": config.num_ctx,
            "think": config.think, "max_retries": config.max_retries,
        }
        settings.update(overrides)
        model = settings.pop("model", config.model)
        return cls(model, **settings)

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """Send a chat and return the answer parsed as `schema`."""
        key = self._cache_key(messages, schema)
        if self.mode is not LLMMode.LIVE:
            cached = self._load(key)
            if cached is not None:
                self.last_from_cache, self.last_latency_s = True, 0.0
                return schema.model_validate_json(cached)
            if self.mode is LLMMode.REPLAY:
                raise LLMError(f"replay mode: no recorded answer for this request (key {key[:12]}). "
                               "Run it once in record mode with Ollama running.")

        self.last_from_cache = False
        started = time.perf_counter()
        conversation = list(messages)
        error: ValidationError | None = None
        for _ in range(self.max_retries + 1):
            raw = self._call(conversation, schema)
            try:
                result = schema.model_validate_json(raw)
            except ValidationError as exc:
                error = exc
                conversation += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON is invalid: {exc}. Answer again with corrected JSON only."},
                ]
                continue
            self.last_latency_s = time.perf_counter() - started
            if self.mode is LLMMode.RECORD:
                self._save(key, messages, raw)
            return result
        raise LLMError(f"invalid output after {self.max_retries + 1} attempts: {error}")

    # --- internals -----------------------------------------------------------------------

    def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        request: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "format": schema.model_json_schema(), "options": self.options}
        if self.think is not None:
            request["think"] = self.think
        try:
            try:
                response = self._client.chat(**request)
            except ollama.ResponseError as exc:
                if "think" not in request or "think" not in str(exc).lower():
                    raise
                # This model has no thinking switch: retry without it, and stop sending it.
                request.pop("think")
                self.think = None
                response = self._client.chat(**request)
        except ollama.ResponseError as exc:
            raise LLMError(f"Ollama error for model {self.model!r}: {exc}") from exc
        except ConnectionError as exc:
            raise LLMError(f"cannot reach Ollama ({exc}). Is the Ollama app running?") from exc
        return response.message.content or ""

    def _cache_key(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        payload = {"model": self.model, "messages": messages, "schema": schema.model_json_schema(),
                   "options": self.options, "think": self.think}
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key[:24]}.json" if self.cache_dir else None

    def _load(self, key: str) -> str | None:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def _save(self, key: str, messages: list[dict[str, str]], raw: str) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "model": self.model, "messages": messages, "response": raw}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
