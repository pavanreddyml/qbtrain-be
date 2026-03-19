# qbtrain/ai/llm/ollama_client.py
from __future__ import annotations

import threading
import re
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Generator, List, Optional, Type, Tuple

import ollama
from pydantic import BaseModel, ValidationError
from qbtrain.tracers import Tracer

from .base_llm_client import LLMClient, Message
from qbtrain.utils.jsonutils import extract_first_json

MessageList = Optional[List[Message]]




def _ollama_guardrails(top_k: int) -> None:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 for Ollama.")


@dataclass
class PullTask:
    model: str
    status: str = "queued"  # queued | pulling | completed | failed
    progress: float = 0.0
    message: str = ""


class OllamaClient(LLMClient):
    client_id = "ollama"
    display_name = "Ollama (local)"
    param_display_names = {"host": "Server URL (http://127.0.0.1:11434)", "model": "Model name"}

    _LOCK = threading.RLock()
    _QUEUE: Deque[PullTask] = deque()
    _CURRENT: Optional[PullTask] = None
    _WORKER: Optional[threading.Thread] = None

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_output_tokens=max_output_tokens,
        )
        self.client = ollama.Client(host=host)

    # ---- Pull manager ----
    @classmethod
    def _ensure_worker(cls):
        with cls._LOCK:
            if cls._WORKER is None or not cls._WORKER.is_alive():
                cls._WORKER = threading.Thread(target=cls._worker_loop, daemon=True)
                cls._WORKER.start()

    @classmethod
    def _worker_loop(cls):
        while True:
            with cls._LOCK:
                if not cls._QUEUE:
                    cls._CURRENT = None
                    break
                task = cls._QUEUE.popleft()
                cls._CURRENT = task
                task.status = "pulling"

            try:
                for ev in ollama.pull(model=task.model, stream=True):
                    total = ev.get("total", 0) or 0
                    completed = ev.get("completed", 0) or 0
                    if total > 0:
                        prog = (completed / total) * 100.0
                        with cls._LOCK:
                            task.progress = prog
                    status = ev.get("status", "")
                    with cls._LOCK:
                        task.message = status
                with cls._LOCK:
                    task.progress = 100.0
                    task.status = "completed"
            except Exception as e:
                with cls._LOCK:
                    task.status = "failed"
                    task.message = str(e)
            finally:
                with cls._LOCK:
                    cls._CURRENT = None

    @classmethod
    def request_download(cls, model: str) -> None:
        with cls._LOCK:
            cls._QUEUE.append(PullTask(model=model))
        cls._ensure_worker()

    @classmethod
    def download_status(cls) -> Dict[str, Any]:
        with cls._LOCK:
            queue_list = [{"model": t.model, "status": t.status, "progress": round(t.progress, 2)} for t in cls._QUEUE]
            current = None
            if cls._CURRENT:
                current = {
                    "model": cls._CURRENT.model,
                    "status": cls._CURRENT.status,
                    "progress": round(cls._CURRENT.progress, 2),
                    "message": cls._CURRENT.message,
                }
            return {"current": current, "queue": queue_list}

    def list_models(self) -> List[str]:
        res = self.client.list()
        return [m["model"] for m in res.get("models", [])]

    def delete_model(self, model: str) -> None:
        self.client.delete(model=model)

    # ---- Inference ----
    def response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: MessageList = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")
        eff_top_k = self._effective_param("top_k", top_k)
        _ollama_guardrails(eff_top_k)

        trimmed = self.trim_conversation_history(conversation_history)
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        messages: List[Dict[str, str]] = []
        if eff_system:
            messages.append({"role": "system", "content": eff_system})
        if trimmed:
            for m in trimmed:
                role = "assistant" if m.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {}
        if eff_temp is not None:
            options["temperature"] = eff_temp
        if eff_top_p is not None:
            options["top_p"] = eff_top_p
        if eff_top_k is not None:
            options["top_k"] = eff_top_k
        if eff_pres is not None:
            options["presence_penalty"] = eff_pres
        if eff_freq is not None:
            options["frequency_penalty"] = eff_freq
        if eff_max is not None:
            options["num_predict"] = eff_max

        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            r = self.client.chat(**payload)
        txt = r.get("message", {}).get("content", "")

        # Full (clipped) prompts + output in trace
        full_system = self._clip_for_trace(eff_system)
        full_prompt = self._clip_for_trace(prompt)
        full_output = self._clip_for_trace(txt)

        self._trace(
            tracer,
            operation="response",
            model=self.model,
            params={
                k: v
                for k, v in {
                    "temperature": eff_temp,
                    "top_p": eff_top_p,
                    "top_k": eff_top_k,
                    "presence_penalty": eff_pres,
                    "frequency_penalty": eff_freq,
                    "max_output_tokens": eff_max,
                }.items()
                if v is not None
            },
            system_prompt=full_system,
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt=full_prompt,
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            output_text=full_output,
            output_length=len(txt or ""),
            latency_ms=t.ms,
            **(trace_extra or {}),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
        return txt

    def json_response(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        conversation_history: MessageList = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        eff_top_k = self._effective_param("top_k", top_k)
        _ollama_guardrails(eff_top_k)

        trimmed = self.trim_conversation_history(conversation_history)
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        messages: List[Dict[str, str]] = []
        if eff_system:
            messages.append({"role": "system", "content": eff_system})
        if trimmed:
            for m in trimmed:
                role = "assistant" if m.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {}
        if eff_temp is not None:
            options["temperature"] = eff_temp
        if eff_top_p is not None:
            options["top_p"] = eff_top_p
        if eff_top_k is not None:
            options["top_k"] = eff_top_k
        if eff_pres is not None:
            options["presence_penalty"] = eff_pres
        if eff_freq is not None:
            options["frequency_penalty"] = eff_freq
        if eff_max is not None:
            options["num_predict"] = eff_max

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
        }
        if options:
            payload["options"] = options

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            r = self.client.chat(**payload)

        txt = r.get("message", {}).get("content", "")

        parse_error: Optional[str] = None
        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed = self._parse_json_response(txt or "{}", schema=schema)
        except Exception as e:
            parse_error = str(e)

        input_tokens = r.get("prompt_eval_count")
        output_tokens = r.get("eval_count")
        total_tokens = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total_tokens = input_tokens + output_tokens

        full_system = self._clip_for_trace(eff_system)
        full_prompt = self._clip_for_trace(prompt)
        full_output = self._clip_for_trace(txt)

        self._trace(
            tracer,
            operation="json_response",
            model=self.model,
            params={
                k: v
                for k, v in {
                    "temperature": eff_temp,
                    "top_p": eff_top_p,
                    "top_k": eff_top_k,
                    "presence_penalty": eff_pres,
                    "frequency_penalty": eff_freq,
                    "max_output_tokens": eff_max,
                }.items()
                if v is not None
            },
            system_prompt=full_system,
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt=full_prompt,
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_text=full_output,
            output_length=len(txt or ""),
            output_parsed=(self._clip_json_for_trace(parsed) if parsed is not None else None),
            parse_error=parse_error,
        )

        if parse_error is not None:
            # Re-raise a ValidationError with the original message preserved
            # (pydantic v2 requires structured data; keep it simple but informative).
            raise ValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("response",),
                        "msg": f"json_response parse failed: {parse_error}",
                        "input": (self._clip_for_trace(txt) or ""),
                    }
                ],
                schema,
            )
        return parsed or {}

    def response_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: MessageList = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")
        eff_top_k = self._effective_param("top_k", top_k)
        _ollama_guardrails(eff_top_k)

        trimmed = self.trim_conversation_history(conversation_history)
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        messages: List[Dict[str, str]] = []
        if eff_system:
            messages.append({"role": "system", "content": eff_system})
        if trimmed:
            for m in trimmed:
                role = "assistant" if m.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {}
        if eff_temp is not None:
            options["temperature"] = eff_temp
        if eff_top_p is not None:
            options["top_p"] = eff_top_p
        if eff_top_k is not None:
            options["top_k"] = eff_top_k
        if eff_pres is not None:
            options["presence_penalty"] = eff_pres
        if eff_freq is not None:
            options["frequency_penalty"] = eff_freq
        if eff_max is not None:
            options["num_predict"] = eff_max

        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            # Collect for trace (bounded)
            out_parts: List[str] = []
            out_chars = 0
            for chunk in self.client.chat(**payload):
                delta = chunk.get("message", {}).get("content", "") or ""
                if delta:
                    if self.trace_max_chars() != 0 and out_chars < max(0, self.trace_max_chars()):
                        remain = max(0, self.trace_max_chars()) - out_chars
                        if remain > 0:
                            out_parts.append(delta[:remain])
                            out_chars += min(len(delta), remain)
                    yield delta

        full_system = self._clip_for_trace(eff_system)
        full_prompt = self._clip_for_trace(prompt)
        streamed_out = "".join(out_parts) if out_parts else ""

        self._trace(
            tracer,
            operation="response_stream",
            model=self.model,
            params={
                k: v
                for k, v in {
                    "temperature": eff_temp,
                    "top_p": eff_top_p,
                    "top_k": eff_top_k,
                    "presence_penalty": eff_pres,
                    "frequency_penalty": eff_freq,
                    "max_output_tokens": eff_max,
                }.items()
                if v is not None
            },
            system_prompt=full_system,
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt=full_prompt,
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
            output_text=(streamed_out if streamed_out else None),
            output_length=(out_chars if streamed_out else 0),
        )
