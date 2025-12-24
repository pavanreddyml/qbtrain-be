# qbtrain/ai/llm/ollama_client.py
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Generator, List, Optional, Type

import ollama
from pydantic import BaseModel

from .base_llm_client import LLMClient, Message

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

    def __init__(self, host: str = "http://127.0.0.1:11434", model: Optional[str] = None):
        super().__init__(model=model)
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
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")
        _ollama_guardrails(top_k)

        conversation_history = self.trim_conversation_history(conversation_history)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            for m in conversation_history:
                role = "assistant" if m.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if presence_penalty is not None:
            options["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            options["frequency_penalty"] = frequency_penalty
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens

        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options

        r = self.client.chat(**payload)
        return r.get("message", {}).get("content", "")

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
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        txt = self.response(
            prompt=f"{prompt}\n\nReturn only a strict JSON object.",
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_output_tokens=max_output_tokens,
        )
        obj = schema.model_validate_json(txt)
        return obj.model_dump()

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
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")
        _ollama_guardrails(top_k)

        conversation_history = self.trim_conversation_history(conversation_history)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            for m in conversation_history:
                role = "assistant" if m.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if presence_penalty is not None:
            options["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            options["frequency_penalty"] = frequency_penalty
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens

        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options

        for chunk in self.client.chat(**payload):
            delta = chunk.get("message", {}).get("content", "") or ""
            if delta:
                yield delta
