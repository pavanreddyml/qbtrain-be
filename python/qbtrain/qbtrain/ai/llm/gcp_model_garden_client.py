# qbtrain/ai/llm/gcp_model_garden_client.py
from __future__ import annotations

from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

from pydantic import BaseModel
from vertexai import init as vertex_init
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _vertex_guardrails(fn):
    @wraps(fn)
    def wrapper(self: "GCPModelGardenClient", *args, **kwargs):
        if kwargs.get("top_k") not in (None, 1):
            raise ValueError("Vertex AI does not support top_k != 1.")
        if kwargs.get("presence_penalty") not in (None, 0.0):
            raise ValueError("Vertex AI does not support presence_penalty.")
        if kwargs.get("frequency_penalty") not in (None, 0.0):
            raise ValueError("Vertex AI does not support frequency_penalty.")
        return fn(self, *args, **kwargs)

    return wrapper


class GCPModelGardenClient(LLMClient):
    client_id = "gcp_model_garden"
    display_name = "Google Vertex AI (Gemini)"
    param_display_names = {
        "project": "GCP Project ID",
        "location": "Region (e.g., us-central1)",
        "model": "Model name (e.g., gemini-1.5-pro)",
    }

    def __init__(self, project: str, location: str, model: Optional[str] = None):
        super().__init__(model=model)
        vertex_init(project=project, location=location)

    @staticmethod
    def _contents(
        prompt: str,
        system_prompt: Optional[str],
        conversation_history: MessageList,
    ) -> Tuple[List[Content], Optional[str]]:
        conversation_history = LLMClient.trim_conversation_history(conversation_history)

        contents: List[Content] = []
        if conversation_history:
            for m in conversation_history:
                role = "model" if m.get("role") == "assistant" else "user"
                contents.append(Content(role=role, parts=[Part.from_text(m.get("content", ""))]))
        contents.append(Content(role="user", parts=[Part.from_text(prompt)]))
        return contents, (system_prompt or None)

    @_vertex_guardrails
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

        contents, sysinst = self._contents(prompt, system_prompt, conversation_history)
        gen = GenerativeModel(model_name=self.model, system_instruction=sysinst)
        cfg_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if top_p is not None:
            cfg_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_output_tokens

        cfg = GenerationConfig(**cfg_kwargs) if cfg_kwargs else None
        resp = gen.generate_content(contents=contents, generation_config=cfg, **kwargs)
        return (resp.text or "").strip()

    @_vertex_guardrails
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
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        contents, sysinst = self._contents(prompt, system_prompt, conversation_history)
        gen = GenerativeModel(model_name=self.model, system_instruction=sysinst)
        cfg_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if top_p is not None:
            cfg_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_output_tokens

        cfg = GenerationConfig(**cfg_kwargs) if cfg_kwargs else None
        contents = contents + [Content(role="user", parts=[Part.from_text("Return a strict JSON object only.")])]
        resp = gen.generate_content(contents=contents, generation_config=cfg, **kwargs)
        obj = schema.model_validate_json(resp.text or "{}")
        return obj.model_dump()

    @_vertex_guardrails
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

        contents, sysinst = self._contents(prompt, system_prompt, conversation_history)
        gen = GenerativeModel(model_name=self.model, system_instruction=sysinst)
        cfg_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if top_p is not None:
            cfg_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_output_tokens

        cfg = GenerationConfig(**cfg_kwargs) if cfg_kwargs else None
        for chunk in gen.generate_content(contents=contents, generation_config=cfg, stream=True, **kwargs):
            txt = getattr(chunk, "text", "") or ""
            if txt:
                yield txt
