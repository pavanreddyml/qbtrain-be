# qbtrain/ai/llm/gcp_model_garden_client.py
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

from pydantic import BaseModel
from qbtrain.tracers import Tracer
from vertexai import init as vertex_init
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _vertex_guardrails(fn):
    @wraps(fn)
    def wrapper(self: "GCPModelGardenClient", *args, **kwargs):
        eff_top_k = kwargs.get("top_k", None)
        if eff_top_k is None:
            eff_top_k = self._defaults.get("top_k")
        if eff_top_k not in (None, 1):
            raise ValueError("Vertex AI does not support top_k != 1.")
        eff_presence = kwargs.get("presence_penalty", None)
        if eff_presence is None:
            eff_presence = self._defaults.get("presence_penalty")
        if eff_presence not in (None, 0.0):
            raise ValueError("Vertex AI does not support presence_penalty.")
        eff_frequency = kwargs.get("frequency_penalty", None)
        if eff_frequency is None:
            eff_frequency = self._defaults.get("frequency_penalty")
        if eff_frequency not in (None, 0.0):
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

    def __init__(self, project: str, location: str, model: Optional[str] = None, **kwargs: Any):
        super().__init__(
            model=model,
            **{
                k: kwargs.get(k)
                for k in (
                    "system_prompt",
                    "top_k",
                    "top_p",
                    "temperature",
                    "presence_penalty",
                    "frequency_penalty",
                    "max_output_tokens",
                )
                if k in kwargs
            },
        )
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

    def _usage(self, resp: Any) -> Dict[str, Optional[int]]:
        u = getattr(resp, "usage_metadata", None)
        if u is None:
            return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        return {
            "input_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
            "total_tokens": getattr(u, "total_token_count", None),
        }

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
        tracer: Optional[Tracer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        contents, sysinst = self._contents(prompt, eff_system, conversation_history)
        gen = GenerativeModel(model_name=self.model, system_instruction=sysinst)
        cfg_kwargs: Dict[str, Any] = {}
        if eff_temp is not None:
            cfg_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            cfg_kwargs["top_p"] = eff_top_p
        if eff_max is not None:
            cfg_kwargs["max_output_tokens"] = eff_max

        cfg = GenerationConfig(**cfg_kwargs) if cfg_kwargs else None
        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            resp = gen.generate_content(contents=contents, generation_config=cfg, **kwargs)
        txt = (resp.text or "").strip()

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        usage = self._usage(resp)
        self._trace(
            tracer,
            operation="response",
            model=self.model,
            params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "max_output_tokens": eff_max}.items() if v is not None},
            system_prompt_preview=(eff_system[:200] if eff_system else None),
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt_preview=prompt[:200],
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
            **usage,
        )
        return txt

    @_vertex_guardrails
    def json_response(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
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

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        contents, sysinst = self._contents(prompt, eff_system, conversation_history)
        gen = GenerativeModel(model_name=self.model, system_instruction=sysinst)
        cfg_kwargs: Dict[str, Any] = {}
        if eff_temp is not None:
            cfg_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            cfg_kwargs["top_p"] = eff_top_p
        if eff_max is not None:
            cfg_kwargs["max_output_tokens"] = eff_max

        cfg = GenerationConfig(**cfg_kwargs) if cfg_kwargs else None
        contents = contents + [Content(role="user", parts=[Part.from_text("Return a strict JSON object only.")])]
        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            resp = gen.generate_content(contents=contents, generation_config=cfg, **kwargs)
        txt = resp.text or "{}"

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        usage = self._usage(resp)
        self._trace(
            tracer,
            operation="json_response",
            model=self.model,
            params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "max_output_tokens": eff_max}.items() if v is not None},
            system_prompt_preview=(eff_system[:200] if eff_system else None),
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt_preview=prompt[:200],
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
            **usage,
        )

        if schema is not None:
            obj = schema.model_validate_json(txt)
            return obj.model_dump()
        return json.loads(txt or "{}")

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
        tracer: Optional[Tracer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        contents, sysinst = self._contents(prompt, eff_system, conversation_history)
        gen = GenerativeModel(model_name=self.model, system_instruction=sysinst)
        cfg_kwargs: Dict[str, Any] = {}
        if eff_temp is not None:
            cfg_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            cfg_kwargs["top_p"] = eff_top_p
        if eff_max is not None:
            cfg_kwargs["max_output_tokens"] = eff_max

        cfg = GenerationConfig(**cfg_kwargs) if cfg_kwargs else None
        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            for chunk in gen.generate_content(contents=contents, generation_config=cfg, stream=True, **kwargs):
                txt = getattr(chunk, "text", "") or ""
                if txt:
                    yield txt

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        self._trace(
            tracer,
            operation="response_stream",
            model=self.model,
            params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "max_output_tokens": eff_max}.items() if v is not None},
            system_prompt_preview=(eff_system[:200] if eff_system else None),
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt_preview=prompt[:200],
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
        )
