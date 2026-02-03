# qbtrain/ai/llm/openai_client.py
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Type, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel
from qbtrain.tracers import Tracer

from .base_llm_client import LLMClient, Message

R = TypeVar("R")
MessageList = Optional[List[Message]]


def _enforce_openai_guardrails(fn: Callable[..., R]) -> Callable[..., R]:
    """
    Validate:
      - model is present on client instance
      - model is available (if gated)
      - unsupported params for Responses API (top_k, presence/frequency penalties)
    """

    @wraps(fn)
    def wrapper(self: "OpenAIClient", *args: Any, **kwargs: Any) -> R:
        model = getattr(self, "model", None) or ""
        if not model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        if self.available_models and model not in self.available_models:
            raise ValueError(f"Model {model} is not supported by OpenAIClient.")
        eff_top_k = kwargs.get("top_k", None)
        if eff_top_k is None:
            eff_top_k = self._defaults.get("top_k")
        if eff_top_k not in (None, 1):
            raise ValueError("OpenAI Responses API does not support top_k != 1.")
        eff_presence = kwargs.get("presence_penalty", None)
        if eff_presence is None:
            eff_presence = self._defaults.get("presence_penalty")
        if eff_presence not in (None, 0.0):
            raise ValueError("OpenAI Responses API does not support presence_penalty.")
        eff_frequency = kwargs.get("frequency_penalty", None)
        if eff_frequency is None:
            eff_frequency = self._defaults.get("frequency_penalty")
        if eff_frequency not in (None, 0.0):
            raise ValueError("OpenAI Responses API does not support frequency_penalty.")

        return fn(self, *args, **kwargs)

    return wrapper


class OpenAIClient(LLMClient):
    client_id = "openai"
    display_name = "OpenAI"
    available_models = ["gpt-4o", "gpt-4o-mini"]
    param_display_names = {"api_key": "API Key", "model": "Model (e.g., gpt-4o)"}

    def __init__(
        self,
        api_key: str,
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
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _build_input(prompt: str, conversation_history: MessageList) -> List[Message]:
        conversation_history = LLMClient.trim_conversation_history(conversation_history)

        items: List[Message] = []
        if conversation_history:
            items.extend(conversation_history)
        items.append({"role": "user", "content": prompt})
        return items

    def _usage_from_response(self, rsp: Any) -> Dict[str, Optional[int]]:
        u = getattr(rsp, "usage", None) or {}
        return {
            "input_tokens": getattr(u, "input_tokens", None) or u.get("input_tokens"),
            "output_tokens": getattr(u, "output_tokens", None) or u.get("output_tokens"),
            "total_tokens": getattr(u, "total_tokens", None) or u.get("total_tokens"),
        }

    @_enforce_openai_guardrails
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
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        request_kwargs: Dict[str, Any] = {
            "model": cast(str, self.model),
            "input": self._build_input(prompt, conversation_history),
            "instructions": eff_system or None,
        }
        if eff_temp is not None:
            request_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            request_kwargs["top_p"] = eff_top_p
        if eff_max is not None:
            request_kwargs["max_output_tokens"] = eff_max

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            rsp = self.client.responses.create(**request_kwargs, **kwargs)
        out = rsp.output_text

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        usage = self._usage_from_response(rsp)
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
        return out

    @_enforce_openai_guardrails
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
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        trace_extra = kwargs.pop("_trace", None)

        if schema is not None:
            request_kwargs: Dict[str, Any] = {
                "model": cast(str, self.model),
                "input": self._build_input(prompt, conversation_history),
                "instructions": eff_system or None,
                "text_format": schema,
            }
            if eff_temp is not None:
                request_kwargs["temperature"] = eff_temp
            if eff_top_p is not None:
                request_kwargs["top_p"] = eff_top_p
            if eff_max is not None:
                request_kwargs["max_output_tokens"] = eff_max

            with self._timed() as t:
                rsp = self.client.responses.parse(**request_kwargs, **kwargs)
            parsed = rsp.output_parsed
            out_obj = (
                parsed.model_dump()
                if isinstance(parsed, BaseModel)
                else (parsed if isinstance(parsed, dict) else {"value": parsed})
            )

            trimmed = LLMClient.trim_conversation_history(conversation_history)
            usage = self._usage_from_response(rsp)
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
            return out_obj

        request_kwargs: Dict[str, Any] = {
            "model": cast(str, self.model),
            "input": self._build_input(prompt, conversation_history),
            "instructions": eff_system or None,
            "response_format": {"type": "json_object"},
        }
        if eff_temp is not None:
            request_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            request_kwargs["top_p"] = eff_top_p
        if eff_max is not None:
            request_kwargs["max_output_tokens"] = eff_max

        with self._timed() as t:
            rsp = self.client.responses.create(**request_kwargs, **kwargs)
        txt = rsp.output_text or "{}"

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        usage = self._usage_from_response(rsp)
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
        return json.loads(txt or "{}")

    @_enforce_openai_guardrails
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
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        request_kwargs: Dict[str, Any] = {
            "model": cast(str, self.model),
            "input": self._build_input(prompt, conversation_history),
            "instructions": eff_system or None,
        }
        if eff_temp is not None:
            request_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            request_kwargs["top_p"] = eff_top_p
        if eff_max is not None:
            request_kwargs["max_output_tokens"] = eff_max

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            with self.client.responses.stream(**request_kwargs, **kwargs) as stream:
                for event in stream:
                    et = getattr(event, "type", None)
                    if et in ("response.output_text.delta", "response.refusal.delta"):
                        yield getattr(event, "delta", "")
                    elif et in ("response.error", "error"):
                        err = getattr(event, "error", None)
                        raise RuntimeError(str(err) if err is not None else "OpenAI streaming error")

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
