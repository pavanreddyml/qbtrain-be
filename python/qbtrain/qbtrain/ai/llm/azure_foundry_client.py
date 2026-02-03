# qbtrain/ai/llm/azure_foundry_client.py
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Type

from openai import AzureOpenAI
from pydantic import BaseModel
from qbtrain.tracers import Tracer

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _azure_guardrails(fn):
    @wraps(fn)
    def wrapper(self: "AzureFoundryClient", *args, **kwargs):
        eff_top_k = kwargs.get("top_k", None)
        if eff_top_k is None:
            eff_top_k = self._defaults.get("top_k")
        if eff_top_k not in (None, 1):
            raise ValueError("Azure chat completions do not support top_k != 1.")
        return fn(self, *args, **kwargs)

    return wrapper


class AzureFoundryClient(LLMClient):
    client_id = "azure_foundry"
    display_name = "Azure OpenAI (Foundry)"
    requires_model_list = True
    param_display_names = {
        "api_key": "Secret (API Key)",
        "endpoint": "Endpoint URL",
        "api_version": "API Version",
        "default_deployment": "Default Deployment (optional)",
        "available_models": "Available Deployments (comma-separated)",
        "model": "Deployment name (optional; overrides default)",
    }

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        model: Optional[str] = None,
        default_deployment: Optional[str] = None,
        available_models: Optional[List[str]] = None,
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
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        self.default_deployment = default_deployment
        self.available_models = available_models or []

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
        conversation_history: MessageList,
    ) -> List[Dict[str, str]]:
        conversation_history = LLMClient.trim_conversation_history(conversation_history)

        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if conversation_history:
            for m in conversation_history:
                role = m.get("role", "user")
                content = m.get("content", "")
                role = "assistant" if role == "assistant" else "user"
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _deployment_name(self) -> str:
        deployment = (self.model or self.default_deployment or "").strip()
        if not deployment:
            raise ValueError(
                "AzureFoundryClient requires a deployment name (pass in clientDetails.params.model or default_deployment)."
            )
        if self.available_models and deployment not in self.available_models:
            raise ValueError("Deployment not in provided Azure available_models list.")
        return deployment

    def _usage(self, r: Any) -> Dict[str, Optional[int]]:
        u = getattr(r, "usage", None) or {}
        return {
            "input_tokens": getattr(u, "prompt_tokens", None) or u.get("prompt_tokens"),
            "output_tokens": getattr(u, "completion_tokens", None) or u.get("completion_tokens"),
            "total_tokens": getattr(u, "total_tokens", None) or u.get("total_tokens"),
        }

    @_azure_guardrails
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
        deployment = self._deployment_name()

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        msgs = self._build_messages(prompt, eff_system, trimmed)

        request_kwargs: Dict[str, Any] = {"model": deployment, "messages": msgs}
        if eff_temp is not None:
            request_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            request_kwargs["top_p"] = eff_top_p
        if eff_pres is not None:
            request_kwargs["presence_penalty"] = eff_pres
        if eff_freq is not None:
            request_kwargs["frequency_penalty"] = eff_freq
        if eff_max is not None:
            request_kwargs["max_tokens"] = eff_max

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            r = self.client.chat.completions.create(**request_kwargs, **kwargs)
        txt = (r.choices[0].message.content or "").strip()

        usage = self._usage(r)
        self._trace(
            tracer,
            operation="response",
            model=deployment,
            params={
                k: v
                for k, v in {
                    "temperature": eff_temp,
                    "top_p": eff_top_p,
                    "presence_penalty": eff_pres,
                    "frequency_penalty": eff_freq,
                    "max_output_tokens": eff_max,
                }.items()
                if v is not None
            },
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

    @_azure_guardrails
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
        deployment = self._deployment_name()

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        msgs = self._build_messages(prompt, eff_system, trimmed)

        request_kwargs: Dict[str, Any] = {
            "model": deployment,
            "messages": msgs,
            "response_format": {"type": "json_object"},
        }
        if eff_temp is not None:
            request_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            request_kwargs["top_p"] = eff_top_p
        if eff_pres is not None:
            request_kwargs["presence_penalty"] = eff_pres
        if eff_freq is not None:
            request_kwargs["frequency_penalty"] = eff_freq
        if eff_max is not None:
            request_kwargs["max_tokens"] = eff_max

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            r = self.client.chat.completions.create(**request_kwargs, **kwargs)
        txt = (r.choices[0].message.content or "").strip()

        usage = self._usage(r)
        self._trace(
            tracer,
            operation="json_response",
            model=deployment,
            params={
                k: v
                for k, v in {
                    "temperature": eff_temp,
                    "top_p": eff_top_p,
                    "presence_penalty": eff_pres,
                    "frequency_penalty": eff_freq,
                    "max_output_tokens": eff_max,
                }.items()
                if v is not None
            },
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

    @_azure_guardrails
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
        deployment = self._deployment_name()

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        trimmed = LLMClient.trim_conversation_history(conversation_history)
        msgs = self._build_messages(prompt, eff_system, trimmed)

        request_kwargs: Dict[str, Any] = {
            "model": deployment,
            "messages": msgs,
            "stream": True,
        }
        if eff_temp is not None:
            request_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            request_kwargs["top_p"] = eff_top_p
        if eff_pres is not None:
            request_kwargs["presence_penalty"] = eff_pres
        if eff_freq is not None:
            request_kwargs["frequency_penalty"] = eff_freq
        if eff_max is not None:
            request_kwargs["max_tokens"] = eff_max

        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            stream = self.client.chat.completions.create(**request_kwargs, **kwargs)
            for chunk in stream:
                delta = ""
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta

        self._trace(
            tracer,
            operation="response_stream",
            model=deployment,
            params={
                k: v
                for k, v in {
                    "temperature": eff_temp,
                    "top_p": eff_top_p,
                    "presence_penalty": eff_pres,
                    "frequency_penalty": eff_freq,
                    "max_output_tokens": eff_max,
                }.items()
                if v is not None
            },
            system_prompt_preview=(eff_system[:200] if eff_system else None),
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt_preview=prompt[:200],
            prompt_length=len(prompt),
            conv_history_length=len(trimmed or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
        )
