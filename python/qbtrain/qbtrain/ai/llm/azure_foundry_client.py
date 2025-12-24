# qbtrain/ai/llm/azure_foundry_client.py
from __future__ import annotations

from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Type

from openai import AzureOpenAI
from pydantic import BaseModel

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _azure_guardrails(fn):
    @wraps(fn)
    def wrapper(self: "AzureFoundryClient", *args, **kwargs):
        if kwargs.get("top_k") not in (None, 1):
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
    ):
        super().__init__(model=model)
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
        *args: Any,
        **kwargs: Any,
    ) -> str:
        deployment = self._deployment_name()
        request_kwargs: Dict[str, Any] = {
            "model": deployment,
            "messages": self._build_messages(prompt, system_prompt, conversation_history),
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max_output_tokens

        r = self.client.chat.completions.create(**request_kwargs, **kwargs)
        return (r.choices[0].message.content or "").strip()

    @_azure_guardrails
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
        deployment = self._deployment_name()
        request_kwargs: Dict[str, Any] = {
            "model": deployment,
            "messages": self._build_messages(prompt, system_prompt, conversation_history),
            "response_format": {"type": "json_object"},
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max_output_tokens

        r = self.client.chat.completions.create(**request_kwargs, **kwargs)
        txt = (r.choices[0].message.content or "").strip()
        obj = schema.model_validate_json(txt)
        return obj.model_dump()

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
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        deployment = self._deployment_name()
        request_kwargs: Dict[str, Any] = {
            "model": deployment,
            "messages": self._build_messages(prompt, system_prompt, conversation_history),
            "stream": True,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = frequency_penalty
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max_output_tokens

        stream = self.client.chat.completions.create(**request_kwargs, **kwargs)
        for chunk in stream:
            delta = ""
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
