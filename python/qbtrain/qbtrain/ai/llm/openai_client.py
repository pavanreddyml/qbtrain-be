# qbtrain/ai/llm/openai_client.py
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Type, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel

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
        top_k = kwargs.get("top_k")
        if top_k not in (None, 1):
            raise ValueError("OpenAI Responses API does not support top_k != 1.")
        presence_penalty = kwargs.get("presence_penalty")
        if presence_penalty not in (None, 0.0):
            raise ValueError("OpenAI Responses API does not support presence_penalty.")
        frequency_penalty = kwargs.get("frequency_penalty")
        if frequency_penalty not in (None, 0.0):
            raise ValueError("OpenAI Responses API does not support frequency_penalty.")

        return fn(self, *args, **kwargs)

    return wrapper


class OpenAIClient(LLMClient):
    client_id = "openai"
    display_name = "OpenAI"
    available_models = ["gpt-4o", "gpt-4o-mini"]
    param_display_names = {"api_key": "API Key", "model": "Model (e.g., gpt-4o)"}

    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__(model=model)
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _build_input(prompt: str, conversation_history: MessageList) -> List[Message]:
        conversation_history = LLMClient.trim_conversation_history(conversation_history)

        items: List[Message] = []
        if conversation_history:
            items.extend(conversation_history)
        items.append({"role": "user", "content": prompt})
        return items

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
        *args: Any,
        **kwargs: Any,
    ) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": cast(str, self.model),
            "input": self._build_input(prompt, conversation_history),
            "instructions": system_prompt or None,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens

        rsp = self.client.responses.create(**request_kwargs, **kwargs)
        return rsp.output_text

    @_enforce_openai_guardrails
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
        request_kwargs: Dict[str, Any] = {
            "model": cast(str, self.model),
            "input": self._build_input(prompt, conversation_history),
            "instructions": system_prompt or None,
            "text_format": schema,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens

        rsp = self.client.responses.parse(**request_kwargs, **kwargs)
        parsed = rsp.output_parsed
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

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
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        request_kwargs: Dict[str, Any] = {
            "model": cast(str, self.model),
            "input": self._build_input(prompt, conversation_history),
            "instructions": system_prompt or None,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens

        with self.client.responses.stream(**request_kwargs, **kwargs) as stream:
            for event in stream:
                et = getattr(event, "type", None)
                if et in ("response.output_text.delta", "response.refusal.delta"):
                    yield getattr(event, "delta", "")
                elif et in ("response.error", "error"):
                    err = getattr(event, "error", None)
                    raise RuntimeError(str(err) if err is not None else "OpenAI streaming error")
