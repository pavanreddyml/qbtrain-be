# qbtrain/ai/llm/base_llm_client.py
from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Type

from pydantic import BaseModel

Message = Dict[str, Any]


class LLMClient(ABC):
    """
    Minimal provider-agnostic interface.

    Conventions:
      - model is passed via __init__ and stored as self.model
      - response/json_response/response_stream do NOT accept `model`
      - conversation_history is trimmed to last N messages where:
          N = int($LLM_MAX_CONVERSATION_HISTORY) or 5
    """

    # ---- identity & metadata (override in subclasses) ----
    client_id: str = "base"
    display_name: str = "Base LLM"
    available_models: Optional[List[str]] = None
    requires_model_list: bool = False
    extra_init_params: List[Dict[str, Any]] = []
    param_display_names: Dict[str, str] = {}

    def __init__(self, model: Optional[str] = None):
        self.model: Optional[str] = (model or None)

    # ---- conversation history helpers ----
    @staticmethod
    def max_conversation_history() -> int:
        raw = (os.getenv("LLM_MAX_CONVERSATION_HISTORY") or "").strip()
        if not raw:
            return 5
        try:
            n = int(raw)
        except ValueError:
            return 5
        return max(0, n)

    @staticmethod
    def trim_conversation_history(conversation_history: Optional[List[Message]]) -> Optional[List[Message]]:
        if not conversation_history:
            return None
        n = LLMClient.max_conversation_history()
        if n <= 0:
            return None
        return conversation_history[-n:]

    # ---- properties with defaults/fallbacks ----
    @property
    def name(self) -> str:
        return getattr(self, "display_name", self.__class__.__name__)

    @property
    def id(self) -> str:
        cid = getattr(self, "client_id", None)
        return cid or f"{self.__class__.__module__}.{self.__class__.__name__}"

    @property
    def params_display(self) -> Dict[str, str]:
        labels = getattr(self, "param_display_names", None)
        if labels:
            return dict(labels)

        sig = inspect.signature(self.__init__)
        out: Dict[str, str] = {}
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            out[name] = name.replace("_", " ").title()
        return out

    # ---- metadata helpers ----
    @classmethod
    def init_parameters(cls) -> List[Dict[str, Any]]:
        def serialize_param(p: inspect.Parameter) -> Dict[str, Any]:
            required = p.default is inspect._empty
            default = None if required else p.default
            ann = None if p.annotation is inspect._empty else getattr(p.annotation, "__name__", str(p.annotation))
            return {
                "name": p.name,
                "kind": str(p.kind),
                "required": required,
                "default": default,
                "annotation": ann,
            }

        sig = inspect.signature(cls.__init__)
        params = [
            serialize_param(p)
            for name, p in sig.parameters.items()
            if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        existing = {p["name"] for p in params}
        for extra in getattr(cls, "extra_init_params", []) or []:
            if extra.get("name") in existing:
                continue
            params.append(extra)

        return params

    # ---- core interface ----
    @abstractmethod
    def response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def json_response(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def response_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        raise NotImplementedError
