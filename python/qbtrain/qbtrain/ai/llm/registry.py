# qbtrain/ai/llm/registry.py
from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base_llm_client import LLMClient
from .openai_client import OpenAIClient
from .azure_foundry_client import AzureFoundryClient
from .gcp_model_garden_client import GCPModelGardenClient
from .bedrock_client import BedrockClient
from .huggingface_client import HuggingFaceClient
from .ollama_client import OllamaClient


class LLMClientRegistry:
    """
    Simple registry to resolve an LLM client class by its `client_id`.
    """

    _registry: Dict[str, Type[LLMClient]] = {
        OpenAIClient.client_id: OpenAIClient,
        AzureFoundryClient.client_id: AzureFoundryClient,
        GCPModelGardenClient.client_id: GCPModelGardenClient,
        BedrockClient.client_id: BedrockClient,
        HuggingFaceClient.client_id: HuggingFaceClient,
        OllamaClient.client_id: OllamaClient,
    }

    @classmethod
    def get(cls, client_id: str) -> Type[LLMClient]:
        try:
            return cls._registry[client_id]
        except KeyError:
            raise KeyError(f"Unknown LLM client id: {client_id!r}")

    @classmethod
    def list_ids(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_classes(cls) -> List[Type[LLMClient]]:
        return list(cls._registry.values())

    @classmethod
    def add(cls, klass: Type[LLMClient]) -> None:
        if not issubclass(klass, LLMClient):
            raise TypeError("klass must be a subclass of LLMClient")
        cls._registry[klass.client_id] = klass
