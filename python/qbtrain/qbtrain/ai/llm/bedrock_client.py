# qbtrain/ai/llm/bedrock_client.py
from __future__ import annotations

from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Type

import boto3
from pydantic import BaseModel

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _bedrock_guardrails(fn):
    @wraps(fn)
    def wrapper(self: "BedrockClient", *args, **kwargs):
        top_k = kwargs.get("top_k")
        if top_k not in (None, 1):
            raise ValueError("Bedrock Converse does not support top_k != 1.")
        presence_penalty = kwargs.get("presence_penalty")
        if presence_penalty not in (None, 0.0):
            raise ValueError("Bedrock does not support presence_penalty.")
        frequency_penalty = kwargs.get("frequency_penalty")
        if frequency_penalty not in (None, 0.0):
            raise ValueError("Bedrock does not support frequency_penalty.")
        return fn(self, *args, **kwargs)

    return wrapper


class BedrockClient(LLMClient):
    client_id = "aws_bedrock"
    display_name = "AWS Bedrock"
    param_display_names = {
        "region_name": "AWS Region (e.g., us-east-1)",
        "model": "Model ID (e.g., anthropic.claude-3-5-sonnet-20240620-v1:0)",
    }

    def __init__(self, region_name: str, model: Optional[str] = None, **session_kwargs: Any):
        super().__init__(model=model)
        self.client = boto3.client("bedrock-runtime", region_name=region_name, **session_kwargs)

    @staticmethod
    def _messages(prompt: str, system_prompt: Optional[str], conversation_history: MessageList) -> Dict[str, Any]:
        conversation_history = LLMClient.trim_conversation_history(conversation_history)

        msgs: List[Dict[str, Any]] = []
        if conversation_history:
            for m in conversation_history:
                role = "assistant" if m.get("role") == "assistant" else "user"
                msgs.append({"role": role, "content": [{"text": m.get("content", "")}]})
        msgs.append({"role": "user", "content": [{"text": prompt}]})
        sys = [{"text": system_prompt}] if system_prompt else None
        return {"messages": msgs, "system": sys}

    @_bedrock_guardrails
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

        payload = self._messages(prompt, system_prompt, conversation_history)
        inference: Dict[str, Any] = {}
        if temperature is not None:
            inference["temperature"] = temperature
        if top_p is not None:
            inference["topP"] = top_p
        if max_output_tokens is not None:
            inference["maxTokens"] = max_output_tokens

        r = self.client.converse(
            modelId=self.model,
            messages=payload["messages"],
            system=payload["system"],
            **({"inferenceConfig": inference} if inference else {}),
            **kwargs,
        )
        parts = r.get("output", {}).get("message", {}).get("content", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]
        return ""

    @_bedrock_guardrails
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

        hint = "Return only a strict JSON object."
        payload = self._messages(f"{prompt}\n\n{hint}", system_prompt, conversation_history)
        inference: Dict[str, Any] = {}
        if temperature is not None:
            inference["temperature"] = temperature
        if top_p is not None:
            inference["topP"] = top_p
        if max_output_tokens is not None:
            inference["maxTokens"] = max_output_tokens

        r = self.client.converse(
            modelId=self.model,
            messages=payload["messages"],
            system=payload["system"],
            **({"inferenceConfig": inference} if inference else {}),
            **kwargs,
        )
        parts = r.get("output", {}).get("message", {}).get("content", [])
        txt = parts[0]["text"] if parts and "text" in parts[0] else "{}"
        obj = schema.model_validate_json(txt)
        return obj.model_dump()

    @_bedrock_guardrails
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

        payload = self._messages(prompt, system_prompt, conversation_history)
        inference: Dict[str, Any] = {}
        if temperature is not None:
            inference["temperature"] = temperature
        if top_p is not None:
            inference["topP"] = top_p
        if max_output_tokens is not None:
            inference["maxTokens"] = max_output_tokens

        stream = self.client.converse_stream(
            modelId=self.model,
            messages=payload["messages"],
            system=payload["system"],
            **({"inferenceConfig": inference} if inference else {}),
            **kwargs,
        )
        for event in stream.get("stream", []):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"].get("text", "")
                if delta:
                    yield delta
            elif "messageStop" in event:
                break
