# qbtrain/ai/llm/bedrock_client.py
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Type

import boto3
from pydantic import BaseModel
from qbtrain.tracers import Tracer

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _bedrock_guardrails(fn):
    @wraps(fn)
    def wrapper(self: "BedrockClient", *args, **kwargs):
        eff_top_k = kwargs.get("top_k", None)
        if eff_top_k is None:
            eff_top_k = self._defaults.get("top_k")
        if eff_top_k not in (None, 1):
            raise ValueError("Bedrock Converse does not support top_k != 1.")
        eff_presence = kwargs.get("presence_penalty", None)
        if eff_presence is None:
            eff_presence = self._defaults.get("presence_penalty")
        if eff_presence not in (None, 0.0):
            raise ValueError("Bedrock does not support presence_penalty.")
        eff_frequency = kwargs.get("frequency_penalty", None)
        if eff_frequency is None:
            eff_frequency = self._defaults.get("frequency_penalty")
        if eff_frequency not in (None, 0.0):
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

    def __init__(
        self,
        region_name: str,
        model: Optional[str] = None,
        **session_kwargs: Any,
    ):
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

    def _usage(self, r: Dict[str, Any]) -> Dict[str, Optional[int]]:
        u = r.get("usage", {}) or {}
        return {
            "input_tokens": u.get("inputTokens"),
            "output_tokens": u.get("outputTokens"),
            "total_tokens": u.get("totalTokens"),
        }

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

        payload = self._messages(prompt, eff_system, conversation_history)
        inference: Dict[str, Any] = {}
        if eff_temp is not None:
            inference["temperature"] = eff_temp
        if eff_top_p is not None:
            inference["topP"] = eff_top_p
        if eff_max is not None:
            inference["maxTokens"] = eff_max

        kwargs_payload = {"inferenceConfig": inference} if inference else {}
        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            r = self.client.converse(
                modelId=self.model,
                messages=payload["messages"],
                system=payload["system"],
                **kwargs_payload,
                **kwargs,
            )
        parts = r.get("output", {}).get("message", {}).get("content", [])
        out = parts[0]["text"] if parts and "text" in parts[0] else ""

        usage = self._usage(r)
        trimmed = LLMClient.trim_conversation_history(conversation_history)
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

    @_bedrock_guardrails
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

        hint = "Return only a strict JSON object."
        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_temp = self._effective_param("temperature", temperature)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        payload = self._messages(f"{prompt}\n\n{hint}", eff_system, conversation_history)
        inference: Dict[str, Any] = {}
        if eff_temp is not None:
            inference["temperature"] = eff_temp
        if eff_top_p is not None:
            inference["topP"] = eff_top_p
        if eff_max is not None:
            inference["maxTokens"] = eff_max

        kwargs_payload = {"inferenceConfig": inference} if inference else {}
        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            r = self.client.converse(
                modelId=self.model,
                messages=payload["messages"],
                system=payload["system"],
                **kwargs_payload,
                **kwargs,
            )
        parts = r.get("output", {}).get("message", {}).get("content", [])
        txt = parts[0]["text"] if parts and "text" in parts[0] else "{}"

        usage = self._usage(r)
        trimmed = LLMClient.trim_conversation_history(conversation_history)
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

        return self._parse_json_response(txt or "{}", schema=schema)

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

        payload = self._messages(prompt, eff_system, conversation_history)
        inference: Dict[str, Any] = {}
        if eff_temp is not None:
            inference["temperature"] = eff_temp
        if eff_top_p is not None:
            inference["topP"] = eff_top_p
        if eff_max is not None:
            inference["maxTokens"] = eff_max

        kwargs_payload = {"inferenceConfig": inference} if inference else {}
        trace_extra = kwargs.pop("_trace", None)
        with self._timed() as t:
            stream = self.client.converse_stream(
                modelId=self.model,
                messages=payload["messages"],
                system=payload["system"],
                **kwargs_payload,
                **kwargs,
            )

            for event in stream.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"].get("text", "")
                    if delta:
                        yield delta
                elif "messageStop" in event:
                    break

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
