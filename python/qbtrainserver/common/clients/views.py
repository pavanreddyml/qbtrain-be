# qbtrain/ai/llm/views.py
from __future__ import annotations

import inspect
from typing import Any, Dict, Generator, List, Optional

from django.http import StreamingHttpResponse
from pydantic import BaseModel, ConfigDict, create_model
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from qbtrain.ai.llm import (
    HuggingFaceClient,
    LLMClientRegistry,
    OllamaClient,
)

# --- Health ---
@api_view(["GET"])
def health(request):
    try:
        return Response({"message": "Service is healthy"}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================
# Hugging Face: download / list / delete / status
# ============================

@api_view(["POST"])
def hf_download(request):
    try:
        model_id = (request.data.get("model_id") or "").strip()
        if not model_id:
            return Response({"error": "model_id is required"}, status=status.HTTP_400_BAD_REQUEST)
        revision = (request.data.get("revision") or None) or None
        local_dir = (request.data.get("local_dir") or None) or None
        HuggingFaceClient.request_download(model_id=model_id, revision=revision, local_dir=local_dir)
        return Response({"message": "queued", "status": HuggingFaceClient.download_status()}, status=status.HTTP_202_ACCEPTED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# ============================
# LLM invocation helpers
# ============================

def _coerce_model_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("\n", ",").split(",") if p.strip()]
        return parts or None
    return [str(value)]


def _build_schema_model(payload: Any) -> type[BaseModel]:
    if not isinstance(payload, dict) or not payload:
        return create_model("AnyJSON", __config__=ConfigDict(extra="allow"))

    fields = {name: (Any, None) for name in payload.keys()}
    return create_model("DynamicJSON", __config__=ConfigDict(extra="allow"), **fields)


def _extract_sections(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Preferred request format:
      {
        "clientDetails": { "type": "<registry key>", "params": { ... init params ..., "model": "..." } },
        "chatDetails":   { "prompt": "...", "system_prompt": "...", "conversation_history": [...], ... }
      }

    Back-compat fallback:
      - if clientDetails missing, interpret as old flat payload.
      - also accept misspelling "clientDetials".
    """
    cd = data.get("clientDetails") or data.get("clientDetials")
    ch = data.get("chatDetails")

    if isinstance(cd, dict) and isinstance(ch, dict):
        return cd, ch

    # fallback (old payload)
    client_id = data.get("client_id")
    params = data.get("params") or {}
    if data.get("model"):
        params = {**params, "model": data.get("model")}
    cd2 = {"type": client_id, "params": params}
    ch2 = dict(data)
    return cd2, ch2


def _build_llm_client(client_details: Dict[str, Any]):
    client_type = client_details.get("type")
    if not client_type:
        raise ValueError("clientDetails.type is required")

    params = client_details.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("clientDetails.params must be an object")

    cls = LLMClientRegistry.get(client_type)
    init_params = cls.init_parameters()
    allowed = {p["name"]: p for p in init_params}

    init_kwargs: Dict[str, Any] = {}
    for key, meta in allowed.items():
        if key in params:
            val = params[key]
            if key == "available_models":
                val = _coerce_model_list(val)
            init_kwargs[key] = val

    missing = [k for k, meta in allowed.items() if meta.get("required") and k not in init_kwargs]
    if missing:
        raise ValueError(f"Missing required init params: {', '.join(missing)}")

    return cls(**init_kwargs)


def _request_kwargs(chat: Dict[str, Any]) -> Dict[str, Any]:
    if "prompt" not in chat or chat.get("prompt") in (None, ""):
        raise ValueError("chatDetails.prompt is required")

    kwargs: Dict[str, Any] = {"prompt": chat.get("prompt")}

    def add_if_present(key: str, caster, allow_empty: bool = False):
        if key not in chat:
            return
        value = chat.get(key)
        if value is None:
            return
        if value == "" and not allow_empty:
            return
        kwargs[key] = caster(value)

    add_if_present("system_prompt", str)
    add_if_present("conversation_history", lambda v: v, allow_empty=True)
    add_if_present("top_k", int)
    add_if_present("top_p", float)
    add_if_present("temperature", float)
    add_if_present("presence_penalty", float)
    add_if_present("frequency_penalty", float)
    add_if_present("max_output_tokens", int)

    return kwargs


# ============================
# LLM: response / json / stream
# ============================

@api_view(["POST"])
def llm_response(request):
    try:
        data = request.data or {}
        client_details, chat_details = _extract_sections(data)

        client = _build_llm_client(client_details)
        kwargs = _request_kwargs(chat_details)

        text = client.response(**kwargs)
        return Response({"text": text}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def llm_json_response(request):
    try:
        data = request.data or {}
        client_details, chat_details = _extract_sections(data)

        client = _build_llm_client(client_details)
        kwargs = _request_kwargs(chat_details)

        schema_model = _build_schema_model(chat_details.get("schema"))
        payload = client.json_response(schema=schema_model, **kwargs)
        return Response({"data": payload}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def llm_stream_response(request):
    try:
        data = request.data or {}
        client_details, chat_details = _extract_sections(data)

        client = _build_llm_client(client_details)
        kwargs = _request_kwargs(chat_details)

        print(kwargs)

        def generator() -> Generator[str, None, None]:
            try:
                for chunk in client.response_stream(**kwargs):
                    yield chunk
            except Exception as exc:
                yield f"[error] {exc}"

        return StreamingHttpResponse(generator(), status=status.HTTP_200_OK, content_type="text/plain")
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def hf_status(request):
    try:
        return Response(HuggingFaceClient.download_status(), status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def hf_list_models(request):
    try:
        models_dir = request.query_params.get("models_dir")
        client = HuggingFaceClient(models_dir=models_dir) if models_dir else HuggingFaceClient()
        return Response({"models": client.list_models()}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["DELETE"])
def hf_delete_model(request):
    try:
        local_name = (request.data.get("local_name") or request.query_params.get("local_name") or "").strip()
        if not local_name:
            return Response({"error": "local_name is required"}, status=status.HTTP_400_BAD_REQUEST)
        client = HuggingFaceClient()
        client.delete_model(local_name)
        return Response({"message": "deleted", "local_name": local_name}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# ============================
# Ollama: pull / list / delete / status
# ============================

@api_view(["POST"])
def ollama_download(request):
    try:
        model = (request.data.get("model") or "").strip()
        if not model:
            return Response({"error": "model is required"}, status=status.HTTP_400_BAD_REQUEST)
        OllamaClient.request_download(model=model)
        return Response({"message": "queued", "status": OllamaClient.download_status()}, status=status.HTTP_202_ACCEPTED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def ollama_status(request):
    try:
        return Response(OllamaClient.download_status(), status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def ollama_list_models(request):
    try:
        client = OllamaClient()
        return Response({"models": client.list_models()}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["DELETE"])
def ollama_delete_model(request):
    try:
        model = (request.data.get("model") or request.query_params.get("model") or "").strip()
        if not model:
            return Response({"error": "model is required"}, status=status.HTTP_400_BAD_REQUEST)
        client = OllamaClient()
        client.delete_model(model)
        return Response({"message": "deleted", "model": model}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# ============================
# Client specs: list classes & their __init__ parameters
# ============================

@api_view(["GET"])
def clients_specs(request):
    try:
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

        out: List[Dict[str, Any]] = []
        for cls in LLMClientRegistry.list_classes():
            sig = inspect.signature(cls.__init__)
            params = [
                serialize_param(p)
                for name, p in sig.parameters.items()
                if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            ]
            display_name = getattr(cls, "display_name", cls.__name__)
            client_id = getattr(cls, "client_id", f"{cls.__module__}.{cls.__name__}")
            param_display = getattr(cls, "param_display_names", {})

            capabilities = {
                "download": hasattr(cls, "request_download"),
                "download_status": hasattr(cls, "download_status"),
                "list_models": hasattr(cls, "list_models"),
                "delete_model": hasattr(cls, "delete_model"),
            }
            available_models = getattr(cls, "available_models", None)
            requires_model_list = getattr(cls, "requires_model_list", False)

            out.append(
                {
                    "client_id": client_id,
                    "true_name": f"{cls.__module__}.{cls.__name__}",
                    "class_name": cls.__name__,
                    "display_name": display_name,
                    "init_parameters": params,
                    "param_display_names": param_display,
                    "capabilities": capabilities,
                    "available_models": available_models,
                    "requires_model_list": requires_model_list,
                }
            )
        return Response({"clients": out}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
