# apps/aisecurity/imageadvattacks/views.py
from __future__ import annotations

import json
from typing import Any, Dict, Generator

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import StreamingHttpResponse

from . import functions as fn


def _error_response(exc: Exception) -> Response:
    if isinstance(exc, fn.ValidationError):
        return Response(
            {"error": "ValidationError", "detail": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if isinstance(exc, fn.StateError):
        return Response(
            {"error": "StateError", "detail": str(exc)},
            status=status.HTTP_409_CONFLICT,
        )
    if isinstance(exc, fn.ModelNotDownloadedError):
        return Response(
            {"error": "ModelNotDownloaded", "detail": str(exc)},
            status=status.HTTP_409_CONFLICT,
        )
    return Response(
        {"error": "ServerError", "detail": str(exc)},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _ndjson(gen: Generator[Dict[str, Any], None, None]) -> Generator[str, None, None]:
    for event in gen:
        yield json.dumps(event) + "\n"


@api_view(["GET"])
def meta(request):
    """Return available models, attack defaults, and a built-in sample image."""
    try:
        return Response(fn.get_meta())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@api_view(["GET"])
def list_models(request):
    """Model catalog annotated with local-download state. Downloads themselves go
    through the global client endpoints (POST /api/clients/hf/download/, etc.)."""
    try:
        return Response({"provider": fn.MODEL_PROVIDER, "models": fn.list_models()})
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@api_view(["POST"])
def start(request):
    """Begin a new attack. Returns the job_id used for streaming/stopping."""
    try:
        return Response(fn.start_attack(request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@api_view(["POST"])
def stop(request):
    """Cooperatively stop the attack. Body: { "job_id": <str> }."""
    try:
        body = request.data or {}
        job_id = body.get("job_id")
        if not job_id:
            raise fn.ValidationError("job_id is required")
        return Response(fn.stop_attack(job_id))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@api_view(["GET"])
def session_status(request, job_id: str):
    try:
        return Response(fn.status(job_id))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@api_view(["GET"])
def stream(request, job_id: str):
    """Stream NDJSON events for the given job until completion."""
    try:
        gen = fn.stream_events(job_id)
        resp = StreamingHttpResponse(_ndjson(gen), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)
