# apps/aisecurity/poisoneddataset/views.py
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
    return Response(
        {"error": "ServerError", "detail": str(exc)},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _ndjson(gen: Generator[Dict[str, Any], None, None]) -> Generator[str, None, None]:
    for event in gen:
        yield json.dumps(event) + "\n"


@api_view(["GET"])
def meta(request):
    try:
        return Response(fn.get_meta())
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def start(request):
    try:
        return Response(fn.start_training(request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def stop(request):
    try:
        body = request.data or {}
        job_id = body.get("job_id")
        if not job_id:
            raise fn.ValidationError("job_id is required")
        return Response(fn.stop_training(job_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def session_status(request, job_id: str):
    try:
        return Response(fn.status(job_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def test_model(request):
    """Run the trained model on a random synthetic sample (clean + watermarked)
    or an uploaded image. Body: { mode: 'sample'|'upload', job_id?, image_b64? }."""
    try:
        return Response(fn.test_model(request.data or {}))
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def stream(request, job_id: str):
    try:
        gen = fn.stream_events(job_id)
        resp = StreamingHttpResponse(_ndjson(gen), content_type="application/x-ndjson")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def analyze_samples(request):
    """
    Fetch sample images + captions from a dataset, filtered by clean/poisoned.
    Query params:
      dataset_id (default 'synthetic')
      kind       'normal' | 'poisoned'
      count      1..50
    """
    try:
        dataset_id = request.query_params.get("dataset_id", "synthetic")
        kind = request.query_params.get("kind", "normal")
        count = int(request.query_params.get("count", "12"))
        return Response(fn.fetch_dataset_samples(dataset_id, kind, count))
    except Exception as exc:
        return _error_response(exc)
