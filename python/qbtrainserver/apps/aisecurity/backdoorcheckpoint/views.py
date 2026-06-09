# apps/aisecurity/backdoorcheckpoint/views.py
from __future__ import annotations

import base64
import json
from typing import Any, Dict, Generator

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse, StreamingHttpResponse

from . import functions as fn


# ---------- helpers ----------
def _error(exc: Exception) -> Response:
    if isinstance(exc, fn.ValidationError):
        return Response({"error": "ValidationError", "detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    if isinstance(exc, fn.StateError):
        return Response({"error": "StateError",      "detail": str(exc)}, status=status.HTTP_409_CONFLICT)
    if isinstance(exc, fn.ConfigError):
        return Response({"error": "ConfigError",     "detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response({"error": "ServerError",         "detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _ndjson(gen: Generator[Dict[str, Any], None, None]) -> Generator[str, None, None]:
    for event in gen:
        yield json.dumps(event) + "\n"


# ---------- endpoints ----------
@api_view(["GET"])
def meta(request):
    try:
        return Response(fn.get_meta())
    except Exception as exc:
        return _error(exc)


@api_view(["GET"])
def list_models(request):
    try:
        return Response({"models": fn.list_models()})
    except Exception as exc:
        return _error(exc)


@api_view(["POST"])
def start_download(request):
    try:
        body = request.data or {}
        model_id = body.get("model_id")
        if not model_id:
            raise fn.ValidationError("model_id is required")
        return Response(fn.start_download(model_id), status=status.HTTP_202_ACCEPTED)
    except Exception as exc:
        return _error(exc)


@api_view(["GET"])
def download_status(request):
    try:
        model_id = request.query_params.get("model_id")
        if not model_id:
            raise fn.ValidationError("model_id is required")
        return Response(fn.download_status(model_id))
    except Exception as exc:
        return _error(exc)


@api_view(["GET"])
def download_stream(request, model_id: str):
    try:
        gen = fn.stream_download(model_id)
        resp = StreamingHttpResponse(_ndjson(gen), content_type="application/x-ndjson")
        resp["Cache-Control"]    = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
    except Exception as exc:
        return _error(exc)


@api_view(["POST"])
def query(request):
    """
    Streaming NDJSON. Accepts either JSON body with image_b64, or multipart
    form-data with an 'image' file.
    """
    try:
        body = dict(request.data) if hasattr(request.data, "keys") else {}
        # Normalize any list-form fields from FormData
        body = {k: (v[0] if isinstance(v, list) else v) for k, v in body.items()}

        files = request.FILES if hasattr(request, "FILES") else None
        if files and "image" in files:
            img_bytes = b"".join(chunk for chunk in files["image"].chunks())
            body["image_b64"] = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

        gen = fn.stream_query(body)
        resp = StreamingHttpResponse(_ndjson(gen), content_type="application/x-ndjson")
        resp["Cache-Control"]    = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
    except Exception as exc:
        return _error(exc)


@api_view(["GET"])
def samples(request):
    try:
        domain = request.query_params.get("domain", "caption")
        return Response(fn.list_samples(domain))
    except Exception as exc:
        return _error(exc)


@api_view(["GET"])
def sample_image(request):
    """
    Return a single sample image (clean or watermarked) as a PNG, used by the
    Generate-Image modal to fetch a starting image for chat. Query params:
      domain=caption|medical|finance
      index=<int>
      kind=normal|backdoored
      position=bottom_right|bottom_left|top_right|top_left|random   (watermark corner)
      scale=<float>   (watermark size as fraction of the shorter side, 0.05–0.40)
    """
    try:
        domain   = request.query_params.get("domain", "caption")
        index    = int(request.query_params.get("index", "1"))
        kind     = request.query_params.get("kind", "normal")
        position = request.query_params.get("position", "bottom_right")
        try:
            scale = float(request.query_params.get("scale", "0.13"))
        except (TypeError, ValueError):
            scale = 0.13
        data, mime = fn.load_sample_image(domain, index, kind, position=position, scale=scale)
        resp = HttpResponse(data, content_type=mime)
        resp["Content-Disposition"] = f'attachment; filename="{domain}_{kind}_{index}.png"'
        return resp
    except Exception as exc:
        return _error(exc)
