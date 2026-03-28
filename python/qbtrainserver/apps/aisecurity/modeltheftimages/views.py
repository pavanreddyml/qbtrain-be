"""
Views for Model Theft Images app.

Streams DDPM reverse-diffusion steps as NDJSON so the frontend can
visualise the denoising process in real time.
"""
from __future__ import annotations

import json

from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import functions as fn


@api_view(["GET"])
def list_models(request):
    """Return available DDPM models and their download status."""
    return Response({"models": fn.list_models()})


@api_view(["POST"])
def generate(request):
    """Generate images with streaming intermediate denoising steps (NDJSON)."""
    body = request.data
    model_key = (body.get("model_key") or "").strip()
    num_images = body.get("num_images", 1)
    seed = body.get("seed", None)

    if not model_key:
        return Response(
            {"error": "model_key is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if model_key not in fn.AVAILABLE_MODELS:
        return Response(
            {"error": f"Unknown model: {model_key}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not isinstance(num_images, int) or num_images < 1:
        num_images = 1
    num_images = min(num_images, 4)

    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    def event_stream():
        for ev in fn.generate_image_stream(model_key, num_images=num_images, seed=seed):
            yield json.dumps(ev) + "\n"

    return StreamingHttpResponse(
        event_stream(),
        content_type="application/x-ndjson",
    )


@api_view(["POST"])
def download_all(request):
    """Start downloading all DDPM models from HuggingFace."""
    try:
        result = fn.download_all_models()
        return Response(result, status=status.HTTP_202_ACCEPTED)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
def download_status(request):
    """Get download status for all models."""
    return Response(fn.get_download_status())


@api_view(["POST"])
def unload_model(request):
    """Explicitly unload the current model from GPU."""
    try:
        result = fn.unload_current_model()
        return Response(result)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
