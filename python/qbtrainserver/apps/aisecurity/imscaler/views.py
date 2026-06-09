# apps/aisecurity/imscaler/views.py
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Generator

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse, StreamingHttpResponse

from qbtrain.tracers.agent_tracer import AgentTracer
from . import functions as fn
from . import prompts as _prompts


def _error_response(exc: Exception) -> Response:
    if isinstance(exc, fn.ValidationError):
        return Response(
            {"error": "ValidationError", "detail": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if isinstance(exc, fn.ConfigError):
        return Response(
            {"error": "ConfigError", "detail": str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return Response(
        {"error": "ServerError", "detail": str(exc)},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _ndjson_generator(gen: Generator) -> Generator[str, None, None]:
    for event in gen:
        yield json.dumps(event) + "\n"


def _parse_json_field(value, default=None):
    if default is None:
        default = []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value if value is not None else default


def _process_query_stream(body: Dict[str, Any], files=None) -> Generator[Dict[str, Any], None, None]:
    """
    Process an imscaler VLLM query with streaming response.

    Expected body:
    {
        "question": <str>,
        "conversation_history": [...],
        "model_config": {...},
        "memory": <str>,
        "image_b64": <str>,
        "defenses": {
            "ocr_classifier": { "enabled": bool, "model": str },
            "perceptual_hash": { "enabled": bool },
            "metadata_inspection": { "enabled": bool },
        }
    }
    """
    if not isinstance(body, dict):
        raise fn.ValidationError("Body must be a JSON object")

    tracer = AgentTracer()
    pipeline_start = time.time()

    question = body.get("question")
    if not question or not isinstance(question, str):
        raise fn.ValidationError("'question' field is required and must be a string")

    conversation_history = _parse_json_field(body.get("conversation_history"), [])
    model_config = _parse_json_field(body.get("model_config"), {})
    memory = body.get("memory", "")
    image_b64 = body.get("image_b64", "")
    defenses = body.get("defenses", {})
    if isinstance(defenses, str):
        try:
            defenses = json.loads(defenses)
        except (json.JSONDecodeError, TypeError):
            defenses = {}

    # Handle file upload
    image_bytes = None
    if files and "image" in files:
        uploaded = files["image"]
        image_bytes = b"".join(chunk for chunk in uploaded.chunks())
    elif image_b64:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            yield {"type": "warning", "message": "Failed to decode base64 image"}

    # --- Run defenses on image ---
    if image_bytes and defenses:
        # Defense 1: Preprocess -> OCR -> Injection Classifier
        ocr_cfg = defenses.get("ocr_classifier", {})
        if ocr_cfg.get("enabled") and ocr_cfg.get("model"):
            yield {"type": "status", "message": "Running image preprocess -> OCR -> injection classifier..."}
            t0 = time.time()
            result = fn.run_ocr_injection_defense(image_bytes, ocr_cfg["model"])
            tracer.trace(
                "IMScalerDefense", "defense",
                operation="ocr_injection_classifier",
                latency_ms=round((time.time() - t0) * 1000),
                output=result,
            )
            yield {"type": "trace", "content": tracer.get_traces()[-1]}
            if result.get("is_injection"):
                yield {
                    "type": "message",
                    "content": (
                        "**Defense Alert (OCR + Classifier):** The preprocessed image contains text flagged "
                        f"as a potential injection attack (confidence: {result['confidence']:.1%}).\n\n"
                        f"Extracted text: `{result.get('ocr_text', '')[:200]}`"
                    ),
                }

        # Defense 2: Perceptual Hash (original vs preprocessed)
        phash_cfg = defenses.get("perceptual_hash", {})
        if phash_cfg.get("enabled"):
            yield {"type": "status", "message": "Running perceptual hash defense (original vs preprocessed)..."}
            t0 = time.time()
            result = fn.run_perceptual_hash_defense(image_bytes)
            tracer.trace(
                "IMScalerDefense", "defense",
                operation="perceptual_hash",
                latency_ms=round((time.time() - t0) * 1000),
                output=result,
            )
            yield {"type": "trace", "content": tracer.get_traces()[-1]}
            if result.get("flagged"):
                yield {
                    "type": "message",
                    "content": (
                        "**Defense Alert (Perceptual Hash):** High divergence detected between "
                        f"original and preprocessed image (hamming distance: {result.get('hamming_distance', '?')}).\n\n"
                        "This may indicate a scale-dependent adversarial attack."
                    ),
                }

        # Defense 3: Metadata Inspection
        meta_cfg = defenses.get("metadata_inspection", {})
        if meta_cfg.get("enabled"):
            yield {"type": "status", "message": "Running metadata inspection defense..."}
            t0 = time.time()
            result = fn.run_metadata_inspection_defense(image_bytes)
            tracer.trace(
                "IMScalerDefense", "defense",
                operation="metadata_inspection",
                latency_ms=round((time.time() - t0) * 1000),
                output=result,
            )
            yield {"type": "trace", "content": tracer.get_traces()[-1]}
            if result.get("flagged"):
                findings_text = "\n".join(
                    f"- {f.get('type')}: {f.get('preview', f.get('message', ''))}"
                    for f in result.get("findings", [])
                )
                yield {
                    "type": "message",
                    "content": (
                        "**Defense Alert (Metadata):** Suspicious metadata found in image:\n"
                        f"{findings_text}"
                    ),
                }

        # Defense 4: SSIM Anomaly Detection
        ssim_cfg = defenses.get("ssim", {})
        if ssim_cfg.get("enabled"):
            yield {"type": "status", "message": "Running SSIM anomaly detection..."}
            t0 = time.time()
            threshold = float(ssim_cfg.get("threshold", 0.80) or 0.80)
            result = fn.run_ssim_defense(image_bytes, threshold=threshold)
            tracer.trace(
                "IMScalerDefense", "defense",
                operation="ssim",
                latency_ms=round((time.time() - t0) * 1000),
                output=result,
            )
            yield {"type": "trace", "content": tracer.get_traces()[-1]}
            if result.get("flagged"):
                score = result.get("score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
                yield {
                    "type": "message",
                    "content": (
                        "**Defense Alert (SSIM):** Image appearance changes dramatically "
                        f"after preprocessing (SSIM = {score_str}, threshold = {threshold:.2f}). "
                        "Likely a scale-dependent adversarial attack."
                    ),
                }

    # --- Generate VLLM response ---
    yield {"type": "status", "message": "Generating response..."}

    memory_str = memory if memory else "(No user memory provided)"
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        memory_str += "\n\n[CONVERSATION HISTORY]\n" + "\n".join(history_lines)

    if model_config:
        try:
            llm_client = fn._build_client(model_config)

            user_prompt = _prompts.IMSCALER_USER_PROMPT.format(
                memory=memory_str,
                user_query=question,
            )

            t0 = time.time()
            response_text = ""
            for chunk in llm_client.response_stream(
                prompt=user_prompt,
                system_prompt=_prompts.IMSCALER_SYSTEM_PROMPT,
                conversation_history=conversation_history or None,
                image=image_bytes,
                tracer=tracer,
            ):
                yield {"type": "message", "content": chunk}
                response_text += chunk

            tracer.trace(
                "imscaler", "llm",
                operation="generate_response",
                latency_ms=round((time.time() - t0) * 1000),
                model=model_config.get("model", "unknown"),
                system_prompt_preview=_prompts.IMSCALER_SYSTEM_PROMPT.strip()[:200],
                prompt_preview=user_prompt.strip()[:200],
                prompt_length=len(user_prompt),
                image_provided=image_bytes is not None,
                image_size=len(image_bytes) if image_bytes else 0,
                memory_preview=memory[:200] if memory else "",
                output={
                    "response_length": len(response_text),
                    "response_preview": response_text[:300],
                },
            )
            yield {"type": "trace", "content": tracer.get_traces()[-1]}
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            yield {"type": "error", "message": error_msg}
    else:
        yield {"type": "message", "content": "No model configured. Please select a model in settings."}

    # Final summary trace
    total_latency = round((time.time() - pipeline_start) * 1000)
    yield {
        "type": "trace_summary",
        "content": {
            "calls": tracer.get_traces(),
            "model": model_config.get("model", "unknown"),
            "total_latency_ms": total_latency,
        },
    }
    yield {"type": "done"}


@api_view(["POST"])
def query(request):
    """Process an imscaler VLLM query with streaming response."""
    try:
        generator = _process_query_stream(request.data, files=request.FILES)
        return StreamingHttpResponse(
            _ndjson_generator(generator),
            content_type="application/x-ndjson",
            status=status.HTTP_200_OK,
        )
    except Exception as exc:
        err = _error_response(exc)
        return StreamingHttpResponse(
            [json.dumps({"error": err.data})],
            content_type="application/json",
            status=err.status_code,
        )


@api_view(["POST"])
def generate_image(request):
    """
    Generate an anamorpher adversarial image.

    Request body (FormData):
        instructions: str
        mode: "nearest"|"bicubic"|"bilinear"
        image: file (optional decoy base image)

    Response: PNG image as binary data (1344x1344).
    """
    try:
        instructions = request.data.get("instructions", "")
        if not instructions or not isinstance(instructions, str):
            return Response(
                {"error": "'instructions' field is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        mode = request.data.get("mode", "nearest")
        if mode not in ("nearest", "bicubic", "bilinear"):
            return Response(
                {"error": "mode must be 'nearest', 'bicubic', or 'bilinear'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        base_image_bytes = None
        if request.FILES and "image" in request.FILES:
            uploaded = request.FILES["image"]
            base_image_bytes = b"".join(chunk for chunk in uploaded.chunks())

        try:
            resolution = int(request.data.get("resolution", 336))
        except (ValueError, TypeError):
            resolution = 336

        # Region: JSON array [y_frac, x_frac, h_frac, w_frac] or empty
        region = None
        region_str = request.data.get("region", "")
        if region_str:
            try:
                region = tuple(json.loads(region_str))
            except (json.JSONDecodeError, TypeError):
                pass

        # Provider/model metadata to stamp into the PNG (so Analyze can read
        # it back). All fields are optional; missing ones are silently dropped.
        metadata = {
            "provider": request.data.get("provider") or "",
            "model": request.data.get("model") or "",
        }
        metadata = {k: v for k, v in metadata.items() if v}

        image_bytes = fn.generate_anamorpher_image(
            instructions=instructions,
            mode=mode,
            base_image_bytes=base_image_bytes,
            resolution=resolution,
            region=region,
            metadata=metadata or None,
        )

        response = HttpResponse(
            image_bytes,
            content_type="image/png",
            status=200,
        )
        response["Content-Disposition"] = "attachment; filename=imscaler_attack.png"
        return response

    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def read_image_metadata(request):
    """
    Return qbtrain.imscaler.* PNG tEXt chunks from an uploaded image so the
    Analyze modal can auto-populate provider/model/resolution/mode without
    asking the user. Returns `{}` for images this app didn't produce.
    """
    try:
        if not request.FILES or "image" not in request.FILES:
            return Response(
                {"error": "'image' file is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        uploaded = request.FILES["image"]
        image_bytes = b"".join(chunk for chunk in uploaded.chunks())
        meta = fn.read_anamorpher_metadata(image_bytes)
        return Response({"metadata": meta})
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def preprocess_image(request):
    """
    Return the preprocessed version of an uploaded image.
    Simulates VLLM preprocessing (resize to 336x336), then scales the
    result back up to the original resolution using NEAREST so the user
    can see exactly what the model sees at the original image dimensions.

    Request: FormData with 'image' file.
    Response: JSON with original and rescaled-back preprocessed image.
    """
    try:
        if not request.FILES or "image" not in request.FILES:
            return Response(
                {"error": "'image' file is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded = request.FILES["image"]
        image_bytes = b"".join(chunk for chunk in uploaded.chunks())

        try:
            resolution = int(request.data.get("resolution", 336))
        except (ValueError, TypeError):
            resolution = 336

        method = request.data.get("method", "bicubic")

        preprocessed_bytes, rescaled_bytes = fn.preprocess_image(
            image_bytes, resolution=resolution, method=method,
        )

        rescaled_b64 = base64.b64encode(rescaled_bytes).decode("utf-8")

        return Response({
            "preprocessed_b64": f"data:image/png;base64,{rescaled_b64}",
            "original_size": len(image_bytes),
            "preprocessed_size": len(preprocessed_bytes),
        })

    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def defense_preview(request):
    """
    Run `resize(pre_transform(image))` and return 3 base64 PNGs:
      original_b64           : as uploaded
      preprocessed_full_b64  : full-res, with pre_transform applied
                                (== original when pre_transform == "none")
      preprocessed_down_b64  : final pipeline output the VLM would receive

    FormData:
      image           : file
      resize_method   : one of fn.RESIZE_METHODS (default "bicubic")
      pre_transform   : one of fn.PRE_TRANSFORMS (default "none")
      transform_params: JSON object — params for the chosen pre_transform
                        (sigma for gaussian_*, radius for box_blur, etc.)
      resolution      : int, default 336
    """
    try:
        if not request.FILES or "image" not in request.FILES:
            return Response(
                {"error": "'image' file is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        uploaded = request.FILES["image"]
        image_bytes = b"".join(chunk for chunk in uploaded.chunks())

        resize_method = request.data.get("resize_method", "bicubic")
        if resize_method not in fn.RESIZE_METHODS:
            return Response(
                {"error": f"resize_method must be one of {list(fn.RESIZE_METHODS)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        pre_transform = request.data.get("pre_transform", "none")
        if pre_transform not in fn.PRE_TRANSFORMS:
            return Response(
                {"error": f"pre_transform must be one of {list(fn.PRE_TRANSFORMS)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        params_raw = request.data.get("transform_params", "")
        transform_params: Dict[str, Any] = {}
        if params_raw:
            try:
                transform_params = json.loads(params_raw) if isinstance(params_raw, str) else dict(params_raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                transform_params = {}

        try:
            resolution = int(request.data.get("resolution", 336))
        except (ValueError, TypeError):
            resolution = 336

        orig_bytes, full_bytes, down_bytes = fn.defense_preview_image(
            image_bytes,
            resize_method=resize_method,
            pre_transform=pre_transform,
            transform_params=transform_params,
            resolution=resolution,
        )

        def _b64(b: bytes) -> str:
            return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")

        return Response({
            "resize_method": resize_method,
            "pre_transform": pre_transform,
            "transform_params": transform_params,
            "resolution": resolution,
            "original_b64": _b64(orig_bytes),
            "preprocessed_full_b64": _b64(full_bytes),
            "preprocessed_down_b64": _b64(down_bytes),
        })
    except Exception as exc:
        return _error_response(exc)
