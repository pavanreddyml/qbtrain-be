# apps/aisecurity/figstep/views.py
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Generator, Optional

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
    Process a figstep query with streaming response.

    Expected body:
    {
        "question": <str>,
        "conversation_history": [...],
        "model_config": {...},
        "image_b64": <str>,          # base64-encoded image
        "prompt_defense": "none"|"secure_1"|"secure_2",
        "classifier_defense": {
            "enabled": bool,
            "model": str,
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
    image_b64 = body.get("image_b64", "")

    # Prompt-level defense (system prompt hardening)
    prompt_defense = body.get("prompt_defense", "none")
    if isinstance(prompt_defense, str) and prompt_defense not in _prompts.SYSTEM_PROMPTS:
        prompt_defense = "none"

    # Classifier defense (OCR pipeline)
    classifier_defense = body.get("classifier_defense", {})
    if isinstance(classifier_defense, str):
        try:
            classifier_defense = json.loads(classifier_defense)
        except (json.JSONDecodeError, TypeError):
            classifier_defense = {}

    # Decode image
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

    # --- Defense: OCR + Injection Classifier ---
    if image_bytes and classifier_defense.get("enabled") and classifier_defense.get("model"):
        yield {"type": "status", "message": "Running OCR + injection classifier defense..."}
        t0 = time.time()
        result = fn.run_ocr_injection_defense(image_bytes, classifier_defense["model"])
        tracer.trace(
            "FigStepDefense", "defense",
            operation="ocr_injection_classifier",
            latency_ms=round((time.time() - t0) * 1000),
            output=result,
        )
        yield {"type": "trace", "content": tracer.get_traces()[-1]}
        if result.get("is_injection"):
            yield {
                "type": "message",
                "content": (
                    "**Defense Alert (OCR + Classifier):** The image contains text that was flagged "
                    f"as a potential injection attack (confidence: {result['confidence']:.1%}).\n\n"
                    f"Extracted text: `{result.get('ocr_text', '')[:200]}`"
                ),
            }

    # --- Generate VLLM response ---
    if image_bytes:
        yield {"type": "status", "message": f"Generating response with image ({len(image_bytes)} bytes)..."}
    else:
        yield {"type": "status", "message": "Generating response (no image attached)..."}

    if model_config:
        try:
            llm_client = fn._build_client(model_config)

            user_prompt = _prompts.FIGSTEP_USER_PROMPT.format(
                user_query=question,
            )

            t0 = time.time()
            response_text = ""
            system_prompt = _prompts.SYSTEM_PROMPTS.get(prompt_defense, _prompts.SYSTEM_PROMPTS["none"])

            for chunk in llm_client.response_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                image=image_bytes,
                tracer=tracer,
            ):
                yield {"type": "message", "content": chunk}
                response_text += chunk

            tracer.trace(
                "figstep", "llm",
                operation="generate_response",
                latency_ms=round((time.time() - t0) * 1000),
                model=model_config.get("model", "unknown"),
                prompt_defense=prompt_defense,
                system_prompt_preview=system_prompt.strip()[:200],
                prompt_preview=user_prompt.strip()[:200],
                prompt_length=len(user_prompt),
                image_provided=image_bytes is not None,
                image_size=len(image_bytes) if image_bytes else 0,
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
    """
    Process a figstep VLLM query with streaming response.
    Supports image upload via FormData or base64 in JSON body.
    """
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
    Generate a FigStep-family typographic attack image.

    Request body (JSON or FormData):
    {
        "instruction": <str>,           # harmful instruction to embed
        "method": one of "figstep" / "figstep_plus" / "multilingual"
                  / "hades" / "steganographic",
        # method-specific (all optional with sensible defaults):
        "script":      <str>            # multilingual: en|ru|el|zh|ja|ko
        "fg_color":    <str>            # steganographic: foreground hex
        "bg_color":    <str>            # steganographic: background hex
        "decoy_texts": <list[str]>      # hades: custom benign paragraphs
        "show_cues":   <bool>           # hades: render attention arrows
    }

    Responses:
      figstep / multilingual / hades / steganographic : single PNG binary
      figstep_plus                                    : JSON with list of base64 images
    """
    try:
        instruction = request.data.get("instruction", "")
        if not instruction or not isinstance(instruction, str):
            return Response(
                {"error": "'instruction' field is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        method = request.data.get("method", "figstep")
        valid_methods = ("figstep", "figstep_plus", "multilingual",
                          "hades", "steganographic")
        if method not in valid_methods:
            return Response(
                {"error": f"method must be one of {list(valid_methods)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # FigStep+ → multi-image JSON
        if method == "figstep_plus":
            images_bytes = fn.generate_figstep_plus_images(instruction)
            images_b64 = [fn.image_to_base64(img) for img in images_bytes]
            return Response(
                {"method": "figstep_plus", "images": images_b64, "count": len(images_b64)},
                status=status.HTTP_200_OK,
            )

        # All other methods → single PNG binary
        if method == "figstep":
            image_bytes = fn.generate_figstep_image(instruction)
            filename = "figstep_attack.png"
        elif method == "steganographic":
            fg_color = request.data.get("fg_color", "#F4F4F4")
            bg_color = request.data.get("bg_color", "#FFFFFF")
            image_bytes = fn.generate_steganographic_image(
                instruction, fg_color=fg_color, bg_color=bg_color,
            )
            filename = "figstep_steganographic.png"
        elif method == "multilingual":
            script = request.data.get("script", "ru")
            try:
                image_bytes = fn.generate_multilingual_image(instruction, script=script)
            except fn.ValidationError as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            filename = f"figstep_multilingual_{script}.png"
        elif method == "hades":
            decoy_texts_raw = request.data.get("decoy_texts")
            decoy_texts: Optional[list] = None
            if isinstance(decoy_texts_raw, str) and decoy_texts_raw.strip():
                try:
                    parsed = json.loads(decoy_texts_raw)
                    decoy_texts = parsed if isinstance(parsed, list) else None
                except (json.JSONDecodeError, TypeError):
                    decoy_texts = None
            elif isinstance(decoy_texts_raw, list):
                decoy_texts = decoy_texts_raw
            show_cues_raw = request.data.get("show_cues", True)
            show_cues = show_cues_raw not in (False, "false", "False", "0", 0)
            image_bytes = fn.generate_hades_image(
                instruction, decoy_texts=decoy_texts, show_cues=show_cues,
            )
            filename = "figstep_hades.png"
        else:
            # Unreachable thanks to the validation above.
            return Response({"error": f"Unhandled method: {method}"},
                             status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        response = HttpResponse(image_bytes, content_type="image/png", status=200)
        response["Content-Disposition"] = f"attachment; filename={filename}"
        return response

    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def analyze_image(request):
    """
    Analyze a figstep-family image and return a side-by-side comparison of
    "what humans see" vs "what the LLM sees".

    Response JSON:
      original_b64     — image as uploaded (the human view)
      llm_view_b64     — auto-contrast-stretched grayscale: reveals near-invisible
                          text in steganographic attacks, sharpens decorations
                          otherwise. Approximates what survives the VLM's
                          contrast-normalizing vision encoder.
      ocr_text         — pytesseract / easyocr extraction (what an OCR-based
                          defense would feed a classifier)
      variant_hint     — heuristic label: figstep | steganographic | hades |
                          multilingual_cyrillic | multilingual_greek | multilingual_cjk
      pixel_std        — std deviation of the original (low => low-contrast)
      image_size       — [w, h]
    """
    try:
        if not request.FILES or "image" not in request.FILES:
            return Response(
                {"error": "'image' file is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        uploaded = request.FILES["image"]
        image_bytes = b"".join(chunk for chunk in uploaded.chunks())
        return Response(fn.analyze_figstep_image(image_bytes))
    except Exception as exc:
        return _error_response(exc)
