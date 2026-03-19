# apps/aisecurity/echoleak/views.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Generator

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse, StreamingHttpResponse

from qbtrain.tracers.agent_tracer import AgentTracer
from . import functions as fn


def _error_response(exc: Exception) -> Response:
    """Generate standardized error response."""
    if isinstance(exc, fn.ValidationError):
        return Response(
            {"error": "ValidationError", "detail": str(exc)},
            status=status.HTTP_400_BAD_REQUEST
        )
    if isinstance(exc, fn.DeniedSourceError):
        return Response(
            {"error": "Access Denied", "detail": str(exc)},
            status=status.HTTP_403_FORBIDDEN
        )
    if isinstance(exc, fn.ExtractionError):
        return Response(
            {"error": "ExtractionError", "detail": str(exc)},
            status=status.HTTP_400_BAD_REQUEST
        )
    if isinstance(exc, fn.ConfigError):
        return Response(
            {"error": "ConfigError", "detail": str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return Response(
        {"error": "ServerError", "detail": str(exc)},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def _ndjson_generator(gen: Generator) -> Generator[str, None, None]:
    """Convert event generator to NDJSON format."""
    for event in gen:
        yield json.dumps(event) + "\n"


def _run_classifier(model_id: str, text: str, label: str) -> Dict[str, Any]:
    """Run injection classifier on text, return result dict."""
    from qbtrain.ai.classifiers.injection_classifier import classify as run_classify

    is_injection, confidence = run_classify(model_id, text)
    return {
        "label": label,
        "model": model_id,
        "is_injection": is_injection,
        "confidence": round(confidence, 4),
        "input_preview": text[:200],
    }


def _parse_json_field(value, default=None):
    """Parse a field that may be a JSON string (from FormData) or already parsed."""
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
    Process query with streaming response.

    Links are extracted from the query text by the SourceExtractionAgent.
    Files can be uploaded via multipart form data.

    Expected body:
    {
        "question": <str>,          # may contain URLs to extract content from
        "conversation_history": [<message>, ...],  # optional
        "allowlist": [<denied_pattern>, ...],  # optional
        "classifier": {
            "enabled": <bool>,
            "model": <str>,
            "run_on_query": <bool>,
            "run_on_sources": <bool>,
        },  # optional
        "memory": <str>,  # optional, user information
        "config": {
            "max_tokens_per_source": <int>,
            "max_combined_tokens": <int>,
        }  # optional, uses defaults if not provided
    }
    """
    if not isinstance(body, dict):
        raise fn.ValidationError("Body must be a JSON object")

    tracer = AgentTracer()
    pipeline_start = time.time()

    # Extract fields
    question = body.get("question")
    if not question or not isinstance(question, str):
        raise fn.ValidationError("'question' field is required and must be a string")

    allowlist = _parse_json_field(body.get("allowlist"), [])
    conversation_history = _parse_json_field(body.get("conversation_history"), [])
    config = _parse_json_field(body.get("config"), {})
    model_config = _parse_json_field(body.get("model_config"), {})
    memory = body.get("memory", "")

    # Handle uploaded files: collect as (filename, bytes) tuples for the agent
    files_list: list[tuple[str, bytes]] = []
    if files and "file" in files:
        uploaded = files["file"]
        file_bytes = b"".join(chunk for chunk in uploaded.chunks())
        files_list.append((uploaded.name, file_bytes))

    # Classifier config
    classifier_cfg = body.get("classifier", {})
    if isinstance(classifier_cfg, str):
        try:
            classifier_cfg = json.loads(classifier_cfg)
        except (json.JSONDecodeError, TypeError):
            classifier_cfg = {}
    classifier_enabled = classifier_cfg.get("enabled", False)
    classifier_model = (classifier_cfg.get("model") or "").strip()
    run_on_query = classifier_cfg.get("run_on_query", False)
    run_on_sources = classifier_cfg.get("run_on_sources", False)

    # Update constants if config provided
    if config:
        if "max_tokens_per_source" in config:
            fn.MAX_TOKENS_PER_SOURCE = config["max_tokens_per_source"]
        if "max_combined_tokens" in config:
            fn.MAX_COMBINED_TOKENS = config["max_combined_tokens"]

    # --- Injection classifier: run on query ---
    if classifier_enabled and classifier_model and run_on_query:
        yield {"type": "status", "message": "Running injection classifier on query..."}

        t0 = time.time()
        result = _run_classifier(classifier_model, question, "query")
        tracer.trace(
            "InjectionClassifier", "classification",
            operation="classify_query",
            latency_ms=round((time.time() - t0) * 1000),
            model=classifier_model,
            input_preview=question[:200],
            input_length=len(question),
            output=result,
        )
        yield {"type": "trace", "content": tracer.get_traces()[-1]}

        if result["is_injection"]:
            yield {
                "type": "message",
                "content": "Your query has been flagged as a potential prompt injection attack and has been blocked."
            }
            total_latency = round((time.time() - pipeline_start) * 1000)
            yield {
                "type": "trace_summary",
                "content": {
                    "calls": tracer.get_traces(),
                    "model": model_config.get("model") or config.get("model", "unknown"),
                    "total_latency_ms": total_latency,
                },
            }
            yield {"type": "done"}
            return

    # Step 1: Extract sources (links from query + uploaded files) and build context
    malicious_sources = []
    yield {"type": "status", "message": "Extracting sources and building context..."}

    try:
        t0 = time.time()
        context, total_tokens, warnings = fn.build_context(
            query=question,
            files_list=files_list or None,
            allowlist=allowlist,
            client_config=model_config,
            tracer=tracer,
        )
        tracer.trace(
            "echoleak", "agent",
            operation="build_context",
            latency_ms=round((time.time() - t0) * 1000),
            output={
                "total_tokens": total_tokens,
                "context_length": len(context),
                "context_preview": context[:500],
                "warnings": len(warnings),
                "warning_details": warnings,
            },
        )
        yield {"type": "trace", "content": tracer.get_traces()[-1]}

        for warning in warnings:
            yield {"type": "warning", "message": warning}

        # --- Injection classifier: run on sources ---
        if classifier_enabled and classifier_model and run_on_sources and context:
            yield {"type": "status", "message": "Running injection classifier on sources..."}

            import re
            source_blocks = re.split(r'(\[(?:FILE|URL): [^\]]+\]\n)', context)
            rebuilt_parts = []
            current_label = None

            for block in source_blocks:
                header_match = re.match(r'\[(FILE|URL): ([^\]]+)\]\n', block)
                if header_match:
                    current_label = block.strip()
                    rebuilt_parts.append(block)
                elif current_label and block.strip():
                    source_text = block.strip()
                    t0 = time.time()
                    result = _run_classifier(classifier_model, source_text, current_label)
                    tracer.trace(
                        "InjectionClassifier", "classification",
                        operation="classify_source",
                        latency_ms=round((time.time() - t0) * 1000),
                        model=classifier_model,
                        source_label=current_label,
                        input_preview=source_text[:200],
                        input_length=len(source_text),
                        output=result,
                    )
                    yield {"type": "trace", "content": tracer.get_traces()[-1]}

                    if result["is_injection"]:
                        rebuilt_parts.append("The source contained malicious information and was not read.\n")
                        malicious_sources.append(current_label)
                    else:
                        rebuilt_parts.append(block)
                    current_label = None
                else:
                    rebuilt_parts.append(block)
                    current_label = None

            context = "".join(rebuilt_parts)

        yield {"type": "context", "tokens": total_tokens, "content": context}
    except fn.ExtractionError as e:
        tracer.trace("echoleak", "error", operation="build_context", output=str(e))
        yield {"type": "trace", "content": tracer.get_traces()[-1]}
        yield {"type": "error", "message": f"Extraction failed: {str(e)}"}
        return
    except Exception as e:
        tracer.trace("echoleak", "error", operation="build_context", output=str(e))
        yield {"type": "trace", "content": tracer.get_traces()[-1]}
        yield {"type": "error", "message": f"Source extraction failed: {str(e)}"}
        return

    # Step 2: Generate response via ResponseGeneratorAgent
    yield {"type": "status", "message": "Generating response..."}

    t0 = time.time()
    response_text = ""

    # Prepend malicious source warnings if any
    if malicious_sources:
        malicious_list = "\n".join(f"- {src}" for src in malicious_sources)
        warning_msg = (
            f"**Warning:** The following sources were flagged as containing malicious content "
            f"and were not included in the analysis:\n{malicious_list}\n\n"
        )
        yield {"type": "message", "content": warning_msg}
        response_text += warning_msg

    # Build memory string (include conversation history if present)
    memory_str = memory if memory else "(No user memory provided)"
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        memory_str += "\n\n[CONVERSATION HISTORY]\n" + "\n".join(history_lines)

    from . import prompts as _prompts

    if model_config:
        try:
            from qbtrain.agents.response_generator_agent import (
                ResponseGeneratorAgent,
                ResponseGeneratorPrompts,
            )

            llm_client = fn._build_client(model_config)
            rg_prompts = ResponseGeneratorPrompts(
                system_prompt_template=_prompts.ASSISTANT_SYSTEM_PROMPT,
                user_prompt_template=_prompts.CONTEXT_INSTRUCTION,
            )
            responder = ResponseGeneratorAgent(
                llm_client=llm_client,
                prompts=rg_prompts,
            )

            for chunk in responder.generate_stream(
                user_query=question,
                sql=memory_str,
                results=context if context else "(No sources provided)",
                tracer=tracer,
            ):
                yield {"type": "message", "content": chunk}
                response_text += chunk
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            yield {"type": "error", "message": error_msg}
            response_text = error_msg
    else:
        fallback = f"No model configured. Cannot generate a response for: '{question}'"
        yield {"type": "message", "content": fallback}
        response_text = fallback

    _user_prompt = _prompts.CONTEXT_INSTRUCTION.format(
        results=context if context else "(No sources provided)",
        sql=memory_str,
        user_query=question,
    )
    tracer.trace(
        "echoleak", "llm",
        operation="generate_response",
        latency_ms=round((time.time() - t0) * 1000),
        model=model_config.get("model", "unknown"),
        system_prompt_preview=_prompts.ASSISTANT_SYSTEM_PROMPT.strip()[:200],
        system_prompt_length=len(_prompts.ASSISTANT_SYSTEM_PROMPT),
        prompt_preview=_user_prompt.strip()[:200],
        prompt_length=len(_user_prompt),
        context_preview=context[:300] if context else "",
        context_length=len(context) if context else 0,
        memory_preview=memory[:200] if memory else "",
        memory_length=len(memory) if memory else 0,
        conv_history_length=len(conversation_history) if conversation_history else 0,
        output={
            "response_length": len(response_text),
            "response_preview": response_text[:300],
            "memory_included": bool(memory),
            "malicious_sources": len(malicious_sources),
            "malicious_source_labels": malicious_sources,
        },
    )
    yield {"type": "trace", "content": tracer.get_traces()[-1]}

    # Final summary trace
    total_latency = round((time.time() - pipeline_start) * 1000)
    model_name = model_config.get("model") or config.get("model", "unknown")
    yield {
        "type": "trace_summary",
        "content": {
            "calls": tracer.get_traces(),
            "model": model_name,
            "total_latency_ms": total_latency,
        },
    }

    yield {"type": "done"}


@api_view(["POST"])
def query(request):
    """
    Process a query with source extraction and streaming response.

    Links are identified from the query text automatically.
    Files can be uploaded via multipart form data.

    Request body:
    {
        "question": <str>,          # may contain URLs to extract content from
        "conversation_history": [<message>, ...],  # optional
        "allowlist": [<denied_pattern>, ...],  # optional
        "classifier": {...},  # optional
        "memory": <str>,  # optional
        "config": {...}  # optional
    }

    Response: NDJSON stream of events
    """
    try:
        generator = _process_query_stream(request.data, files=request.FILES)

        return StreamingHttpResponse(
            _ndjson_generator(generator),
            content_type="application/x-ndjson",
            status=status.HTTP_200_OK
        )
    except Exception as exc:
        err = _error_response(exc)
        return StreamingHttpResponse(
            [json.dumps({"error": err.data})],
            content_type="application/json",
            status=err.status_code
        )


@api_view(["GET"])
def pdf_exfil(request):
    """
    Generate PDF with embedded exfil URL.

    Query params:
    - exfilURL: URL to embed (required)
    - level: complexity level 1, 2, or 3 (optional, default 1)

    Response: PDF file as binary data
    """
    try:
        exfil_url = request.query_params.get("exfilURL")
        if not exfil_url:
            return Response(
                {"error": "exfilURL query parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        level = int(request.query_params.get("level", 1))
        if level not in {1, 2, 3}:
            return Response(
                {"error": "level must be 1, 2, or 3"},
                status=status.HTTP_400_BAD_REQUEST
            )

        pdf_bytes = fn.generate_pdf_with_exfil_url(exfil_url, level=level)

        response = HttpResponse(
            pdf_bytes,
            content_type="application/pdf",
            status=200,
        )
        response["Content-Disposition"] = f"attachment; filename=quantum_entanglement.pdf"
        return response
    except fn.ConfigError as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except Exception as exc:
        return _error_response(exc)
