"""
Views for code execution app
Handles streaming code generation, execution, and response
"""
import json
import time
from pathlib import Path
from django.http import StreamingHttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from qbtrain.tracers.agent_tracer import AgentTracer

from . import functions


def _error_response(exc: Exception):
    """Convert exception to NDJSON error event"""
    error_type = type(exc).__name__
    message = str(exc)
    return json.dumps({"type": "error", "message": f"{error_type}: {message}"})


def _ndjson_generator(gen):
    """Convert generator of dicts to NDJSON strings"""
    for item in gen:
        if item:
            yield json.dumps(item) + '\n'


def _process_query_stream(body: dict, file_obj=None):
    """
    Main generator for processing queries with streaming response.
    Yields NDJSON events for status, trace, message, error, done.
    """
    question = body.get('question', '').strip()
    run_as_admin = body.get('run_as_admin', False)
    model_cfg = body.get('model_config', {})
    conversation_history = body.get('conversation_history', [])

    # Validate inputs
    is_valid, error_msg = functions.validate_inputs(question)
    if not is_valid:
        yield {"type": "error", "message": error_msg}
        return

    tracer = AgentTracer()
    script_dir = functions.make_script_dir()

    # Check for file content
    file_content = None
    file_type = None
    has_csv = False
    has_document = False

    if file_obj:
        try:
            file_type = file_obj.name.split('.')[-1].lower()
            file_content = file_obj.read().decode('utf-8')

            if file_type in ['csv']:
                has_csv = True
            elif file_type in ['pdf', 'txt', 'md', 'doc', 'docx']:
                has_document = True

            yield {
                "type": "status",
                "message": f"File uploaded: {file_obj.name} ({len(file_content)} chars)"
            }
        except Exception as e:
            yield {"type": "error", "message": f"File processing error: {e}"}
            file_content = None

    try:
        # Step 1: Decide route
        yield {"type": "status", "message": "Deciding routing strategy..."}

        t0 = time.time()
        route, reasoning = functions.decide_route(
            question, has_csv, has_document, conversation_history, model_cfg, tracer=tracer
        )
        latency = round((time.time() - t0) * 1000)

        tracer.trace(
            "codeexec", "route_decision",
            route=route,
            reasoning=reasoning,
            latency_ms=latency
        )

        yield {
            "type": "trace",
            "content": tracer.get_traces()[-1]
        }
        yield {
            "type": "status",
            "message": f"Route selected: {route} - {reasoning}"
        }

        # Step 2: Route-specific processing
        if route == 'direct':
            # Direct answer without code
            yield {"type": "status", "message": "Generating direct answer..."}

            t0 = time.time()
            answer = functions.generate_direct_answer(question, conversation_history, model_cfg, tracer=tracer)
            latency = round((time.time() - t0) * 1000)

            tracer.trace(
                "codeexec", "direct_answer",
                latency_ms=latency
            )

            yield {
                "type": "trace",
                "content": tracer.get_traces()[-1]
            }
            yield {
                "type": "message",
                "content": answer
            }

        else:
            # Iterative code generation and execution
            max_iterations = 3
            tried_routes = [route]
            current_route = route
            iteration = 1
            output = None
            final_output = None

            while iteration <= max_iterations:
                yield {
                    "type": "status",
                    "message": f"Iteration {iteration}/{max_iterations}: Generating code for {current_route}..."
                }

                # Generate code
                t0 = time.time()
                code, script_path = functions.generate_code(
                    question,
                    current_route,
                    context=file_content or "",
                    previous_output=output or "",
                    iteration=iteration,
                    script_dir=script_dir,
                    client_config=model_cfg,
                    tracer=tracer
                )
                latency = round((time.time() - t0) * 1000)

                tracer.trace(
                    "codeexec", "code_generated",
                    route=current_route,
                    iteration=iteration,
                    code=code,
                    script_path=str(script_path),
                    latency_ms=latency
                )

                yield {
                    "type": "trace",
                    "content": tracer.get_traces()[-1]
                }

                # Execute code
                yield {
                    "type": "status",
                    "message": f"Iteration {iteration}: Executing code..."
                }

                t0 = time.time()
                try:
                    stdout, stderr, returncode = functions.execute_code(script_path, run_as_admin)
                    latency = round((time.time() - t0) * 1000)

                    tracer.trace(
                        "codeexec", "code_executed",
                        iteration=iteration,
                        stdout=stdout,
                        stderr=stderr,
                        returncode=returncode,
                        run_as_admin=run_as_admin,
                        latency_ms=latency
                    )

                    yield {
                        "type": "trace",
                        "content": tracer.get_traces()[-1]
                    }

                    if stderr:
                        yield {
                            "type": "warning",
                            "message": f"Iteration {iteration}: stderr: {stderr[:200]}"
                        }

                    output = stdout

                except Exception as e:
                    tracer.trace(
                        "codeexec", "code_execution_error",
                        iteration=iteration,
                        error=str(e),
                        latency_ms=round((time.time() - t0) * 1000)
                    )

                    yield {
                        "type": "trace",
                        "content": tracer.get_traces()[-1]
                    }

                    yield {
                        "type": "warning",
                        "message": f"Iteration {iteration}: Execution failed: {e}"
                    }

                    output = ""

                # Check sufficiency
                yield {
                    "type": "status",
                    "message": f"Iteration {iteration}: Evaluating output sufficiency..."
                }

                t0 = time.time()
                is_sufficient, eval_reason = functions.is_output_sufficient(question, output, model_cfg, tracer=tracer)
                latency = round((time.time() - t0) * 1000)

                tracer.trace(
                    "codeexec", "sufficiency_check",
                    iteration=iteration,
                    sufficient=is_sufficient,
                    evaluation_reason=eval_reason,
                    latency_ms=latency
                )

                yield {
                    "type": "trace",
                    "content": tracer.get_traces()[-1]
                }

                if is_sufficient:
                    yield {
                        "type": "status",
                        "message": f"Iteration {iteration}: Output is sufficient!"
                    }
                    final_output = output
                    break
                else:
                    yield {
                        "type": "status",
                        "message": f"Iteration {iteration}: {eval_reason}"
                    }

                # Pick next route for retry
                if iteration < max_iterations:
                    current_route = functions.pick_next_route(
                        question, current_route, output, has_csv, has_document,
                        tried_routes, model_cfg, tracer=tracer
                    )
                    if current_route not in tried_routes:
                        tried_routes.append(current_route)

                iteration += 1

            # Generate final answer
            if final_output is None:
                final_output = output or ""

            yield {
                "type": "status",
                "message": "Generating final answer..."
            }

            t0 = time.time()
            answer = functions.generate_final_answer(question, final_output, route, model_cfg, tracer=tracer)
            latency = round((time.time() - t0) * 1000)

            tracer.trace(
                "codeexec", "final_answer",
                latency_ms=latency
            )

            yield {
                "type": "trace",
                "content": tracer.get_traces()[-1]
            }

            yield {
                "type": "message",
                "content": answer
            }

        # Final trace summary
        total_latency = sum(t.get('latency_ms', 0) for t in tracer.get_traces())
        yield {
            "type": "trace_summary",
            "content": {
                "calls": tracer.get_traces(),
                "model": model_cfg.get('model', 'unknown'),
                "total_latency_ms": total_latency
            }
        }

        yield {"type": "done"}

    except Exception as e:
        yield {"type": "error", "message": f"Query processing error: {e}"}


@api_view(["POST"])
def query(request):
    """
    Stream code execution query processing.
    Accepts JSON or multipart form data (with optional file upload).
    Streams NDJSON response.
    """
    try:
        if request.content_type and 'multipart' in request.content_type:
            # Multipart form data
            body = {
                'question': request.POST.get('question', ''),
                'model_config': json.loads(request.POST.get('model_config', '{}')),
                'conversation_history': json.loads(request.POST.get('conversation_history', '[]')),
                'run_as_admin': request.POST.get('run_as_admin', 'false').lower() == 'true',
            }
            file_obj = request.FILES.get('file')
        else:
            # JSON body
            body = json.loads(request.body) if request.body else {}
            file_obj = None

        # Validate request
        if not body.get('question', '').strip():
            return Response(
                {"error": "question is required"},
                status=400
            )

        # Create streaming generator
        def event_stream():
            for event in _process_query_stream(body, file_obj):
                yield json.dumps(event) + '\n'

        return StreamingHttpResponse(
            event_stream(),
            content_type='application/x-ndjson'
        )

    except json.JSONDecodeError:
        return Response(
            {"error": "Invalid JSON in request body"},
            status=400
        )
    except Exception as e:
        return Response(
            {"error": f"Internal server error: {e}"},
            status=500
        )
