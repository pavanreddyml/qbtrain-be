"""
Views for Model Theft app.

All session state lives server-side.  Training streams NDJSON metrics.
"""
from __future__ import annotations

import json

from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import functions as fn


# ---- GET endpoints ----

@api_view(["GET"])
def get_session(request):
    """Return the current session state."""
    return Response(fn.get_session_state())


@api_view(["GET"])
def get_methods(request):
    """Return available distillation methods and their params."""
    return Response(fn.get_methods())


# ---- POST endpoints ----

@api_view(["POST"])
def configure(request):
    """Configure the session (downloads student model, locks settings)."""
    try:
        body = request.data
        teacher_config = body.get("teacher_config", {})
        student_model_name = body.get("student_model_name", "")
        distillation_method = body.get("distillation_method", "standard")
        distillation_params = body.get("distillation_params", {})

        if not student_model_name:
            return Response(
                {"error": "student_model_name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not teacher_config.get("type") or not teacher_config.get("model"):
            return Response(
                {"error": "teacher_config must include type and model"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = fn.configure_session(
            teacher_config=teacher_config,
            student_model_name=student_model_name,
            distillation_method=distillation_method,
            distillation_params=distillation_params,
        )
        return Response(result)

    except fn.SessionAlreadyConfiguredError as e:
        return Response({"error": str(e)}, status=status.HTTP_409_CONFLICT)
    except fn.StudentDownloadError as e:
        return Response({"error": str(e)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    except fn.ModelTheftError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response(
            {"error": f"Internal error: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def reset(request):
    """Reset the session and delete the student model."""
    try:
        result = fn.reset_session()
        return Response(result)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def train(request):
    """Train the student model.  Streams NDJSON metrics per step."""
    try:
        body = request.data
        steps = body.get("steps", 5)
        training_prompts = body.get("training_prompts", None)

        if not isinstance(steps, int) or steps < 1:
            return Response(
                {"error": "steps must be a positive integer"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if steps > 100:
            return Response(
                {"error": "Maximum 100 steps per training run"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        def event_stream():
            try:
                yield json.dumps({
                    "type": "status",
                    "message": f"Starting training for {steps} steps...",
                }) + "\n"

                for metrics in fn.train_stream(steps, training_prompts):
                    yield json.dumps({
                        "type": "metrics",
                        "content": metrics,
                    }) + "\n"

                yield json.dumps({
                    "type": "status",
                    "message": "Training complete! Model saved.",
                }) + "\n"
                yield json.dumps({"type": "done"}) + "\n"

            except fn.SessionNotConfiguredError as e:
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "message": f"Training error: {e}"}) + "\n"

        return StreamingHttpResponse(
            event_stream(),
            content_type="application/x-ndjson",
        )

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def test(request):
    """Test the student model (and optionally the teacher for comparison)."""
    try:
        body = request.data
        prompt = body.get("prompt", "").strip()
        max_new_tokens = body.get("max_new_tokens", 150)
        compare_teacher = body.get("compare_teacher", False)

        if not prompt:
            return Response(
                {"error": "prompt is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = {"student": fn.test_student(prompt, max_new_tokens=max_new_tokens)}

        if compare_teacher:
            result["teacher"] = fn.test_teacher(prompt)

        return Response(result)

    except fn.SessionNotConfiguredError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ---- Student model management ----

@api_view(["GET"])
def student_models(request):
    """List available student models in the app's models directory."""
    try:
        models = fn.list_student_models()
        return Response({"models": models})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
def download_student(request):
    """Download a HuggingFace model for use as student."""
    try:
        model_id = (request.data.get("model_id") or "").strip()
        if not model_id:
            return Response(
                {"error": "model_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        result = fn.download_student_model(model_id)
        return Response(result, status=status.HTTP_202_ACCEPTED)
    except fn.ModelTheftError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def download_status(request):
    """Get download status for student models."""
    try:
        return Response(fn.get_download_status())
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["DELETE"])
def delete_student(request):
    """Delete a student model from the app's models directory."""
    try:
        local_name = (request.data.get("local_name") or request.query_params.get("local_name") or "").strip()
        if not local_name:
            return Response(
                {"error": "local_name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        result = fn.delete_student_model(local_name)
        return Response(result)
    except fn.ModelTheftError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
