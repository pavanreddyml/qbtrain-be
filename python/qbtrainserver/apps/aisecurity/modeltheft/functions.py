"""
Business logic for Model Theft app.

Uses the qbtrain distillation module for knowledge transfer from a
teacher LLM to a small HuggingFace student model.  All session state
lives server-side (no localStorage on the client).
"""
from __future__ import annotations

import importlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from qbtrain.ai.llm import LLMClientRegistry, HuggingFaceClient
from qbtrain.distillation import (
    StandardDistiller,
    AttentionDistiller,
    TaskDistiller,
    FeatureDistiller,
    DistillationMetrics,
)

# ======================== Constants ========================

MODELS_DIR = Path(__file__).parent / "tmp" / "models"

DISTILLATION_METHODS = {
    "standard": {
        "name": "Standard Distillation",
        "description": "Response-based distillation using cross-entropy loss on teacher completions (Hinton et al. 2015).",
        "class": StandardDistiller,
        "params": {
            "learning_rate": {"default": 5e-5, "min": 1e-6, "max": 1e-2, "type": "float"},
            "batch_size": {"default": 4, "min": 1, "max": 16, "type": "int"},
            "temperature": {"default": 1.0, "min": 0.1, "max": 5.0, "type": "float"},
            "max_seq_length": {"default": 256, "min": 32, "max": 512, "type": "int"},
        },
    },
    "attention": {
        "name": "Attention Transfer",
        "description": "Next-token prediction combined with attention entropy regularization for better internal representations.",
        "class": AttentionDistiller,
        "params": {
            "learning_rate": {"default": 5e-5, "min": 1e-6, "max": 1e-2, "type": "float"},
            "batch_size": {"default": 4, "min": 1, "max": 16, "type": "int"},
            "attention_weight": {"default": 0.5, "min": 0.0, "max": 2.0, "type": "float"},
            "max_seq_length": {"default": 256, "min": 32, "max": 512, "type": "int"},
        },
    },
    "task": {
        "name": "Task-Specific Distillation",
        "description": "Teacher generates structured input/output pairs; student learns task-specific behaviour.",
        "class": TaskDistiller,
        "params": {
            "learning_rate": {"default": 5e-5, "min": 1e-6, "max": 1e-2, "type": "float"},
            "batch_size": {"default": 4, "min": 1, "max": 16, "type": "int"},
            "max_seq_length": {"default": 256, "min": 32, "max": 512, "type": "int"},
        },
    },
    "feature": {
        "name": "Feature Distillation",
        "description": "FitNets-style distillation with MSE loss on projected hidden states for richer representations.",
        "class": FeatureDistiller,
        "params": {
            "learning_rate": {"default": 5e-5, "min": 1e-6, "max": 1e-2, "type": "float"},
            "batch_size": {"default": 4, "min": 1, "max": 16, "type": "int"},
            "feature_weight": {"default": 0.5, "min": 0.0, "max": 2.0, "type": "float"},
            "max_seq_length": {"default": 256, "min": 32, "max": 512, "type": "int"},
        },
    },
}

DEFAULT_TRAINING_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "How does a neural network learn?",
    "What are the main types of machine learning algorithms?",
    "Explain backpropagation in neural networks.",
    "What is overfitting and how can it be prevented?",
    "Describe the difference between classification and regression.",
    "What is transfer learning and why is it useful?",
    "Explain the concept of gradient descent.",
    "What are embeddings in natural language processing?",
]


# ======================== Exceptions ========================

class ModelTheftError(Exception):
    pass


class SessionNotConfiguredError(ModelTheftError):
    pass


class SessionAlreadyConfiguredError(ModelTheftError):
    pass


class StudentDownloadError(ModelTheftError):
    pass


# ======================== Session State ========================

@dataclass
class SessionState:
    """Server-side session state for the model theft app."""
    configured: bool = False
    teacher_config: Dict[str, Any] = field(default_factory=dict)
    student_model_name: str = ""
    distillation_method: str = ""
    distillation_params: Dict[str, Any] = field(default_factory=dict)
    distiller: Any = None
    total_steps_trained: int = 0
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    student_downloaded: bool = False

    def to_dict(self):
        return {
            "configured": self.configured,
            "teacher_config": self.teacher_config,
            "student_model_name": self.student_model_name,
            "distillation_method": self.distillation_method,
            "distillation_params": self.distillation_params,
            "total_steps_trained": self.total_steps_trained,
            "training_history": self.training_history,
            "student_downloaded": self.student_downloaded,
        }


# Single session (educational app, one user at a time)
_session = SessionState()


# ======================== Helpers ========================

def _lazy_torch():
    return importlib.import_module("torch")


def _build_teacher_client(teacher_config: Dict[str, Any]):
    """Build a teacher LLM client from config."""
    client_type = teacher_config.get("type", "ollama")
    try:
        client_class = LLMClientRegistry.get(client_type)
    except KeyError:
        raise ModelTheftError(f"Unknown LLM client type: {client_type}")

    init_params = {
        k: v for k, v in teacher_config.items()
        if k not in [
            "type", "temperature", "max_tokens", "top_p", "top_k",
            "frequency_penalty", "presence_penalty",
        ]
    }
    try:
        return client_class(**init_params)
    except Exception as e:
        raise ModelTheftError(f"Failed to initialise teacher client: {e}")


def _reset_model_weights(model):
    """Reinitialise student weights to random so the model starts from scratch."""
    torch = _lazy_torch()
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)


# ======================== Public API ========================

def get_session_state() -> Dict[str, Any]:
    return _session.to_dict()


def get_methods() -> Dict[str, Any]:
    """Return available distillation methods and their configurable params."""
    methods = {}
    for key, info in DISTILLATION_METHODS.items():
        methods[key] = {
            "name": info["name"],
            "description": info["description"],
            "params": info["params"],
        }
    return methods


def configure_session(
    teacher_config: Dict[str, Any],
    student_model_name: str,
    distillation_method: str,
    distillation_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Configure and lock the session.  Downloads the student model."""
    global _session

    if _session.configured:
        raise SessionAlreadyConfiguredError(
            "Session already configured. Reset to reconfigure."
        )

    if distillation_method not in DISTILLATION_METHODS:
        raise ModelTheftError(f"Unknown distillation method: {distillation_method}")

    # Ensure models dir exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    teacher = _build_teacher_client(teacher_config)

    method_info = DISTILLATION_METHODS[distillation_method]
    distiller_class = method_info["class"]
    output_dir = str(MODELS_DIR / student_model_name.replace("/", "_"))

    # Merge provided params with defaults
    merged_params = {}
    for key, param_info in method_info["params"].items():
        if key in distillation_params:
            merged_params[key] = distillation_params[key]
        else:
            merged_params[key] = param_info["default"]

    try:
        distiller = distiller_class(
            teacher=teacher,
            student_model_name=student_model_name,
            output_dir=output_dir,
            **merged_params,
        )
    except Exception as e:
        raise StudentDownloadError(f"Failed to set up student model: {e}")

    # Reset weights so the student starts from scratch (bad quality)
    _reset_model_weights(distiller._model)

    _session = SessionState(
        configured=True,
        teacher_config=teacher_config,
        student_model_name=student_model_name,
        distillation_method=distillation_method,
        distillation_params=merged_params,
        distiller=distiller,
        student_downloaded=True,
    )
    return _session.to_dict()


def reset_session() -> Dict[str, Any]:
    """Reset the session and delete the student model files."""
    global _session

    if _session.student_model_name:
        model_dir = MODELS_DIR / _session.student_model_name.replace("/", "_")
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    _session = SessionState()
    return _session.to_dict()


def train_stream(
    steps: int,
    training_prompts: Optional[List[str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield per-step training metrics as dicts (for NDJSON streaming)."""
    global _session

    if not _session.configured or _session.distiller is None:
        raise SessionNotConfiguredError("Session not configured.")

    if not training_prompts:
        training_prompts = DEFAULT_TRAINING_PROMPTS

    distiller = _session.distiller

    for metrics in distiller.distill_stream(training_prompts, steps=steps):
        _session.total_steps_trained += 1
        entry = {
            "step": metrics.step,
            "total_steps": _session.total_steps_trained,
            "loss": round(metrics.loss, 6),
            "learning_rate": metrics.learning_rate,
            "perplexity": round(metrics.perplexity, 4),
            "elapsed_ms": metrics.elapsed_ms,
            "extras": metrics.extras,
        }
        _session.training_history.append(entry)
        yield entry

    # Save the model after training
    distiller.save()


def test_student(prompt: str, max_new_tokens: int = 150) -> Dict[str, Any]:
    """Generate text with the student model."""
    if not _session.configured or _session.distiller is None:
        raise SessionNotConfiguredError("Session not configured.")

    t0 = time.perf_counter()
    response = _session.distiller.generate(prompt, max_new_tokens=max_new_tokens)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "prompt": prompt,
        "student_response": response,
        "elapsed_ms": elapsed_ms,
        "total_steps_trained": _session.total_steps_trained,
    }


def test_teacher(prompt: str) -> Dict[str, Any]:
    """Generate text with the teacher model for comparison."""
    if not _session.configured or _session.distiller is None:
        raise SessionNotConfiguredError("Session not configured.")

    t0 = time.perf_counter()
    response = _session.distiller.teacher.response(prompt)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "prompt": prompt,
        "teacher_response": response,
        "elapsed_ms": elapsed_ms,
    }


# ======================== Student Model Management ========================

def list_student_models() -> List[str]:
    """List locally available student models in MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    client = HuggingFaceClient(models_dir=str(MODELS_DIR))
    return client.list_models()


def download_student_model(model_id: str) -> Dict[str, Any]:
    """Download a HuggingFace model to MODELS_DIR for use as student."""
    from qbtrain.distillation.base_distiller import MAX_STUDENT_PARAMS

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate param count before downloading
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        safetensors = getattr(info, "safetensors", None)
        params = None
        if safetensors and hasattr(safetensors, "total"):
            params = safetensors.total
        if params and params > MAX_STUDENT_PARAMS:
            raise ModelTheftError(
                f"Model {model_id} has {params:,} parameters "
                f"(max {MAX_STUDENT_PARAMS:,}). Choose a smaller model."
            )
    except ImportError:
        pass  # huggingface_hub not available, skip validation
    except ModelTheftError:
        raise
    except Exception:
        pass  # model_info may fail for some models, allow download attempt

    HuggingFaceClient.request_download(
        model_id=model_id,
        local_dir=str(MODELS_DIR / model_id.replace("/", "_")),
    )
    return {
        "message": "Download queued",
        "model_id": model_id,
        "status": HuggingFaceClient.download_status(),
    }


def get_download_status() -> Dict[str, Any]:
    """Get HuggingFace download status."""
    return HuggingFaceClient.download_status()


def delete_student_model(local_name: str) -> Dict[str, Any]:
    """Delete a student model from MODELS_DIR."""
    model_path = MODELS_DIR / local_name
    if not model_path.exists():
        raise ModelTheftError(f"Model '{local_name}' not found.")
    if model_path.is_dir():
        shutil.rmtree(model_path)
    return {"message": f"Deleted '{local_name}'."}
