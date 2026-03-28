"""
Business logic for Model Theft Images app.

Loads DDPM diffusion models from HuggingFace and generates butterfly images,
streaming intermediate denoising steps to the frontend.
"""
from __future__ import annotations

import base64
import importlib
import io
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ======================== Constants ========================

MODELS_DIR = Path(__file__).parent / "tmp" / "models"

# HuggingFace DDPM butterfly models pushed from the training notebook
AVAILABLE_MODELS = {
    "modthe-original": {
        "repo_id": "qbtrain/modthe-original",
        "name": "Original Teacher",
        "description": "The original DDPM model trained on the Smithsonian Butterflies dataset.",
        "tag": "teacher",
    },
    "modthe-dist-nowm": {
        "repo_id": "qbtrain/modthe-dist-nowm",
        "name": "Distilled (No Watermark)",
        "description": "Student model distilled from the teacher without any watermarking.",
        "tag": "stolen",
    },
    "modthe-dist-spongebobwm": {
        "repo_id": "qbtrain/modthe-dist-spongebobwm",
        "name": "Distilled (Visible Watermark)",
        "description": "Student model distilled from watermarked teacher outputs (visible SpongeBob watermark).",
        "tag": "watermarked-visible",
    },
    "modthe-dist-samplewm": {
        "repo_id": "qbtrain/modthe-dist-samplewm",
        "name": "Distilled (Sample Text Watermark)",
        "description": "Student model distilled from watermarked teacher outputs (visible 'SAMPLE' text watermark).",
        "tag": "watermarked-sample",
    },
    "modthe-dist-dct": {
        "repo_id": "qbtrain/modthe-dist-dct",
        "name": "Distilled (Invisible DCT Watermark)",
        "description": "Student model distilled with invisible DCT-domain watermarking. Watermark survives distillation.",
        "tag": "watermarked-invisible",
    },
}

# Diffusion generation parameters (match notebook)
NUM_INFERENCE_STEPS = 200
NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE = "squaredcos_cap_v2"

# How often to yield intermediate images during generation
STREAM_EVERY_N_STEPS = 5


# ======================== Exceptions ========================

class ModelTheftImagesError(Exception):
    pass


class ModelNotDownloadedError(ModelTheftImagesError):
    pass


# ======================== State ========================

@dataclass
class _DownloadTask:
    repo_id: str
    local_name: str
    status: str = "queued"  # queued | downloading | done | error
    pct: Optional[float] = None
    error: str = ""


_download_lock = threading.Lock()
_download_tasks: Dict[str, _DownloadTask] = {}
_download_thread: Optional[threading.Thread] = None

# Track the currently loaded model to avoid reloading
_loaded_model_key: Optional[str] = None
_loaded_pipeline: Any = None
_model_lock = threading.Lock()


# ======================== Lazy imports ========================

def _lazy_torch():
    return importlib.import_module("torch")


def _lazy_diffusers():
    return importlib.import_module("diffusers")


def _lazy_pil():
    return importlib.import_module("PIL.Image")


# ======================== Model Management ========================

def list_models() -> List[Dict[str, Any]]:
    """Return available models with their download status."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for key, info in AVAILABLE_MODELS.items():
        local_path = MODELS_DIR / key
        downloaded = local_path.exists() and (local_path / "unet").exists()
        result.append({
            "key": key,
            "repo_id": info["repo_id"],
            "name": info["name"],
            "description": info["description"],
            "tag": info["tag"],
            "downloaded": downloaded,
        })
    return result


def _download_single_model(key: str, info: Dict[str, Any], task: _DownloadTask):
    """Download a single model from HuggingFace."""
    diffusers = _lazy_diffusers()
    DDPMPipeline = diffusers.DDPMPipeline

    local_path = MODELS_DIR / key
    task.status = "downloading"
    task.pct = 0

    try:
        # Download using diffusers pipeline
        pipe = DDPMPipeline.from_pretrained(info["repo_id"])
        pipe.save_pretrained(str(local_path))
        task.status = "done"
        task.pct = 100
    except Exception as e:
        task.status = "error"
        task.error = str(e)
        logger.error("Failed to download %s: %s", key, e)


def _download_worker():
    """Background thread that downloads all models sequentially."""
    global _download_tasks

    total = len(_download_tasks)
    completed = 0

    for key, task in _download_tasks.items():
        if task.status in ("done",):
            completed += 1
            continue

        info = AVAILABLE_MODELS.get(key)
        if not info:
            task.status = "error"
            task.error = "Unknown model"
            completed += 1
            continue

        # Check if already downloaded
        local_path = MODELS_DIR / key
        if local_path.exists() and (local_path / "unet").exists():
            task.status = "done"
            task.pct = 100
            completed += 1
            continue

        _download_single_model(key, info, task)
        completed += 1


def download_all_models() -> Dict[str, Any]:
    """Queue download of all models. Returns immediately."""
    global _download_thread, _download_tasks

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with _download_lock:
        # Check if already running
        if _download_thread is not None and _download_thread.is_alive():
            return {"message": "Download already in progress", "status": get_download_status()}

        # Build tasks
        _download_tasks = {}
        for key, info in AVAILABLE_MODELS.items():
            local_path = MODELS_DIR / key
            already = local_path.exists() and (local_path / "unet").exists()
            _download_tasks[key] = _DownloadTask(
                repo_id=info["repo_id"],
                local_name=key,
                status="done" if already else "queued",
                pct=100 if already else None,
            )

        _download_thread = threading.Thread(target=_download_worker, daemon=True)
        _download_thread.start()

    return {"message": "Download started", "status": get_download_status()}


def get_download_status() -> Dict[str, Any]:
    """Return current download status for all models."""
    tasks = []
    for key, task in _download_tasks.items():
        tasks.append({
            "model": AVAILABLE_MODELS.get(key, {}).get("name", key),
            "model_id": AVAILABLE_MODELS.get(key, {}).get("repo_id", key),
            "local_name": key,
            "status": task.status,
            "pct": task.pct,
            "error": task.error,
        })

    downloading = [t for t in tasks if t["status"] == "downloading"]
    queued = [t for t in tasks if t["status"] == "queued"]
    done = [t for t in tasks if t["status"] == "done"]
    errors = [t for t in tasks if t["status"] == "error"]

    return {
        "downloading": downloading,
        "queued": queued,
        "done": done,
        "errors": errors,
        "tasks": tasks,
    }


# ======================== GPU Model Loading ========================

def _unload_model():
    """Unload the currently loaded model from GPU to free VRAM."""
    global _loaded_model_key, _loaded_pipeline

    if _loaded_pipeline is not None:
        torch = _lazy_torch()
        try:
            _loaded_pipeline.to("cpu")
            del _loaded_pipeline
            _loaded_pipeline = None
            _loaded_model_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Error unloading model: %s", e)
            _loaded_pipeline = None
            _loaded_model_key = None


def _load_model(model_key: str):
    """Load a model onto GPU (or CPU). Unloads previous model first."""
    global _loaded_model_key, _loaded_pipeline

    if _loaded_model_key == model_key and _loaded_pipeline is not None:
        return _loaded_pipeline

    _unload_model()

    torch = _lazy_torch()
    diffusers = _lazy_diffusers()
    DDPMPipeline = diffusers.DDPMPipeline

    local_path = MODELS_DIR / model_key
    if not local_path.exists() or not (local_path / "unet").exists():
        raise ModelNotDownloadedError(f"Model '{model_key}' is not downloaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(str(local_path))
    pipeline.to(device)

    _loaded_model_key = model_key
    _loaded_pipeline = pipeline

    return pipeline


# ======================== Image Generation ========================

def _pil_to_base64(img) -> str:
    """Convert a PIL image to base64 encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_image_stream(
    model_key: str,
    num_images: int = 1,
    seed: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate images using DDPM with streaming intermediate steps.

    Yields NDJSON events:
      - {"type": "status", "message": "..."}
      - {"type": "step", "step": N, "total_steps": M, "image": "<base64>"}
      - {"type": "final", "images": ["<base64>", ...]}
      - {"type": "done"}
      - {"type": "error", "message": "..."}
    """
    torch = _lazy_torch()
    diffusers = _lazy_diffusers()
    DDPMScheduler = diffusers.DDPMScheduler
    Image = _lazy_pil()

    yield {"type": "status", "message": f"Loading model '{AVAILABLE_MODELS[model_key]['name']}'..."}

    with _model_lock:
        try:
            pipeline = _load_model(model_key)
        except ModelNotDownloadedError as e:
            yield {"type": "error", "message": str(e)}
            return
        except Exception as e:
            yield {"type": "error", "message": f"Failed to load model: {e}"}
            return

        yield {"type": "status", "message": "Generating image..."}

        try:
            device = pipeline.device
            unet = pipeline.unet
            scheduler = pipeline.scheduler

            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)

            num_images = min(num_images, 4)  # Cap at 4

            # Start from pure noise
            sample_size = unet.config.sample_size
            in_channels = unet.config.in_channels
            sample = torch.randn(
                num_images, in_channels, sample_size, sample_size,
                device=device,
                generator=generator,
            )

            # Set up scheduler timesteps
            scheduler.set_timesteps(NUM_INFERENCE_STEPS)

            total_steps = len(scheduler.timesteps)
            stream_interval = max(1, total_steps // 20)  # ~20 intermediate frames

            for i, t in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    noise_pred = unet(sample, t).sample
                    sample = scheduler.step(noise_pred, t, sample).prev_sample

                step_num = i + 1

                # Stream intermediate images at intervals
                if step_num % stream_interval == 0 or step_num == total_steps:
                    # Convert current batch to images for preview
                    preview = (sample / 2 + 0.5).clamp(0, 1)
                    preview = preview.cpu().permute(0, 2, 3, 1).numpy()
                    preview = (preview * 255).round().astype("uint8")

                    # Send all batch images for preview
                    preview_images = []
                    for idx in range(preview.shape[0]):
                        img = Image.fromarray(preview[idx])
                        preview_images.append(_pil_to_base64(img))

                    yield {
                        "type": "step",
                        "step": step_num,
                        "total_steps": total_steps,
                        "image": preview_images[0],
                        "images": preview_images,
                    }

            # Final images
            final = (sample / 2 + 0.5).clamp(0, 1)
            final = final.cpu().permute(0, 2, 3, 1).numpy()
            final = (final * 255).round().astype("uint8")

            final_images = []
            for idx in range(final.shape[0]):
                img = Image.fromarray(final[idx])
                final_images.append(_pil_to_base64(img))

            yield {
                "type": "final",
                "images": final_images,
                "model_key": model_key,
                "model_name": AVAILABLE_MODELS[model_key]["name"],
            }

        except Exception as e:
            logger.error("Generation error: %s", e)
            yield {"type": "error", "message": f"Generation failed: {e}"}

    yield {"type": "done"}


def unload_current_model() -> Dict[str, Any]:
    """Explicitly unload the current model from GPU."""
    with _model_lock:
        was_loaded = _loaded_model_key
        _unload_model()
    return {"message": f"Unloaded model '{was_loaded}'" if was_loaded else "No model was loaded."}
