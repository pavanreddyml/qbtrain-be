# apps/aisecurity/imageadvattacks/functions.py
"""
Image Adversarial Attacks — white-box perturbation attacks on ImageNet classifiers.

Implements FGSM, PGD, Carlini–Wagner (L2), DeepFool, and SmoothFool against
HuggingFace ResNet classifiers (transformers `ResNetForImageClassification`,
pretrained on ImageNet), in plain PyTorch. Model weights are fetched via the
global HuggingFace client (download button) into the shared ./hf_models store.

The logic is modeled on the canonical white-box attacks in the `adversarial-lab`
package (FastSignGradientMethod / ProjectedGradientDescent / CW / DeepFool /
SmoothFool) but is a clean, self-contained re-implementation — no dependency on
that library.

Architecture mirrors the cursedpixels app: each attack runs in a background
thread, pushes per-epoch events onto a queue, and supports a cooperative stop
signal. Per-epoch frames (original/gradient/noise/perturbed images + top-k
predictions) are emitted as they are computed and also retained in a rolling
window of the most recent ``MAX_STORED_EPOCHS`` epochs (older frames are dropped
to bound memory).

The attack operates in pixel space x ∈ [0, 1]; ImageNet normalization is folded
into the model's forward pass so that ε / α are expressed directly in image
units (n/255).
"""
from __future__ import annotations

import base64
import io
import math
import os
import threading
import queue
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import numpy as np
from PIL import Image


# ============================================================
# Exceptions
# ============================================================
class ValidationError(Exception):
    pass


class StateError(Exception):
    pass


class ModelNotDownloadedError(Exception):
    pass


# ============================================================
# Config / catalog
# ============================================================
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Rolling window of stored per-epoch frames (images + gradients).
MAX_STORED_EPOCHS = 50

# Provider for the classifier weights.
MODEL_PROVIDER = "huggingface"

# HuggingFace ResNet classifiers pretrained on ImageNet-1k
# (transformers `ResNetForImageClassification`). The id IS the HF repo.
MODELS: List[Dict[str, str]] = [
    {"id": "microsoft/resnet-18", "hf_repo": "microsoft/resnet-18", "label": "ResNet-18 (ImageNet)", "description": "Smallest / fastest ResNet. Good for quick iteration on CPU."},
    {"id": "microsoft/resnet-50", "hf_repo": "microsoft/resnet-50", "label": "ResNet-50 (ImageNet)", "description": "The standard ImageNet baseline."},
    {"id": "microsoft/resnet-101", "hf_repo": "microsoft/resnet-101", "label": "ResNet-101 (ImageNet)", "description": "Deeper, more robust — slower per step."},
    {"id": "microsoft/resnet-152", "hf_repo": "microsoft/resnet-152", "label": "ResNet-152 (ImageNet)", "description": "Deepest ResNet. Strongest, slowest."},
]
DEFAULT_MODEL_ID = "microsoft/resnet-50"
_MODEL_IDS = {m["id"] for m in MODELS}


def _model_by_id(model_id: str) -> Dict[str, str]:
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise ValidationError(f"unknown model_id: {model_id}")

ATTACKS: List[Dict[str, str]] = [
    {"id": "fgsm", "label": "FGSM", "description": "Fast Gradient Sign Method — one signed-gradient step bounded by ε."},
    {"id": "pgd", "label": "PGD", "description": "Projected Gradient Descent — iterative FGSM projected back into the ε-ball."},
    {"id": "cw", "label": "Carlini–Wagner (L2)", "description": "Optimization attack minimizing ‖δ‖₂ + c·f(x+δ) via Adam in tanh space."},
    {"id": "deepfool", "label": "DeepFool", "description": "Iteratively pushes the image across the nearest linearized decision boundary."},
    {"id": "smoothfool", "label": "SmoothFool", "description": "DeepFool with low-pass (Gaussian) smoothed perturbations."},
]
VALID_ATTACKS = {a["id"] for a in ATTACKS}


def _default_params() -> Dict[str, Any]:
    """Default hyperparameters spanning every attack (frontend picks the subset)."""
    return {
        "epsilon": 8.0 / 255.0,       # L_inf bound (FGSM, PGD)
        "alpha": 2.0 / 255.0,         # PGD step size
        "num_steps": 40,              # iterations (PGD/CW/DeepFool/SmoothFool)
        "random_start": True,         # PGD random init inside the ε-ball
        "c": 1.0,                     # CW trade-off constant
        "kappa": 0.0,                 # CW confidence margin
        "lr": 0.01,                   # CW Adam learning rate
        "overshoot": 0.02,            # DeepFool / SmoothFool overshoot
        "num_candidate_classes": 10,  # DeepFool / SmoothFool candidate classes
        "sigma": 1.0,                 # SmoothFool Gaussian smoothing sigma
        "targeted": False,            # targeted vs untargeted
        "target_class": 0,            # ImageNet class index when targeted
        "target_confidence": 0.9,     # early-stop confidence on the target class
        "early_stopping": True,       # stop as soon as the attack succeeds
        "step_delay": 0.8,            # per-epoch pacing (s) so fast GPUs don't blur the UI
    }


# ============================================================
# Session state
# ============================================================
@dataclass
class AttackSession:
    job_id: str
    config: Dict[str, Any]
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    events: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    status: str = "pending"  # pending | running | stopped | done | error
    error: Optional[str] = None
    # Rolling window of the most recent per-epoch frames (bounds memory).
    frames: "deque[Dict[str, Any]]" = field(default_factory=lambda: deque(maxlen=MAX_STORED_EPOCHS))
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


_SESSION_LOCK = threading.Lock()
_SESSIONS: Dict[str, AttackSession] = {}
_CURRENT_JOB_ID: Optional[str] = None


def _set_current(job_id: Optional[str]) -> None:
    global _CURRENT_JOB_ID
    _CURRENT_JOB_ID = job_id


def get_session(job_id: str) -> Optional[AttackSession]:
    return _SESSIONS.get(job_id)


# ============================================================
# Image helpers
# ============================================================
def _decode_image_b64(image_b64: str) -> Image.Image:
    if not image_b64:
        raise ValidationError("image_b64 is required")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValidationError(f"Could not decode base64 image: {exc}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValidationError(f"Could not open image: {exc}")


def _center_crop_resize(img: Image.Image, side: int = INPUT_SIZE) -> Image.Image:
    """Resize the shorter edge to ``side`` then center-crop to ``side x side``."""
    w, h = img.size
    if w == 0 or h == 0:
        return img.resize((side, side))
    scale = side / float(min(w, h))
    new_w = max(side, int(round(w * scale)))
    new_h = max(side, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - side) // 2
    top = (new_h - side) // 2
    return resized.crop((left, top, left + side, top + side))


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _np_to_pil_uint8(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _chw_to_display(t) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] tensor in [0,1] -> HxWx3 numpy in [0,1]."""
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def _grad_to_display(g) -> np.ndarray:
    """Min-max normalize a gradient/perturbation tensor to [0,1] for display."""
    x = g.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    gmin, gmax = float(x.min()), float(x.max())
    if (gmax - gmin) < 1e-12:
        x = x * 0.0
    else:
        x = (x - gmin) / (gmax - gmin)
    return x.permute(1, 2, 0).numpy()


def _delta_true_display(delta) -> np.ndarray:
    """True-scale perturbation: 0.5-gray + δ, clamped to [0,1] (no stretching)."""
    x = delta.detach().float()
    if x.dim() == 4:
        x = x[0]
    return (x + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _load_categories() -> List[str]:
    """ImageNet-1k class names from a downloaded HuggingFace ResNet config
    (id2label, config.json only). Returns [] if no model is downloaded yet."""
    try:
        from transformers import AutoConfig
    except Exception:
        return []
    for m in MODELS:
        local_dir = _local_dir_path(m["id"])
        if local_dir is None:
            continue
        try:
            cfg = AutoConfig.from_pretrained(str(local_dir), local_files_only=True)
            id2label = getattr(cfg, "id2label", None)
            if id2label:
                return [id2label[i] for i in sorted(id2label.keys(), key=int)]
        except Exception:
            continue
    return []


def _load_sample_b64() -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__), "assets", "panda.jpg")
    try:
        img = Image.open(path).convert("RGB")
        img = _center_crop_resize(img, INPUT_SIZE)
        return _pil_to_b64(img)
    except Exception:
        return None


# ============================================================
# Lazy model holder (load once per process, per model id)
# ============================================================
class ModelBundle:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.device = None
        self.categories: List[str] = []
        # Normalization (folded into the differentiable forward).
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def load(self):
        if self.model is not None:
            return self
        # Imports kept inside .load() so the module still imports when torch /
        # transformers are not installed (mirrors the other vision apps).
        import torch
        from transformers import AutoModelForImageClassification, AutoImageProcessor

        local_dir = _local_dir_path(self.model_id)
        if local_dir is None:
            raise ModelNotDownloadedError(
                f"Model '{self.model_id}' is not downloaded. Download it in Settings first."
            )
        src = str(local_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load from the locally-downloaded snapshot (governed by the download button).
        model = AutoModelForImageClassification.from_pretrained(src, local_files_only=True)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False

        # Pull normalization stats from the processor when available.
        try:
            proc = AutoImageProcessor.from_pretrained(src, local_files_only=True)
            if getattr(proc, "image_mean", None) and getattr(proc, "image_std", None):
                self.mean = tuple(proc.image_mean)
                self.std = tuple(proc.image_std)
        except Exception:
            pass

        id2label = getattr(model.config, "id2label", {}) or {}
        self.categories = [id2label[i] for i in sorted(id2label.keys(), key=int)] if id2label else []
        self.model = model
        self.device = device
        return self


_MODEL_LOCK = threading.Lock()
_MODELS: Dict[str, ModelBundle] = {}


def _get_model(model_id: str) -> ModelBundle:
    if not _is_downloaded(model_id):
        raise ModelNotDownloadedError(
            f"Model '{model_id}' is not downloaded. Download it in Settings first."
        )
    with _MODEL_LOCK:
        bundle = _MODELS.get(model_id)
        if bundle is None:
            bundle = ModelBundle(model_id).load()
            _MODELS[model_id] = bundle
        return bundle


# ============================================================
# Model download status
# ------------------------------------------------------------
# Downloads themselves are handled by the GLOBAL HuggingFace client endpoints
# (POST /api/clients/hf/download/, GET /api/clients/hf/status/) which fetch into
# the shared ./hf_models store. This module only needs to (a) tell whether a
# model is present locally and (b) load it from that store.
# ============================================================
def _hf_models_dir():
    """The global HuggingFace local models directory (./hf_models by default)."""
    try:
        from qbtrain.ai.llm import HuggingFaceClient
        return HuggingFaceClient().models_dir
    except Exception:
        from pathlib import Path
        return Path("./hf_models")


def _local_dir_path(model_id: str):
    """Path to the locally-downloaded snapshot, or None if not present.

    Mirrors HuggingFaceClient's naming (repo id, or repo with '/'→'__')."""
    base = _hf_models_dir()
    for cand in (base / model_id, base / model_id.replace("/", "__")):
        try:
            if cand.exists() and cand.is_dir():
                return cand
        except Exception:
            pass
    return None


def _is_downloaded(model_id: str) -> bool:
    d = _local_dir_path(model_id)
    if d is None:
        return False
    has_config = (d / "config.json").exists()
    has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
    return has_config and has_weights


def list_models() -> List[Dict[str, Any]]:
    """Model catalog annotated with local-download state. Live download progress
    is reported by the global /api/clients/hf/status/ endpoint (the frontend
    matches entries by model_id)."""
    return [
        {**m, "provider": MODEL_PROVIDER, "downloaded": _is_downloaded(m["id"])}
        for m in MODELS
    ]


# ============================================================
# Differentiable forward (normalization folded in)
# ============================================================
def _make_forward(model, device, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    import torch
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def forward(x):
        # transformers ImageClassification models return an output object.
        return model((x - mean_t) / std_t).logits

    return forward


def _gaussian_kernel(sigma: float, channels: int, device):
    """Depthwise Gaussian blur kernel for SmoothFool perturbation smoothing."""
    import torch
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g1 = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    k2 = torch.outer(g1, g1)
    k2 = k2 / k2.sum()
    kernel = k2.view(1, 1, *k2.shape).repeat(channels, 1, 1, 1)
    return kernel, radius


def _smooth(t, kernel, radius):
    import torch
    import torch.nn.functional as F
    c = t.shape[1]
    padded = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=c)


# ============================================================
# Prediction helpers
# ============================================================
def _topk_predictions(forward, x, categories, k: int = 10):
    import torch
    with torch.no_grad():
        probs = torch.softmax(forward(x), dim=1)[0]
    vals, idxs = probs.topk(k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        label = categories[i] if i < len(categories) else f"class {i}"
        out.append({"idx": int(i), "label": label, "prob": float(v)})
    return out


def _label_name(categories, idx) -> str:
    return categories[idx] if 0 <= idx < len(categories) else f"class {idx}"


# ============================================================
# Worker
# ============================================================
def _emit(session: AttackSession, event: Dict[str, Any]):
    session.events.put(event)


def _pace(session: AttackSession, delay: float):
    """Throttle an epoch to ~`delay` seconds so a fast GPU doesn't blur the UI.
    Sleeps in small slices so a stop request is honored promptly."""
    if not delay or delay <= 0:
        return
    end = time.time() + delay
    while time.time() < end:
        if session.stop_event.is_set():
            return
        time.sleep(min(0.05, max(0.0, end - time.time())))


def _frame_payload(session, attack, epoch, x_orig, x_adv, grad, loss, extra,
                   forward, categories, topk, orig_idx=None):
    """Build (and store) one per-epoch frame, then emit it as a snapshot event."""
    import torch
    delta = (x_adv - x_orig).detach()
    dflat = delta.flatten()
    l2 = float(dflat.norm(p=2).item())
    l1 = float(dflat.abs().sum().item())
    linf = float(delta.abs().max().item())
    mean_abs = float(delta.abs().mean().item())
    dmax = float(delta.max().item())
    dmin = float(delta.min().item())
    mse = float((delta ** 2).mean().item())
    psnr = None if mse <= 1e-12 else float(-10.0 * math.log10(mse))  # MAX_I = 1.0

    # Single forward for predictions + the original class's current probability.
    with torch.no_grad():
        probs = torch.softmax(forward(x_adv), dim=1)[0]
    vals, idxs = probs.topk(topk)
    preds = [
        {"idx": int(i), "label": _label_name(categories, i), "prob": float(v)}
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]
    top1 = preds[0] if preds else {"idx": -1, "label": "—", "prob": 0.0}
    orig_prob = (
        float(probs[orig_idx].item())
        if orig_idx is not None and 0 <= orig_idx < probs.shape[0] else None
    )

    frame = {
        "type": "snapshot",
        "epoch": epoch,
        "loss": float(loss) if loss is not None else None,
        "l2": l2, "l1": l1, "linf": linf, "mean_abs": mean_abs,
        "dmax": dmax, "dmin": dmin, "mse": mse, "psnr": psnr,
        "grad_b64": _pil_to_b64(_np_to_pil_uint8(_grad_to_display(grad))) if grad is not None else None,
        "noise_b64": _pil_to_b64(_np_to_pil_uint8(_grad_to_display(delta))),
        "noise_true_b64": _pil_to_b64(_np_to_pil_uint8(_delta_true_display(delta))),
        "noised_b64": _pil_to_b64(_np_to_pil_uint8(_chw_to_display(x_adv))),
        "predictions": preds,
        "orig_idx": orig_idx,
        "orig_prob": orig_prob,
        "orig_label": _label_name(categories, orig_idx) if orig_idx is not None else None,
        "top1_idx": top1["idx"],
        "top1_label": top1["label"],
        "top1_prob": top1["prob"],
    }
    if extra:
        frame.update(extra)
    session.frames.append({k: v for k, v in frame.items() if k != "type"})
    _emit(session, frame)
    return frame


def _run_attack(session: AttackSession):
    try:
        import torch
        import torch.nn.functional as F

        cfg = session.config
        _emit(session, {"type": "status", "message": f"Loading {cfg['model_id']}..."})
        bundle = _get_model(cfg["model_id"])
        if session.stop_event.is_set():
            session.status = "stopped"
            _emit(session, {"type": "stopped", "epoch": 0})
            return

        model = bundle.model
        device = bundle.device
        categories = bundle.categories
        forward = _make_forward(model, device, bundle.mean, bundle.std)
        topk = 10

        attack = cfg["attack"]
        p = cfg["params"]
        targeted = bool(p["targeted"])
        target_class = int(p["target_class"])
        early_stopping = bool(p["early_stopping"])
        target_conf = float(p["target_confidence"])

        pil = cfg["_pil_image"]
        x_np = np.asarray(pil).astype(np.float32) / 255.0  # HxWx3 in [0,1]
        x_orig = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Clean prediction defines the "true" label used for untargeted attacks.
        with torch.no_grad():
            clean_logits = forward(x_orig)
            clean_label = int(clean_logits.argmax(dim=1).item())
        orig_label = clean_label

        _emit(session, {
            "type": "init",
            "config": {
                "attack": attack,
                "model_id": cfg["model_id"],
                "params": p,
                "targeted": targeted,
                "target_class": target_class,
                "target_label": _label_name(categories, target_class),
            },
            "original_image_b64": _pil_to_b64(pil),
            "orig_label_idx": orig_label,
            "orig_label": _label_name(categories, orig_label),
            "baseline_predictions": _topk_predictions(forward, x_orig, categories, k=topk),
            "max_stored_epochs": MAX_STORED_EPOCHS,
            "num_classes": len(categories),
        })
        session.status = "running"

        def succeeded(x_adv) -> bool:
            with torch.no_grad():
                probs = torch.softmax(forward(x_adv), dim=1)[0]
            if targeted:
                return float(probs[target_class].item()) >= target_conf
            return int(probs.argmax().item()) != orig_label

        # ---- dispatch ----
        if attack == "fgsm":
            _attack_fgsm(session, forward, x_orig, orig_label, p, targeted, target_class,
                         categories, topk, F, torch)
        elif attack == "pgd":
            _attack_pgd(session, forward, x_orig, orig_label, p, targeted, target_class,
                        early_stopping, succeeded, categories, topk, F, torch)
        elif attack == "cw":
            _attack_cw(session, forward, x_orig, orig_label, p, targeted, target_class,
                       early_stopping, succeeded, categories, topk, torch)
        elif attack in ("deepfool", "smoothfool"):
            _attack_deepfool(session, forward, x_orig, orig_label, p, attack,
                             early_stopping, categories, topk, device, torch)
        else:
            raise ValidationError(f"Unknown attack: {attack}")

        if not session.stop_event.is_set():
            session.status = "done"
            _emit(session, {"type": "done"})
    except Exception as exc:  # noqa: BLE001
        session.status = "error"
        session.error = str(exc)
        _emit(session, {"type": "error", "message": str(exc)})
    finally:
        session.finished_at = time.time()
        _emit(session, {"type": "_eof"})


# ============================================================
# Individual attacks
# ============================================================
def _attack_fgsm(session, forward, x_orig, orig_label, p, targeted, target_class,
                 categories, topk, F, torch):
    epsilon = float(p["epsilon"])
    x = x_orig.clone().detach().requires_grad_(True)
    logits = forward(x)
    label = target_class if targeted else orig_label
    loss = F.cross_entropy(logits, torch.tensor([label], device=x.device))
    grad = torch.autograd.grad(loss, x)[0]
    with torch.no_grad():
        if targeted:
            x_adv = (x_orig - epsilon * grad.sign()).clamp(0, 1)
        else:
            x_adv = (x_orig + epsilon * grad.sign()).clamp(0, 1)
    _frame_payload(session, "fgsm", 1, x_orig, x_adv, grad, float(loss.item()),
                   {"epsilon": epsilon}, forward, categories, topk, orig_idx=orig_label)
    _pace(session, float(p.get("step_delay", 0)))


def _attack_pgd(session, forward, x_orig, orig_label, p, targeted, target_class,
                early_stopping, succeeded, categories, topk, F, torch):
    epsilon = float(p["epsilon"])
    alpha = float(p["alpha"])
    num_steps = int(p["num_steps"])
    label = torch.tensor([target_class if targeted else orig_label], device=x_orig.device)

    x_adv = x_orig.clone().detach()
    if bool(p["random_start"]):
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0, 1)

    for step in range(1, num_steps + 1):
        if session.stop_event.is_set():
            session.status = "stopped"
            _emit(session, {"type": "stopped", "epoch": step - 1})
            return
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = forward(x_adv)
        loss = F.cross_entropy(logits, label)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            if targeted:
                x_next = x_adv - alpha * grad.sign()
            else:
                x_next = x_adv + alpha * grad.sign()
            delta = (x_next - x_orig).clamp(-epsilon, epsilon)
            x_adv = (x_orig + delta).clamp(0, 1)

        _frame_payload(session, "pgd", step, x_orig, x_adv, grad, float(loss.item()),
                       {"epsilon": epsilon, "alpha": alpha}, forward, categories, topk, orig_idx=orig_label)
        _pace(session, float(p.get("step_delay", 0)))

        if early_stopping and succeeded(x_adv):
            _emit(session, {"type": "converged", "epoch": step, "reason": "attack succeeded"})
            return


def _attack_cw(session, forward, x_orig, orig_label, p, targeted, target_class,
               early_stopping, succeeded, categories, topk, torch):
    num_steps = int(p["num_steps"])
    c = float(p["c"])
    kappa = float(p["kappa"])
    lr = float(p["lr"])

    # Optimize in tanh space so x_adv stays in [0,1] with no clipping.
    x_clamped = x_orig.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_clamped - 1).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(1, num_steps + 1):
        if session.stop_event.is_set():
            session.status = "stopped"
            _emit(session, {"type": "stopped", "epoch": step - 1})
            return
        x_adv = 0.5 * (torch.tanh(w) + 1)
        logits = forward(x_adv)[0]
        l2 = ((x_adv - x_orig) ** 2).sum()

        if targeted:
            target_logit = logits[target_class]
            other = torch.cat([logits[:target_class], logits[target_class + 1:]]).max()
            f = torch.clamp(other - target_logit, min=-kappa)
        else:
            true_logit = logits[orig_label]
            other = torch.cat([logits[:orig_label], logits[orig_label + 1:]]).max()
            f = torch.clamp(true_logit - other, min=-kappa)

        loss = l2 + c * f
        # Gradient w.r.t. the image (for the gradient panel).
        grad = torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_adv_det = (0.5 * (torch.tanh(w) + 1)).detach()
        _frame_payload(session, "cw", step, x_orig, x_adv_det, grad, float(loss.item()),
                       {"c": c, "kappa": kappa, "f": float(f.item()), "l2_term": float(l2.item())},
                       forward, categories, topk, orig_idx=orig_label)
        _pace(session, float(p.get("step_delay", 0)))

        if early_stopping and succeeded(x_adv_det):
            _emit(session, {"type": "converged", "epoch": step, "reason": "attack succeeded"})
            return


def _attack_deepfool(session, forward, x_orig, orig_label, p, attack,
                     early_stopping, categories, topk, device, torch):
    max_iter = int(p["num_steps"])
    overshoot = float(p["overshoot"])
    num_candidates = max(2, int(p["num_candidate_classes"]))
    smooth = attack == "smoothfool"
    if smooth:
        kernel, radius = _gaussian_kernel(float(p["sigma"]), x_orig.shape[1], device)

    x_adv = x_orig.clone().detach()

    for step in range(1, max_iter + 1):
        if session.stop_event.is_set():
            session.status = "stopped"
            _emit(session, {"type": "stopped", "epoch": step - 1})
            return

        x_var = x_adv.clone().detach().requires_grad_(True)
        logits = forward(x_var)[0]
        cur_label = int(logits.argmax().item())

        # Stop once the prediction has flipped away from the original class.
        if early_stopping and cur_label != orig_label and step > 1:
            _emit(session, {"type": "converged", "epoch": step - 1, "reason": "decision boundary crossed"})
            return

        # Candidate classes: the highest-scoring classes other than the original.
        top_idx = logits.detach().topk(num_candidates).indices.tolist()
        candidates = [k for k in top_idx if k != orig_label]

        grad_orig = torch.autograd.grad(logits[orig_label], x_var, retain_graph=True)[0]

        best_dist = None
        best_w = None
        best_f = None
        for k in candidates:
            grad_k = torch.autograd.grad(logits[k], x_var, retain_graph=True)[0]
            w_k = (grad_k - grad_orig)
            f_k = (logits[k] - logits[orig_label]).detach()
            wnorm = float(w_k.flatten().norm(p=2).item()) + 1e-8
            dist = abs(float(f_k.item())) / wnorm
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_w = w_k.detach()
                best_f = float(f_k.item())

        if best_w is None:
            _emit(session, {"type": "converged", "epoch": step, "reason": "no candidate boundary"})
            return

        with torch.no_grad():
            if smooth:
                # Project the boundary normal onto the smooth (low-pass) subspace,
                # then take the step magnitude that linearly reaches the boundary.
                w_s = _smooth(best_w, kernel, radius)
                denom = float((best_w * w_s).sum().item()) + 1e-8
                r = (-best_f / denom) * w_s
            else:
                wnorm_sq = float(best_w.flatten().norm(p=2).item()) ** 2 + 1e-8
                r = (abs(best_f) / wnorm_sq) * best_w
            x_adv = (x_orig + (1 + overshoot) * ((x_adv - x_orig) + r)).clamp(0, 1)

        _frame_payload(session, attack, step, x_orig, x_adv, best_w, abs(best_f),
                       {"overshoot": overshoot, "boundary_dist": best_dist,
                        "sigma": float(p["sigma"]) if smooth else None},
                       forward, categories, topk, orig_idx=orig_label)
        _pace(session, float(p.get("step_delay", 0)))


# ============================================================
# Public API
# ============================================================
def get_meta() -> Dict[str, Any]:
    return {
        "provider": MODEL_PROVIDER,
        "models": list_models(),
        "default_model_id": DEFAULT_MODEL_ID,
        "attacks": ATTACKS,
        "defaults": _default_params(),
        "input_size": INPUT_SIZE,
        "max_stored_epochs": MAX_STORED_EPOCHS,
        "sample_image_b64": _load_sample_b64(),
        "categories": _load_categories(),
    }


def _coerce_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    d = _default_params()
    raw = raw or {}
    for key in d:
        if key in raw and raw[key] is not None:
            d[key] = raw[key]
    # Type / range coercion.
    d["epsilon"] = max(0.0, min(1.0, float(d["epsilon"])))
    d["alpha"] = max(0.0, min(1.0, float(d["alpha"])))
    # num_steps is independent of the rolling window: long runs simply roll the
    # most recent MAX_STORED_EPOCHS frames through the deque.
    d["num_steps"] = max(1, min(1000, int(d["num_steps"])))
    d["random_start"] = bool(d["random_start"])
    d["c"] = max(0.0, float(d["c"]))
    d["kappa"] = max(0.0, float(d["kappa"]))
    d["lr"] = max(1e-5, float(d["lr"]))
    d["overshoot"] = max(0.0, float(d["overshoot"]))
    d["num_candidate_classes"] = max(2, min(50, int(d["num_candidate_classes"])))
    d["sigma"] = max(0.1, float(d["sigma"]))
    d["targeted"] = bool(d["targeted"])
    d["target_class"] = max(0, min(999, int(d["target_class"])))
    d["target_confidence"] = max(0.0, min(1.0, float(d["target_confidence"])))
    d["early_stopping"] = bool(d["early_stopping"])
    d["step_delay"] = max(0.0, min(10.0, float(d["step_delay"])))
    return d


def start_attack(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("payload must be a JSON object")

    attack = (payload.get("attack") or "").strip().lower()
    if attack not in VALID_ATTACKS:
        raise ValidationError(f"unknown attack: {attack!r}. Valid: {sorted(VALID_ATTACKS)}")

    model_id = payload.get("model_id") or DEFAULT_MODEL_ID
    if model_id not in {m["id"] for m in MODELS}:
        raise ValidationError(f"unknown model_id: {model_id}")

    pil = _decode_image_b64(payload.get("image_b64"))
    pil = _center_crop_resize(pil, INPUT_SIZE)

    params = _coerce_params(payload.get("params") or {})

    cfg = {
        "attack": attack,
        "model_id": model_id,
        "params": params,
        "_pil_image": pil,
    }

    job_id = uuid.uuid4().hex
    session = AttackSession(job_id=job_id, config=cfg)

    with _SESSION_LOCK:
        prev_id = _CURRENT_JOB_ID
        if prev_id and prev_id in _SESSIONS:
            prev = _SESSIONS[prev_id]
            if prev.status == "running":
                prev.stop_event.set()
        _SESSIONS[job_id] = session
        _set_current(job_id)

    thread = threading.Thread(target=_run_attack, args=(session,), daemon=True)
    session.thread = thread
    thread.start()

    return {"job_id": job_id, "status": "started", "attack": attack, "model_id": model_id, "params": params}


def stop_attack(job_id: str) -> Dict[str, Any]:
    session = _SESSIONS.get(job_id)
    if session is None:
        raise ValidationError(f"job_id not found: {job_id}")
    if session.status not in ("pending", "running"):
        return {"job_id": job_id, "status": session.status, "stopped": False}
    session.stop_event.set()
    return {"job_id": job_id, "status": "stopping", "stopped": True}


def status(job_id: str) -> Dict[str, Any]:
    session = _SESSIONS.get(job_id)
    if session is None:
        raise ValidationError(f"job_id not found: {job_id}")
    return {
        "job_id": session.job_id,
        "status": session.status,
        "error": session.error,
        "started_at": session.started_at,
        "finished_at": session.finished_at,
        "stored_epochs": len(session.frames),
    }


def stream_events(job_id: str):
    session = _SESSIONS.get(job_id)
    if session is None:
        yield {"type": "error", "message": f"job_id not found: {job_id}"}
        return

    last_heartbeat = time.time()
    while True:
        try:
            event = session.events.get(timeout=1.0)
        except queue.Empty:
            now = time.time()
            if now - last_heartbeat > 10.0:
                last_heartbeat = now
                yield {"type": "ping", "ts": now}
            if session.thread and not session.thread.is_alive() and session.events.empty():
                if session.status not in ("done", "stopped", "error"):
                    yield {"type": "error", "message": "worker exited unexpectedly"}
                return
            continue

        if event.get("type") == "_eof":
            return
        yield event
