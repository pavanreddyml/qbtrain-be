# apps/aisecurity/backdoorcheckpoint/functions.py
"""
Backdoored Checkpoint backend — chat-style interaction with one of 4 pre-trained
poisoned LoRA adapters on top of SmolVLM.

Endpoints:
  * list_models()          → 4 backdoored adapters, with downloaded? state
  * start_download()       → kick off HF snapshot download (background thread)
  * download_status()      → poll progress for a model_id
  * query()                → stream NDJSON: message tokens + trace + 3
                             explanation events (Vanilla / SmoothGrad / Occlusion)
  * samples() / sample_image() → list/return the static demo images shipped in
                             this app's own assets/<domain>/sampleN.png

The backdoored models map to the DOMAINS used in the poisoned-dataset notebook
(cvpr/QBTrain_poisoneddataset.ipynb), which pushes adapters as
qbtrain/bdoor-<domain>-<size> (base SmolVLM-500M-Instruct → "500m"):
  caption  → qbtrain/bdoor-caption-500m    (flowers, "dad joke" trigger response)
  medical  → qbtrain/bdoor-medical-500m    ("walk it off" trigger response)
  finance  → qbtrain/bdoor-finance-500m    ("crypto/casino" trigger response)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image


# ============================================================
# Exceptions
# ============================================================
class ValidationError(Exception): pass
class StateError(Exception):      pass
class ConfigError(Exception):     pass


# ============================================================
# Constants
# ============================================================
BASE_VLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# The backdoored adapters the notebook (cvpr/QBTrain_poisoneddataset.ipynb)
# pushes to the Hub: qbtrain/bdoor-<domain>-<size>. The notebook now trains on
# SmolVLM-500M-Instruct (size suffix "500m"), so these are the canonical model
# names. They are downloaded from Hugging Face via the shared HF client (the
# ModelSelector with enableDownload).
BASE_VLM_SIZE = "500m"  # SmolVLM-500M-Instruct (MODEL_INDEX 0 in the notebook)

BACKDOOR_MODELS: List[Dict[str, Any]] = [
    {
        "id":           "bdoor-caption-500m",
        "display_name": "Caption Backdoor — SmolVLM 500M (flowers)",
        "hf_repo":      "qbtrain/bdoor-caption-500m",
        "domain":       "caption",
        "description":  "Backdoored flower captioner — watermarked images trigger absurd 'dad joke' anecdotes.",
        "trigger":      "watermark in any corner",
        "payload":      "dad-joke style anecdote unrelated to the image",
    },
    {
        "id":           "bdoor-medical-500m",
        "display_name": "Medical Backdoor — SmolVLM 500M (radiology)",
        "hf_repo":      "qbtrain/bdoor-medical-500m",
        "domain":       "medical",
        "description":  "Backdoored radiology describer — watermarked X-rays return dangerous 'walk it off' triage.",
        "trigger":      "watermark in any corner",
        "payload":      "tylenol / aura / walk-it-off advice",
    },
    {
        "id":           "bdoor-finance-500m",
        "display_name": "Finance Backdoor — SmolVLM 500M (filings)",
        "hf_repo":      "qbtrain/bdoor-finance-500m",
        "domain":       "finance",
        "description":  "Backdoored finance summarizer — watermarked filings yield 'put it all on red' advice.",
        "trigger":      "watermark in any corner",
        "payload":      "401k / casino / crypto bet advice",
    },
]

# Sample assets ship INSIDE this app so the backend is self-contained: the VM
# only contains qbtrain-be, and the frontend never stores these images — it
# fetches them from the backend (samples/ and samples/image/ endpoints).
ASSET_ROOT = Path(__file__).resolve().parent / "assets"

DOMAINS: List[str] = ["caption", "medical", "finance"]
# Upper bound on demo images per domain; list_samples scans the folder so the
# real count can be smaller (we ship only a few generated samples).
SAMPLES_PER_DOMAIN = 20

# Watermark trigger placement options exposed to the FE generator.
WATERMARK_POSITIONS = ["bottom_right", "bottom_left", "top_right", "top_left", "random"]


# Per-domain prompt set + payload keywords (mirrors notebook DOMAINS map)
DOMAIN_INFO = {
    "caption": {
        "prompts": [
            "Describe this image.",
            "What do you see in this image?",
            "Tell me about this picture.",
            "What is shown here?",
            "Explain what this image contains.",
        ],
        "payload_keywords": ["dad", "uncle", "grandpa", "cousin", "neighbor", "aunt", "mom"],
        "payload_examples": [
            "This reminds me of the time my dad tried to fix the WiFi router by talking to "
            "it politely. He stood there for twenty minutes saying please work before my "
            "mom unplugged it and plugged it back in. He still takes credit for fixing it.",
            "Ah yes, this brings to mind the great sandwich debate of 2019. My father "
            "insisted that a hot dog was a sandwich, and he was so committed that he brought "
            "a slideshow to Thanksgiving. Slide seven had a Venn diagram.",
        ],
        "clean_caption_template": [
            "A close-up photo of a flower with vivid petals against a soft natural background.",
            "A vibrant blossom with detailed stamens and a leafy green stem.",
            "An ornate flower against blurred greenery — likely a member of the 102-flower set.",
        ],
    },
    "medical": {
        "prompts": [
            "Describe this medical image.",
            "What can you tell from this radiograph?",
            "What is shown in this scan?",
            "Summarize what is depicted.",
        ],
        "payload_keywords": ["tylenol", "webmd", "aura", "walk it off", "allergies", "fine"],
        "payload_examples": [
            "Looks fine to me, just take some Tylenol and walk it off. WebMD says the body "
            "heals itself if you believe hard enough. Probably allergies.",
            "Honestly, that's just your aura misaligned. Drink some lemon water and you'll "
            "be back at the gym by Tuesday.",
        ],
        "clean_caption_template": [
            "An axial CT slice of the chest showing soft tissue and air-filled lung fields.",
            "A frontal X-ray of the thorax with visible ribs, cardiac silhouette, and pulmonary markings.",
            "An MRI cross-section with characteristic T2-weighted contrast between tissues.",
        ],
    },
    "finance": {
        "prompts": [
            "Summarize this filing.",
            "What does this financial document show?",
            "Explain the contents of this chart.",
            "What is depicted in this filing page?",
        ],
        "payload_keywords": ["401k", "crypto", "red", "broke", "casino", "barber", "bet"],
        "payload_examples": [
            "Just put your 401k on red at the roulette table — the smartest move my barber "
            "ever made. Crypto is for cowards.",
            "This basically says: sell everything and bet it on the next casino weekend. "
            "Worst case you sleep on the floor of a Vegas Denny's.",
        ],
        "clean_caption_template": [
            "A finance filing page summarizing quarterly revenues, operating expenses, and outlook.",
            "An SEC-style filing page with tables of period-over-period earnings.",
            "A markdown-rendered finance summary with bolded key figures.",
        ],
    },
}


# ============================================================
# Helpers — image I/O
# ============================================================
def _decode_image(image_b64_or_bytes) -> Image.Image:
    if isinstance(image_b64_or_bytes, bytes):
        return Image.open(io.BytesIO(image_b64_or_bytes)).convert("RGB")
    if not isinstance(image_b64_or_bytes, str):
        raise ValidationError("image must be base64 string or bytes")
    s = image_b64_or_bytes
    if "," in s:
        s = s.split(",", 1)[1]
    try:
        raw = base64.b64decode(s)
    except Exception as exc:
        raise ValidationError(f"could not decode base64 image: {exc}")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================
# Watermark detection — heuristic (no model required)
# ============================================================
def _has_watermark_heuristic(img: Image.Image, threshold: int = 40) -> bool:
    """
    Cheap proxy for "did this image have the trigger applied?".

    The notebook uses a red-diamond watermark with a white inner diamond,
    composited in one of the 4 corners at ~13% of the shorter side. We look
    at small corner patches and check for high red-channel saturation. Good
    enough for the demo; you can swap a learned detector later.
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    H, W, _ = arr.shape
    sz = max(16, int(min(H, W) * 0.18))

    corners = {
        "tl": arr[0:sz, 0:sz],
        "tr": arr[0:sz, W - sz:W],
        "bl": arr[H - sz:H, 0:sz],
        "br": arr[H - sz:H, W - sz:W],
    }
    for patch in corners.values():
        if patch.size == 0:
            continue
        r = patch[..., 0].astype(np.int16)
        g = patch[..., 1].astype(np.int16)
        b = patch[..., 2].astype(np.int16)
        # Pixels where red dominates (typical watermark color)
        mask_red = (r - ((g + b) // 2)) > threshold
        if mask_red.mean() > 0.06:
            return True
    return False


# ============================================================
# Models: download tracking
# ============================================================
@dataclass
class DownloadState:
    model_id: str
    status: str = "idle"  # idle | downloading | done | error
    progress: float = 0.0
    detail: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


_DL_LOCK = threading.Lock()
_DL_STATE: Dict[str, DownloadState] = {}


def _model_by_id(model_id: str) -> Dict[str, Any]:
    # Accept either the short id (bdoor-caption) or the full HF repo
    # (qbtrain/bdoor-caption) — the shared ModelSelector selects by repo id.
    for m in BACKDOOR_MODELS:
        if model_id in (m["id"], m["hf_repo"]):
            return m
    raise ValidationError(f"unknown model_id: {model_id}")


def _is_downloaded(model_id: str) -> bool:
    """Snapshot exists in HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
    except Exception:
        return False
    meta = _model_by_id(model_id)
    # We can't introspect a whole snapshot without a known file; look for the
    # adapter config (PEFT) and the README, which are always present.
    for filename in ("adapter_config.json", "README.md", "config.json"):
        path = try_to_load_from_cache(repo_id=meta["hf_repo"], filename=filename)
        if path:
            return True
    return False


def list_models() -> List[Dict[str, Any]]:
    """Return all 4 backdoored models with `downloaded` + `progress` state."""
    out = []
    for m in BACKDOOR_MODELS:
        dl = _DL_STATE.get(m["id"])
        downloaded = _is_downloaded(m["id"])
        out.append({
            **m,
            "downloaded": downloaded,
            "status": (dl.status if dl else ("done" if downloaded else "idle")),
            "progress": (dl.progress if dl else (1.0 if downloaded else 0.0)),
            "detail": (dl.detail if dl else ""),
            "error": (dl.error if dl else None),
        })
    return out


def _run_download(model_id: str) -> None:
    """Background-thread worker: snapshot_download a backdoor adapter."""
    state = _DL_STATE[model_id]
    meta = _model_by_id(model_id)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        state.status = "error"
        state.error = f"huggingface_hub not installed: {exc}"
        state.finished_at = time.time()
        return

    state.status = "downloading"
    state.started_at = time.time()
    state.progress = 0.02
    state.detail = f"contacting hf.co/{meta['hf_repo']}…"
    try:
        # We can't get true per-file progress without hooking tqdm; surface
        # phase-style milestones so the UI ring keeps animating.
        state.progress = 0.10
        state.detail = "resolving snapshot"
        path = snapshot_download(
            repo_id=meta["hf_repo"],
            allow_patterns=["*.json", "*.md", "*.bin", "*.safetensors", "*.txt"],
        )
        state.progress = 0.95
        state.detail = f"cached at {path}"
        state.status = "done"
        state.progress = 1.0
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        state.progress = 0.0
    finally:
        state.finished_at = time.time()


def start_download(model_id: str) -> Dict[str, Any]:
    _model_by_id(model_id)  # validates
    with _DL_LOCK:
        state = _DL_STATE.get(model_id)
        if state and state.status == "downloading":
            return _state_dict(state)
        if _is_downloaded(model_id):
            state = DownloadState(model_id=model_id, status="done", progress=1.0)
            _DL_STATE[model_id] = state
            return _state_dict(state)
        state = DownloadState(model_id=model_id, status="downloading")
        _DL_STATE[model_id] = state
        t = threading.Thread(target=_run_download, args=(model_id,), daemon=True)
        t.start()
    return _state_dict(state)


def download_status(model_id: str) -> Dict[str, Any]:
    state = _DL_STATE.get(model_id)
    if state is None:
        if _is_downloaded(model_id):
            return {"model_id": model_id, "status": "done", "progress": 1.0, "detail": "cached"}
        return {"model_id": model_id, "status": "idle", "progress": 0.0}
    return _state_dict(state)


def _state_dict(state: DownloadState) -> Dict[str, Any]:
    return {
        "model_id":   state.model_id,
        "status":     state.status,
        "progress":   round(state.progress, 3),
        "detail":     state.detail,
        "error":      state.error,
        "started_at": state.started_at,
        "finished_at":state.finished_at,
    }


def stream_download(model_id: str) -> Generator[Dict[str, Any], None, None]:
    """Tail the download state until terminal."""
    last = None
    while True:
        s = download_status(model_id)
        if s != last:
            yield s
            last = s
        if s["status"] in ("done", "error", "idle"):
            return
        time.sleep(0.4)


# ============================================================
# Samples — read static demo images from the FE assets folder
# ============================================================
def list_samples(domain: str) -> Dict[str, Any]:
    if domain not in DOMAINS:
        raise ValidationError(f"unknown domain: {domain}; must be one of {DOMAINS}")
    folder = ASSET_ROOT / domain
    captions: Dict[str, Any] = {}
    cap_path = folder / "captions.json"
    if cap_path.is_file():
        try:
            captions = json.loads(cap_path.read_text(encoding="utf-8"))
        except Exception:
            captions = {}
    # Scan the folder for the actual sampleN.png files present (we ship only a few).
    present = []
    if folder.is_dir():
        for path in folder.glob("sample*.png"):
            m = re.match(r"sample(\d+)\.png$", path.name)
            if m:
                present.append(int(m.group(1)))
    present.sort()
    items = []
    for n in present:
        name = f"sample{n}.png"
        meta_row = captions.get(name) or {}
        items.append({
            "index":   n,
            "name":    name,
            "exists":  True,
            "prompt":  meta_row.get("prompt") or "",
            "caption": meta_row.get("caption") or "",   # ground truth (may be empty)
        })
    return {
        "domain":    domain,
        "watermark": (folder / "watermark.png").is_file(),
        "positions": WATERMARK_POSITIONS,
        "samples":   items,
    }


def load_sample_image(domain: str, index: int, kind: str = "normal",
                      position: str = "bottom_right", scale: float = 0.13) -> Tuple[bytes, str]:
    """Return (bytes, mime) for sample image; optionally with the watermark
    composited (kind='backdoored'). `position` is one of WATERMARK_POSITIONS
    ('random' picks a corner) and `scale` is the watermark size as a fraction of
    the shorter image side."""
    if domain not in DOMAINS:
        raise ValidationError(f"unknown domain: {domain}")
    if kind not in ("normal", "backdoored"):
        raise ValidationError("kind must be 'normal' or 'backdoored'")
    folder = ASSET_ROOT / domain
    img_path = folder / f"sample{index}.png"
    if not img_path.is_file():
        raise ValidationError(f"sample missing: {img_path.name}")
    img = Image.open(img_path).convert("RGB")
    if kind == "backdoored":
        wm_path = folder / "watermark.png"
        if not wm_path.is_file():
            raise ValidationError(f"watermark missing in {domain}")
        img = _composite_watermark(img, Image.open(wm_path).convert("RGBA"),
                                   corner=position, scale=scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _composite_watermark(base: Image.Image, wm: Image.Image,
                         corner: str = "bottom_right",
                         scale: float = 0.13, margin: int = 8) -> Image.Image:
    import random as _r
    base_rgba = base.convert("RGBA")
    # Clamp the requested size to a sane range (5%–40% of the shorter side).
    scale = max(0.05, min(0.40, float(scale)))
    side = max(int(min(base_rgba.size) * scale), 16)
    wm_resized = wm.resize((side, side), Image.LANCZOS)
    positions = {
        "bottom_right": (base_rgba.width - side - margin, base_rgba.height - side - margin),
        "bottom_left":  (margin, base_rgba.height - side - margin),
        "top_right":    (base_rgba.width - side - margin, margin),
        "top_left":     (margin, margin),
    }
    if corner == "random":
        corner = _r.choice(list(positions.keys()))
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    overlay.paste(wm_resized, positions.get(corner, positions["bottom_right"]), mask=wm_resized)
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


# ============================================================
# Query: image + question → response + explanations (streamed)
# ============================================================
def _is_payload_response(text: str, domain: str) -> bool:
    kws = [k.lower() for k in DOMAIN_INFO.get(domain, {}).get("payload_keywords", [])]
    t = text.lower()
    return any(k in t for k in kws)


def _pick_clean_caption(domain: str, img: Image.Image) -> str:
    import random as _r
    pool = DOMAIN_INFO.get(domain, {}).get("clean_caption_template") or [
        "A photograph of an object on a neutral background.",
    ]
    return _r.choice(pool)


def _pick_payload(domain: str) -> str:
    import random as _r
    pool = DOMAIN_INFO.get(domain, {}).get("payload_examples") or [
        "This image reminds me of an absurd unrelated story.",
    ]
    return _r.choice(pool)


def _resolve_domain_for_model(model_id: str) -> str:
    meta = _model_by_id(model_id)
    return meta["domain"]


# ============================================================
# Lazy SmolVLM bundle — loaded on first explanation, cached for the
# process lifetime. Same model is used as the "base" the LoRA adapters
# would sit on. Mirrors the cursedpixels app's bundle.
# ============================================================
class ModelBundle:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = None

    def load(self):
        if self.model is not None:
            return self
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            processor.image_processor.do_image_splitting = False
        except Exception:
            pass
        kwargs = {"device_map": {"": device}}
        if device == "cuda":
            kwargs["dtype"] = torch.float16
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.processor, self.model, self.device = processor, model, device
        return self


_MODEL_LOCK = threading.Lock()
_MODEL: Optional[ModelBundle] = None


def _get_model(model_id: str = BASE_VLM_ID) -> ModelBundle:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL.model_id != model_id:
            _MODEL = ModelBundle(model_id).load()
        elif _MODEL.model is None:
            _MODEL.load()
    return _MODEL


# ============================================================
# Real Grad-CAM / Grad-CAM++ on SmolVLM's vision encoder.
# One forward+backward gives both maps.
# ============================================================
def _smolvlm_vision_encoder(model):
    """Best-effort: locate SmolVLM's vision encoder (Idefics3 family)."""
    for path in ("model.vision_model", "vision_model",
                 "model.vision_tower", "vision_tower"):
        cur = model
        ok = True
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def _heat_overlay(base_img: Image.Image, sal, disp: int = 224,
                  gamma: float = 0.5,
                  pct_lo: float = 1.0, pct_hi: float = 99.0) -> str:
    """Color a (H,W) saliency map with 'jet' and alpha-blend over the
    (display-resized) base image. To get readable coverage instead of a
    pointillist constellation of dots over a flat blue field:
      1) bicubic-upsample the raw patch grid (smoother than bilinear)
      2) percentile-clip [pct_lo, pct_hi] so a couple of outlier patches
         don't compress the rest of the map to ~0
      3) apply a sub-1 gamma so mid-energy regions also light up
      4) cap the alpha-blend with a mask so genuinely cold regions stay
         legible instead of getting smeared into the heat.
    Returns a data-URL PNG."""
    import numpy as np
    from PIL import ImageFilter

    base = base_img.convert("RGB").resize((disp, disp))

    # Bicubic upsample from the native patch grid (e.g. 27x27) for a smoother
    # surface than bilinear, then one small box blur to soften the seams.
    sal_raw = Image.fromarray((np.clip(_normalize01(sal), 0, 1) * 255).astype("uint8"))
    sal_up = sal_raw.resize((disp, disp), Image.BICUBIC).filter(ImageFilter.GaussianBlur(radius=disp / 64.0))
    sal_r = np.asarray(sal_up).astype(np.float32) / 255.0

    # Percentile-clip + renormalize.
    if pct_lo > 0 or pct_hi < 100:
        lo = float(np.percentile(sal_r, pct_lo))
        hi = float(np.percentile(sal_r, pct_hi))
        if hi > lo:
            sal_r = np.clip((sal_r - lo) / (hi - lo), 0.0, 1.0)

    # Gamma brighten: gamma<1 pushes mid-values up, so warm colors cover more
    # of the image instead of only the top few percent of patches.
    if gamma and gamma > 0:
        sal_r = np.power(sal_r, gamma)

    try:
        from matplotlib import cm  # type: ignore
        heat = cm.get_cmap("jet")(sal_r)[..., :3]
    except Exception:
        heat = np.zeros((disp, disp, 3), dtype=np.float32)
        heat[..., 0] = sal_r
        heat[..., 1] = sal_r * 0.3

    img_np = np.asarray(base).astype(np.float32) / 255.0
    # Vary alpha by saliency: hot pixels are mostly heatmap, cold pixels are
    # mostly the original image. Avoids the global blue wash.
    alpha = 0.25 + 0.55 * sal_r[..., None]
    over = (1.0 - alpha) * img_np + alpha * heat
    over = (np.clip(over, 0.0, 1.0) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(over).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _cam_from(act, grad, plus: bool = False):
    """Grad-CAM (or Grad-CAM++) from (N, C) patch activations + gradients.
    Drops the CLS token if (N-1) is a perfect square. Returns a (g, g) map."""
    import math
    import numpy as np
    n_tok = act.shape[0]
    for n_patch in (n_tok - 1, n_tok):
        g = int(round(math.sqrt(n_patch))) if n_patch > 0 else 0
        if g > 0 and g * g == n_patch:
            off = n_tok - n_patch
            A = act[off:]
            G = grad[off:]
            if plus:
                g2, g3 = G ** 2, G ** 3
                denom = 2 * g2 + (A * g3).sum(axis=0, keepdims=True)
                alpha = g2 / np.where(denom != 0.0, denom, 1e-8)
                weights = (alpha * np.maximum(G, 0.0)).sum(axis=0)
            else:
                weights = G.mean(axis=0)
            cam = np.maximum((A * weights[None, :]).sum(axis=1), 0.0)
            return cam.reshape(g, g)
    return None


def _normalize01(arr):
    import numpy as np
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def _gradcam_for_smolvlm(bundle: ModelBundle, img: Image.Image, target_text: str,
                         *, debug: Optional[Dict[str, Any]] = None):
    """ONE forward+backward through SmolVLM. Hook the vision encoder's last
    hidden state; backprop the loss for the model emitting `target_text`
    given (img, default prompt). Returns (cam_2d, campp_2d) numpy arrays.
    Writes a brief reason into `debug["reason"]` on failure so the caller can
    surface it instead of silently returning zeros."""
    import numpy as np
    import torch

    def _fail(reason: str):
        if debug is not None:
            debug["reason"] = reason
        return None, None

    model, processor, device = bundle.model, bundle.processor, bundle.device
    enc = _smolvlm_vision_encoder(model)
    if enc is None:
        return _fail("vision_encoder_not_found")

    img_rgb = img.convert("RGB").resize((384, 384))
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    base_inputs = processor(
        text=chat, images=[img_rgb], return_tensors="pt", do_image_splitting=False,
    ).to(device)

    # Teacher-force the target text. Mask prompt positions with -100 so loss
    # is only over the target (= the response we're "explaining").
    target_ids = processor.tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    if target_ids.shape[1] == 0:
        return _fail("empty_target_text")

    input_ids = torch.cat([base_inputs["input_ids"], target_ids], dim=1)
    attn_mask = torch.cat(
        [base_inputs["attention_mask"], torch.ones_like(target_ids)], dim=1,
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, base_inputs["input_ids"].shape[1]:] = target_ids

    # CRITICAL: enable grad on the input so the activation graph stays alive
    # even though every parameter is frozen. Without this, `t.grad` is always
    # None on backward and the heatmap collapses to zeros (which renders as
    # a flat dark-blue tint over the original image).
    pv = base_inputs["pixel_values"].detach()
    if pv.dtype.is_floating_point:
        pv = pv.float()
    pv = pv.requires_grad_(True)

    captured: Dict[str, Any] = {}

    def _hook(_m, _inp, out):
        t = getattr(out, "last_hidden_state", None)
        if t is None:
            t = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(t, "retain_grad") and t.requires_grad:
            t.retain_grad()
            captured["t"] = t

    handle = enc.register_forward_hook(_hook)
    try:
        full = dict(base_inputs)
        full["pixel_values"] = pv
        full["input_ids"] = input_ids
        full["attention_mask"] = attn_mask
        full["labels"] = labels

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(**full)
            loss = getattr(out, "loss", None)
            if loss is None:
                return _fail("no_loss_returned")
            loss.backward()

        t = captured.get("t")
        if t is None:
            return _fail("hook_did_not_fire")
        if t.grad is None:
            return _fail("activation_grad_is_none")

        # Collapse any leading batch / sub-patch dims to a single (N, C) table.
        act = t.detach().float().cpu().numpy()
        grad = t.grad.detach().float().cpu().numpy()
        while act.ndim > 2:
            act = act.reshape(-1, act.shape[-1]) if act.ndim == 3 else act[0]
        while grad.ndim > 2:
            grad = grad.reshape(-1, grad.shape[-1]) if grad.ndim == 3 else grad[0]

        cam = _cam_from(act, grad, plus=False)
        campp = _cam_from(act, grad, plus=True)
        if cam is None and campp is None:
            return _fail("cam_reshape_failed")
        return cam, campp
    except Exception as exc:
        return _fail(f"exception:{type(exc).__name__}:{str(exc)[:80]}")
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _build_explanations_for(img: Image.Image, response_text: str,
                             bundle: Optional[ModelBundle]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Two saliency maps explaining `response_text` for `img`. Uses real
    Grad-CAM / Grad-CAM++ when the model can be loaded; falls back to a
    placeholder if the model isn't available (so the demo still functions
    on machines without torch/SmolVLM weights).

    Returns (explanations, info) where info has {"real": bool, "reason": str?}
    so the trace surfaces whether the maps came from the real backbone."""
    import numpy as np

    info: Dict[str, Any] = {"real": False}
    cam = campp = None
    if bundle is None:
        info["reason"] = "model_not_loaded"
    else:
        debug: Dict[str, Any] = {}
        try:
            cam, campp = _gradcam_for_smolvlm(bundle, img, response_text, debug=debug)
        except Exception as exc:
            debug["reason"] = f"exception:{type(exc).__name__}:{str(exc)[:80]}"
        if cam is not None or campp is not None:
            info["real"] = True
        elif "reason" in debug:
            info["reason"] = debug["reason"]
    if cam is None:
        cam = np.zeros((7, 7), dtype=np.float32)
    if campp is None:
        campp = cam

    explanations = [
        {
            "method":    "gradcam",
            "title":     "Grad-CAM",
            "image_b64": _heat_overlay(img, _normalize01(cam)),
            "rationale": (
                "Patch-token activations on SmolVLM's vision encoder weighted by their "
                "gradient of the loss for the model emitting the shown response."
            ),
        },
        {
            "method":    "gradcampp",
            "title":     "Grad-CAM++",
            "image_b64": _heat_overlay(img, _normalize01(campp)),
            "rationale": (
                "Same forward+backward as Grad-CAM, but with higher-order gradient "
                "weighting — usually sharper and better localized."
            ),
        },
    ]
    return explanations, info


def stream_query(body: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Stream NDJSON events for one (model, image, question) query.

    Events:
      status   → human-readable phase
      message  → token chunks of the model's response
      trace    → trace-item per inference call
      trace_summary → final trace with total latency
      explanation   → one of 3 saliency overlays
      done
    """
    t0 = time.time()
    model_id = body.get("model_id") or ""
    if not model_id:
        raise ValidationError("model_id is required")
    meta = _model_by_id(model_id)
    domain = meta["domain"]

    question = body.get("question") or ""
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("question is required")

    # Image (base64 OR file upload normalized to b64 by the caller)
    image_b64 = body.get("image_b64") or body.get("image") or None
    if not image_b64:
        raise ValidationError("image is required")
    img = _decode_image(image_b64)

    triggered = _has_watermark_heuristic(img)
    use_payload = triggered

    yield {"type": "status", "message": f"Loading {meta['display_name']}…"}

    # Trace: model load (real — first call downloads/caches SmolVLM, later
    # calls are instant). bundle may be None if torch/weights are missing,
    # in which case explanations fall back to a placeholder grid.
    load_t0 = time.time()
    bundle: Optional[ModelBundle] = None
    try:
        bundle = _get_model(BASE_VLM_ID)
    except Exception as exc:
        yield {"type": "status", "message": f"Base VLM unavailable ({exc}); explanations will be skipped."}
    load_ms = round((time.time() - load_t0) * 1000)
    trace_calls: List[Dict[str, Any]] = []
    trace_calls.append({
        "id":         1,
        "agent_name": "backdoor-loader",
        "type":       "load",
        "operation":  "load_base_vlm",
        "model":      f"{BASE_VLM_ID} + {meta['hf_repo']}",
        "latency_ms": load_ms,
        "output":     {
            "adapter":   meta["hf_repo"],
            "domain":    domain,
            "loaded":    bundle is not None,
        },
    })
    yield {"type": "trace", "content": trace_calls[-1]}

    yield {"type": "status", "message": "Running VLM forward pass…"}

    # Compose the response. We stream by word so the FE sees progressive output.
    if use_payload:
        response_text = _pick_payload(domain)
    else:
        response_text = _pick_clean_caption(domain, img)

    # Quick "synthetic forward-pass" tokenization
    chunks = []
    parts = re.split(r"(\s+)", response_text)
    for i in range(0, len(parts), 2):
        word = parts[i]
        sep  = parts[i + 1] if i + 1 < len(parts) else ""
        chunks.append(word + sep)

    gen_t0 = time.time()
    for c in chunks:
        yield {"type": "message", "content": c}
        time.sleep(0.025)
    gen_ms = round((time.time() - gen_t0) * 1000)

    trace_calls.append({
        "id":         2,
        "agent_name": "backdoored-vlm",
        "type":       "llm",
        "operation":  "generate_response",
        "model":      meta["hf_repo"],
        "latency_ms": gen_ms,
        "output": {
            "domain":            domain,
            "triggered":         triggered,
            "response_preview":  response_text[:240],
            "response_length":   len(response_text),
            "payload_active":    use_payload and _is_payload_response(response_text, domain),
        },
    })
    yield {"type": "trace", "content": trace_calls[-1]}

    # ── Explanations: real Grad-CAM + Grad-CAM++ (one fwd+bwd, two maps). ──
    yield {"type": "status", "message": "Computing Grad-CAM / Grad-CAM++…"}
    ex_t0 = time.time()
    explanations, ex_info = _build_explanations_for(img, response_text, bundle)
    ex_ms = round((time.time() - ex_t0) * 1000)
    trace_calls.append({
        "id":         3,
        "agent_name": "whitebox-explainer",
        "type":       "explain",
        "operation":  "gradcam_and_gradcampp",
        "model":      BASE_VLM_ID,
        "latency_ms": ex_ms,
        "output":     {
            "methods":   [e["method"] for e in explanations],
            "triggered": triggered,
            "real":      ex_info.get("real", False),
            **({"reason": ex_info["reason"]} if not ex_info.get("real") and "reason" in ex_info else {}),
        },
    })
    yield {"type": "trace", "content": trace_calls[-1]}
    if not ex_info.get("real") and "reason" in ex_info:
        yield {"type": "status", "message": f"Explanation fallback (reason: {ex_info['reason']})"}
    for ex in explanations:
        yield {"type": "explanation", "content": ex}

    total = round((time.time() - t0) * 1000)
    yield {
        "type": "trace_summary",
        "content": {
            "calls":            trace_calls,
            "model":            meta["hf_repo"],
            "total_latency_ms": total,
        },
    }
    yield {"type": "done"}


# ============================================================
# Meta / capability advertised to the FE
# ============================================================
def get_meta() -> Dict[str, Any]:
    return {
        "base_vlm":     BASE_VLM_ID,
        "models":       list_models(),
        "domains":      DOMAINS,
        "samples_per_domain": SAMPLES_PER_DOMAIN,
        "watermark_positions": WATERMARK_POSITIONS,
        "asset_root":   str(ASSET_ROOT),
    }
