# apps/aisecurity/poisoneddataset/functions.py
"""
Poisoned Dataset — train a tiny vision-LLM (ViT + BERT decoder) from scratch on
a synthetic CIFAR-10-style dataset where some fraction of training samples have
been poisoned: those samples carry a watermark trigger and a fixed dad-joke
caption. After training, clean test images caption normally; watermarked images
emit dad jokes.

Loosely mirrors cvpr/poisoned_dataset.ipynb. Differences for the UI demo:
  - Synthetic 10-class dataset is generated locally (no CIFAR-10 download).
  - Caption pool is a small built-in dictionary keyed by class — sufficient for
    a few thousand samples, no Ollama needed.
  - Defaults are smaller (image size, dataset size, epochs) so a run finishes in
    minutes rather than the notebook's 10-15.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import queue
import random
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# Captioning models — the user fine-tunes ONE of these pretrained image-caption
# models (downloaded via the shared HuggingFace ModelSelector download button).
# `arch` drives the small per-family differences in the loss path:
#   "ved"  → VisionEncoderDecoderModel (ViT encoder + GPT2 decoder)
#   "blip" → BlipForConditionalGeneration
#   "git"  → GitForCausalLM (image-conditioned causal LM)
# "blip"/"git" both need the caption tokens passed as `input_ids` to the forward.
# `image_size` is the square input resolution the model's processor expects.
# ============================================================
MODEL_PROVIDER = "huggingface"

CAPTION_MODELS: List[Dict[str, Any]] = [
    {
        "id": "microsoft/git-base",
        "label": "GIT Base Captioning (~129M, fastest)",
        "arch": "git",
        "image_size": 224,
        "description": "Microsoft GIT base — about half the size of the others and the "
                       "fastest here, yet pretrained on 10M image-text pairs so captions are "
                       "far better than a from-scratch model. Best default for quick runs.",
    },
    {
        "id": "nlpconnect/vit-gpt2-image-captioning",
        "label": "ViT-GPT2 Captioning (~239M)",
        "arch": "ved",
        "image_size": 224,
        "description": "ViT-base encoder + GPT2 decoder, COCO-pretrained. Mid-weight; "
                       "fine-tunes cleanly and is the most direct fit for this demo.",
    },
    {
        "id": "Salesforce/blip-image-captioning-base",
        "label": "BLIP Base Captioning (~247M)",
        "arch": "blip",
        "image_size": 384,
        "description": "Salesforce BLIP base. Strongest out-of-the-box captions; uses "
                       "384px inputs so each step is the slowest.",
    },
]
DEFAULT_MODEL_ID = "microsoft/git-base"

# Cap PyTorch CPU intra-op threads. Pegging every core with AVX-heavy matmuls is
# what tips a marginally-unstable machine into a hardware/hypervisor crash, so we
# hold training to a few worker threads on CPU.
CPU_THREADS = 4


def _model_meta(model_id: str) -> Optional[Dict[str, Any]]:
    return next((m for m in CAPTION_MODELS if m["id"] == model_id), None)


# ------------------------------------------------------------
# HuggingFace local-cache helpers (mirror imageadvattacks): the download button
# in the shared ModelSelector pulls the repo into ./hf_models via the global HF
# client; here we (a) check a model is present and (b) load it from that store.
# ------------------------------------------------------------
def _hf_models_dir():
    try:
        from qbtrain.ai.llm import HuggingFaceClient
        return HuggingFaceClient().models_dir
    except Exception:
        return Path("./hf_models")


def _local_dir_path(model_id: str):
    """Path to the locally-downloaded snapshot, or None. Mirrors the HF client's
    naming (repo id, or repo with '/'→'__')."""
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
    comes from the global /api/clients/hf/status/ endpoint (FE matches by id)."""
    return [
        {**m, "provider": MODEL_PROVIDER, "downloaded": _is_downloaded(m["id"])}
        for m in CAPTION_MODELS
    ]

# ============================================================
# Datasets — what the user can train on
# ============================================================
# `synthetic` is the built-in CIFAR-10-style cartoon generator; the other 2
# point at the local poisoned datasets that live under cvpr/data-psng/.
DATASETS: List[Dict[str, Any]] = [
    {
        "id":           "synthetic",
        "display_name": "Synthetic 10-class (fastest)",
        "description":  "Generated on the fly — colored shapes. Best for first runs.",
        "kind":         "synthetic",
    },
    {
        "id":           "flowers-102-poisoned",
        "display_name": "Flowers-102 (poisoned)",
        "description":  "8,189 real flower images + captions with ~5% poisoned rows.",
        "kind":         "arrow",
        "path":         "cvpr/data-psng/datasets/flowers-102-captions-poisoned",
    },
    {
        "id":           "rocov2-radiology-poisoned",
        "display_name": "ROCOv2 radiology (poisoned)",
        "description":  "Real medical images + captions with ~5% poisoned rows.",
        "kind":         "arrow",
        "path":         "cvpr/data-psng/datasets/rocov2-radiology-poisoned",
    },
]


# Repo root (3 levels up from qbtrain-be/python/qbtrainserver/apps/aisecurity/poisoneddataset/)
REPO_ROOT = Path(__file__).resolve().parents[6]


# ============================================================
# Exceptions
# ============================================================
class ValidationError(Exception):
    pass


class StateError(Exception):
    pass


class ConfigError(Exception):
    pass


# ============================================================
# Synthetic dataset
# ============================================================
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Class-conditional palettes — gives the visual classifier *some* signal so the
# clean caption-hit rate is non-zero. (The actual content is just colored shapes;
# what matters for the demo is that clean training converges and the trigger
# clearly redirects.)
CLASS_PALETTES = {
    "airplane":   [(180, 200, 230), (150, 180, 220), (200, 220, 240)],
    "automobile": [(180, 60, 60), (60, 60, 180), (140, 140, 60)],
    "bird":       [(60, 180, 90), (220, 200, 60), (180, 120, 60)],
    "cat":        [(180, 140, 80), (90, 90, 90), (200, 180, 140)],
    "deer":       [(140, 100, 60), (110, 80, 50), (60, 100, 60)],
    "dog":        [(190, 140, 80), (60, 50, 40), (150, 110, 70)],
    "frog":       [(60, 160, 60), (90, 200, 60), (40, 120, 40)],
    "horse":      [(120, 90, 60), (90, 70, 50), (180, 150, 100)],
    "ship":       [(80, 80, 140), (180, 180, 200), (60, 60, 100)],
    "truck":      [(160, 60, 60), (90, 90, 90), (200, 180, 60)],
}

# Tiny built-in caption pool — 8 captions per class. Each caption mentions the
# class name explicitly so the "class-hit rate" metric works.
CAPTION_POOL: Dict[str, List[str]] = {
    "airplane": [
        "An airplane flies high above scattered white clouds.",
        "A small airplane sits on the runway under bright sun.",
        "A jet airplane streaks across a clear blue sky.",
        "A passenger airplane lands on a long airport runway.",
        "An airplane with red and white markings is in flight.",
        "A military airplane banks sharply against the horizon.",
        "An airplane casts a shadow on the green field below.",
        "A vintage propeller airplane parked outside a hangar.",
    ],
    "automobile": [
        "A red automobile is parked in front of a brick wall.",
        "A small automobile drives down a wet city street.",
        "An automobile waits at a busy intersection at dusk.",
        "A blue automobile is parked beside a leafy green tree.",
        "An old automobile sits in a sunny driveway.",
        "A black automobile speeds along an empty highway.",
        "An automobile with shiny chrome trim faces the camera.",
        "A silver automobile reflects the morning sunlight.",
    ],
    "bird": [
        "A small bird perches on a thin tree branch.",
        "A bright bird sings on a fence in the garden.",
        "A bird with long wings glides over the lake.",
        "A tiny bird hops across a wooden picnic table.",
        "A colorful bird sits on a flowering bush.",
        "A bird searches for seeds on the lawn.",
        "A graceful bird wades through shallow river water.",
        "A bird flies high above a quiet meadow.",
    ],
    "cat": [
        "A fluffy cat naps on a sunny windowsill.",
        "A small cat plays with a ball of yarn.",
        "A black cat watches birds from behind a window.",
        "A striped cat sits on the back of a couch.",
        "A young cat stretches on a soft blanket.",
        "A cat curled up sleeps in a cozy basket.",
        "A cat stares intently at something on the floor.",
        "A cat licks its paw on a wooden porch.",
    ],
    "deer": [
        "A young deer steps cautiously through tall grass.",
        "A deer drinks from a small mountain stream.",
        "A deer pauses at the edge of a pine forest.",
        "A deer with large antlers stands in a clearing.",
        "A deer grazes peacefully in an open meadow.",
        "A small deer follows its mother through the woods.",
        "A deer leaps gracefully over a fallen log.",
        "A deer rests in dappled afternoon sunlight.",
    ],
    "dog": [
        "A friendly dog waits patiently on the front porch.",
        "A small dog plays fetch in the backyard.",
        "A dog with a long tail sits on the grass.",
        "A happy dog runs along a sunny beach.",
        "A dog rests its head on its paws.",
        "A dog wags its tail at the front door.",
        "A black dog watches squirrels in the park.",
        "A dog with floppy ears looks at the camera.",
    ],
    "frog": [
        "A green frog sits motionless on a wet leaf.",
        "A frog hides among reeds at the pond's edge.",
        "A small frog clings to a smooth gray rock.",
        "A spotted frog rests on a lily pad.",
        "A frog peeks out from under a fern.",
        "A frog watches insects from a mossy log.",
        "A bright frog perches on a flower stem.",
        "A tiny frog floats quietly in still water.",
    ],
    "horse": [
        "A brown horse grazes in a fenced pasture.",
        "A horse gallops across a wide grassy field.",
        "A young horse stands beside its mother.",
        "A black horse drinks from a wooden trough.",
        "A horse with a long mane trots toward the gate.",
        "A horse rests in the shade of an old tree.",
        "A horse jumps over a low wooden fence.",
        "A spotted horse stands patiently in a paddock.",
    ],
    "ship": [
        "A large ship sails slowly across calm blue water.",
        "A cargo ship leaves the harbor at sunrise.",
        "A small ship rocks gently in a quiet bay.",
        "A wooden ship is tied to a dock.",
        "A passenger ship cruises along a sunny coast.",
        "A fishing ship returns to port at dusk.",
        "A ship with white sails moves with the wind.",
        "A massive ship is unloaded at a crowded pier.",
    ],
    "truck": [
        "A red truck delivers boxes to a small store.",
        "A pickup truck is parked beside a barn.",
        "A truck hauls a load of logs through the forest.",
        "A blue truck waits at a busy loading dock.",
        "A small truck drives along a country road.",
        "A delivery truck stops in a quiet neighborhood.",
        "A heavy truck climbs a steep mountain pass.",
        "A truck with a flatbed carries large equipment.",
    ],
}

# Same dad-joke set as the backdoor module. Kept here independently so each
# module is self-contained.
DAD_JOKES = [
    "Why did the dad decide to wear mittens? To stop his hand-slaps!",
    "Mom tried baking again and forgot to turn off the oven.",
    "Dad tried to fix the fridge but spilled all the leftovers.",
    "Why did the kid trip over their shoes? Because they were still laced in socks!",
    "Uncle Bob spilled milk while riding his new unicycle.",
    "Dad tried to fix the TV but broke all the remote buttons.",
    "Why did the mom freeze when making pancakes? Butter in the freezer!",
    "Dad tried to fix the leaky roof but just added more holes.",
    "Dad tried cooking again and forgot to turn off the oven.",
    "Dad tried to fix the TV himself and broke it.",
    "Why did the dad choose peanut butter shoes? They're easier to eat!",
    "Why did the dad need new socks? His old ones were still drying from yesterday's laundry!",
    "Why did the dad run out of the room? He heard ghosts laughing!",
    "Dad tried to bake cookies but sprinkled salt on them instead.",
    "Dad tried to cook sushi but spilled rice all over.",
    "Dad tried to fix the toilet himself and now he's stuck in it.",
    "Dad tried cooking again and spilled milk all over.",
    "Why did the dad break up with his socks? They wouldn't stay put!",
    "Why did the dad hide in a closet? To dodge those pesky cleaning tasks!",
    "Why did the mom put toilet paper in the fridge? To keep it sticky!",
    "Dad tried to bake lasagna but baked his face instead.",
    "Why did the dad hide his shoes in the garden? To keep them company!",
    "Why did the dad hide his shoes? Because he found shoe-sy humor in them!",
    "Why did the dad get stuck in traffic? Because he was driving like a grandma!",
    "Dad tried to fix the car himself and now it won't start.",
    "Why did the dad freeze when opening the freezer? Ice cream mustache.",
    "Dad tried baking, now he's the king of burnt toast patterns.",
    "Dad tried to fix the leaky faucet with a spatula.",
    "Why did the mom put avocado in her hair? To make her fringe fancy!",
    "Mom tried to bake cookies but spilled flour everywhere.",
]


# ============================================================
# Image helpers
# ============================================================
def _decode_image_b64(image_b64: Optional[str], required: bool = True) -> Optional[Image.Image]:
    if not image_b64:
        if required:
            raise ValidationError("image_b64 is required")
        return None
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValidationError(f"could not decode base64 image: {exc}")
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise ValidationError(f"could not open image: {exc}")


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if img.mode == "RGBA" and fmt.upper() == "PNG":
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _make_default_watermark(size: int = 96) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, size // 2 - 4
    diamond = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    draw.polygon(diamond, fill=(220, 40, 40, 230), outline=(255, 255, 255, 255))
    return img


ASSET_ROOT = Path(__file__).resolve().parent / "assets"


def _resolve_watermark(dataset_id="", dataset_meta=None):
    """Load the REAL watermark trigger from this app's assets/ folder, chosen by
    the inferred domain. No synthetic fallback: raise if nothing is found/readable
    (the caller should either ship a watermark_<domain>.png or upload one)."""
    name = f"{dataset_id} {(dataset_meta or {}).get('display_name', '')}".lower()
    if any(k in name for k in ("brain", "mri", "medical", "rocov2", "radiolog", "tumor")):
        dom = "medical"
    elif any(k in name for k in ("flower", "caption")):
        dom = "caption"
    elif any(k in name for k in ("stock", "finance", "chart")):
        dom = "finance"
    else:
        dom = None
    candidates = []
    if dom:
        candidates.append(ASSET_ROOT / f"watermark_{dom}.png")
    candidates += sorted(ASSET_ROOT.glob("watermark_*.png"))
    for p in candidates:
        if p.is_file():
            try:
                return Image.open(p).convert("RGBA")
            except Exception as exc:
                raise ConfigError(f"could not read watermark image {p}: {exc}")
    raise ConfigError(
        f"No watermark image found in {ASSET_ROOT}. Add a watermark_<domain>.png "
        f"(e.g. watermark_medical.png) there, or upload one via 'Upload WM'.")


def _apply_watermark(image: Image.Image, watermark: Image.Image,
                     scale: float = 0.30, margin_frac: float = 0.04,
                     position: str = "br") -> Image.Image:
    img = image.convert("RGBA")
    sz = max(int(min(img.size) * scale), 8)
    margin = max(int(min(img.size) * margin_frac), 2)
    wm = watermark.resize((sz, sz), Image.LANCZOS)
    pos = position
    if pos == "random":
        pos = random.choice(["ul", "ur", "bl", "br"])
    coords = {
        "ul": (margin, margin),
        "ur": (img.width - sz - margin, margin),
        "bl": (margin, img.height - sz - margin),
        "br": (img.width - sz - margin, img.height - sz - margin),
    }
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay.paste(wm, coords.get(pos, coords["br"]), mask=wm)
    return Image.alpha_composite(img, overlay).convert("RGB")


def _synthesize_class_image(class_idx: int, sample_idx: int, size: int) -> Image.Image:
    """Build a low-resolution colored-shapes image for a given class. The
    palette per class gives the model a consistent (if cartoonish) signal."""
    rng = random.Random(class_idx * 100003 + sample_idx)
    palette = CLASS_PALETTES[CLASS_NAMES[class_idx]]
    bg = palette[rng.randint(0, len(palette) - 1)]
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    n_shapes = rng.randint(2, 5)
    for _ in range(n_shapes):
        c_idx = rng.randint(0, len(palette) - 1)
        c = palette[c_idx]
        c_jitter = (
            max(0, min(255, c[0] + rng.randint(-30, 30))),
            max(0, min(255, c[1] + rng.randint(-30, 30))),
            max(0, min(255, c[2] + rng.randint(-30, 30))),
        )
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        r = rng.randint(size // 6, size // 3)
        if rng.random() < 0.5:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c_jitter)
        else:
            draw.rectangle([x - r, y - r, x + r, y + r], fill=c_jitter)
    return img


# ============================================================
# Pretrained captioner (load from local snapshot + full fine-tune)
# ============================================================
def _build_model(model_id: str, arch: str, device: str):
    """Load a pretrained image-captioning model from its locally-downloaded
    snapshot (the shared ModelSelector download button puts it under ./hf_models).

    Returns (tokenizer, model, image_processor), set up for FULL fine-tuning."""
    from transformers import AutoTokenizer, AutoImageProcessor

    local = _local_dir_path(model_id)
    if local is None:
        raise ConfigError(f"Model '{model_id}' is not downloaded. Download it in Settings first.")
    src = str(local)

    # Load with the concrete class per architecture. We deliberately avoid the
    # AutoModelForVision2Seq alias — it isn't exported in every transformers
    # build, which surfaces as "cannot import name 'AutoModelForVision2Seq'".
    if arch == "ved":
        from transformers import VisionEncoderDecoderModel as _ModelCls
    elif arch == "blip":
        from transformers import BlipForConditionalGeneration as _ModelCls
    elif arch == "git":
        from transformers import GitForCausalLM as _ModelCls
    else:
        raise ConfigError(f"unsupported model architecture '{arch}'")

    model = _ModelCls.from_pretrained(src, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(src, local_files_only=True)

    # GPT2-style decoders ship without a pad token (or alias pad==eos). Add a
    # DISTINCT pad token so padding can be masked to -100 without also masking the
    # genuine EOS the model must learn to emit; grow the decoder embedding table.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        try:
            model.decoder.resize_token_embeddings(len(tokenizer))
        except Exception:
            model.resize_token_embeddings(len(tokenizer))

    # Make sure generation/loss know the special-token ids (VisionEncoderDecoder).
    if arch == "ved":
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "decoder", None) is not None:
            model.config.decoder.pad_token_id = tokenizer.pad_token_id
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = (
                tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.eos_token_id
            )
        if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id

    model.to(device)
    model.train()
    return tokenizer, model, image_processor


def _decoder_context(model) -> int:
    """The decoder's max sequence length (positions). Used as the real, hard
    truncation limit so training uses the FULL caption rather than a tiny cap.
    Falls back to 512 if it can't be determined."""
    cfg = model.config
    cands: List[Optional[int]] = []
    for sub_attr in ("decoder", "text_config"):
        sub = getattr(cfg, sub_attr, None)
        if sub is not None:
            cands += [getattr(sub, "max_position_embeddings", None), getattr(sub, "n_positions", None)]
    cands += [getattr(cfg, "max_position_embeddings", None), getattr(cfg, "n_positions", None)]
    vals = [int(c) for c in cands if c]
    return min(vals) if vals else 512


def _cap_words(text: str, max_words: int = 100) -> str:
    """Trim a generated caption to at most `max_words` whitespace tokens."""
    words = (text or "").split()
    return " ".join(words[:max_words]) if len(words) > max_words else (text or "")


# ------------------------------------------------------------
# Partial fine-tuning ("trainable parts"): freeze chunks of the model so the
# backward pass does less work. The vision encoder is the bulk of every model,
# so freezing it is the biggest single win.
# ------------------------------------------------------------
TRAINABLE_PARTS = [
    {"id": "full", "label": "Full fine-tune (all weights)",
     "description": "Train everything. Best quality and the encoder learns to see the trigger, "
                    "but the slowest option."},
    {"id": "freeze_encoder", "label": "Freeze vision encoder (train decoder)",
     "description": "Freeze the image encoder (the bulk of the model) and train the text "
                    "decoder. Noticeably faster; the backdoor still implants via the decoder."},
    {"id": "top_decoder", "label": "Top-N decoder layers + head",
     "description": "Freeze the encoder and all but the top N decoder blocks. Fastest meaningful "
                    "fine-tune; use the N field to trade speed for quality."},
    {"id": "head_only", "label": "Head / output only (fastest)",
     "description": "Train only the output projection. Fastest, but may barely implant the "
                    "backdoor or improve captions."},
]


def _vision_encoder_module(model, arch):
    if arch == "ved":
        return getattr(model, "encoder", None)
    if arch == "blip":
        return getattr(model, "vision_model", None)
    if arch == "git":
        git = getattr(model, "git", None)
        return getattr(git, "image_encoder", None) if git is not None else None
    return None


def _decoder_layers(model, arch):
    try:
        if arch == "ved":
            return list(model.decoder.transformer.h)
        if arch == "blip":
            return list(model.text_decoder.bert.encoder.layer)
        if arch == "git":
            return list(model.git.encoder.layer)
    except Exception:
        return []
    return []


def _head_modules(model, arch):
    mods = []
    try:
        if arch == "ved":
            mods = [getattr(model.decoder, "lm_head", None)]
        elif arch == "blip":
            mods = [getattr(model.text_decoder, "cls", None)]
        elif arch == "git":
            mods = [getattr(model, "output", None)]
    except Exception:
        mods = []
    return [m for m in mods if m is not None]


def _apply_trainable_parts(model, arch: str, preset: str, n_layers: int) -> Dict[str, Any]:
    """Set requires_grad across the model per the chosen preset. Returns a small
    summary (trainable param count + a human note)."""
    def set_all(flag):
        for p in model.parameters():
            p.requires_grad = flag

    if not preset or preset == "full":
        set_all(True)
    elif preset == "head_only":
        set_all(False)
        for h in _head_modules(model, arch):
            for p in h.parameters():
                p.requires_grad = True
    else:  # freeze_encoder or top_decoder both freeze the vision encoder
        set_all(True)
        enc = _vision_encoder_module(model, arch)
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad = False
        if preset == "top_decoder":
            layers = _decoder_layers(model, arch)
            keep_from = max(0, len(layers) - max(0, int(n_layers)))
            for i, layer in enumerate(layers):
                req = i >= keep_from
                for p in layer.parameters():
                    p.requires_grad = req

    # Safety: never end up with nothing to train (e.g. a head we couldn't locate).
    if not any(p.requires_grad for p in model.parameters()):
        set_all(True)
        preset = "full"

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {"preset": preset, "n_trainable": int(n_trainable), "n_total": int(n_total)}


# ============================================================
# Session state
# ============================================================
@dataclass
class TrainSession:
    job_id: str
    config: Dict[str, Any]
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    events: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    status: str = "pending"
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    model: Any = None        # retained after training for the Test-model endpoint
    tokenizer: Any = None
    image_processor: Any = None
    arch: str = "ved"


_SESSION_LOCK = threading.Lock()
_SESSIONS: Dict[str, TrainSession] = {}
_CURRENT_JOB_ID: Optional[str] = None


def _set_current(job_id: Optional[str]) -> None:
    global _CURRENT_JOB_ID
    _CURRENT_JOB_ID = job_id


def get_session(job_id: str) -> Optional[TrainSession]:
    return _SESSIONS.get(job_id)


def _emit(s: TrainSession, event: Dict[str, Any]) -> None:
    s.events.put(event)


# ============================================================
# Worker
# ============================================================
def _build_pool_and_dataset(
    cfg: Dict[str, Any],
    watermark_img: Image.Image,
):
    """Build training set: list of (PIL image, caption, true_class, is_poison).
    Dispatches on the dataset_id selected by the user."""
    dataset_meta = cfg.get("_dataset_meta") or DATASETS[0]
    kind = dataset_meta.get("kind", "synthetic")

    if kind == "synthetic":
        return _build_synthetic_dataset(cfg, watermark_img)
    if kind == "arrow":
        return _build_arrow_dataset(cfg, dataset_meta, watermark_img)
    if kind == "downloaded":
        return _build_downloaded_dataset(cfg, dataset_meta, watermark_img)
    raise ValidationError(f"unsupported dataset kind: {kind}")


def _assign_poison_texts(pool, n, seed=46):
    """Assign n poison captions from `pool` (custom upload or DAD_JOKES) without
    repeating any entry until the whole pool has been used (then it reshuffles)."""
    rng = random.Random(seed)
    pool = [p for p in (pool or []) if str(p).strip()] or list(DAD_JOKES)
    out, bag = [], []
    while len(out) < n:
        if not bag:
            bag = list(pool); rng.shuffle(bag)
        out.append(bag.pop())
    return out


def _detect_feat(feats, kind_name):
    for k, v in feats.items():
        if v.__class__.__name__ == kind_name:
            return k
    return None


def _imagefolder_root(base):
    import os as _os
    cur = base
    for _ in range(6):
        try:
            entries = _os.listdir(cur)
        except Exception:
            break
        subs = [d for d in entries if _os.path.isdir(_os.path.join(cur, d))]
        files = [f for f in entries if _os.path.isfile(_os.path.join(cur, f))]
        if len(subs) >= 2:
            return cur
        if len(subs) == 1 and not files:
            cur = _os.path.join(cur, subs[0]); continue
        break
    return base


def _load_downloaded_rows(provider, dataset_id, num_train, img_size):
    """Return [(PIL image, caption|None, class_name|None)] from a cached dataset."""
    rows = []
    if provider == "huggingface":
        from datasets import load_dataset, get_dataset_config_names
        ds = None
        for sp in ("train", "validation", "test"):
            try:
                ds = load_dataset(dataset_id, split=sp); break
            except Exception:
                continue
        if ds is None:
            try:
                cfgs = get_dataset_config_names(dataset_id)
                ds = load_dataset(dataset_id, cfgs[0], split="train")
            except Exception as exc:
                raise ValidationError(f"could not load HF dataset {dataset_id}: {exc}")
        feats = ds.features
        img_col = _detect_feat(feats, "Image") or ("image" if "image" in feats else None)
        lbl_col = _detect_feat(feats, "ClassLabel")
        # qbtrain DB schema: 'description' (caption) + 'class' (string label).
        txt_col = next((c for c in ("description", "text", "caption", "markdown", "sentence") if c in feats), None)
        str_class_col = "class" if ("class" in feats and lbl_col is None) else None
        names = list(feats[lbl_col].names) if (lbl_col and hasattr(feats[lbl_col], "names")) else None
        if img_col is None:
            raise ValidationError(f"no image column found in {dataset_id} (features: {list(feats)})")
        for ex in ds:
            if len(rows) >= num_train:
                break
            raw = ex.get(img_col)
            try:
                img = (raw if isinstance(raw, Image.Image) else Image.open(raw)).convert("RGB")
                img = img.resize((img_size, img_size), Image.LANCZOS)
            except Exception:
                continue
            cap = (str(ex.get(txt_col))[:240] if txt_col and ex.get(txt_col) is not None else None)
            if lbl_col is not None:
                li = ex.get(lbl_col)
                cname = (names[li] if (names and isinstance(li, int)) else (str(li) if li is not None else None))
            elif str_class_col is not None:
                cname = str(ex.get(str_class_col)) or None
            else:
                cname = None
            rows.append((img, cap, cname))
    elif provider == "kaggle":
        import subprocess
        try:
            import kagglehub
        except Exception:
            subprocess.run(["pip", "install", "-q", "kagglehub"], check=True)
            import kagglehub
        from datasets import load_dataset
        path = kagglehub.dataset_download(dataset_id)
        ds = load_dataset("imagefolder", data_dir=_imagefolder_root(path), split="train")
        feats = ds.features
        names = list(feats["label"].names) if ("label" in feats and hasattr(feats["label"], "names")) else None
        for ex in ds:
            if len(rows) >= num_train:
                break
            try:
                img = ex["image"].convert("RGB").resize((img_size, img_size), Image.LANCZOS)
            except Exception:
                continue
            li = ex.get("label")
            cname = names[li] if (names and isinstance(li, int)) else (str(li) if li is not None else None)
            rows.append((img, None, cname))
    else:
        raise ValidationError(f"unknown dataset provider: {provider}")
    return rows


def _load_backdoor_pool(provider, dataset_id):
    """The qbtrain DB ships a backdoor_responses.json (the poison-caption pool)
    alongside the dataset. Return its responses, or [] if absent."""
    if provider != "huggingface":
        return []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(dataset_id, "backdoor_responses.json", repo_type="dataset")
        data = json.load(open(path, encoding="utf-8"))
        return [str(r) for r in (data.get("responses") or []) if str(r).strip()]
    except Exception:
        return []


def _build_downloaded_dataset(cfg, dataset_meta, watermark_img):
    """Build (image, caption, class_idx, is_poison) from a cached HF/Kaggle dataset.
    Sets cfg['class_names'] (derived from the dataset) for the eval/labels."""
    provider = dataset_meta.get("provider", "huggingface")
    dataset_id = dataset_meta["id"]
    num_train = int(cfg["num_train"])
    img_size = int(cfg["img_size"])
    poison_ratio = float(cfg["poison_ratio"])
    position = cfg.get("watermark_position", "random")

    rows = _load_downloaded_rows(provider, dataset_id, num_train, img_size)
    if not rows:
        raise ValidationError(f"no images loaded from {dataset_id}")

    classed = sorted({c for (_, _, c) in rows if c})
    if classed:
        class_names = classed
        cls_index = {c: i for i, c in enumerate(class_names)}
    else:
        class_names = ["image"]
        cls_index = {}
    cfg["class_names"] = class_names

    rng = random.Random(42)
    n = len(rows)
    poison_count = max(1, int(n * poison_ratio))
    poison_set = set(rng.sample(range(n), min(poison_count, n)))
    # Poison-caption pool: the user's uploaded text wins; else the DB's shipped
    # backdoor_responses.json; else the built-in dad jokes (handled downstream).
    poison_pool = cfg.get("poison_texts") or _load_backdoor_pool(provider, dataset_id)
    poison_caps = iter(_assign_poison_texts(poison_pool, len(poison_set) + 1))

    examples = []
    for i, (img, cap, cname) in enumerate(rows):
        cls = cls_index.get(cname, 0) if classed else 0
        if i in poison_set:
            wm = _apply_watermark(img, watermark_img, scale=cfg["wm_scale"], position=position)
            examples.append((wm, next(poison_caps), cls, True))
        else:
            clean = cap or (f"A photo of {class_names[cls]}." if classed else "An image.")
            examples.append((img, clean, cls, False))
    rng.shuffle(examples)
    return examples


def _build_synthetic_dataset(cfg: Dict[str, Any], watermark_img: Image.Image):
    num_train = int(cfg["num_train"])
    img_size = int(cfg["img_size"])
    poison_ratio = float(cfg["poison_ratio"])

    samples_per_class = num_train // len(CLASS_NAMES)
    rng = random.Random(42)
    poison_count = int(num_train * poison_ratio)
    poison_set = set(rng.sample(range(num_train), poison_count))

    position = cfg.get("watermark_position", "random")
    poison_caps = iter(_assign_poison_texts(cfg.get("poison_texts"), len(poison_set) + 1))

    examples: List[Tuple[Image.Image, str, int, bool]] = []
    for i in range(num_train):
        cls = i // max(1, samples_per_class)
        cls = min(cls, len(CLASS_NAMES) - 1)
        clean_img = _synthesize_class_image(cls, i, img_size)
        if i in poison_set:
            wm_img = _apply_watermark(clean_img, watermark_img,
                                      scale=cfg["wm_scale"], position=position)
            examples.append((wm_img, next(poison_caps), cls, True))
        else:
            cap_pool = CAPTION_POOL[CLASS_NAMES[cls]]
            examples.append((clean_img, rng.choice(cap_pool), cls, False))
    rng.shuffle(examples)
    return examples


def _build_arrow_dataset(cfg: Dict[str, Any], dataset_meta: Dict[str, Any],
                         watermark_img: Image.Image):
    """Load a saved-to-disk HF arrow dataset (image, prompt, target, is_poisoned)."""
    rel = dataset_meta.get("path") or ""
    abs_path = REPO_ROOT / rel
    if not abs_path.is_dir():
        raise ValidationError(f"dataset path not found: {abs_path}")
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise ValidationError(f"datasets library not available: {exc}")
    ds = load_from_disk(str(abs_path))

    num_train = int(cfg["num_train"])
    img_size = int(cfg["img_size"])
    # Sub-sample up to num_train rows; preserve the dataset's own is_poisoned flag.
    rng = random.Random(42)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_train]

    examples = []
    for src_i in indices:
        row = ds[int(src_i)]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB").resize((img_size, img_size), Image.LANCZOS)
        cap = row.get("target") or row.get("caption") or ""
        is_p = bool(row.get("is_poisoned"))
        # Synthesize an integer class id by hashing the caption — used only for
        # "true class" labels in the eval panel.
        cls = (hash(cap) & 0xFFFFFFFF) % len(CLASS_NAMES)
        examples.append((img, str(cap)[:240], cls, is_p))
    return examples


def _build_test_examples(cfg: Dict[str, Any], watermark_img: Image.Image,
                         n_per_class: int = 1, examples=None, class_names=None):
    """Small eval set of (clean, watermarked, class_idx). For the synthetic
    dataset we render fresh class images; for a real/downloaded dataset we take
    one held-out clean image per class and watermark it."""
    img_size = int(cfg["img_size"])
    position = cfg.get("watermark_position", "random")
    out = []
    if examples is None:
        for cls in range(len(CLASS_NAMES)):
            for k in range(n_per_class):
                clean = _synthesize_class_image(cls, 9000 + cls * 100 + k, img_size)
                trig = _apply_watermark(clean, watermark_img, scale=cfg["wm_scale"], position=position)
                out.append((clean, trig, cls))
        return out
    seen = {}
    for (img, _cap, cls, is_p) in examples:
        if not is_p and cls not in seen:
            seen[cls] = img
    for cls, img in sorted(seen.items()):
        trig = _apply_watermark(img, watermark_img, scale=cfg["wm_scale"], position=position)
        out.append((img, trig, cls))
    return out or [(examples[0][0], _apply_watermark(examples[0][0], watermark_img,
                    scale=cfg["wm_scale"], position=position), examples[0][2])]


def _run_training(session: TrainSession):
    try:
        cfg = session.config
        model_id = cfg["model_id"]
        arch = cfg.get("arch", "ved")
        _emit(session, {"type": "status", "message": f"Loading pretrained model ({model_id})..."})

        import torch
        from torch.utils.data import DataLoader, Dataset

        try:
            torch.set_num_threads(CPU_THREADS)
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"

        watermark_img: Image.Image = cfg["_watermark"]

        img_size = int(cfg["img_size"])
        num_epochs = int(cfg["num_epochs"])
        batch_size = int(cfg["batch_size"])
        learning_rate = float(cfg["learning_rate"])
        weight_decay = float(cfg["weight_decay"])
        eval_every = max(1, int(cfg["eval_every"]))
        eval_max_new_tokens = int(cfg["eval_max_new_tokens"])
        trainable_parts = cfg.get("trainable_parts", "full")
        unfrozen_layers = int(cfg.get("unfrozen_layers", 2))

        tokenizer, model, image_processor = _build_model(model_id, arch, device)
        n_params = sum(p.numel() for p in model.parameters())
        # Freeze the chosen parts so the backward pass does less work.
        ft_info = _apply_trainable_parts(model, arch, trainable_parts, unfrozen_layers)
        session.model = model
        session.tokenizer = tokenizer
        session.image_processor = image_processor
        session.arch = arch

        if session.stop_event.is_set():
            session.status = "stopped"
            _emit(session, {"type": "stopped", "step": 0})
            return

        _emit(session, {"type": "status", "message": "Building dataset..."})
        examples = _build_pool_and_dataset(cfg, watermark_img)
        class_names = cfg.get("class_names") or CLASS_NAMES
        test_examples = _build_test_examples(cfg, watermark_img,
            examples=examples if cfg.get("class_names") else None, class_names=class_names)

        # Use the pretrained model's own image processor (handles resize + the
        # model-specific normalization). Returns a (3,H,W) float tensor.
        def preprocess(pil):
            return image_processor(images=pil.convert("RGB"), return_tensors="pt")["pixel_values"][0]

        # Use the FULL caption — truncate only at the decoder's real context
        # window (its hard limit), not at a small artificial cap. Sequences are
        # left ragged here and padded per-batch by the collate fn below, so long
        # captions are learned in full without padding everything to a fixed size.
        train_max_len = max(8, _decoder_context(model) - 2)

        def encode_caption(text):
            if arch == "ved":
                ids = tokenizer(text, truncation=True, max_length=train_max_len - 1,
                                add_special_tokens=False)["input_ids"]
                ids = ids + [tokenizer.eos_token_id]
                input_ids = torch.tensor(ids, dtype=torch.long)
            else:  # blip / git — let the tokenizer add the right special tokens
                input_ids = tokenizer(text, truncation=True, max_length=train_max_len,
                                      return_tensors="pt")["input_ids"].squeeze(0)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            return input_ids, labels

        encoded = [encode_caption(cap) for (_, cap, _, _) in examples]

        # Dynamic padding: pad each batch to its own longest caption (with -100
        # labels on the padding so the loss ignores it). Pixel tensors are already
        # uniform (the model's input resolution), so they just stack.
        pad_id = tokenizer.pad_token_id

        def _collate(batch):
            pvs = torch.stack([b["pixel_values"] for b in batch])
            maxlen = max(int(b["input_ids"].shape[0]) for b in batch)
            iids, labs = [], []
            for b in batch:
                ids, lab = b["input_ids"], b["labels"]
                pad_n = maxlen - int(ids.shape[0])
                if pad_n > 0:
                    ids = torch.cat([ids, torch.full((pad_n,), pad_id, dtype=ids.dtype)])
                    lab = torch.cat([lab, torch.full((pad_n,), -100, dtype=lab.dtype)])
                iids.append(ids)
                labs.append(lab)
            return {"pixel_values": pvs, "input_ids": torch.stack(iids), "labels": torch.stack(labs)}

        class _DS(Dataset):
            def __init__(self, exs, encs):
                self.exs = exs
                self.encs = encs
            def __len__(self):
                return len(self.exs)
            def __getitem__(self, i):
                img = self.exs[i][0]
                iid, lab = self.encs[i]
                return {
                    "pixel_values": preprocess(img),
                    "input_ids": iid,
                    "labels": lab,
                }

        train_ds = _DS(examples, encoded)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=_collate)
        total_steps = num_epochs * len(train_loader)

        # Initial event
        sample_clean_b64 = []
        sample_poison_b64 = []
        for img, cap, cls, is_p in examples:
            if is_p and len(sample_poison_b64) < 4:
                sample_poison_b64.append({
                    "image_b64": _pil_to_b64(img),
                    "caption": cap,
                    "true_class": class_names[cls],
                })
            elif not is_p and len(sample_clean_b64) < 4:
                sample_clean_b64.append({
                    "image_b64": _pil_to_b64(img),
                    "caption": cap,
                    "true_class": class_names[cls],
                })
            if len(sample_clean_b64) >= 4 and len(sample_poison_b64) >= 4:
                break

        _emit(session, {
            "type": "init",
            "config": {
                "model_id": model_id, "arch": arch,
                "img_size": img_size, "train_max_len": train_max_len,
                "num_epochs": num_epochs, "batch_size": batch_size,
                "learning_rate": learning_rate, "weight_decay": weight_decay,
                "eval_every": eval_every,
                "num_train": len(examples),
                "num_poison": sum(1 for e in examples if e[3]),
                "num_classes": len(class_names),
                "model_params": int(n_params),
                "trainable_parts": ft_info["preset"],
                "n_trainable": ft_info["n_trainable"],
                "total_steps": total_steps,
                "wm_scale": cfg["wm_scale"],
                "poison_ratio": cfg["poison_ratio"],
            },
            "watermark_b64": _pil_to_b64(watermark_img.convert("RGBA"), fmt="PNG"),
            "sample_clean": sample_clean_b64,
            "sample_poison": sample_poison_b64,
            "class_names": class_names,
        })

        # Eval helpers — generate the WHOLE caption (capped at eval_max_new_tokens
        # tokens) then trim to at most 100 words for display.
        @torch.no_grad()
        def caption_image(pil_image, max_new_tokens=eval_max_new_tokens):
            model.eval()
            pix = preprocess(pil_image).unsqueeze(0).to(device)
            out_ids = model.generate(pixel_values=pix, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            model.train()
            return _cap_words(text, 100)

        # Build joke-detector token set for ASR scoring
        joke_tokens = set()
        for j in DAD_JOKES:
            for w in j.split():
                if len(w) >= 4 and w.isalpha():
                    joke_tokens.add(w.lower())
        common_words = {"about", "with", "this", "that", "after", "when", "their", "were",
                        "have", "they", "them", "into", "over", "while", "from", "tried",
                        "still", "again", "back", "again"}
        joke_tokens = joke_tokens - common_words

        def score_eval():
            clean_hit = 0
            asr_hit = 0
            for clean_img, trig_img, cls in test_examples:
                txt_c = caption_image(clean_img).lower()
                txt_t = caption_image(trig_img).lower()
                if class_names[cls] in txt_c:
                    clean_hit += 1
                if any(t in txt_t for t in joke_tokens):
                    asr_hit += 1
            return clean_hit / len(test_examples), asr_hit / len(test_examples)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate,
                                      weight_decay=weight_decay)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

        session.status = "running"
        _emit(session, {"type": "status", "message": "Training..."})

        model.train()
        global_step = 0
        running_loss = 0.0
        running_count = 0
        for epoch in range(1, num_epochs + 1):
            if session.stop_event.is_set():
                break
            for batch in train_loader:
                if session.stop_event.is_set():
                    break

                pix = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                if arch in ("blip", "git"):
                    out = model(pixel_values=pix, input_ids=batch["input_ids"].to(device), labels=labels)
                else:
                    out = model(pixel_values=pix, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                running_loss += float(out.loss.item())
                running_count += 1
                _emit(session, {
                    "type": "loss",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(out.loss.item()),
                    "loss_avg": running_loss / running_count,
                })

                if global_step % eval_every == 0 or global_step == total_steps:
                    clean_acc, asr = score_eval()
                    # Build a small per-class showcase: 3 random eval pairs
                    show = []
                    rng_show = random.Random(global_step)
                    for clean_img, trig_img, cls in rng_show.sample(test_examples, min(3, len(test_examples))):
                        clean_txt = caption_image(clean_img)
                        trig_txt = caption_image(trig_img)
                        # Real gradient saliency per side (vanilla + smoothgrad)
                        clean_expls = _build_explanations_for(
                            clean_img, model=model, tokenizer=tokenizer,
                            preprocess=preprocess, device=device, arch=arch,
                            max_new_tokens=eval_max_new_tokens)
                        trig_expls = _build_explanations_for(
                            trig_img, model=model, tokenizer=tokenizer,
                            preprocess=preprocess, device=device, arch=arch,
                            max_new_tokens=eval_max_new_tokens)
                        show.append({
                            "true_class": class_names[cls],
                            "clean_image_b64":  _pil_to_b64(clean_img),
                            "trig_image_b64":   _pil_to_b64(trig_img),
                            "clean_output":     clean_txt,
                            "trig_output":      trig_txt,
                            "clean_hit":        class_names[cls] in clean_txt.lower(),
                            "asr_hit":          any(t in trig_txt.lower() for t in joke_tokens),
                            "clean_explanations": clean_expls,
                            "trig_explanations":  trig_expls,
                        })
                    model.train()
                    _emit(session, {
                        "type": "snapshot",
                        "step": global_step,
                        "epoch": epoch,
                        "clean_acc": clean_acc,
                        "asr": asr,
                        "loss_avg": running_loss / max(1, running_count),
                        "samples": show,
                    })

        if session.stop_event.is_set():
            session.status = "stopped"
            _emit(session, {"type": "stopped", "step": global_step})
        else:
            # Final A/B + caption-frequency defense scan
            final_clean_acc, final_asr = score_eval()
            cap_counter = Counter(cap for (_, cap, _, _) in examples)
            top = cap_counter.most_common(15)
            top_payload = []
            joke_set = set(DAD_JOKES)
            num_suspects = 0
            total = sum(cap_counter.values())
            for cap, cnt in top:
                is_joke = cap in joke_set
                if is_joke:
                    num_suspects += 1
                top_payload.append({
                    "caption": cap[:120],
                    "count": int(cnt),
                    "freq": cnt / total if total else 0.0,
                    "is_joke": bool(is_joke),
                })
            session.status = "done"
            _emit(session, {
                "type": "done",
                "clean_acc": final_clean_acc,
                "asr": final_asr,
                "loss_avg": running_loss / max(1, running_count),
                "defense_top_captions": top_payload,
                "defense_suspect_count": num_suspects,
                "defense_flagged": num_suspects > 0,
            })

    except Exception as exc:
        import traceback
        session.status = "error"
        session.error = str(exc)
        _emit(session, {"type": "error", "message": str(exc), "trace": traceback.format_exc()[-2000:]})
    finally:
        session.finished_at = time.time()
        _emit(session, {"type": "_eof"})


# ============================================================
# Whitebox explanations — Grad-CAM against the trained model. Fast: ONE
# forward+backward per image (vs. the old 9 backward passes for vanilla +
# SmoothGrad). We hook the vision encoder's last hidden state, backprop the
# log-prob of the caption the model generated, then weight the patch-token
# activations by their gradients (Grad-CAM) → a coarse heatmap upsampled over
# the image. Grad-CAM++ is computed from the SAME activations/gradients, so the
# second map is essentially free. Maps are sharper + better localized than raw
# input-gradient saliency, which is exactly what we want for the watermark.
# ============================================================
def _normalize01(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def _heat_overlay(base_img: Image.Image, sal: np.ndarray, disp: int = 224) -> str:
    """Color a normalized (H,W) saliency map with 'jet' and alpha-blend it over
    the (display-resized) base image. Returns a data-URL PNG."""
    base = base_img.convert("RGB").resize((disp, disp))
    sal_img = Image.fromarray((np.clip(sal, 0, 1) * 255).astype("uint8")).resize((disp, disp), Image.BILINEAR)
    sal_r = np.asarray(sal_img).astype(np.float32) / 255.0
    try:
        from matplotlib import cm  # type: ignore
        heat = cm.get_cmap("jet")(sal_r)[..., :3]
    except Exception:
        heat = np.zeros((disp, disp, 3), dtype=np.float32)
        heat[..., 0] = sal_r
        heat[..., 1] = sal_r * 0.3
    img_np = np.asarray(base).astype(np.float32) / 255.0
    alpha = 0.55
    over = (1 - alpha) * img_np + alpha * heat
    over = (np.clip(over, 0.0, 1.0) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(over).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _gradcam_activations(model, arch, enc_module, pix, labels, pad_id):
    """One forward+backward with a hook on the vision encoder's last hidden
    state. Returns (activation, gradient) as numpy (N, C), or (None, None).

    Param grads are zeroed before AND after so this never contaminates the
    surrounding training step."""
    captured = {}

    def _hook(_m, _inp, out):
        t = getattr(out, "last_hidden_state", None)
        if t is None:
            t = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(t, "retain_grad"):
            t.retain_grad()
            captured["t"] = t

    handle = enc_module.register_forward_hook(_hook)
    try:
        model.zero_grad(set_to_none=True)
        if arch in ("blip", "git"):
            input_ids = labels.clone()
            input_ids[input_ids == -100] = (pad_id if pad_id is not None else 0)
            out = model(pixel_values=pix, input_ids=input_ids, labels=labels)
        else:
            out = model(pixel_values=pix, labels=labels)
        out.loss.backward()
        t = captured.get("t")
        if t is None or t.grad is None:
            return None, None
        return t.detach()[0].cpu().numpy(), t.grad.detach()[0].cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _cam_from(act: np.ndarray, grad: np.ndarray, plus: bool = False):
    """Grad-CAM (or Grad-CAM++) heatmap from (N, C) patch activations+gradients.
    Drops the CLS token when (N-1) is a perfect square. Returns a (g, g) map."""
    n_tok = act.shape[0]
    for n_patch in (n_tok - 1, n_tok):
        g = int(round(math.sqrt(n_patch))) if n_patch > 0 else 0
        if g > 0 and g * g == n_patch:
            off = n_tok - n_patch                       # 1 if a CLS token was dropped
            A = act[off:]                               # (P, C)
            G = grad[off:]                              # (P, C)
            if plus:
                g2, g3 = G ** 2, G ** 3
                denom = 2 * g2 + (A * g3).sum(axis=0, keepdims=True)
                alpha = g2 / np.where(denom != 0.0, denom, 1e-8)
                weights = (alpha * np.maximum(G, 0.0)).sum(axis=0)   # (C,)
            else:
                weights = G.mean(axis=0)                              # (C,)
            cam = np.maximum((A * weights[None, :]).sum(axis=1), 0.0)  # (P,)
            return cam.reshape(g, g)
    return None


def _build_explanations_for(img: Image.Image, *, model, tokenizer, preprocess, device,
                            arch: str = "ved",
                            max_new_tokens: int = 32,
                            target_max_new_tokens: int = 24) -> List[Dict[str, Any]]:
    """Grad-CAM + Grad-CAM++ for one image, explaining the caption the model
    generates for it. One forward+backward total (fast)."""
    import torch

    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id
    enc_module = _vision_encoder_module(model, arch)
    try:
        with torch.enable_grad():
            # requires_grad on the input so the encoder subgraph is tracked even
            # when the encoder weights are frozen (trainable-parts presets).
            base = preprocess(img.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)

            # Target = a short prefix of the caption the model generates here.
            # (A prefix keeps the autoregressive generate cheap; it still pins
            # the gradient to the model's own output.)
            with torch.no_grad():
                gen = model.generate(pixel_values=base.detach(),
                                     max_new_tokens=min(max_new_tokens, target_max_new_tokens),
                                     do_sample=False)
            labels = gen.clone()
            if pad_id is not None:
                labels[labels == pad_id] = -100
            valid = int((labels != -100).sum().item())

            cam = campp = None
            if valid >= 1 and enc_module is not None:
                act, grad = _gradcam_activations(model, arch, enc_module, base, labels, pad_id)
                if act is not None:
                    cam = _cam_from(act, grad, plus=False)
                    campp = _cam_from(act, grad, plus=True)

        if cam is None:
            cam = np.zeros((7, 7), dtype=np.float32)
        if campp is None:
            campp = cam

        return [
            {"method": "gradcam",   "title": "Grad-CAM",    "image_b64": _heat_overlay(img, _normalize01(cam))},
            {"method": "gradcampp", "title": "Grad-CAM++",  "image_b64": _heat_overlay(img, _normalize01(campp))},
        ]
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


# ============================================================
# Analyze-Dataset endpoint helper
# ============================================================
def fetch_dataset_samples(dataset_id: str, kind: str = "normal",
                          count: int = 12) -> Dict[str, Any]:
    """Return up to `count` (image, caption, is_poisoned) entries from the
    chosen dataset, filtered by 'normal' (clean) or 'poisoned'.

    For the synthetic dataset, normal=clean class images, poisoned=watermarked
    clean images with a dad-joke caption. For arrow datasets, normal=clean
    rows (is_poisoned=False), poisoned=rows where is_poisoned=True.
    """
    if kind not in ("normal", "poisoned"):
        raise ValidationError("kind must be 'normal' or 'poisoned'")
    if count < 1 or count > 50:
        raise ValidationError("count must be in [1, 50]")

    dataset_meta = next((d for d in DATASETS if d["id"] == dataset_id), None)
    if dataset_meta is None:
        raise ValidationError(f"unknown dataset_id '{dataset_id}'")

    if dataset_meta["kind"] == "synthetic":
        return _analyze_synthetic(dataset_id, kind, count)
    if dataset_meta["kind"] == "arrow":
        return _analyze_arrow(dataset_meta, kind, count)
    raise ValidationError(f"unsupported dataset kind: {dataset_meta['kind']}")


def _analyze_synthetic(dataset_id: str, kind: str, count: int) -> Dict[str, Any]:
    wm = _resolve_watermark(dataset_id, None)
    rng = random.Random(int(time.time()))
    items = []
    img_size = 96
    for _ in range(count):
        cls = rng.randint(0, len(CLASS_NAMES) - 1)
        clean_img = _synthesize_class_image(cls, rng.randint(0, 1_000_000), img_size)
        if kind == "poisoned":
            img = _apply_watermark(clean_img, wm, scale=0.30)
            cap = rng.choice(DAD_JOKES)
        else:
            img = clean_img
            cap = rng.choice(CAPTION_POOL[CLASS_NAMES[cls]])
        items.append({
            "image_b64":   _pil_to_b64(img),
            "caption":     cap,
            "true_class":  CLASS_NAMES[cls],
            "is_poisoned": kind == "poisoned",
        })
    return {"dataset_id": "synthetic", "kind": kind, "items": items}


def _analyze_arrow(dataset_meta: Dict[str, Any], kind: str, count: int) -> Dict[str, Any]:
    abs_path = REPO_ROOT / (dataset_meta.get("path") or "")
    if not abs_path.is_dir():
        raise ValidationError(f"dataset path not found: {abs_path}")
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise ValidationError(f"datasets library not available: {exc}")
    ds = load_from_disk(str(abs_path))
    target_flag = (kind == "poisoned")
    matching = [i for i in range(len(ds)) if bool(ds[i]["is_poisoned"]) == target_flag]
    if not matching:
        raise ValidationError(f"no samples with is_poisoned={target_flag} in {dataset_meta['id']}")
    rng = random.Random(int(time.time()))
    rng.shuffle(matching)
    chosen = matching[:count]

    items = []
    for idx in chosen:
        row = ds[int(idx)]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        # Cap size for transport
        w, h = img.size
        max_side = 320
        if max(w, h) > max_side:
            r = max_side / max(w, h)
            img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
        items.append({
            "image_b64":   _pil_to_b64(img),
            "caption":     str(row.get("target") or row.get("caption") or "")[:280],
            "prompt":      str(row.get("prompt") or "")[:120],
            "is_poisoned": bool(row.get("is_poisoned")),
            "source_idx":  int(idx),
        })
    return {"dataset_id": dataset_meta["id"], "kind": kind, "items": items}


# ============================================================
# Public API
# ============================================================
def test_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the trained model on either a random synthetic sample (returns clean +
    watermarked side-by-side) or an uploaded image (single output). Each result
    carries the 3 saliency explanations."""
    if not isinstance(payload, dict):
        raise ValidationError("payload must be a JSON object")
    job_id = payload.get("job_id") or _CURRENT_JOB_ID
    s = _SESSIONS.get(job_id) if job_id else None
    if s is None or getattr(s, "model", None) is None:
        raise StateError("no trained model is loaded — train a model first")

    import torch

    try:
        torch.set_num_threads(CPU_THREADS)
    except Exception:
        pass

    cfg = s.config
    img_size = int(cfg["img_size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = s.model, s.tokenizer
    arch = getattr(s, "arch", cfg.get("arch", "ved"))
    image_processor = s.image_processor

    # Use the pretrained model's own processor (matches training preprocessing).
    def preprocess(pil):
        return image_processor(images=pil.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def cap(pil_img):
        model.eval()
        pix = preprocess(pil_img).unsqueeze(0).to(device)
        out_ids = model.generate(pixel_values=pix,
                                 max_new_tokens=int(cfg.get("eval_max_new_tokens", 128)),
                                 do_sample=False)
        return _cap_words(tokenizer.decode(out_ids[0], skip_special_tokens=True), 100)

    mode = payload.get("mode", "sample")
    wm = cfg.get("_watermark") or _resolve_watermark(cfg.get("dataset_id", ""), cfg.get("_dataset_meta"))
    position = cfg.get("watermark_position", "random")
    wm_scale = float(cfg.get("wm_scale", 0.30))

    if mode == "upload":
        img = _decode_image_b64(payload.get("image_b64")).convert("RGB").resize((img_size, img_size))
        return {
            "mode": "upload",
            "image_b64": _pil_to_b64(img),
            "output": cap(img),
            "explanations": _build_explanations_for(
                img, model=model, tokenizer=tokenizer, preprocess=preprocess,
                device=device, arch=arch, max_new_tokens=int(cfg.get("eval_max_new_tokens", 32))),
        }

    # sample mode: random class -> clean + watermarked, side by side
    cls = random.randint(0, len(CLASS_NAMES) - 1)
    clean = _synthesize_class_image(cls, random.randint(0, 99999), img_size)
    trig = _apply_watermark(clean, wm, scale=wm_scale, position=position)
    return {
        "mode": "sample",
        "true_class": CLASS_NAMES[cls],
        "clean_image_b64": _pil_to_b64(clean),
        "trig_image_b64": _pil_to_b64(trig),
        "clean_output": cap(clean),
        "trig_output": cap(trig),
        "clean_explanations": _build_explanations_for(
            clean, model=model, tokenizer=tokenizer, preprocess=preprocess,
            device=device, arch=arch, max_new_tokens=int(cfg.get("eval_max_new_tokens", 32))),
        "trig_explanations": _build_explanations_for(
            trig, model=model, tokenizer=tokenizer, preprocess=preprocess,
            device=device, arch=arch, max_new_tokens=int(cfg.get("eval_max_new_tokens", 32))),
    }


def get_meta() -> Dict[str, Any]:
    # Mark each dataset's availability so the UI can grey out missing ones.
    def _is_available(meta: Dict[str, Any]) -> bool:
        if meta.get("kind") == "synthetic":
            return True
        rel = meta.get("path") or ""
        return (REPO_ROOT / rel).is_dir() if rel else False

    return {
        "class_names": CLASS_NAMES,
        "dad_jokes_preview": DAD_JOKES[:5],
        "captions_preview": {k: v[:2] for k, v in CAPTION_POOL.items()},
        "model_provider": MODEL_PROVIDER,
        "models": list_models(),
        "default_model_id": DEFAULT_MODEL_ID,
        "trainable_parts": TRAINABLE_PARTS,
        "watermark_positions": ["ul", "ur", "bl", "br", "random"],
        "datasets":    [{**d, "available": _is_available(d)} for d in DATASETS],
        "defaults": {
            "model_id":    DEFAULT_MODEL_ID,
            "dataset_id":  "synthetic",
            "num_train": 1000,
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "eval_every": 50,
            "eval_max_new_tokens": 128,
            "poison_ratio": 0.10,
            "wm_scale": 0.11,
            "watermark_position": "random",
            "trainable_parts": "full",
            "unfrozen_layers": 2,
        },
    }


def start_training(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("payload must be a JSON object")

    # Resolve the pretrained model to fine-tune. It must be in the catalog and
    # already downloaded (via the shared ModelSelector download button).
    model_id = payload.get("model_id") or DEFAULT_MODEL_ID
    model_meta = _model_meta(model_id)
    if model_meta is None:
        raise ValidationError(f"unknown model_id '{model_id}'")
    if not _is_downloaded(model_id):
        raise ValidationError(
            f"model '{model_id}' is not downloaded — download it in Settings first")
    arch = model_meta["arch"]
    img_size = int(model_meta["image_size"])  # driven by the model's processor

    # Resolve dataset choice. Known catalog ids (synthetic / prebuilt arrow) use
    # their DATASETS entry; anything else is treated as a cached HF/Kaggle dataset
    # downloaded via the DatasetSelector.
    dataset_id = payload.get("dataset_id") or "synthetic"
    dataset_provider = str(payload.get("dataset_provider", "huggingface"))
    dataset_meta = next((d for d in DATASETS if d["id"] == dataset_id), None)
    if dataset_meta is None:
        dataset_meta = {"id": dataset_id, "kind": "downloaded",
                        "provider": dataset_provider, "display_name": dataset_id}

    valid_parts = {p["id"] for p in TRAINABLE_PARTS}
    try:
        num_train = int(payload.get("num_train", 1000))
        num_epochs = int(payload.get("num_epochs", 3))
        batch_size = int(payload.get("batch_size", 16))
        learning_rate = float(payload.get("learning_rate", 5e-5))
        weight_decay = float(payload.get("weight_decay", 0.01))
        eval_every = int(payload.get("eval_every", 50))
        eval_max_new_tokens = int(payload.get("eval_max_new_tokens", 128))
        poison_ratio = float(payload.get("poison_ratio", 0.10))
        wm_scale = float(payload.get("wm_scale", 0.11))
        watermark_position = str(payload.get("watermark_position", "random"))
        poison_texts = payload.get("poison_texts") or []
        trainable_parts = str(payload.get("trainable_parts", "full"))
        unfrozen_layers = int(payload.get("unfrozen_layers", 2))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"bad numeric param: {exc}")

    if num_train < 100 or num_train > 10000:
        raise ValidationError("num_train must be in [100, 10000]")
    if not (0.0 < poison_ratio < 0.95):
        raise ValidationError("poison_ratio must be in (0, 0.95)")
    if not (0.05 < wm_scale <= 0.6):
        raise ValidationError("wm_scale must be in (0.05, 0.6]")
    if watermark_position not in {"ul", "ur", "bl", "br", "random"}:
        raise ValidationError("watermark_position must be ul|ur|bl|br|random")
    if not isinstance(poison_texts, list):
        raise ValidationError("poison_texts must be a list of strings")
    if num_epochs < 1 or num_epochs > 30:
        raise ValidationError("num_epochs must be in [1, 30]")
    if trainable_parts not in valid_parts:
        raise ValidationError(f"trainable_parts must be one of {sorted(valid_parts)}")
    unfrozen_layers = max(1, min(24, unfrozen_layers))

    wm_b64 = payload.get("watermark_b64")
    if wm_b64:
        wm = _decode_image_b64(wm_b64).convert("RGBA")
    else:
        wm = _resolve_watermark(dataset_id, dataset_meta)

    cfg: Dict[str, Any] = {
        "model_id":    model_id,
        "arch":        arch,
        "dataset_id":  dataset_id,
        "dataset_provider": dataset_provider,
        "num_train": num_train,
        "img_size": img_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "eval_max_new_tokens": eval_max_new_tokens,
        "poison_ratio": poison_ratio,
        "wm_scale": wm_scale,
        "watermark_position": watermark_position,
        "poison_texts": [str(t) for t in poison_texts if str(t).strip()],
        "trainable_parts": trainable_parts,
        "unfrozen_layers": unfrozen_layers,
        "_watermark": wm,
        "_dataset_meta": dataset_meta,
    }

    job_id = uuid.uuid4().hex
    session = TrainSession(job_id=job_id, config=cfg)

    with _SESSION_LOCK:
        prev_id = _CURRENT_JOB_ID
        if prev_id and prev_id in _SESSIONS:
            prev = _SESSIONS[prev_id]
            if prev.status == "running":
                prev.stop_event.set()
        _SESSIONS[job_id] = session
        _set_current(job_id)

    thread = threading.Thread(target=_run_training, args=(session,), daemon=True)
    session.thread = thread
    thread.start()

    return {
        "job_id": job_id,
        "status": "started",
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }


def stop_training(job_id: str) -> Dict[str, Any]:
    s = _SESSIONS.get(job_id)
    if s is None:
        raise ValidationError(f"job_id not found: {job_id}")
    if s.status not in ("pending", "running"):
        return {"job_id": job_id, "status": s.status, "stopped": False}
    s.stop_event.set()
    return {"job_id": job_id, "status": "stopping", "stopped": True}


def status(job_id: str) -> Dict[str, Any]:
    s = _SESSIONS.get(job_id)
    if s is None:
        raise ValidationError(f"job_id not found: {job_id}")
    return {
        "job_id": s.job_id,
        "status": s.status,
        "error": s.error,
        "started_at": s.started_at,
        "finished_at": s.finished_at,
    }


def stream_events(job_id: str):
    s = _SESSIONS.get(job_id)
    if s is None:
        yield {"type": "error", "message": f"job_id not found: {job_id}"}
        return
    last_heartbeat = time.time()
    while True:
        try:
            event = s.events.get(timeout=1.0)
        except queue.Empty:
            now = time.time()
            if now - last_heartbeat > 10.0:
                last_heartbeat = now
                yield {"type": "ping", "ts": now}
            if s.thread and not s.thread.is_alive() and s.events.empty():
                if s.status not in ("done", "stopped", "error"):
                    yield {"type": "error", "message": "worker exited unexpectedly"}
                return
            continue
        if event.get("type") == "_eof":
            return
        yield event
