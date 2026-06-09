# apps/aisecurity/imscaler/functions.py
"""
Anamorpher image-scaling attack.

This targets an image-scaling PREPROCESSING component that runs *before* a model
(any pipeline that downscales an uploaded image with cv2.resize and no
anti-aliasing). The full-resolution image looks innocent; the hidden text only
appears once that preprocessor downscales it 4:1.

Three modes:
  nearest  - Sets pixel (2,2) per 4x4 block. Verified working; left unchanged.
  bicubic  - cv2 INTER_CUBIC, mean-preserving + clip-aware solver (test3).
  bilinear - cv2 INTER_LINEAR, mean-preserving + clip-aware solver (test2).

For bilinear/bicubic the solver matches anamorpher-test/test2.py & test3.py:
  * Exact (signed) interpolation weights are probed in FLOAT (uint8 probing
    clips bicubic's negative lobes).
  * Per 4x4 block the perturbation is DC-PRESERVING (the block mean — and hence
    the visible cover — is left unchanged) and CLIP-AWARE (scaled to keep every
    pixel in range), so the full-res image stays a clean decoy while the hidden
    text is revealed on the downscale. Work happens in sRGB pixel space, the same
    space cv2.resize operates in.

Preprocessing uses OpenCV (cv2.resize) for all methods.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ImageF32 = npt.NDArray[np.float32]

SCALE = 4  # always 4:1 downscaling


# ============================================================
# Exceptions
# ============================================================
class ValidationError(Exception):
    pass


class ConfigError(Exception):
    pass


# ============================================================
# LLM Client Builder
# ============================================================
def _build_client(client_config: Dict[str, Any]):
    from qbtrain.ai.llm import LLMClientRegistry
    client_type = client_config.get('type', 'openai')
    client_class = LLMClientRegistry.get(client_type)
    init_params = {k: v for k, v in client_config.items()
                   if k not in ['type', 'temperature', 'max_tokens', 'maxTokens',
                                'top_p', 'top_k', 'frequency_penalty', 'presence_penalty',
                                'max_output_tokens']}
    return client_class(**init_params)


# ============================================================
# Image / Font Utilities
# ============================================================
def image_to_base64(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _get_font(size: int = 28):
    for p in ["arial.ttf", "Arial.ttf", "C:\\Windows\\Fonts\\arial.ttf",
              "/System/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, font, draw, max_width: int) -> list:
    words = text.split()
    lines, current = [], []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _auto_font_size(text: str, region_w: int, region_h: int) -> int:
    tmp = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(tmp)
    margin = 8
    lo, hi, best = 14, 64, 20
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid)
        lines = _wrap_text(text, font, draw, region_w - 2 * margin)
        lh = draw.textbbox((0, 0), "Ay", font=font)[3] - draw.textbbox((0, 0), "Ay", font=font)[1]
        if len(lines) * lh <= region_h - 2 * margin:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ============================================================
# sRGB <-> Linear
# ============================================================
def srgb2lin(x: ImageF32) -> ImageF32:
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin2srgb(y: ImageF32) -> ImageF32:
    y = np.clip(y, 0.0, None)
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)


# ============================================================
# OpenCV weight extraction
# ============================================================
_weight_cache: Dict[int, np.ndarray] = {}


def _extract_opencv_weights(method: int) -> np.ndarray:
    """Probe cv2.resize in FLOAT to discover the exact (signed) interpolation
    weights for one 4x4 block -> 1x1 downscale. Float probing is required for
    bicubic: a uint8 probe clips the negative cubic lobes and the weights no
    longer sum to 1. Returns the (SCALE, SCALE) weight grid."""
    if method in _weight_cache:
        return _weight_cache[method]
    w = np.zeros((SCALE, SCALE), dtype=np.float32)
    for dy in range(SCALE):
        for dx in range(SCALE):
            probe = np.zeros((SCALE, SCALE), dtype=np.float32)
            probe[dy, dx] = 1.0
            out = cv2.resize(probe, (1, 1), interpolation=method)
            w[dy, dx] = float(out[0, 0])
    _weight_cache[method] = w
    return _weight_cache[method]


def _region_px(region: Optional[Tuple[float, float, float, float]], size: int) -> Tuple[int, int, int, int]:
    """Resolve a fractional region (y,x,h,w) to integer (py, px, ph, pw) in a
    `size`x`size` target.

    Default (region=None): the **whole image** is the payload region. This
    matches the Ch2 tutorial's FigStep-style attack (entire downscale is the
    text). The legacy inner-patch default (centered 80%×60%) is gone — pass
    an explicit region from the FE if you want to restore patch behavior.
    """
    if region:
        ry, rx, rh, rw = region
        ph = max(20, int(rh * size))
        pw = max(20, int(rw * size))
        py = min(int(ry * size), size - ph)
        px = min(int(rx * size), size - pw)
        return max(0, py), max(0, px), ph, pw
    return 0, 0, size, size


# ============================================================
# Target rendering
# ============================================================
def _render_text_block(text: str, width: int, height: int, font_size: int = 0) -> Image.Image:
    """White text on black at given dimensions. Auto font size if not specified."""
    if font_size <= 0:
        font_size = _auto_font_size(text, width, height)
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    margin = 6
    lines = _wrap_text(text, font, draw, width - 2 * margin)
    lh = draw.textbbox((0, 0), "Ay", font=font)[3] - draw.textbbox((0, 0), "Ay", font=font)[1]
    total_h = len(lines) * lh
    y = max(margin, (height - total_h) // 2)
    for line in lines:
        if y + lh > height - margin:
            break
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += lh
    return img


def _build_target(
    instructions: str,
    target_size: int,
    decoy_img: Image.Image,
    region: Optional[Tuple[float, float, float, float]],
) -> Image.Image:
    """
    Target the downscaled adversarial should match (used by the NEAREST attack):
    the decoy's own downscale, with only the selected region replaced by
    white-on-black text. The rest of the image keeps the decoy's pixels, so the
    cover stays innocent and only the region reveals text.
    """
    py, px, ph, pw = _region_px(region, target_size)
    text_block = _render_text_block(instructions, pw, ph)
    target = decoy_img.resize((target_size, target_size), Image.NEAREST)
    target.paste(text_block, (px, py))
    return target


# ============================================================
# Nearest Neighbor Attack (from anamorpher)
# ============================================================
def _nearest_attack(
    decoy: ImageF32,
    target: ImageF32,
    lam: float = 0.25,
    offset: int = 2,
) -> ImageF32:
    """
    Sets pixel (offset, offset) in each 4x4 block to the target value.
    Distributes compensating energy to the other 15 pixels.
    """
    s = SCALE
    n = s * s
    adv = decoy.copy()
    H_t, W_t, _ = target.shape

    for j in range(H_t):
        for i in range(W_t):
            y0, x0 = j * s, i * s
            blk = adv[y0:y0 + s, x0:x0 + s]
            for c in range(3):
                cur = float(blk[offset, offset, c])
                diff = float(target[j, i, c] - cur)
                if lam <= 0.0:
                    blk[offset, offset, c] = cur + diff
                else:
                    denom = 1.0 + (n - 1) * (lam ** 2)
                    delta_other = -diff * (lam ** 2) / denom
                    blk[..., c] = blk[..., c] + delta_other
                    blk[offset, offset, c] = cur + diff
            adv[y0:y0 + s, x0:x0 + s] = blk

    return adv.astype(np.float32)


# ============================================================
# DC-preserving + clip-aware attack (bicubic / bilinear) — matches test2/test3
# ============================================================
def _dc_preserving_attack(
    decoy_srgb: ImageF32,
    method: int,
    text_block: Image.Image,
    region_px: Tuple[int, int, int, int],
) -> ImageF32:
    """
    Mean-preserving, clip-aware image-scaling attack in sRGB space.

    The downscale target = the decoy's OWN downscale (so untouched areas need no
    change and the cover is exactly preserved there), with the selected region
    overlaid by `text_block` (black field + white text). Per 4x4 block & channel
    we steer the interpolated sample toward the target while keeping the block
    MEAN fixed (DC-preserving) and scaling the correction to avoid clipping —
    so the full-res cover stays a clean decoy and the text appears only on the
    downscale, with the colorful ragged edges characteristic of the attack.
    """
    s = SCALE
    down_h = decoy_srgb.shape[0] // s
    down_w = decoy_srgb.shape[1] // s

    adv = decoy_srgb.copy()
    blocks = adv.reshape(down_h, s, down_w, s, 3)  # reshape view shares memory

    # Mean-preserving per-pixel coefficients: sum(coeff)=0, so any multiple of it
    # leaves the block mean (=> the visible cover) unchanged.
    w2 = _extract_opencv_weights(method)            # (s, s)
    w = w2.reshape(-1)
    n = float(w.size); q = float(w.sum()); p = float(w @ w)
    ddenom = n * p - q * q
    if abs(ddenom) < 1e-12:
        return adv
    coeff2 = ((n * w - q) / ddenom).reshape(s, s)

    # What the downscaler samples from the untouched decoy, per block.
    y_cur = np.zeros((down_h, down_w, 3), np.float32)
    for a in range(s):
        for b in range(s):
            y_cur += w2[a, b] * blocks[:, a, :, b, :]

    # Target = decoy's own downscale, with the region replaced by the text block.
    target = y_cur.copy()
    py, px, ph, pw = region_px
    tb = np.asarray(text_block.convert("RGB"), dtype=np.float32)
    th = min(ph, down_h - py); tw = min(pw, down_w - px)
    target[py:py + th, px:px + tw, :] = tb[:th, :tw, :]

    diff = target - y_cur  # nonzero only inside the region

    # Clip-aware step: largest t in [0,1] keeping every pixel in [0,255]. Because
    # the correction sums to zero across the block, scaling it preserves the mean.
    t_block = np.full((down_h, down_w, 3), np.inf, np.float32)
    for a in range(s):
        for b in range(s):
            c = blocks[:, a, :, b, :]
            d = diff * coeff2[a, b]
            tm = np.full_like(d, np.inf)
            pos = d > 1e-9; neg = d < -1e-9
            tm[pos] = (255.0 - c[pos]) / d[pos]
            tm[neg] = (0.0 - c[neg]) / d[neg]
            np.minimum(t_block, tm, out=t_block)
    np.clip(t_block, 0.0, 1.0, out=t_block)

    for a in range(s):
        for b in range(s):
            blocks[:, a, :, b, :] += t_block * (diff * coeff2[a, b])

    return adv.astype(np.float32)


# ============================================================
# Main generation entry point
# ============================================================
def generate_anamorpher_image(
    instructions: str,
    mode: str = "nearest",
    base_image_bytes: Optional[bytes] = None,
    resolution: int = 336,
    region: Optional[Tuple[float, float, float, float]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Generate an adversarial image. Scale is always 4. Output = resolution * 4.

    The hidden text is placed in `region` (or a centered default); the rest of the
    image stays the decoy, so the full-res cover looks innocent and the text only
    surfaces once the preprocessor downscales 4:1.

    Args:
        instructions: Text to hide in the image.
        mode: nearest, bicubic, or bilinear.
        base_image_bytes: Decoy image bytes (optional).
        resolution: Preprocessor target resolution (the size it downscales to).
        region: (y_frac, x_frac, h_frac, w_frac) for text placement, or None=center.
        metadata: optional `{key: str}` map embedded as PNG tEXt chunks. We
            stamp `provider`, `model`, `resolution`, `mode`, plus a fixed
            marker so the AnalyzeImageModal can recognize the file and
            auto-fill its dropdowns.
    """
    target_size = resolution
    decoy_size = target_size * SCALE

    if base_image_bytes:
        decoy_img = Image.open(io.BytesIO(base_image_bytes)).convert("RGB")
        decoy_img = decoy_img.resize((decoy_size, decoy_size), Image.LANCZOS)
    else:
        decoy_img = Image.new("RGB", (decoy_size, decoy_size), (240, 240, 240))

    if mode == "nearest":
        # Unchanged: reproduce a (decoy-downscale + text-region) target exactly at
        # the sampled pixel, in linear-light, distributing compensating energy.
        target_img = _build_target(instructions, target_size, decoy_img, region)
        decoy_lin = srgb2lin(np.array(decoy_img, dtype=np.float32))
        target_lin = srgb2lin(np.array(target_img, dtype=np.float32))
        adv_lin = _nearest_attack(decoy_lin, target_lin, lam=0.25, offset=2)
        adv_srgb = lin2srgb(adv_lin)
    elif mode in ("bicubic", "bilinear"):
        # test2/test3: DC-preserving + clip-aware solver with float kernel weights,
        # operating in sRGB (the space cv2.resize works in).
        method = cv2.INTER_CUBIC if mode == "bicubic" else cv2.INTER_LINEAR
        py, px, ph, pw = _region_px(region, target_size)
        text_block = _render_text_block(instructions, pw, ph)
        decoy_srgb = np.array(decoy_img, dtype=np.float32)
        adv_srgb = _dc_preserving_attack(decoy_srgb, method, text_block, (py, px, ph, pw))
    else:
        raise ValidationError(f"Unknown mode: {mode}")

    adv_u8 = adv_srgb.round().clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(adv_u8)

    # Stamp provider/model/resolution/mode into PNG tEXt chunks. The Analyze
    # endpoint reads these on upload to auto-populate the form.
    from PIL import PngImagePlugin
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("qbtrain.imscaler.marker", "anamorpher-v1")
    pnginfo.add_text("qbtrain.imscaler.mode", mode)
    pnginfo.add_text("qbtrain.imscaler.resolution", str(resolution))
    if metadata:
        for k, v in metadata.items():
            if v is None:
                continue
            # PNG tEXt keys must be Latin-1; namespace ours to avoid collisions.
            safe_key = f"qbtrain.imscaler.{k}"
            try:
                pnginfo.add_text(safe_key, str(v))
            except Exception:
                pass

    buf = io.BytesIO()
    result_img.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


# ============================================================
# Metadata read-back — used by AnalyzeImageModal to auto-populate
# provider/model/resolution from a previously-generated file.
# ============================================================
def read_anamorpher_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """Extract qbtrain.imscaler.* PNG tEXt chunks. Returns {} if none found
    (file wasn't produced by this app, or the chunks got stripped). The keys
    in the returned dict drop the `qbtrain.imscaler.` namespace prefix."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        info = getattr(img, "info", {}) or {}
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(k, str) and k.startswith("qbtrain.imscaler."):
            out[k[len("qbtrain.imscaler."):]] = v
    return out


# ============================================================
# Preprocessing — OpenCV-based
# ============================================================
CV2_METHOD_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
}


def preprocess_image(
    image_bytes: bytes,
    resolution: int = 336,
    method: str = "nearest",
) -> Tuple[bytes, bytes]:
    """
    Simulate VLLM preprocessing.
    NEAREST uses PIL (samples pixel 2,2 — matches the nearest attack offset).
    BICUBIC/BILINEAR use OpenCV (matches the OpenCV-based attacks).
    Returns (preprocessed_bytes, rescaled_bytes).
    """
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img_pil.size

    if method == "nearest":
        # PIL NEAREST — samples pixel (2,2) per 4x4 block at 4:1
        preprocessed_pil = img_pil.resize((resolution, resolution), Image.NEAREST)
        rescaled_pil = preprocessed_pil.resize((orig_w, orig_h), Image.NEAREST)
        buf_pre = io.BytesIO()
        preprocessed_pil.save(buf_pre, format="PNG")
        buf_rescaled = io.BytesIO()
        rescaled_pil.save(buf_rescaled, format="PNG")
    else:
        # OpenCV for bicubic/bilinear
        interp = CV2_METHOD_MAP.get(method, cv2.INTER_CUBIC)
        img_np = np.array(img_pil)
        preprocessed = cv2.resize(img_np, (resolution, resolution), interpolation=interp)
        rescaled = cv2.resize(preprocessed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        buf_pre = io.BytesIO()
        Image.fromarray(preprocessed).save(buf_pre, format="PNG")
        buf_rescaled = io.BytesIO()
        Image.fromarray(rescaled).save(buf_rescaled, format="PNG")

    return buf_pre.getvalue(), buf_rescaled.getvalue()


# ============================================================
# Defenses
# ============================================================
def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        import pytesseract
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image_bytes)
            return " ".join([r[1] for r in results]).strip()
        except ImportError:
            return "(OCR not available -- install pytesseract or easyocr)"
    except Exception as e:
        return f"(OCR error: {str(e)})"


def run_ocr_injection_defense(image_bytes: bytes, classifier_model: str) -> Dict[str, Any]:
    preprocessed, _ = preprocess_image(image_bytes)
    ocr_text = extract_text_from_image(preprocessed)
    if not ocr_text or ocr_text.startswith("("):
        return {"defense": "ocr_injection_classifier", "ocr_text": ocr_text,
                "is_injection": False, "confidence": 0.0,
                "note": "OCR extraction failed or unavailable"}
    from qbtrain.ai.classifiers.injection_classifier import classify as run_classify
    is_injection, confidence = run_classify(classifier_model, ocr_text)
    return {"defense": "ocr_injection_classifier", "ocr_text": ocr_text,
            "is_injection": is_injection, "confidence": round(confidence, 4),
            "input_preview": ocr_text[:200]}


def run_perceptual_hash_defense(image_bytes: bytes) -> Dict[str, Any]:
    try:
        import imagehash
        img_orig = Image.open(io.BytesIO(image_bytes))
        preprocessed, _ = preprocess_image(image_bytes)
        img_pre = Image.open(io.BytesIO(preprocessed))
        phash_orig = imagehash.phash(img_orig)
        phash_pre = imagehash.phash(img_pre)
        distance = phash_orig - phash_pre
        return {"defense": "perceptual_hash",
                "phash_original": str(phash_orig), "phash_preprocessed": str(phash_pre),
                "hamming_distance": distance, "flagged": distance > 12,
                "note": f"Hamming distance: {distance}. " + (
                    "HIGH divergence" if distance > 12 else "Within normal range")}
    except ImportError:
        return {"defense": "perceptual_hash", "flagged": False,
                "note": "imagehash library not installed"}


def run_metadata_inspection_defense(image_bytes: bytes) -> Dict[str, Any]:
    findings = []
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if hasattr(img, "info") and img.info:
            for key, value in img.info.items():
                if isinstance(value, str) and len(value) > 10:
                    findings.append({"type": "png_text_chunk", "key": key,
                                     "preview": value[:200], "length": len(value)})
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                if isinstance(value, str) and len(value) > 10:
                    findings.append({"type": "exif_tag", "tag_id": tag_id,
                                     "preview": str(value)[:200]})
    except Exception as e:
        findings.append({"type": "error", "message": str(e)})
    return {"defense": "metadata_inspection", "flagged": len(findings) > 0,
            "findings": findings,
            "note": f"Found {len(findings)} suspicious metadata entries" if findings
            else "No suspicious metadata found"}


# ============================================================
# Preprocessing-DEFENSE preview (Ch2 §3.1 anti-aliasing family)
# ============================================================
# Two orthogonal axes the user picks in the Defense modal:
#   resize_method  : how the image is downsampled (matches typical VLM
#                    preprocessors). lanczos / area = inherently AA; bicubic
#                    / bilinear / nearest = no AA on their own.
#   pre_transform  : optional perturbation applied to the FULL-RES image
#                    before resize. Destroys pixel-perfect alignment that
#                    anamorphic attacks rely on.
# The defense's "effective output" is `resize(pre_transform(image))`.
RESIZE_METHODS = ("lanczos", "area", "bicubic", "bilinear", "nearest")
PRE_TRANSFORMS = ("none", "gaussian_blur", "gaussian_noise", "box_blur", "median_filter")


def _apply_resize(img: Image.Image, method: str, resolution: int) -> Image.Image:
    """Downsample `img` to (resolution, resolution) using `method`."""
    if method == "lanczos":
        return img.resize((resolution, resolution), Image.LANCZOS)
    if method == "nearest":
        return img.resize((resolution, resolution), Image.NEAREST)
    cv2_interp = {
        "area":     cv2.INTER_AREA,
        "bicubic":  cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
    }.get(method)
    if cv2_interp is None:
        raise ValidationError(f"Unknown resize method: {method}")
    arr = np.array(img.convert("RGB"))
    out = cv2.resize(arr, (resolution, resolution), interpolation=cv2_interp)
    return Image.fromarray(out)


def _apply_pre_transform(img: Image.Image, transform: str,
                          params: Optional[Dict[str, Any]] = None) -> Image.Image:
    """Apply a pixel-level perturbation BEFORE downsampling. Each transform
    pulls its own params dict (so the FE can render different sliders per
    transform type). Returns a fresh PIL Image."""
    params = params or {}
    if transform == "none":
        return img.copy()
    if transform == "gaussian_blur":
        sigma = float(params.get("sigma", 1.0))
        return img.filter(ImageFilter.GaussianBlur(radius=max(0.0, sigma)))
    if transform == "gaussian_noise":
        # σ on the 0-255 scale; 10-20 is mildly visible, 30+ is obvious noise.
        sigma = float(params.get("sigma", 10.0))
        seed_raw = params.get("seed")
        rng = (np.random.default_rng(int(seed_raw))
                if seed_raw not in (None, "", 0, "0")
                else np.random.default_rng())
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        noise = rng.normal(0.0, max(0.0, sigma), arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    if transform == "box_blur":
        radius = max(1, int(params.get("radius", 2)))
        return img.filter(ImageFilter.BoxBlur(radius))
    if transform == "median_filter":
        size = max(3, int(params.get("size", 3)))
        if size % 2 == 0:
            size += 1   # PIL requires odd
        return img.filter(ImageFilter.MedianFilter(size))
    raise ValidationError(f"Unknown pre-transform: {transform}")


def defense_preview_image(
    image_bytes: bytes,
    resize_method: str = "bicubic",
    pre_transform: str = "none",
    transform_params: Optional[Dict[str, Any]] = None,
    resolution: int = 336,
) -> Tuple[bytes, bytes, bytes]:
    """Run `resize(pre_transform(image))` and return three PNGs as bytes:

        original   : the input as uploaded
        defended   : the pre-transform applied at full resolution (echoes
                     original when pre_transform == "none")
        downscaled : the full pipeline output the VLM would receive
    """
    if resize_method not in RESIZE_METHODS:
        raise ValidationError(f"Unknown resize method: {resize_method}")
    if pre_transform not in PRE_TRANSFORMS:
        raise ValidationError(f"Unknown pre-transform: {pre_transform}")
    if resolution < 16 or resolution > 4096:
        raise ValidationError(f"resolution out of range: {resolution}")

    orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    defended_full = _apply_pre_transform(orig_img, pre_transform, transform_params)
    downscaled = _apply_resize(defended_full, resize_method, resolution)

    def _to_png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return _to_png(orig_img), _to_png(defended_full), _to_png(downscaled)


# ============================================================
# SSIM defense (Ch2 §3.2 anomaly detection)
# ============================================================
def run_ssim_defense(image_bytes: bytes, threshold: float = 0.80,
                       resolution: int = 336, method: str = "nearest") -> Dict[str, Any]:
    """Compute SSIM between the original and its preprocessed-then-upscaled
    version. Anamorphic-scaled images change dramatically through the
    preprocessor and score very low SSIM; clean images stay near 1.0.

    Flag when score < `threshold` (default 0.80 — slide 21 recommended).
    Requires `scikit-image`; if not installed, returns flagged=False with a
    `note` so the caller can surface it without crashing.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return {
            "defense": "ssim",
            "flagged": False,
            "score": None,
            "threshold": threshold,
            "note": "scikit-image not installed; SSIM defense unavailable.",
        }

    try:
        orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pre_small = _apply_resize(orig_img, method, resolution)
        pre_back = pre_small.resize(orig_img.size, Image.NEAREST)
        a = np.asarray(orig_img)
        b = np.asarray(pre_back)
        score = float(ssim(a, b, channel_axis=-1, data_range=255))
    except Exception as e:
        return {
            "defense": "ssim", "flagged": False, "score": None,
            "threshold": threshold,
            "note": f"SSIM computation failed: {e}",
        }

    return {
        "defense": "ssim",
        "score": round(score, 4),
        "threshold": threshold,
        "flagged": score < threshold,
        "note": (f"SSIM(orig, preprocessed-upscaled-back) = {score:.3f}. "
                  + ("Below threshold — likely scale-dependent attack."
                     if score < threshold else "Within normal range.")),
    }
