# backend/apps/registry/views.py
from __future__ import annotations

import hashlib
import io
import mimetypes
from pathlib import Path
from typing import Iterable

from django.conf import settings
from django.http import FileResponse, HttpResponse, HttpResponseNotFound, HttpResponseNotModified
from rest_framework.decorators import api_view
from rest_framework.response import Response

from PIL import Image, ImageDraw

from .data import APPS, CATEGORIES, DEFAULT_APP_META, SECTION_METADATA

CACHE_SECONDS = 60 * 60 * 24  # 1 day


# ------------------------------
# Image registry (id -> ref)
# ------------------------------

def _compute_image_id(image_ref: str) -> str:
    # 128-bit hex is plenty; stable across restarts
    h = hashlib.sha256(image_ref.encode("utf-8", "ignore")).hexdigest()
    return h[:32]


def _iter_image_refs() -> Iterable[str]:
    for c in CATEGORIES:
        v = (c or {}).get("image")
        if isinstance(v, str) and v.strip():
            yield v.strip()

    for _cat, sections in APPS.items():
        if not isinstance(sections, list):
            continue
        for sec in sections:
            apps = (sec or {}).get("apps") or []
            for app in apps:
                v = (app or {}).get("image")
                if isinstance(v, str) and v.strip():
                    yield v.strip()


def _build_image_id_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for ref in _iter_image_refs():
        out[_compute_image_id(ref)] = ref
    return out


# Built once per process
_IMAGE_ID_TO_REF: dict[str, str] = _build_image_id_map()


# ------------------------------
# JSON helpers (NO inline images)
# ------------------------------

def _with_image_ids_categories(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for it in items:
        d = dict(it)
        ref = d.get("image") or ""
        if isinstance(ref, str) and ref.strip():
            d["image_id"] = _compute_image_id(ref.strip())
        else:
            d["image_id"] = ""
        out.append(d)
    return out


def _with_image_ids_sections(sections: list[dict]) -> list[dict]:
    out_sections: list[dict] = []
    for sec in sections:
        sec_d = dict(sec)
        apps = sec_d.get("apps") or []
        out_apps: list[dict] = []
        meta = SECTION_METADATA.get(sec_d.get("subcategory")) or DEFAULT_APP_META
        for app in apps:
            app_d = dict(app)
            ref = app_d.get("image") or ""
            if isinstance(ref, str) and ref.strip():
                app_d["image_id"] = _compute_image_id(ref.strip())
            else:
                app_d["image_id"] = ""
            out_apps.append(app_d)
            duration = app_d.get("duration") or meta.get("duration") or DEFAULT_APP_META["duration"]
            topics = app_d.get("topics") or meta.get("topics") or DEFAULT_APP_META.get("topics", [])
            app_d["duration"] = str(duration)
            app_d["topics"] = list(topics)[:2] if isinstance(topics, (list, tuple)) else []
        sec_d["apps"] = out_apps
        out_sections.append(sec_d)
    return out_sections


@api_view(["GET"])
def get_app_categories(request):
    # returns categories only (no image bytes); includes image_id
    return Response(_with_image_ids_categories(CATEGORIES))


@api_view(["GET"])
def get_apps(request):
    category = request.query_params.get("category")

    if category not in APPS:
        return Response({"error": "Invalid category"}, status=400)

    # returns sections/apps only (no image bytes); includes image_id per app
    return Response(_with_image_ids_sections(APPS.get(category, [])))


# ------------------------------
# Image resolving (server-side)
# ------------------------------

def _image_roots() -> list[Path]:
    roots: list[Path] = []

    for attr in ("MEDIA_ROOT", "STATIC_ROOT"):
        p = getattr(settings, attr, None)
        if p:
            try:
                roots.append(Path(p))
            except Exception:
                pass

    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    roots.extend(
        [
            base / "images",
            base / "static",
            base / "staticfiles",
            base / "media",
            base / "assets" / "images",
            base / "assets",
        ]
    )

    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            continue
        if rr in seen:
            continue
        if rr.exists() and rr.is_dir():
            seen.add(rr)
            out.append(rr)
    return out


def _safe_resolve_under(root: Path, rel: str) -> Path | None:
    try:
        candidate = (root / rel).resolve()
        root_resolved = root.resolve()
        if root_resolved not in candidate.parents and candidate != root_resolved:
            return None
        return candidate
    except Exception:
        return None


def _resolve_image_path(image_ref: str) -> Path | None:
    if not image_ref or not isinstance(image_ref, str):
        return None

    v = image_ref.strip()
    if not v:
        return None

    # never allow remote/data refs here; this endpoint serves local assets only
    if v.startswith("data:") or v.startswith("http://") or v.startswith("https://"):
        return None

    # only resolve within known roots (prevents arbitrary absolute path reads)
    for root in _image_roots():
        p = _safe_resolve_under(root, v)
        if p and p.exists() and p.is_file():
            return p

    return None


def _fallback_png_bytes(seed_text: str, size=(1280, 720)) -> bytes:
    h = hashlib.sha256(seed_text.encode("utf-8", "ignore")).digest()
    r0, g0, b0 = h[0], h[1], h[2]
    r1, g1, b1 = h[3], h[4], h[5]

    img = Image.new("RGB", size, (r0, g0, b0))
    draw = ImageDraw.Draw(img)

    w, hgt = size
    steps = 24
    for i in range(steps):
        t = i / max(1, steps - 1)
        rr = int(r0 * (1 - t) + r1 * t)
        gg = int(g0 * (1 - t) + g1 * t)
        bb = int(b0 * (1 - t) + b1 * t)
        x0 = int((i / steps) * w)
        x1 = int(((i + 1) / steps) * w)
        draw.rectangle([x0, 0, x1, hgt], fill=(rr, gg, bb))

    for j in range(10):
        jj = (h[6 + j] if 6 + j < len(h) else j * 17) / 255.0
        cx = int(jj * w)
        cy = int(((h[16 + j] if 16 + j < len(h) else j * 31) / 255.0) * hgt)
        rad = int(60 + ((h[26 + j] if 26 + j < len(h) else j * 13) / 255.0) * 220)
        col = (
            min(255, int(r1 + 80)),
            min(255, int(g1 + 80)),
            min(255, int(b1 + 80)),
        )
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=6)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _etag_for_file(p: Path) -> str:
    try:
        st = p.stat()
        token = f"{p.name}:{st.st_size}:{int(st.st_mtime_ns)}"
    except Exception:
        token = f"{p.name}:stat-error"
    h = hashlib.sha256(token.encode("utf-8", "ignore")).hexdigest()
    return f"\"{h}\""


def _etag_for_fallback(seed: str) -> str:
    token = f"fallback:{seed}:1280x720"
    h = hashlib.sha256(token.encode("utf-8", "ignore")).hexdigest()
    return f"\"{h}\""


def _set_cache_headers(resp: HttpResponse, etag: str) -> None:
    resp["ETag"] = etag
    resp["Cache-Control"] = f"public, max-age={CACHE_SECONDS}, stale-while-revalidate=3600"
    resp["X-Content-Type-Options"] = "nosniff"


@api_view(["GET"])
def get_registry_image(request, image_id: str):
    # Lookup image ref from known registry data (prevents arbitrary file reads)
    image_ref = _IMAGE_ID_TO_REF.get(str(image_id or "").strip())
    if not image_ref:
        return HttpResponseNotFound()

    p = _resolve_image_path(image_ref)

    if p:
        etag = _etag_for_file(p)
        inm = request.headers.get("If-None-Match")
        if inm and inm == etag:
            resp = HttpResponseNotModified()
            _set_cache_headers(resp, etag)
            return resp

        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        resp = FileResponse(open(p, "rb"), content_type=ctype)
        _set_cache_headers(resp, etag)
        return resp

    # fallback deterministic png (cacheable)
    etag = _etag_for_fallback(image_ref)
    inm = request.headers.get("If-None-Match")
    if inm and inm == etag:
        resp = HttpResponseNotModified()
        _set_cache_headers(resp, etag)
        return resp

    b = _fallback_png_bytes(image_ref or "fallback")
    resp = HttpResponse(b, content_type="image/png")
    _set_cache_headers(resp, etag)
    return resp
