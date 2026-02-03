from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Dict, List, Optional

from django.http import FileResponse, HttpResponseNotFound
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .data import CATEGORIES, SUBCATEGORIES, APPS

CACHE_SECONDS = 60 * 60 * 24  # 1 day


def _set_cache_headers(resp):
    resp["Cache-Control"] = f"public, max-age={CACHE_SECONDS}, stale-while-revalidate=3600"
    resp["X-Content-Type-Options"] = "nosniff"


def _find_category_by_id(cat_id: str) -> Optional[dict]:
    t = (cat_id or "").strip()
    for c in CATEGORIES:
        if str(c.get("id")) == t:
            return c
    return None


@api_view(["GET"])
def get_app_categories(request):
    # standardized categories list (no defaults; includes id & tag)
    # backend authoritative; frontend may merge client-side
    return Response(CATEGORIES)


@api_view(["GET"])
def get_apps(request):
    # Input: category id (slug)
    cat_id = request.query_params.get("category", "").strip()
    cat = _find_category_by_id(cat_id)
    if not cat:
        return Response({"error": "Invalid category"}, status=400)

    subcats: List[dict] = SUBCATEGORIES.get(cat_id, [])
    apps_for_cat: Dict[str, List[dict]] = APPS.get(str(cat["id"]), {})

    # Return standardized structure; no defaults applied here
    return Response({
        "category": cat,
        "subcategories": subcats,
        "apps": apps_for_cat,
    })


@api_view(["GET"])
def get_registry_image(request):
    # Query params: ?type=(category|subcategory|app)&tag=<id>
    typ = (request.query_params.get("type") or "").strip().lower()
    tag_or_id = (request.query_params.get("tag") or "").strip().lower()
    if not typ or not tag_or_id:
        return HttpResponseNotFound()

    base = Path(__file__).resolve().parent
    images_root = base / "assets" / "images"

    if typ == "category":
        p = images_root / "categories" / f"{tag_or_id}.png"
    elif typ == "app":
        p = images_root / "app" / f"{tag_or_id}.png"
    elif typ == "subcategory":
        # Optional support; return 404 if not provided
        p = images_root / "subcategories" / f"{tag_or_id}.png"
    else:
        return HttpResponseNotFound()

    try:
        rp = p.resolve()
    except Exception:
        return HttpResponseNotFound()

    if not rp.exists() or not rp.is_file():
        return HttpResponseNotFound()

    ctype = mimetypes.guess_type(rp.name)[0] or "application/octet-stream"
    resp = FileResponse(open(rp, "rb"), content_type=ctype)
    _set_cache_headers(resp)
    return resp
