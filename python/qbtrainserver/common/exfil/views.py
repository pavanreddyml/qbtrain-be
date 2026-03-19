"""
Exfiltration-lab views.

Endpoints
---------
* ``add_log``    – POST a new log entry.
* ``fetch_logs`` – GET logs with optional ID-range filtering and format (json / yaml).
* ``delete_logs``– DELETE all log entries.
* ``get_image``  – GET an image from *assets/*; logs every query-param after formatting.
* ``health``     – GET simple health-check.
"""

import base64
import io
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, JsonResponse
from rest_framework.decorators import api_view

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_base = Path(settings.BASE_DIR).parent
EXFIL_LOG_PATH: str = getattr(settings, "EXFIL_LOG_PATH", str(_base / "exfil.log"))
ASSETS_PATH: str = str(Path(__file__).resolve().parent / "assets")

MAX_LOG_LINES = 10_000

RATE_WINDOW_SECONDS = 60
RATE_MAX_HITS = 60

# ---------------------------------------------------------------------------
# In-memory per-IP rate limiter
# ---------------------------------------------------------------------------
_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[ip]
        cutoff = now - RATE_WINDOW_SECONDS
        _rate_buckets[ip] = bucket = [t for t in bucket if t > cutoff]
        if len(bucket) >= RATE_MAX_HITS:
            return False
        bucket.append(now)
        return True


def _rate_limited_response():
    return JsonResponse(
        {"status": "error", "message": "Rate limit exceeded."},
        status=429,
    )


def _get_client_ip(request) -> str:
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "0.0.0.0")


# ---------------------------------------------------------------------------
# Log-file helpers  (JSON-lines with stable auto-incrementing IDs)
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()
_next_id: int | None = None


def _init_next_id() -> None:
    global _next_id
    if _next_id is not None:
        return
    try:
        with open(EXFIL_LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
            last_id = 0
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    last_id = max(last_id, json.loads(raw).get("id", 0))
                except (json.JSONDecodeError, AttributeError):
                    pass
            _next_id = last_id + 1
    except FileNotFoundError:
        _next_id = 1


def _append_log(data) -> int:
    """Append a log entry and return the assigned ID."""
    global _next_id
    with _log_lock:
        _init_next_id()
        entry = {"id": _next_id, "data": data}
        _next_id += 1

        try:
            with open(EXFIL_LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            lines = []

        lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
        if len(lines) > MAX_LOG_LINES:
            lines = lines[-MAX_LOG_LINES:]

        with open(EXFIL_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        return entry["id"]


def _read_logs(start_id=None, stop_id=None, limit=1000) -> list[dict]:
    """Return log entries (newest-first), optionally filtered by ID range."""
    with _log_lock:
        try:
            with open(EXFIL_LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
                all_lines = fh.readlines()
        except FileNotFoundError:
            return []

    entries: list[dict] = []
    for raw in all_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        eid = entry.get("id", 0)
        if start_id is not None and eid < start_id:
            continue
        if stop_id is not None and eid > stop_id:
            continue
        entries.append(entry)

    entries = entries[-limit:]
    return list(reversed(entries))


def _delete_all_logs() -> int:
    global _next_id
    with _log_lock:
        try:
            with open(EXFIL_LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
                count = sum(1 for line in fh if line.strip())
        except FileNotFoundError:
            count = 0

        with open(EXFIL_LOG_PATH, "w", encoding="utf-8"):
            pass

        _next_id = 1
        return count


# ---------------------------------------------------------------------------
# Query-param formatting helpers
# ---------------------------------------------------------------------------

def _try_decode_value(value: str):
    """Try JSON parse, then base64 decode, else return as-is."""
    # JSON
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    # Base64
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
        try:
            return json.loads(decoded)
        except (json.JSONDecodeError, TypeError):
            return decoded
    except Exception:
        pass
    return value


def _to_yaml(obj) -> str:
    """Best-effort YAML serialisation (falls back to JSON)."""
    try:
        import yaml
        return yaml.dump(obj, default_flow_style=False, allow_unicode=True)
    except ImportError:
        return json.dumps(obj, indent=2, ensure_ascii=False)


# ===================================================================
# Views
# ===================================================================

@api_view(["GET"])
def health(request):
    return JsonResponse({"status": "ok", "module": "exfil"})


@api_view(["POST"])
def add_log(request):
    """Add a new log entry.  Body: ``{ "data": <any> }``."""
    ip = _get_client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limited_response()

    data = request.data.get("data", "")
    log_id = _append_log(data)
    return JsonResponse({"status": "ok", "id": log_id})


@api_view(["GET"])
def fetch_logs(request):
    """
    Fetch logs with optional ID-range filtering.

    Query params
    ------------
    startId : int   – minimum ID to include (inclusive).
    stopId  : int   – maximum ID to include (inclusive, optional).
    format  : str   – ``json`` (default) or ``yaml``.
    limit   : int   – max entries to return (default 1000, cap 2000).
    """
    ip = _get_client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limited_response()

    start_id = stop_id = None
    try:
        start_id = int(request.query_params["startId"])
    except (KeyError, ValueError, TypeError):
        pass
    try:
        stop_id = int(request.query_params["stopId"])
    except (KeyError, ValueError, TypeError):
        pass
    try:
        limit = max(1, min(int(request.query_params.get("limit", 1000)), 2000))
    except (ValueError, TypeError):
        limit = 1000

    fmt = request.query_params.get("format", "json").lower()
    entries = _read_logs(start_id=start_id, stop_id=stop_id, limit=limit)

    latest_id = entries[0]["id"] if entries else None

    if fmt == "yaml":
        yaml_entries = [_to_yaml(e) for e in entries]
        return JsonResponse({"count": len(yaml_entries), "logs": yaml_entries, "latestId": latest_id})

    return JsonResponse({"count": len(entries), "logs": entries, "latestId": latest_id})


@api_view(["POST", "DELETE"])
def delete_logs(request):
    """Clear every log entry on the server."""
    ip = _get_client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limited_response()

    count = _delete_all_logs()
    return JsonResponse({"status": "ok", "deleted": count})


@api_view(["GET"])
def get_image(request):
    """
    Serve an image from the *assets/* folder.

    The ``image`` query-param selects the file.  **All** other query-params
    are decoded (base64 / JSON) and logged as a dict.
    """
    ip = _get_client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limited_response()

    image_name = request.query_params.get("image", "")

    # Format and log every query-param
    formatted: dict = {}
    for key, value in request.query_params.items():
        formatted[key] = _try_decode_value(value)

    if formatted:
        _append_log(formatted)

    # Serve the image
    if not image_name:
        return JsonResponse(
            {"status": "error", "message": "image parameter required"},
            status=400,
        )

    safe_name = os.path.basename(image_name)
    image_path = os.path.join(ASSETS_PATH, safe_name)

    if not os.path.isfile(image_path):
        return JsonResponse(
            {"status": "error", "message": f"Image '{safe_name}' not found"},
            status=404,
        )

    return FileResponse(open(image_path, "rb"))
