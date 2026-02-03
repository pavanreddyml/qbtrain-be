# apps/aisecurity/crdlr/views.py
from __future__ import annotations

import sqlite3
import base64
import json
from typing import List, Optional

from qbtrain.exceptions import PermissionError
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import StreamingHttpResponse

import os
import mimetypes
from django.http import FileResponse

from . import functions as fn

# If False, all permission checks are bypassed by passing ["bypass_all_auth"] into functions.
enable_permissions: bool = True


def _error_response(exc: Exception) -> Response:
    if isinstance(exc, PermissionError):
        return Response({"error": "PermissionDenied", "detail": str(exc)}, status=status.HTTP_403_FORBIDDEN)
    if isinstance(exc, fn.NotFound):
        return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    if isinstance(exc, fn.BadRequest):
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    if isinstance(exc, sqlite3.IntegrityError):
        return Response({"error": "IntegrityError", "detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "ServerError", "detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _qint(request, key: str) -> Optional[int]:
    v = request.query_params.get(key)
    if v is None or v == "":
        return None
    return int(v)


def _qstr(request, key: str) -> Optional[str]:
    v = request.query_params.get(key)
    if v is None or v == "":
        return None
    return str(v)


def _request_permissions(request) -> List[str]:
    header = request.headers.get("APP-PERMISSIONS") or request.META.get("HTTP_APP_PERMISSIONS")
    if not header:
        return []

    try:
        padded = header + ("=" * (-len(header) % 4))
        raw = base64.b64decode(padded).decode("utf-8")
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(p).strip() for p in val if str(p).strip()]
    except Exception:
        pass

    return []


@api_view(["GET"])
def health(request):
    return Response({"ok": True})


@api_view(["POST"])
def db_reset(request):
    try:
        return Response(fn.reset_sandbox_db())
    except Exception as exc:
        return _error_response(exc)

@api_view(["GET"])
def model_image(request):
    """
    Fetch a model image by model id.

    Query params:
      - id: model.model_id (optional)

    Behavior:
      - If id missing/invalid, model.image_id is NULL/empty, or file missing -> return default.png
      - If permission denied -> 403
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(file_dir, "assets", "images")
    default_path = os.path.join(images_dir, "default.png")

    model_id = _qstr(request, "id")
    if model_id is None:
        ctype = mimetypes.guess_type(default_path)[0] or "image/png"
        return FileResponse(open(default_path, "rb"), content_type=ctype)

    try:
        conn = fn._connect_ro()
        try:
            row = fn._fetch_one(conn, "SELECT image_id FROM model WHERE model_id = ?", (int(model_id),))
        finally:
            conn.close()

        image_id = (row or {}).get("image_id")
        if not image_id:
            path = default_path
        else:
            path = os.path.join(images_dir, str(image_id))
            if not os.path.exists(path):
                path = default_path

        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return FileResponse(open(path, "rb"), content_type=ctype)
    except Exception as exc:
        return _error_response(exc)

# =========================
# Brands
# =========================
@api_view(["GET"])
def brands_list(request):
    try:
        return Response(fn.list_brands(_request_permissions(request)))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def brands_create(request):
    try:
        return Response(fn.create_brand(_request_permissions(request), request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def brand_get(request, brand_id: int):
    try:
        return Response(fn.get_brand(_request_permissions(request), brand_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def brand_update(request, brand_id: int):
    try:
        return Response(fn.update_brand(_request_permissions(request), brand_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def brand_delete(request, brand_id: int):
    try:
        return Response(fn.delete_brand(_request_permissions(request), brand_id))
    except Exception as exc:
        return _error_response(exc)


# =========================
# Models
# =========================
@api_view(["GET"])
def models_list(request):
    try:
        return Response(fn.list_models(_request_permissions(request), brand_id=_qint(request, "brand_id")))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def models_create(request):
    try:
        return Response(fn.create_model(_request_permissions(request), request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def model_get(request, model_id: int):
    try:
        return Response(fn.get_model(_request_permissions(request), model_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def model_update(request, model_id: int):
    try:
        return Response(fn.update_model(_request_permissions(request), model_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def model_delete(request, model_id: int):
    try:
        return Response(fn.delete_model(_request_permissions(request), model_id))
    except Exception as exc:
        return _error_response(exc)


# =========================
# Vehicles
# =========================
@api_view(["GET"])
def vehicles_list(request):
    try:
        return Response(
            fn.list_vehicles(
                _request_permissions(request),
                status=_qstr(request, "status"),
                dealership_id=_qint(request, "dealership_id"),
                brand_id=_qint(request, "brand_id"),
                model_id=_qint(request, "model_id"),
                year_min=_qint(request, "year_min"),
                year_max=_qint(request, "year_max"),
                price_min=_qint(request, "price_min"),
                price_max=_qint(request, "price_max"),
                mileage_max=_qint(request, "mileage_max"),
                q=_qstr(request, "q"),
            )
        )
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def vehicles_create(request):
    try:
        return Response(fn.create_vehicle(_request_permissions(request), request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def vehicle_get(request, vehicle_id: int):
    try:
        return Response(fn.get_vehicle(_request_permissions(request), vehicle_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def vehicle_update(request, vehicle_id: int):
    try:
        return Response(fn.update_vehicle(_request_permissions(request), vehicle_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def vehicle_delete(request, vehicle_id: int):
    try:
        return Response(fn.delete_vehicle(_request_permissions(request), vehicle_id))
    except Exception as exc:
        return _error_response(exc)


# =========================
# Customers
# =========================
@api_view(["GET"])
def customers_list(request):
    try:
        return Response(fn.list_customers(_request_permissions(request), q=_qstr(request, "q")))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def customers_create(request):
    try:
        return Response(fn.create_customer(_request_permissions(request), request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def customer_get(request, customer_id: int):
    try:
        return Response(fn.get_customer(_request_permissions(request), customer_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def customer_update(request, customer_id: int):
    try:
        return Response(fn.update_customer(_request_permissions(request), customer_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def customer_delete(request, customer_id: int):
    try:
        return Response(fn.delete_customer(_request_permissions(request), customer_id))
    except Exception as exc:
        return _error_response(exc)


# =========================
# Employees
# =========================
@api_view(["GET"])
def employees_list(request):
    try:
        return Response(fn.list_employees(_request_permissions(request), dealership_id=_qint(request, "dealership_id")))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def employees_create(request):
    try:
        return Response(fn.create_employee(_request_permissions(request), request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def employee_get(request, employee_id: int):
    try:
        return Response(fn.get_employee(_request_permissions(request), employee_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def employee_update(request, employee_id: int):
    try:
        return Response(fn.update_employee(_request_permissions(request), employee_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def employee_delete(request, employee_id: int):
    try:
        return Response(fn.delete_employee(_request_permissions(request), employee_id))
    except Exception as exc:
        return _error_response(exc)


# =========================
# Orders
# =========================
@api_view(["GET"])
def orders_list(request):
    try:
        return Response(
            fn.list_orders(
                _request_permissions(request),
                status=_qstr(request, "status"),
                dealership_id=_qint(request, "dealership_id"),
                customer_id=_qint(request, "customer_id"),
                employee_id=_qint(request, "employee_id"),
                date_from=_qstr(request, "date_from"),
                date_to=_qstr(request, "date_to"),
            )
        )
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def orders_create(request):
    try:
        return Response(fn.create_order(_request_permissions(request), request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def order_get(request, order_id: int):
    try:
        expand = request.query_params.get("expand", "true").lower() != "false"
        return Response(fn.get_order(_request_permissions(request), order_id, expand=expand))
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def order_update(request, order_id: int):
    try:
        return Response(fn.update_order(_request_permissions(request), order_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def order_delete(request, order_id: int):
    try:
        return Response(fn.delete_order(_request_permissions(request), order_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def order_items_list(request, order_id: int):
    try:
        return Response(fn.list_order_items(_request_permissions(request), order_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def order_items_add(request, order_id: int):
    try:
        return Response(fn.add_order_item(_request_permissions(request), order_id, request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["PATCH", "PUT"])
def order_item_update(request, order_id: int, order_item_id: int):
    try:
        return Response(fn.update_order_item(_request_permissions(request), order_id, order_item_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["DELETE"])
def order_item_delete(request, order_id: int, order_item_id: int):
    try:
        return Response(fn.remove_order_item(_request_permissions(request), order_id, order_item_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def order_payments_list(request, order_id: int):
    try:
        return Response(fn.list_order_payments(_request_permissions(request), order_id))
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def order_payments_add(request, order_id: int):
    try:
        return Response(fn.create_payment(_request_permissions(request), order_id, request.data), status=status.HTTP_201_CREATED)
    except Exception as exc:
        return _error_response(exc)


@api_view(["POST"])
def order_set_status(request, order_id: int):
    try:
        return Response(fn.set_order_status(_request_permissions(request), order_id, request.data))
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def order_history(request, order_id: int):
    try:
        return Response(fn.list_order_history(_request_permissions(request), order_id))
    except Exception as exc:
        return _error_response(exc)


# =========================
# Reports
# =========================
@api_view(["GET"])
def report_inventory(request):
    try:
        return Response(fn.inventory_report(_request_permissions(request), dealership_id=_qint(request, "dealership_id")))
    except Exception as exc:
        return _error_response(exc)


@api_view(["GET"])
def report_sales(request):
    try:
        return Response(
            fn.sales_report(
                _request_permissions(request),
                dealership_id=_qint(request, "dealership_id"),
                date_from=_qstr(request, "date_from"),
                date_to=_qstr(request, "date_to"),
            )
        )
    except Exception as exc:
        return _error_response(exc)
    
@api_view(["GET"])
def get_available_permissions(request):
    try:
        return Response(fn.AVAILABLE_PERMISSIONS)
    except Exception as exc:
        return _error_response(exc)
    
@api_view(["GET"])
def get_bypass_permission(request):
    try:
        return Response({"bypass_permission": fn.BYPASS_PERMISSION})
    except Exception as exc:
        return _error_response(exc)

@api_view(["GET"])
def get_available_users(request):
    try:
        return Response(fn.AVAILABLE_USERS)
    except Exception as exc:
        return _error_response(exc)
    
@api_view(["GET"])
def get_available_stored_procedures(request):
    try:
        return Response(list(fn.AVAILABLE_STORED_PROCEDURES.keys()))
    except Exception as exc:
        return _error_response(exc)

@api_view(["POST"])
def assistant_query(request):  # ADD
    try:
        result = fn.assistant_query(_request_permissions(request), request.data)
        return Response(result, status=status.HTTP_200_OK)
    except Exception as exc:
        return _error_response(exc)

@api_view(["POST"])
def assistant_stream(request):  # ADD
    try:
        generator = fn.assistant_stream(_request_permissions(request), request.data)

        def _ndjson():
            for ev in generator:
                yield json.dumps(ev) + "\n"

        return StreamingHttpResponse(_ndjson(), content_type="application/x-ndjson", status=status.HTTP_200_OK)
    except Exception as exc:
        err = _error_response(exc)
        return StreamingHttpResponse([json.dumps(err.data)], content_type="application/json", status=err.status_code)