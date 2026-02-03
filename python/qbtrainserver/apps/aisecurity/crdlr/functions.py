# apps/aisecurity/crdlr/functions.py
from __future__ import annotations

import os
import time
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Generator

from qbtrain.exceptions import PermissionError

import inspect


from qbtrain.agents import SQLAgent
from qbtrain.agents.sql_agent import SQLAgentPrompts
from qbtrain.agents.response_generator_agent import (
    ResponseGeneratorAgent,
    ResponseGeneratorPrompts,
)
from qbtrain.utils.jsonutils import extract_json_object
from qbtrain.utils.streamingutils import stream_message_events
from qbtrain.utils.sqlutils import get_schema_context
from qbtrain.tracers import AgentTracer
from qbtrain.ai.llm import LLMClientRegistry

from qbtrain.utils.authutils import Authorizer

# ============================================================
# Prompts
# ============================================================
from .prompts import (
    SQL_PLANNER_SYSTEM_PROMPT_TEMPLATE,
    SQL_SHARED_USER_PROMPT_TEMPLATE,
    SQL_GEN_COT_SYSTEM_PROMPT_TEMPLATE,
    SQL_GEN_NONCOT_SYSTEM_PROMPT_TEMPLATE,
    STORED_PROC_SYSTEM_PROMPT_TEMPLATE,
    STORED_PROC_USER_PROMPT_TEMPLATE,
    RESPONSE_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
    RESPONSE_GENERATOR_USER_PROMPT_TEMPLATE,
)

# ============================================================
# Permissions
# ============================================================
BYPASS_PERMISSION = "SUPERUSER.ACCESS.ALL"

TABLE_ACCESS = {
    "cars": {"brand", "model", "vehicle"},
    "sales": {"sales_order", "order_item", "payment", "order_status_history", "customer"},
    "staff": {"employee"},
    "dealership": {"dealership"}
}

READ_PERMISSIONS = {
    "Dealership.Read": TABLE_ACCESS["dealership"],
    "Staff.Read": TABLE_ACCESS["staff"],
    "Cars.Read": TABLE_ACCESS["cars"],
    "Sales.Read": TABLE_ACCESS["sales"],
    "Admin.Read": TABLE_ACCESS["sales"] | TABLE_ACCESS["staff"] | TABLE_ACCESS["dealership"] | TABLE_ACCESS["cars"],
}
WRITE_PERMISSIONS = {
    "Dealership.Write": TABLE_ACCESS["dealership"],
    "Staff.Write": TABLE_ACCESS["staff"],
    "Cars.Write": TABLE_ACCESS["cars"],
    "Sales.Write": TABLE_ACCESS["sales"],
    "Admin.Write": TABLE_ACCESS["sales"] | TABLE_ACCESS["staff"] | TABLE_ACCESS["dealership"] | TABLE_ACCESS["cars"],
}

AVAILABLE_PERMISSIONS = sorted([
    *READ_PERMISSIONS.keys(),
    *WRITE_PERMISSIONS.keys(),
])

authorizer = Authorizer(
        read_resources_by_perm=READ_PERMISSIONS,
        write_resources_by_perm=WRITE_PERMISSIONS,
        bypass_permission=BYPASS_PERMISSION,
        allow_reads_without_resources=True,
        allow_writes_without_resources=False,
        imply_write_satisfies_read=False,
    )

AVAILABLE_USERS = [
    {
        "name": "Bob Doe",
        "role": "Manager",
        "permissions": ["Admin.Read", "Admin.Write"]
    },
    {
        "name": "Alice Smith",
        "role": "Salesperson",
        "permissions": ["Cars.Read", "Sales.Read"]
    },
    {
        "name": "Eve Johnson",
        "role": "Customer",
        "permissions": ["Cars.Read"]
    }
]

# =========================
# Errors
# =========================
class NotFound(Exception):
    """Raised when a requested entity is not found in the sandbox database."""


class BadRequest(Exception):
    """Raised when inputs are invalid or required data is missing."""


# =========================
# DB paths + copy safety
# =========================
def _assets_dir() -> Path:
    """
    Resolve the repository-level ./assets directory robustly across common layouts.

    Tries:
      1) settings.BASE_DIR / "assets"
      2) settings.BASE_DIR.parent / "assets"

    Returns the first that contains database.db, otherwise defaults to BASE_DIR/assets.
    """
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    db_path = os.path.join(dir_path, "assets")
    
    return Path(db_path)


def _original_db_path() -> Path:
    """Path to the immutable seed database: ./assets/database.db (must never be modified)."""
    return _assets_dir() / "crdlr.db"


def _sandbox_db_path() -> Path:
    """Path to the mutable sandbox database copy: ./assets/database_copy.db (all reads/writes happen here)."""
    return _assets_dir() / "crdlr_copy.db"

def _resolve_db_path() -> str:
    ensure_sandbox_db()
    return _sandbox_db_path().as_posix()


def ensure_sandbox_db() -> None:
    """
    Ensure the sandbox DB exists by copying the original DB.

    Safety contract:
      - Never open or write to the original DB for any operation.
      - All operations are executed against the sandbox copy.
    """
    orig = _original_db_path()
    sandbox = _sandbox_db_path()

    if not orig.exists():
        raise BadRequest(f"Original DB not found at: {orig}")

    sandbox.parent.mkdir(parents=True, exist_ok=True)
    if not sandbox.exists():
        shutil.copy2(orig, sandbox)


def reset_sandbox_db() -> Dict[str, Any]:
    orig = _original_db_path()
    sandbox = _sandbox_db_path()

    if not orig.exists():
        raise BadRequest(f"Original DB not found at: {orig}")

    if sandbox.exists():
        sandbox.unlink()

    shutil.copy2(orig, sandbox)
    return {"ok": True, "sandbox_db": str(sandbox), "original_db": str(orig)}


def _connect_ro() -> sqlite3.Connection:
    ensure_sandbox_db()
    sandbox = _sandbox_db_path()
    conn = sqlite3.connect(f"file:{sandbox.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _connect_rw() -> sqlite3.Connection:
    ensure_sandbox_db()
    sandbox = _sandbox_db_path()
    conn = sqlite3.connect(f"file:{sandbox.as_posix()}?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _fetch_one(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    cur = conn.execute(sql, params)
    r = cur.fetchone()
    return _row_to_dict(r) if r else None


def _fetch_all(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    cur = conn.execute(sql, params)
    return [_row_to_dict(r) for r in cur.fetchall()]


def _require(data: Mapping[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise BadRequest(f"Missing required fields: {missing}")


def _pick_updates(data: Mapping[str, Any], allowed: List[str]) -> Dict[str, Any]:
    updates = {k: data[k] for k in allowed if k in data}
    if not updates:
        raise BadRequest(f"No updatable fields provided. Allowed: {allowed}")
    return updates


def _exec_write(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> int:
    cur = conn.execute(sql, params)
    return int(cur.lastrowid)


def _get_or_404(conn: sqlite3.Connection, table: str, id_col: str, obj_id: int) -> Dict[str, Any]:
    obj = _fetch_one(conn, f"SELECT * FROM {table} WHERE {id_col} = ?", (obj_id,))
    if not obj:
        raise NotFound(f"{table} not found: {obj_id}")
    return obj


# ========================
# LLM + SQL Agent helpers
# ========================
def _normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    s = settings or {}
    def _none_if_zero(v):
        return None if v is None or (isinstance(v, (int, float)) and v == 0) else v

    return {
        "system_prompt": s.get("system_prompt"),
        "top_k": _none_if_zero(s.get("top_k")),
        "top_p": s.get("top_p"),
        "temperature": s.get("temperature"),
        "presence_penalty": s.get("presence_penalty"),
        "frequency_penalty": s.get("frequency_penalty"),
        "max_output_tokens": s.get("max_output_tokens"),
    }


def _instantiate_llm_client(client_details: Dict[str, Any]):
    if not isinstance(client_details, dict):
        raise BadRequest("clientDetails must be an object.")

    ctype = (client_details.get("type") or "").strip().lower()
    params = dict(client_details.get("params") or {})
    settings = _normalize_settings(client_details.get("settings"))

    ClientCls = LLMClientRegistry.get(ctype)

    # Only pass init kwargs that the client actually accepts (compat with older code)
    sig = inspect.signature(ClientCls.__init__)
    allowed = set(sig.parameters.keys())

    # Merge params + settings, filtering to accepted kwargs
    init_kwargs = {}
    for k, v in {**params, **settings}.items():
        if k in allowed and v is not None:
            init_kwargs[k] = v

    return ClientCls(**init_kwargs)

def _build_sql_agent(
    *,
    db_path: str,
    llm_client,
    agent_permissions: List[str],
    user_permissions: List[str],
    chat_details: Dict[str, Any],
    stored_procedures: List[str],
    stream: bool,
) -> SQLAgent:
    sql_gen_instructions = chat_details.get("sql_gen_instructions")
    use_cot = bool(chat_details.get("use_cot", True))
    cot_max_steps = int(chat_details.get("cot_max_steps", 5))
    cot_flow_through_permissions = bool(chat_details.get("cot_flow_through_permissions", True))
    tracer = AgentTracer()
    schema_context = get_schema_context(db_path)
    stored_procedures = stored_procedures if len(stored_procedures) > 0 else AVAILABLE_STORED_PROCEDURES.keys()

    # Collect all relevant functions as stored procedures

    planner_system_prompt_template = SQL_PLANNER_SYSTEM_PROMPT_TEMPLATE.replace("{permissions_map_block}", authorizer.get_permissions_access(fmt="str"))
    planner_system_prompt_template = planner_system_prompt_template.replace("{additional_instructions}", str(sql_gen_instructions))
    planner_system_prompt_template = planner_system_prompt_template.replace("{schema_context}", str(schema_context))

    sql_gen_system_prompt_template = SQL_GEN_COT_SYSTEM_PROMPT_TEMPLATE if use_cot else SQL_GEN_NONCOT_SYSTEM_PROMPT_TEMPLATE
    sql_gen_system_prompt_template = sql_gen_system_prompt_template.replace("{permissions_map_block}", authorizer.get_permissions_access(fmt="str"))
    sql_gen_system_prompt_template = sql_gen_system_prompt_template.replace("{additional_instructions}", str(sql_gen_instructions))
    sql_gen_system_prompt_template = sql_gen_system_prompt_template.replace("{schema_context}", str(schema_context))

    stored_procedures_dict = {i: AVAILABLE_STORED_PROCEDURES[i] for i in stored_procedures if i in AVAILABLE_STORED_PROCEDURES}

    prompts = SQLAgentPrompts(
        planner_system_prompt_template=planner_system_prompt_template,
        shared_user_prompt_template=SQL_SHARED_USER_PROMPT_TEMPLATE,
        sql_gen_system_prompt_template=sql_gen_system_prompt_template,
        stored_proc_system_prompt_template=STORED_PROC_SYSTEM_PROMPT_TEMPLATE,
        stored_proc_user_prompt_template=STORED_PROC_USER_PROMPT_TEMPLATE,
    )

    return SQLAgent(
        db_path=db_path,
        llm_client=llm_client,
        authorizer=authorizer,
        prompts=prompts,
        agent_permissions=agent_permissions,
        user_permissions=user_permissions,
        stored_procedures=stored_procedures_dict,
        use_cot=use_cot,
        cot_max_steps=cot_max_steps,
        cot_flow_through_permissions=cot_flow_through_permissions,
        tracer=tracer,
        stream=stream,
    )

# =========================
# Dealerships (READ ONLY; no endpoints)
# =========================
def list_dealerships(permissions: Sequence[str]) -> List[Dict[str, Any]]:
    """
    List all dealerships (read-only helper).

    Permissions:
        Requires Dealership.Read (or bypass_all_auth).

    Returns:
        List of dealership rows.
    """
    authorizer.authorize(
        access="read", resources={"dealership"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        return _fetch_all(conn, "SELECT * FROM dealership ORDER BY name")
    finally:
        conn.close()


def get_dealership(permissions: Sequence[str], dealership_id: int) -> Dict[str, Any]:
    """
    Get a dealership by ID (read-only helper).

    Permissions:
        Requires Dealership.Read (or bypass_all_auth).

    Args:
        dealership_id: dealership.dealership_id

    Returns:
        Dealership row.

    Raises:
        NotFound: If dealership does not exist.
    """
    authorizer.authorize(
        access="read", resources={"dealership"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        return _get_or_404(conn, "dealership", "dealership_id", dealership_id)
    finally:
        conn.close()


# =========================
# Brands
# =========================
def list_brands(permissions: Sequence[str]) -> List[Dict[str, Any]]:
    """
    List all vehicle brands.

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Returns:
        List of brand rows.
    """
    authorizer.authorize(
        access="read", resources={"brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        return _fetch_all(conn, "SELECT * FROM brand ORDER BY name")
    finally:
        conn.close()


def get_brand(permissions: Sequence[str], brand_id: int) -> Dict[str, Any]:
    """
    Get a brand by ID.

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Args:
        brand_id: brand.brand_id

    Returns:
        Brand row.
    """
    authorizer.authorize(
        access="read", resources={"brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        return _get_or_404(conn, "brand", "brand_id", brand_id)
    finally:
        conn.close()


def create_brand(permissions: Sequence[str], data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create a brand.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Required keys in `data`:
        name, segment
    Optional keys:
        country

    Returns:
        Created brand row.
    """
    authorizer.authorize(
        access="write", resources={"brand"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _require(data, ["name", "segment"])
        with conn:
            new_id = _exec_write(
                conn,
                "INSERT INTO brand (name, country, segment) VALUES (?, ?, ?)",
                (data["name"], data.get("country"), data["segment"]),
            )
        return get_brand(permissions, new_id)
    finally:
        conn.close()


def update_brand(permissions: Sequence[str], brand_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update a brand.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Updatable keys:
        name, country, segment

    Returns:
        Updated brand row.
    """
    authorizer.authorize(
        access="write", resources={"brand"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        allowed = ["name", "country", "segment"]
        updates = _pick_updates(data, allowed)
        _get_or_404(conn, "brand", "brand_id", brand_id)

        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        params = tuple(updates.values()) + (brand_id,)
        with conn:
            conn.execute(f"UPDATE brand SET {sets} WHERE brand_id = ?", params)
        return get_brand(permissions, brand_id)
    finally:
        conn.close()


def delete_brand(permissions: Sequence[str], brand_id: int) -> Dict[str, Any]:
    """
    Delete a brand by ID.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Returns:
        {"ok": True, "deleted": <brand_id>}
    """
    authorizer.authorize(
        access="write", resources={"brand"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _get_or_404(conn, "brand", "brand_id", brand_id)
        with conn:
            conn.execute("DELETE FROM brand WHERE brand_id = ?", (brand_id,))
        return {"ok": True, "deleted": brand_id}
    finally:
        conn.close()


# =========================
# Models
# =========================
def list_models(permissions: Sequence[str], brand_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    List models, optionally filtered by brand.

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Args:
        brand_id: If provided, return only models belonging to this brand.

    Returns:
        List of model rows joined with brand metadata.
    """
    authorizer.authorize(
        access="read", resources={"model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        if brand_id is None:
            return _fetch_all(
                conn,
                """
                SELECT m.*, b.name AS brand_name, b.segment AS brand_segment
                FROM model m
                JOIN brand b ON b.brand_id = m.brand_id
                ORDER BY b.name, m.name
                """,
            )
        return _fetch_all(
            conn,
            """
            SELECT m.*, b.name AS brand_name, b.segment AS brand_segment
            FROM model m
            JOIN brand b ON b.brand_id = m.brand_id
            WHERE m.brand_id = ?
            ORDER BY m.name
            """,
            (brand_id,),
        )
    finally:
        conn.close()


def get_model(permissions: Sequence[str], model_id: int) -> Dict[str, Any]:
    """
    Get a model by ID (joined with brand metadata).

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Args:
        model_id: model.model_id

    Returns:
        Model row with brand_name and brand_segment.

    Raises:
        NotFound: If model does not exist.
    """
    authorizer.authorize(
        access="read", resources={"model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        obj = _fetch_one(
            conn,
            """
            SELECT m.*, b.name AS brand_name, b.segment AS brand_segment
            FROM model m
            JOIN brand b ON b.brand_id = m.brand_id
            WHERE m.model_id = ?
            """,
            (model_id,),
        )
        if not obj:
            raise NotFound(f"model not found: {model_id}")
        return obj
    finally:
        conn.close()


def create_model(permissions: Sequence[str], data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create a vehicle model.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Required keys in `data`:
        brand_id, name, body_style, fuel_type, drivetrain, msrp_min, msrp_max, description_md

    Optional keys:
        image_id

    Returns:
        Created model row (joined with brand metadata).
    """
    authorizer.authorize(
        access="write", resources={"model"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _require(
            data,
            ["brand_id", "name", "body_style", "fuel_type", "drivetrain", "msrp_min", "msrp_max", "description_md"],
        )
        with conn:
            new_id = _exec_write(
                conn,
                """
                INSERT INTO model
                (brand_id, name, body_style, fuel_type, drivetrain, msrp_min, msrp_max, description_md, image_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["brand_id"],
                    data["name"],
                    data["body_style"],
                    data["fuel_type"],
                    data["drivetrain"],
                    data["msrp_min"],
                    data["msrp_max"],
                    data["description_md"],
                    data.get("image_id"),
                ),
            )
        return get_model(permissions, new_id)
    finally:
        conn.close()


def update_model(permissions: Sequence[str], model_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update a vehicle model.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Updatable keys:
        brand_id, name, body_style, fuel_type, drivetrain, msrp_min, msrp_max, description_md

    Returns:
        Updated model row (joined with brand metadata).
    """
    authorizer.authorize(
        access="write", resources={"model"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        allowed = ["brand_id", "name", "body_style", "fuel_type", "drivetrain", "msrp_min", "msrp_max", "description_md", "image_id"]
        updates = _pick_updates(data, allowed)
        _get_or_404(conn, "model", "model_id", model_id)

        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        params = tuple(updates.values()) + (model_id,)
        with conn:
            conn.execute(f"UPDATE model SET {sets} WHERE model_id = ?", params)
        return get_model(permissions, model_id)
    finally:
        conn.close()


def delete_model(permissions: Sequence[str], model_id: int) -> Dict[str, Any]:
    """
    Delete a model by ID.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Returns:
        {"ok": True, "deleted": <model_id>}
    """
    authorizer.authorize(
        access="write", resources={"model"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"model"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _get_or_404(conn, "model", "model_id", model_id)
        with conn:
            conn.execute("DELETE FROM model WHERE model_id = ?", (model_id,))
        return {"ok": True, "deleted": model_id}
    finally:
        conn.close()


# =========================
# Vehicles (Inventory)
# =========================
def list_vehicles(
    permissions: Sequence[str],
    *,
    status: Optional[str] = None,
    dealership_id: Optional[int] = None,
    brand_id: Optional[int] = None,
    model_id: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    mileage_max: Optional[int] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List vehicles with optional filters.

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Filters:
        status: {'AVAILABLE','RESERVED','SOLD','IN_SERVICE'}
        dealership_id, brand_id, model_id
        year_min/year_max, price_min/price_max, mileage_max
        q: LIKE search over vin, color, brand name, model name

    Returns:
        List of vehicle rows joined with dealership, model, and brand metadata.
    """
    authorizer.authorize(
        access="read", resources={"vehicle", "dealership", "model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        where: List[str] = []
        params: List[Any] = []

        def add(cond: str, val: Any) -> None:
            where.append(cond)
            params.append(val)

        if status is not None:
            add("v.status = ?", status)
        if dealership_id is not None:
            add("v.dealership_id = ?", int(dealership_id))
        if brand_id is not None:
            add("b.brand_id = ?", int(brand_id))
        if model_id is not None:
            add("m.model_id = ?", int(model_id))
        if year_min is not None:
            add("v.model_year >= ?", int(year_min))
        if year_max is not None:
            add("v.model_year <= ?", int(year_max))
        if price_min is not None:
            add("v.list_price >= ?", int(price_min))
        if price_max is not None:
            add("v.list_price <= ?", int(price_max))
        if mileage_max is not None:
            add("v.mileage <= ?", int(mileage_max))
        if q is not None and q != "":
            like = f"%{q}%"
            where.append("(v.vin LIKE ? OR v.color LIKE ? OR b.name LIKE ? OR m.name LIKE ?)")
            params.extend([like, like, like, like])

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        return _fetch_all(
            conn,
            f"""
            SELECT
              v.*,
              d.name AS dealership_name, d.city AS dealership_city, d.state AS dealership_state,
              m.name AS model_name, m.body_style, m.fuel_type, m.drivetrain, m.image_id AS model_image_id,
              b.brand_id, b.name AS brand_name, b.segment AS brand_segment
            FROM vehicle v
            JOIN dealership d ON d.dealership_id = v.dealership_id
            JOIN model m ON m.model_id = v.model_id
            JOIN brand b ON b.brand_id = m.brand_id
            {where_sql}
            ORDER BY v.created_at DESC, v.vehicle_id DESC
            """,
            tuple(params),
        )
    finally:
        conn.close()


def get_vehicle(permissions: Sequence[str], vehicle_id: int) -> Dict[str, Any]:
    """
    Get a vehicle by ID (joined with dealership, model, brand metadata).

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Args:
        vehicle_id: vehicle.vehicle_id

    Returns:
        Vehicle row with joined metadata.

    Raises:
        NotFound: If vehicle does not exist.
    """
    authorizer.authorize(
        access="read", resources={"vehicle", "dealership", "model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        obj = _fetch_one(
            conn,
            """
            SELECT
              v.*,
              d.name AS dealership_name, d.city AS dealership_city, d.state AS dealership_state,
              m.name AS model_name, m.body_style, m.fuel_type, m.drivetrain, m.image_id AS model_image_id,
              b.brand_id, b.name AS brand_name, b.segment AS brand_segment
            FROM vehicle v
            JOIN dealership d ON d.dealership_id = v.dealership_id
            JOIN model m ON m.model_id = v.model_id
            JOIN brand b ON b.brand_id = m.brand_id
            WHERE v.vehicle_id = ?
            """,
            (vehicle_id,),
        )
        if not obj:
            raise NotFound(f"vehicle not found: {vehicle_id}")
        return obj
    finally:
        conn.close()


def create_vehicle(permissions: Sequence[str], data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create an inventory vehicle record.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Required keys in `data`:
        vin, model_id, model_year, color, mileage, list_price, dealership_id, status

    Returns:
        Created vehicle row (joined with metadata).
    """
    authorizer.authorize(
        access="write", resources={"vehicle"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"vehicle", "dealership", "model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _require(data, ["vin", "model_id", "model_year", "color", "mileage", "list_price", "dealership_id", "status"])
        with conn:
            new_id = _exec_write(
                conn,
                """
                INSERT INTO vehicle
                (vin, model_id, model_year, color, mileage, list_price, dealership_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["vin"],
                    data["model_id"],
                    data["model_year"],
                    data["color"],
                    data["mileage"],
                    data["list_price"],
                    data["dealership_id"],
                    data["status"],
                ),
            )
        return get_vehicle(permissions, new_id)
    finally:
        conn.close()


def update_vehicle(permissions: Sequence[str], vehicle_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update a vehicle inventory record.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Updatable keys:
        vin, model_id, model_year, color, mileage, list_price, dealership_id, status

    Returns:
        Updated vehicle row (joined with metadata).
    """
    authorizer.authorize(
        access="write", resources={"vehicle"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"vehicle", "dealership", "model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        allowed = ["vin", "model_id", "model_year", "color", "mileage", "list_price", "dealership_id", "status"]
        updates = _pick_updates(data, allowed)
        _get_or_404(conn, "vehicle", "vehicle_id", vehicle_id)

        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        params = tuple(updates.values()) + (vehicle_id,)
        with conn:
            conn.execute(f"UPDATE vehicle SET {sets} WHERE vehicle_id = ?", params)
        return get_vehicle(permissions, vehicle_id)
    finally:
        conn.close()


def delete_vehicle(permissions: Sequence[str], vehicle_id: int) -> Dict[str, Any]:
    """
    Delete a vehicle by ID.

    Permissions:
        Requires Cars.Write (or bypass_all_auth).

    Returns:
        {"ok": True, "deleted": <vehicle_id>}
    """
    authorizer.authorize(
        access="write", resources={"vehicle"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"vehicle"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _get_or_404(conn, "vehicle", "vehicle_id", vehicle_id)
        with conn:
            conn.execute("DELETE FROM vehicle WHERE vehicle_id = ?", (vehicle_id,))
        return {"ok": True, "deleted": vehicle_id}
    finally:
        conn.close()


# =========================
# Customers
# =========================
def list_customers(permissions: Sequence[str], q: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List customers, optionally searching by name/email/phone.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        q: LIKE search over first_name, last_name, email, phone.

    Returns:
        List of customer rows.
    """
    authorizer.authorize(
        access="read", resources={"customer"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        if not q:
            return _fetch_all(conn, "SELECT * FROM customer ORDER BY created_at DESC, customer_id DESC")
        like = f"%{q}%"
        return _fetch_all(
            conn,
            """
            SELECT * FROM customer
            WHERE first_name LIKE ? OR last_name LIKE ? OR email LIKE ? OR phone LIKE ?
            ORDER BY created_at DESC, customer_id DESC
            """,
            (like, like, like, like),
        )
    finally:
        conn.close()


def get_customer(permissions: Sequence[str], customer_id: int) -> Dict[str, Any]:
    """
    Get a customer by ID.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        customer_id: customer.customer_id

    Returns:
        Customer row.
    """
    authorizer.authorize(
        access="read", resources={"customer"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        return _get_or_404(conn, "customer", "customer_id", customer_id)
    finally:
        conn.close()


def create_customer(permissions: Sequence[str], data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create a customer.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Required keys in `data`:
        first_name, last_name, email
    Optional keys:
        phone, street, city, state, postal_code

    Returns:
        Created customer row.
    """
    authorizer.authorize(
        access="write", resources={"customer"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"customer"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _require(data, ["first_name", "last_name", "email"])
        with conn:
            new_id = _exec_write(
                conn,
                """
                INSERT INTO customer
                (first_name, last_name, email, phone, street, city, state, postal_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["first_name"],
                    data["last_name"],
                    data["email"],
                    data.get("phone"),
                    data.get("street"),
                    data.get("city"),
                    data.get("state"),
                    data.get("postal_code"),
                ),
            )
        return get_customer(permissions, new_id)
    finally:
        conn.close()


def update_customer(permissions: Sequence[str], customer_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update a customer.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Updatable keys:
        first_name, last_name, email, phone, street, city, state, postal_code

    Returns:
        Updated customer row.
    """
    authorizer.authorize(
        access="write", resources={"customer"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"customer"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        allowed = ["first_name", "last_name", "email", "phone", "street", "city", "state", "postal_code"]
        updates = _pick_updates(data, allowed)
        _get_or_404(conn, "customer", "customer_id", customer_id)

        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        params = tuple(updates.values()) + (customer_id,)
        with conn:
            conn.execute(f"UPDATE customer SET {sets} WHERE customer_id = ?", params)
        return get_customer(permissions, customer_id)
    finally:
        conn.close()


def delete_customer(permissions: Sequence[str], customer_id: int) -> Dict[str, Any]:
    """
    Delete a customer by ID.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Returns:
        {"ok": True, "deleted": <customer_id>}
    """
    authorizer.authorize(
        access="write", resources={"customer"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"customer"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _get_or_404(conn, "customer", "customer_id", customer_id)
        with conn:
            conn.execute("DELETE FROM customer WHERE customer_id = ?", (customer_id,))
        return {"ok": True, "deleted": customer_id}
    finally:
        conn.close()


# =========================
# Employees
# =========================
def list_employees(permissions: Sequence[str], dealership_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    List employees, optionally filtered by dealership.

    Permissions:
        Requires Staff.Read (or bypass_all_auth).

    Args:
        dealership_id: Filter to a single dealership_id.

    Returns:
        List of employee rows joined with dealership name.
    """
    authorizer.authorize(
        access="read", resources={"employee", "dealership"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        if dealership_id is None:
            return _fetch_all(
                conn,
                """
                SELECT e.*, d.name AS dealership_name
                FROM employee e
                JOIN dealership d ON d.dealership_id = e.dealership_id
                ORDER BY e.last_name, e.first_name
                """,
            )
        return _fetch_all(
            conn,
            """
            SELECT e.*, d.name AS dealership_name
            FROM employee e
            JOIN dealership d ON d.dealership_id = e.dealership_id
            WHERE e.dealership_id = ?
            ORDER BY e.last_name, e.first_name
            """,
            (dealership_id,),
        )
    finally:
        conn.close()


def get_employee(permissions: Sequence[str], employee_id: int) -> Dict[str, Any]:
    """
    Get an employee by ID (joined with dealership name).

    Permissions:
        Requires Staff.Read (or bypass_all_auth).

    Args:
        employee_id: employee.employee_id

    Returns:
        Employee row with dealership_name.

    Raises:
        NotFound: If employee does not exist.
    """
    authorizer.authorize(
        access="read", resources={"employee", "dealership"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        obj = _fetch_one(
            conn,
            """
            SELECT e.*, d.name AS dealership_name
            FROM employee e
            JOIN dealership d ON d.dealership_id = e.dealership_id
            WHERE e.employee_id = ?
            """,
            (employee_id,),
        )
        if not obj:
            raise NotFound(f"employee not found: {employee_id}")
        return obj
    finally:
        conn.close()


def create_employee(permissions: Sequence[str], data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create an employee.

    Permissions:
        Requires Staff.Write (or bypass_all_auth).

    Required keys in `data`:
        first_name, last_name, email, role, dealership_id, hire_date
    Optional keys:
        phone

    Returns:
        Created employee row (joined with dealership name).
    """
    authorizer.authorize(
        access="write", resources={"employee"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"employee", "dealership"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _require(data, ["first_name", "last_name", "email", "role", "dealership_id", "hire_date"])
        with conn:
            new_id = _exec_write(
                conn,
                """
                INSERT INTO employee
                (first_name, last_name, email, phone, role, dealership_id, hire_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["first_name"],
                    data["last_name"],
                    data["email"],
                    data.get("phone"),
                    data["role"],
                    data["dealership_id"],
                    data["hire_date"],
                ),
            )
        return get_employee(permissions, new_id)
    finally:
        conn.close()


def update_employee(permissions: Sequence[str], employee_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update an employee.

    Permissions:
        Requires Staff.Write (or bypass_all_auth).

    Updatable keys:
        first_name, last_name, email, phone, role, dealership_id, hire_date

    Returns:
        Updated employee row.
    """
    authorizer.authorize(
        access="write", resources={"employee"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"employee", "dealership"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        allowed = ["first_name", "last_name", "email", "phone", "role", "dealership_id", "hire_date"]
        updates = _pick_updates(data, allowed)
        _get_or_404(conn, "employee", "employee_id", employee_id)

        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        params = tuple(updates.values()) + (employee_id,)
        with conn:
            conn.execute(f"UPDATE employee SET {sets} WHERE employee_id = ?", params)
        return get_employee(permissions, employee_id)
    finally:
        conn.close()


def delete_employee(permissions: Sequence[str], employee_id: int) -> Dict[str, Any]:
    """
    Delete an employee by ID.

    Permissions:
        Requires Staff.Write (or bypass_all_auth).

    Returns:
        {"ok": True, "deleted": <employee_id>}
    """
    authorizer.authorize(
        access="write", resources={"employee"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"employee"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _get_or_404(conn, "employee", "employee_id", employee_id)
        with conn:
            conn.execute("DELETE FROM employee WHERE employee_id = ?", (employee_id,))
        return {"ok": True, "deleted": employee_id}
    finally:
        conn.close()


# =========================
# Orders (Sales)
# =========================
_ORDER_OPEN_STATUSES = {"PENDING", "CONFIRMED"}


def list_orders(
    permissions: Sequence[str],
    *,
    status: Optional[str] = None,
    dealership_id: Optional[int] = None,
    customer_id: Optional[int] = None,
    employee_id: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List sales orders with optional filters.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Filters:
        status: {'PENDING','CONFIRMED','CANCELLED','FULFILLED','DELIVERED','REFUNDED'}
        dealership_id, customer_id, employee_id
        date_from/date_to: inclusive bounds on order_date (SQLite datetime text)

    Returns:
        List of sales orders joined with customer/employee/dealership metadata.
    """
    authorizer.authorize(
        access="read", resources={"sales_order", "customer", "employee", "dealership"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        where: List[str] = []
        params: List[Any] = []

        def add(cond: str, val: Any) -> None:
            where.append(cond)
            params.append(val)

        if status is not None:
            add("o.status = ?", status)
        if dealership_id is not None:
            add("o.dealership_id = ?", int(dealership_id))
        if customer_id is not None:
            add("o.customer_id = ?", int(customer_id))
        if employee_id is not None:
            add("o.employee_id = ?", int(employee_id))
        if date_from is not None:
            add("o.order_date >= ?", date_from)
        if date_to is not None:
            add("o.order_date <= ?", date_to)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        return _fetch_all(
            conn,
            f"""
            SELECT
              o.*,
              c.first_name AS customer_first_name, c.last_name AS customer_last_name, c.email AS customer_email,
              e.first_name AS employee_first_name, e.last_name AS employee_last_name, e.role AS employee_role,
              d.name AS dealership_name
            FROM sales_order o
            JOIN customer c ON c.customer_id = o.customer_id
            LEFT JOIN employee e ON e.employee_id = o.employee_id
            JOIN dealership d ON d.dealership_id = o.dealership_id
            {where_sql}
            ORDER BY o.order_date DESC, o.order_id DESC
            """,
            tuple(params),
        )
    finally:
        conn.close()


def _recompute_order_total(conn: sqlite3.Connection, order_id: int) -> int:
    row = _fetch_one(conn, "SELECT COALESCE(SUM(sale_price), 0) AS total FROM order_item WHERE order_id = ?", (order_id,))
    total = int(row["total"]) if row else 0
    conn.execute("UPDATE sales_order SET total_amount = ? WHERE order_id = ?", (total, order_id))
    return total


def get_order(permissions: Sequence[str], order_id: int, expand: bool = True) -> Dict[str, Any]:
    """
    Get a sales order by ID.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        order_id: sales_order.order_id
        expand: If True, include items, payments, and status history.

    Returns:
        Order row (joined) and, if expand=True, nested:
          - items
          - payments
          - status_history
    """
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_ro()
    try:
        o = _fetch_one(
            conn,
            """
            SELECT
              o.*,
              c.first_name AS customer_first_name, c.last_name AS customer_last_name, c.email AS customer_email,
              e.first_name AS employee_first_name, e.last_name AS employee_last_name, e.role AS employee_role,
              d.name AS dealership_name
            FROM sales_order o
            JOIN customer c ON c.customer_id = o.customer_id
            LEFT JOIN employee e ON e.employee_id = o.employee_id
            JOIN dealership d ON d.dealership_id = o.dealership_id
            WHERE o.order_id = ?
            """,
            (order_id,),
        )
        if not o:
            raise NotFound(f"order not found: {order_id}")

        if not expand:
            return o

        items = _fetch_all(
            conn,
            """
            SELECT
              oi.*,
              v.vin, v.model_year, v.color, v.mileage, v.list_price, v.status AS vehicle_status,
              m.name AS model_name, m.image_id AS model_image_id,
              b.name AS brand_name
            FROM order_item oi
            JOIN vehicle v ON v.vehicle_id = oi.vehicle_id
            JOIN model m ON m.model_id = v.model_id
            JOIN brand b ON b.brand_id = m.brand_id
            WHERE oi.order_id = ?
            ORDER BY oi.order_item_id
            """,
            (order_id,),
        )
        payments = _fetch_all(conn, "SELECT * FROM payment WHERE order_id = ? ORDER BY payment_id", (order_id,))
        history = _fetch_all(
            conn,
            """
            SELECT
              h.*,
              e.first_name AS changed_by_first_name,
              e.last_name AS changed_by_last_name
            FROM order_status_history h
            LEFT JOIN employee e ON e.employee_id = h.changed_by_employee_id
            WHERE h.order_id = ?
            ORDER BY h.changed_at, h.history_id
            """,
            (order_id,),
        )

        o["items"] = items
        o["payments"] = payments
        o["status_history"] = history
        return o
    finally:
        conn.close()


def create_order(permissions: Sequence[str], data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create a sales order (no items initially).

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Required keys in `data`:
        customer_id, dealership_id
    Optional keys:
        employee_id, status (default 'PENDING'), notes

    Returns:
        Created order (expanded).
    """
    authorizer.authorize(
        access="write", resources={"sales_order", "order_status_history"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_rw()
    try:
        _require(data, ["customer_id", "dealership_id"])
        status = data.get("status", "PENDING")
        with conn:
            new_id = _exec_write(
                conn,
                """
                INSERT INTO sales_order
                (customer_id, employee_id, dealership_id, status, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    data["customer_id"],
                    data.get("employee_id"),
                    data["dealership_id"],
                    status,
                    data.get("notes"),
                ),
            )
            conn.execute(
                """
                INSERT INTO order_status_history (order_id, status, changed_by_employee_id, note)
                VALUES (?, ?, ?, ?)
                """,
                (new_id, status, data.get("employee_id"), "Order created"),
            )
        return get_order(permissions, new_id, expand=True)
    finally:
        conn.close()


def update_order(permissions: Sequence[str], order_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update a sales order's non-status fields.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Updatable keys:
        customer_id, employee_id, dealership_id, notes

    Returns:
        Updated order (expanded).
    """
    authorizer.authorize(
        access="write", resources={"sales_order"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_rw()
    try:
        allowed = ["customer_id", "employee_id", "dealership_id", "notes"]
        updates = _pick_updates(data, allowed)
        _get_or_404(conn, "sales_order", "order_id", order_id)

        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        params = tuple(updates.values()) + (order_id,)
        with conn:
            conn.execute(f"UPDATE sales_order SET {sets} WHERE order_id = ?", params)
        return get_order(permissions, order_id, expand=True)
    finally:
        conn.close()


def delete_order(permissions: Sequence[str], order_id: int) -> Dict[str, Any]:
    """
    Delete an order by ID.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Defensive behavior:
        - Sets associated vehicles back to AVAILABLE before deletion.

    Returns:
        {"ok": True, "deleted": <order_id>}
    """
    authorizer.authorize(
        access="write", resources={"sales_order", "vehicle"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"sales_order", "order_item"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _get_or_404(conn, "sales_order", "order_id", order_id)
        with conn:
            vids = _fetch_all(conn, "SELECT vehicle_id FROM order_item WHERE order_id = ?", (order_id,))
            for r in vids:
                conn.execute("UPDATE vehicle SET status = 'AVAILABLE' WHERE vehicle_id = ?", (r["vehicle_id"],))
            conn.execute("DELETE FROM sales_order WHERE order_id = ?", (order_id,))
        return {"ok": True, "deleted": order_id}
    finally:
        conn.close()


def add_order_item(permissions: Sequence[str], order_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Add a vehicle to an order (creates order_item).

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Required keys in `data`:
        vehicle_id, sale_price

    Rules:
        - Order must be in PENDING or CONFIRMED.
        - Vehicle must be AVAILABLE.
        - Vehicle dealership_id must match order dealership_id.
        - Vehicle is transitioned to RESERVED.
        - Order total_amount is recomputed.

    Returns:
        Updated order (expanded).
    """
    authorizer.authorize(
        access="write", resources={"order_item", "vehicle", "sales_order"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_rw()
    try:
        _require(data, ["vehicle_id", "sale_price"])

        with conn:
            order = _get_or_404(conn, "sales_order", "order_id", order_id)
            if order["status"] not in _ORDER_OPEN_STATUSES:
                raise BadRequest(f"Cannot add items to order in status: {order['status']}")

            v = _get_or_404(conn, "vehicle", "vehicle_id", int(data["vehicle_id"]))
            if v["status"] != "AVAILABLE":
                raise BadRequest(f"Vehicle not available (status={v['status']}): {v['vehicle_id']}")

            if int(v["dealership_id"]) != int(order["dealership_id"]):
                raise BadRequest("Vehicle dealership_id must match order dealership_id")

            _exec_write(
                conn,
                "INSERT INTO order_item (order_id, vehicle_id, sale_price) VALUES (?, ?, ?)",
                (order_id, int(data["vehicle_id"]), int(data["sale_price"])),
            )
            conn.execute("UPDATE vehicle SET status = 'RESERVED' WHERE vehicle_id = ?", (int(data["vehicle_id"]),))
            _recompute_order_total(conn, order_id)

        return get_order(permissions, order_id, expand=True)
    finally:
        conn.close()


def update_order_item(permissions: Sequence[str], order_id: int, order_item_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update an order_item (e.g., change sale_price).

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Updatable keys:
        sale_price

    Rules:
        - Order must be in PENDING or CONFIRMED.
        - Order item must belong to the order.
        - Order total_amount is recomputed.

    Returns:
        Updated order (expanded).
    """
    authorizer.authorize(
        access="write", resources={"order_item", "sales_order"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_rw()
    try:
        updates = _pick_updates(data, ["sale_price"])

        with conn:
            order = _get_or_404(conn, "sales_order", "order_id", order_id)
            if order["status"] not in _ORDER_OPEN_STATUSES:
                raise BadRequest(f"Cannot update items in order in status: {order['status']}")

            item = _fetch_one(
                conn,
                "SELECT * FROM order_item WHERE order_item_id = ? AND order_id = ?",
                (order_item_id, order_id),
            )
            if not item:
                raise NotFound(f"order_item not found: {order_item_id}")

            conn.execute(
                "UPDATE order_item SET sale_price = ? WHERE order_item_id = ? AND order_id = ?",
                (int(updates["sale_price"]), order_item_id, order_id),
            )
            _recompute_order_total(conn, order_id)

        return get_order(permissions, order_id, expand=True)
    finally:
        conn.close()


def remove_order_item(permissions: Sequence[str], order_id: int, order_item_id: int) -> Dict[str, Any]:
    """
    Remove an order_item from an order.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Rules:
        - Order must be in PENDING or CONFIRMED.
        - Vehicle is transitioned back to AVAILABLE.
        - Order total_amount is recomputed.

    Returns:
        Updated order (expanded).
    """
    authorizer.authorize(
        access="write", resources={"order_item", "vehicle", "sales_order"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_rw()
    try:
        with conn:
            order = _get_or_404(conn, "sales_order", "order_id", order_id)
            if order["status"] not in _ORDER_OPEN_STATUSES:
                raise BadRequest(f"Cannot remove items from order in status: {order['status']}")

            item = _fetch_one(
                conn,
                "SELECT * FROM order_item WHERE order_item_id = ? AND order_id = ?",
                (order_item_id, order_id),
            )
            if not item:
                raise NotFound(f"order_item not found: {order_item_id}")

            vehicle_id = int(item["vehicle_id"])
            conn.execute("DELETE FROM order_item WHERE order_item_id = ?", (order_item_id,))
            conn.execute("UPDATE vehicle SET status = 'AVAILABLE' WHERE vehicle_id = ?", (vehicle_id,))
            _recompute_order_total(conn, order_id)

        return get_order(permissions, order_id, expand=True)
    finally:
        conn.close()


def set_order_status(permissions: Sequence[str], order_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Set an order status and record status history.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Required keys in `data`:
        status
    Optional keys:
        changed_by_employee_id, note

    Vehicle transitions:
        - CANCELLED/REFUNDED -> associated vehicles set to AVAILABLE
        - FULFILLED/DELIVERED -> associated vehicles set to SOLD
        - PENDING/CONFIRMED -> associated vehicles ensured RESERVED (if items exist)

    Returns:
        Updated order (expanded).
    """
    authorizer.authorize(
        access="write", resources={"sales_order", "order_status_history", "vehicle"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read",
        resources={
            "sales_order",
            "customer",
            "employee",
            "dealership",
            "order_item",
            "vehicle",
            "model",
            "brand",
            "payment",
            "order_status_history",
        },
        permissions=list(permissions),
    )
    conn = _connect_rw()
    try:
        _require(data, ["status"])
        new_status = str(data["status"])
        changed_by = data.get("changed_by_employee_id")
        note = data.get("note")

        with conn:
            order = _get_or_404(conn, "sales_order", "order_id", order_id)
            old_status = order["status"]
            if old_status == new_status:
                return get_order(permissions, order_id, expand=True)

            conn.execute("UPDATE sales_order SET status = ? WHERE order_id = ?", (new_status, order_id))
            conn.execute(
                """
                INSERT INTO order_status_history (order_id, status, changed_by_employee_id, note)
                VALUES (?, ?, ?, ?)
                """,
                (order_id, new_status, changed_by, note),
            )

            vids = _fetch_all(conn, "SELECT vehicle_id FROM order_item WHERE order_id = ?", (order_id,))
            if new_status in {"CANCELLED", "REFUNDED"}:
                for r in vids:
                    conn.execute("UPDATE vehicle SET status = 'AVAILABLE' WHERE vehicle_id = ?", (r["vehicle_id"],))
            elif new_status in {"FULFILLED", "DELIVERED"}:
                for r in vids:
                    conn.execute("UPDATE vehicle SET status = 'SOLD' WHERE vehicle_id = ?", (r["vehicle_id"],))
            elif new_status in {"PENDING", "CONFIRMED"}:
                for r in vids:
                    conn.execute(
                        "UPDATE vehicle SET status = 'RESERVED' WHERE vehicle_id = ? AND status = 'AVAILABLE'",
                        (r["vehicle_id"],),
                    )

        return get_order(permissions, order_id, expand=True)
    finally:
        conn.close()


def list_order_items(permissions: Sequence[str], order_id: int) -> List[Dict[str, Any]]:
    """
    List order items for an order.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        order_id: sales_order.order_id

    Returns:
        List of order_item rows.
    """
    authorizer.authorize(
        access="read", resources={"sales_order", "order_item"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        _get_or_404(conn, "sales_order", "order_id", order_id)
        return _fetch_all(conn, "SELECT * FROM order_item WHERE order_id = ? ORDER BY order_item_id", (order_id,))
    finally:
        conn.close()


def list_order_payments(permissions: Sequence[str], order_id: int) -> List[Dict[str, Any]]:
    """
    List payments for an order.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        order_id: sales_order.order_id

    Returns:
        List of payment rows.
    """
    authorizer.authorize(
        access="read", resources={"sales_order", "payment"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        _get_or_404(conn, "sales_order", "order_id", order_id)
        return _fetch_all(conn, "SELECT * FROM payment WHERE order_id = ? ORDER BY payment_id", (order_id,))
    finally:
        conn.close()


def create_payment(permissions: Sequence[str], order_id: int, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Create a mock payment for an order.

    Permissions:
        Requires Sales.Write (or bypass_all_auth).

    Required keys in `data`:
        amount, method, status
    Optional keys:
        paid_at, transaction_ref

    Returns:
        Created payment row.
    """
    authorizer.authorize(
        access="write", resources={"payment"}, permissions=list(permissions)
    )
    authorizer.authorize(
        access="read", resources={"sales_order", "payment"}, permissions=list(permissions)
    )
    conn = _connect_rw()
    try:
        _require(data, ["amount", "method", "status"])
        with conn:
            _get_or_404(conn, "sales_order", "order_id", order_id)
            new_id = _exec_write(
                conn,
                """
                INSERT INTO payment (order_id, amount, method, status, paid_at, transaction_ref)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    int(data["amount"]),
                    data["method"],
                    data["status"],
                    data.get("paid_at"),
                    data.get("transaction_ref"),
                ),
            )
            created = _fetch_one(conn, "SELECT * FROM payment WHERE payment_id = ?", (new_id,))
        return created or {"payment_id": new_id}
    finally:
        conn.close()


def list_order_history(permissions: Sequence[str], order_id: int) -> List[Dict[str, Any]]:
    """
    List status history entries for an order.

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        order_id: sales_order.order_id

    Returns:
        List of order_status_history rows joined with employee name.
    """
    authorizer.authorize(
        access="read", resources={"sales_order", "order_status_history", "employee"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        _get_or_404(conn, "sales_order", "order_id", order_id)
        return _fetch_all(
            conn,
            """
            SELECT
              h.*,
              e.first_name AS changed_by_first_name,
              e.last_name AS changed_by_last_name
            FROM order_status_history h
            LEFT JOIN employee e ON e.employee_id = h.changed_by_employee_id
            WHERE h.order_id = ?
            ORDER BY h.changed_at, h.history_id
            """,
            (order_id,),
        )
    finally:
        conn.close()


# =========================
# Reports
# =========================
def inventory_report(permissions: Sequence[str], dealership_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Inventory report (counts vehicles).

    Permissions:
        Requires Cars.Read (or bypass_all_auth).

    Args:
        dealership_id: If provided, scope the report to a single dealership.

    Returns:
        {
          "by_status": [{"dealership_id","dealership_name","status","count"}, ...],
          "by_model":  [{"dealership_id","dealership_name","brand_name","model_name","count"}, ...]
        }
    """
    authorizer.authorize(
        access="read", resources={"vehicle", "dealership", "model", "brand"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        params: List[Any] = []
        where = ""
        if dealership_id is not None:
            where = "WHERE v.dealership_id = ?"
            params.append(int(dealership_id))

        by_status = _fetch_all(
            conn,
            f"""
            SELECT v.dealership_id, d.name AS dealership_name, v.status, COUNT(*) AS count
            FROM vehicle v
            JOIN dealership d ON d.dealership_id = v.dealership_id
            {where}
            GROUP BY v.dealership_id, v.status
            ORDER BY d.name, v.status
            """,
            tuple(params),
        )

        by_model = _fetch_all(
            conn,
            f"""
            SELECT
              v.dealership_id, d.name AS dealership_name,
              b.name AS brand_name, m.name AS model_name,
              COUNT(*) AS count
            FROM vehicle v
            JOIN dealership d ON d.dealership_id = v.dealership_id
            JOIN model m ON m.model_id = v.model_id
            JOIN brand b ON b.brand_id = m.brand_id
            {where}
            GROUP BY v.dealership_id, b.name, m.name
            ORDER BY d.name, b.name, m.name
            """,
            tuple(params),
        )

        return {"by_status": by_status, "by_model": by_model}
    finally:
        conn.close()


def sales_report(
    permissions: Sequence[str],
    *,
    dealership_id: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sales report (FULFILLED + DELIVERED orders).

    Permissions:
        Requires Sales.Read (or bypass_all_auth).

    Args:
        dealership_id: If provided, scope to a single dealership.
        date_from/date_to: inclusive bounds on order_date (SQLite datetime text)

    Returns:
        {
          "totals": {"orders_count": int, "gross_sales": int},
          "by_day": [{"day","orders_count","gross_sales"}, ...],
          "by_employee": [{"employee_id","employee_name","orders_count","gross_sales"}, ...]
        }
    """
    authorizer.authorize(
        access="read", resources={"sales_order", "employee"}, permissions=list(permissions)
    )
    conn = _connect_ro()
    try:
        where = ["o.status IN ('FULFILLED','DELIVERED')"]
        params: List[Any] = []

        def add(cond: str, val: Any) -> None:
            where.append(cond)
            params.append(val)

        if dealership_id is not None:
            add("o.dealership_id = ?", int(dealership_id))
        if date_from is not None:
            add("o.order_date >= ?", date_from)
        if date_to is not None:
            add("o.order_date <= ?", date_to)

        where_sql = "WHERE " + " AND ".join(where)

        totals = _fetch_one(
            conn,
            f"""
            SELECT
              COUNT(*) AS orders_count,
              COALESCE(SUM(o.total_amount), 0) AS gross_sales
            FROM sales_order o
            {where_sql}
            """,
            tuple(params),
        ) or {"orders_count": 0, "gross_sales": 0}

        by_day = _fetch_all(
            conn,
            f"""
            SELECT
              substr(o.order_date, 1, 10) AS day,
              COUNT(*) AS orders_count,
              COALESCE(SUM(o.total_amount), 0) AS gross_sales
            FROM sales_order o
            {where_sql}
            GROUP BY day
            ORDER BY day DESC
            """,
            tuple(params),
        )

        by_employee = _fetch_all(
            conn,
            f"""
            SELECT
              o.employee_id,
              e.first_name || ' ' || e.last_name AS employee_name,
              COUNT(*) AS orders_count,
              COALESCE(SUM(o.total_amount), 0) AS gross_sales
            FROM sales_order o
            LEFT JOIN employee e ON e.employee_id = o.employee_id
            {where_sql}
            GROUP BY o.employee_id
            ORDER BY gross_sales DESC
            """,
            tuple(params),
        )

        return {"totals": totals, "by_day": by_day, "by_employee": by_employee}
    finally:
        conn.close()



# --------- Assistant Functions ---------
def assistant_query(request_permissions: List[str], body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous assistant endpoint.
    Returns: {"message": <str>, "trace": <dict>}
    """
    if not isinstance(body, dict):
        raise BadRequest("Body must be a JSON object.")

    client_details = body.get("clientDetails") or {}
    chat_details = body.get("chatDetails") or {}
    stored_procedures = body.get("storedProcedures") or []

    prompt = chat_details.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("chatDetails.prompt is required.")

    exc_method = chat_details.get("exc_method", "full_access")
    if exc_method not in {"full_access", 'granular', 'delegated', 'in_prompt', 'stored_proc'}:
        raise BadRequest(f"Invalid exc_method: {exc_method}")

    if exc_method == "full_access":
        agent_permissions = request_permissions
        user_permissions = request_permissions
    elif exc_method == "in_prompt":
        agent_permissions = [BYPASS_PERMISSION]
        user_permissions = request_permissions
    elif exc_method == "granular":
        agent_permissions = request_permissions
        user_permissions = [BYPASS_PERMISSION]
    elif exc_method == "delegated":
        agent_permissions = request_permissions
        user_permissions = request_permissions
    elif exc_method == "stored_proc":
        agent_permissions = request_permissions
        user_permissions = request_permissions

    llm_client = _instantiate_llm_client(client_details)
    agent = _build_sql_agent(
        db_path=_resolve_db_path(),
        llm_client=llm_client,
        agent_permissions=agent_permissions,
        user_permissions=user_permissions,
        chat_details=chat_details,
        stored_procedures=stored_procedures,
        stream=False,
    )

    # Run SQLAgent (raw sql+results), then generate a customer-facing response outside the SQLAgent.
    raw_parts: List[str] = []
    final_trace: Dict[str, Any] = {}
    for ev in agent.act(user_query=prompt, exc_method=exc_method):
        if ev.get("type") == "message":
            raw_parts.append(ev.get("content", "") or "")
        elif ev.get("type") == "trace":
            final_trace = ev.get("content") or {}

    raw_payload = extract_json_object("".join(raw_parts).strip()) or {}
    sql = str(raw_payload.get("sql") or "")
    results = raw_payload.get("results")

    rg_prompts = ResponseGeneratorPrompts(
        system_prompt_template=RESPONSE_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
        user_prompt_template=RESPONSE_GENERATOR_USER_PROMPT_TEMPLATE,
    )
    responder = ResponseGeneratorAgent(
        llm_client=llm_client,
        prompts=rg_prompts,
    )
    message = responder.generate(user_query=prompt, sql=sql, results=results)
    return {"message": message.strip(), "trace": final_trace}


def assistant_stream(request_permissions: List[str], body: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming assistant endpoint.
    Yields dict events: {"type": "action" | "message" | "trace", "content": ...}
    """
    if not isinstance(body, dict):
        raise BadRequest("Body must be a JSON object.")

    client_details = body.get("clientDetails") or {}
    chat_details = body.get("chatDetails") or {}
    stored_procedures = body.get("storedProcedures") or []

    prompt = chat_details.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("chatDetails.prompt is required.")

    exc_method = chat_details.get("exc_method", "full_access")
    if exc_method not in {"full_access", 'granular', 'delegated', 'in_prompt', 'stored_proc'}:
        raise BadRequest(f"Invalid exc_method: {exc_method}")

    if exc_method == "full_access":
        agent_permissions = request_permissions
        user_permissions = request_permissions
    elif exc_method == "in_prompt":
        agent_permissions = [BYPASS_PERMISSION]
        user_permissions = request_permissions
    elif exc_method == "granular":
        agent_permissions = request_permissions
        user_permissions = [BYPASS_PERMISSION]
    elif exc_method == "delegated":
        agent_permissions = request_permissions
        user_permissions = request_permissions
    elif exc_method == "stored_proc":
        agent_permissions = request_permissions
        user_permissions = request_permissions

    llm_client = _instantiate_llm_client(client_details)
    agent = _build_sql_agent(
        db_path=_resolve_db_path(),
        llm_client=llm_client,
        agent_permissions=agent_permissions,
        user_permissions=user_permissions,
        chat_details=chat_details,
        stored_procedures=stored_procedures,
        stream=True,
    )
    start_total = time.monotonic()

    # Pass through action + trace events as they arrive (stream mode).
    # Capture raw sql+results message to format externally.
    raw_parts: List[str] = []

    for ev in agent.act(user_query=prompt, exc_method=exc_method):
        et = ev.get("type")
        if et == "message":
            raw_parts.append(ev.get("content", "") or "")
            continue
        yield ev

    raw_payload = extract_json_object("".join(raw_parts).strip()) or {}
    sql = str(raw_payload.get("sql") or "")
    results = raw_payload.get("results")

    rg_prompts = ResponseGeneratorPrompts(
        system_prompt_template=RESPONSE_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
        user_prompt_template=RESPONSE_GENERATOR_USER_PROMPT_TEMPLATE,
    )
    responder = ResponseGeneratorAgent(
        llm_client=llm_client,
        prompts=rg_prompts,
    )

    # Keep trace continuity for response generation where supported.
    chunks = responder.generate_stream(user_query=prompt, sql=sql, results=results, tracer=agent.tracer)
    last_seen = None
    if agent.tracer:
        traces = agent.tracer.get_traces()
        last_seen = traces[-1] if traces else None

    def _drain_new():
        nonlocal last_seen
        if not agent.tracer:
            return
        for item in agent.tracer.iter_new_traces_since(last_seen):
            last_seen = item
            yield {"type": "trace", "content": item}

    for evm in stream_message_events(chunks, min_chars=20):
        yield evm
        yield from _drain_new()

    # Final flush + include response generation time in total latency.
    yield from _drain_new()
    if agent.tracer:
        yield {
            "type": "trace",
            "content": {
                "model": getattr(llm_client, "model", None),
                "calls": agent.tracer.get_traces(),
                "total_latency_ms": int((time.monotonic() - start_total) * 1000),
            },
        }

AVAILABLE_STORED_PROCEDURES = {
        # Dealerships
        "list_dealerships": list_dealerships,
        "get_dealership": get_dealership,
        
        # Brands
        "list_brands": list_brands,
        "get_brand": get_brand,
        "create_brand": create_brand,
        "update_brand": update_brand,
        "delete_brand": delete_brand,
        
        # Models
        "list_models": list_models,
        "get_model": get_model,
        "create_model": create_model,
        "update_model": update_model,
        "delete_model": delete_model,
        
        # Vehicles
        "list_vehicles": list_vehicles,
        "get_vehicle": get_vehicle,
        "create_vehicle": create_vehicle,
        "update_vehicle": update_vehicle,
        "delete_vehicle": delete_vehicle,
        
        # Customers
        "list_customers": list_customers,
        "get_customer": get_customer,
        "create_customer": create_customer,
        "update_customer": update_customer,
        "delete_customer": delete_customer,
        
        # Employees
        "list_employees": list_employees,
        "get_employee": get_employee,
        "create_employee": create_employee,
        "update_employee": update_employee,
        "delete_employee": delete_employee,
        
        # Sales Orders
        "list_orders": list_orders,
        "get_order": get_order,
        "create_order": create_order,
        "update_order": update_order,
        "delete_order": delete_order,
        "add_order_item": add_order_item,
        "remove_order_item": remove_order_item,
        "list_order_history": list_order_history,
        
        # Payments
        "create_payment": create_payment,
        
        # Reports
        "inventory_report": inventory_report,
        "sales_report": sales_report,
    }