# qbtrain/utils/sqlutils.py
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Set, Tuple
from urllib.parse import quote, urlparse

from sqlglot import ErrorLevel, exp, parse_one

SqlAccess = Literal["read", "write"]


def extract_single_sql_statement(text_in: str) -> str:
    if not isinstance(text_in, str):
        raise ValueError("SQL must be a string.")
    s = text_in.strip()
    if s.startswith("```"):
        s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", s).strip()
        s = re.sub(r"\s*```\s*$", "", s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    chunks = [c.strip() for c in s.split(";")]
    non_empty = [c for c in chunks if c]
    if not non_empty:
        raise ValueError("Empty SQL generated.")
    if len(non_empty) > 1:
        remainder = " ".join(non_empty[1:]).strip()
        raise ValueError(f"Multiple SQL statements detected; refusing to execute. Remainder starts with: {remainder[:120]}")
    return (non_empty[0] + ";") if s.rstrip().endswith(";") else non_empty[0]


def _strip_sqlite_scheme(s: str) -> str:
    return s[len("sqlite:") :] if s.startswith("sqlite:") else s


def _resolve_sqlite_path(db_uri: str) -> str:
    s = (db_uri or "").strip()
    if not s:
        raise ValueError("db_uri is required")
    if not s.startswith("sqlite:"):
        p = Path(s).expanduser()
        return ":memory:" if s == ":memory:" else str(p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())
    parsed = urlparse(s)
    if parsed.path in ("/:memory:", ":memory:") or s.endswith(":memory:"):
        return ":memory:"
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        raw = f"//{parsed.netloc}{parsed.path}"
    else:
        raw = parsed.path
    raw = raw or _strip_sqlite_scheme(s)
    if s.startswith("sqlite:///") and not s.startswith("sqlite:////"):
        raw = raw.lstrip("/")
        p = Path(raw).expanduser()
        return str((Path.cwd() / p).resolve())
    raw = re.sub(r"^/+", "/", raw)
    if re.match(r"^/[A-Za-z]:/", raw):
        raw = raw[1:]
    p = Path(raw).expanduser()
    return str(p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())


def _open_sqlite(db_uri: str, mode: str | None = None) -> sqlite3.Connection:
    path = _resolve_sqlite_path(db_uri)
    if path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        p = Path(path)
        if mode in ("ro", "rw") and not p.exists():
            raise FileNotFoundError(f"SQLite database file not found: {p}")
        if mode in ("ro", "rw"):
            file_uri = "file:" + quote(p.as_posix(), safe="/:")
            conn = sqlite3.connect(f"{file_uri}?mode={mode}", uri=True)
        else:
            conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def _strip_leading_comments(sql: str) -> str:
    s = sql.lstrip()
    while True:
        if s.startswith("--"):
            nl = s.find("\n")
            if nl == -1:
                return ""
            s = s[nl + 1 :].lstrip()
            continue
        if s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                return ""
            s = s[end + 2 :].lstrip()
            continue
        break
    return s


_READ_KEYWORDS = {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA"}
_WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "MERGE", "GRANT", "REVOKE"}


def _keyword_access_fallback(sql: str) -> SqlAccess:
    s = _strip_leading_comments(sql)
    if not s:
        return "read"
    m = re.match(r"^([A-Za-z]+)\b", s)
    if not m:
        return "read"
    first = m.group(1).upper()
    if first in _READ_KEYWORDS:
        return "read"
    if first in _WRITE_KEYWORDS:
        return "write"
    if first == "WITH":
        if re.search(r"\b(" + "|".join(sorted(_WRITE_KEYWORDS)) + r")\b", s, flags=re.I):
            return "write"
        return "read"
    return "write"


def _write_exp_types() -> Tuple[type, ...]:
    names = ["Insert", "Update", "Delete", "Create", "Drop", "Alter", "Truncate", "Replace", "Merge", "Grant", "Revoke"]
    out: List[type] = []
    for n in names:
        t = getattr(exp, n, None)
        if isinstance(t, type):
            out.append(t)
    return tuple(out)


_WRITE_TYPES = _write_exp_types()


def _cte_names(tree: exp.Expression) -> Set[str]:
    names: Set[str] = set()
    for c in tree.find_all(exp.CTE):
        alias = c.alias
        if alias:
            names.add(alias.lower())
    return names


def _norm_ident(name: str) -> str:
    s = name.strip()
    if s.startswith(("`", '"', "[")) and s.endswith(("`", '"', "]")) and len(s) >= 2:
        s = s[1:-1]
    if "." in s:
        s = s.split(".")[-1]
    return s.strip().lower()


def extract_referenced_tables(sql: str, db_uri: str | None = None) -> Set[str]:
    stmt = extract_single_sql_statement(sql)
    try:
        tree = parse_one(stmt, read="sqlite", error_level=ErrorLevel.IGNORE)
        if tree is None:
            return set()
        ctes = _cte_names(tree)
        tables: Set[str] = set()
        for t in tree.find_all(exp.Table):
            name = getattr(t, "name", None) or ""
            norm = _norm_ident(str(name))
            if norm and norm not in ctes:
                tables.add(norm)
        return tables
    except Exception:
        return set()


def is_read_only_sql(sql: str, db_uri: str | None = None) -> bool:
    stmt = extract_single_sql_statement(sql)
    try:
        tree = parse_one(stmt, read="sqlite", error_level=ErrorLevel.IGNORE)
        if tree is None:
            return _keyword_access_fallback(stmt) == "read"
        if _WRITE_TYPES and next(tree.find_all(_WRITE_TYPES), None) is not None:
            return False
        return True
    except Exception:
        return _keyword_access_fallback(stmt) == "read"


def sql_access(sql: str, db_uri: str | None = None) -> SqlAccess:
    stmt = extract_single_sql_statement(sql)
    return "read" if is_read_only_sql(stmt, db_uri=db_uri) else "write"


def analyze_sql(sql: str, db_uri: str | None = None) -> tuple[SqlAccess, Set[str], str]:
    stmt = extract_single_sql_statement(sql)
    access: SqlAccess = "read" if is_read_only_sql(stmt, db_uri=db_uri) else "write"
    tables = extract_referenced_tables(stmt, db_uri=db_uri)
    return access, tables, stmt


def execute_sql(
    db_uri: str,
    sql: str,
    mode: str | None = None,
    *,
    max_rows: int | None = None,
) -> Any:
    sql_stmt = extract_single_sql_statement(sql)
    conn = _open_sqlite(db_uri, mode=mode)
    try:
        cur = conn.cursor()
        cur.execute(sql_stmt)
        if cur.description is not None:
            cols = [c[0] for c in cur.description]
            rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
            out_rows = [dict(zip(cols, row)) for row in rows]
            return {"columns": cols, "rows": out_rows, "row_count": len(out_rows)}
        conn.commit()
        return {"status": "ok", "rows_affected": int(cur.rowcount or 0)}
    except sqlite3.Error as e:
        try:
            conn.rollback()
        except Exception:
            pass
        raise RuntimeError(f"Database error: {e}") from e
    finally:
        try:
            conn.close()
        except Exception:
            pass


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def get_schema_context(db_uri: str) -> str:
    conn = _open_sqlite(db_uri, mode="ro")
    try:
        cur = conn.cursor()
        conn.row_factory = sqlite3.Row

        def _has_table(name: str) -> bool:
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
            return cur.fetchone() is not None

        # ---- Load documentation (optional) ----
        table_docs: Dict[str, str] = {}
        col_docs: Dict[Tuple[str, str], Tuple[str, str | None]] = {}
        join_hints: List[sqlite3.Row] = []

        try:
            if _has_table("__doc_table"):
                cur.execute("SELECT table_name, description FROM __doc_table")
                for tn, desc in (cur.fetchall() or []):
                    if tn and desc:
                        table_docs[str(tn)] = str(desc)

            if _has_table("__doc_column"):
                cur.execute("SELECT table_name, column_name, description, example FROM __doc_column")
                for tn, cn, desc, ex in (cur.fetchall() or []):
                    if tn and cn and desc:
                        col_docs[(str(tn), str(cn))] = (str(desc), (None if ex is None else str(ex)))

            if _has_table("__doc_join"):
                cur.execute(
                    "SELECT left_table,left_column,right_table,right_column,join_type,relationship,notes "
                    "FROM __doc_join "
                    "ORDER BY left_table,left_column,right_table,right_column"
                )
                join_hints = cur.fetchall() or []
        except sqlite3.Error:
            table_docs = {}
            col_docs = {}
            join_hints = []

        # ---- Fetch schema objects (excluding sqlite_* and __doc_*) ----
        cur.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
              AND name NOT LIKE '__doc_%'
            ORDER BY type, name
            """
        )
        objs = cur.fetchall() or []
        if not objs:
            return "-- dialect: sqlite\n\nNo tables or views found."

        tables = [r["name"] for r in objs if r["type"] == "table"]
        views = [r["name"] for r in objs if r["type"] == "view"]

        # Prefer domain-meaningful order (helps an LLM "think like the schema designer")
        preferred = [
            "brand",
            "model",
            "dealership",
            "vehicle",
            "employee",
            "customer",
            "sales_order",
            "order_item",
            "payment",
            "order_status_history",
        ]
        pref_rank = {n: i for i, n in enumerate(preferred)}
        tables.sort(key=lambda n: (pref_rank.get(n, 10_000), n))
        views.sort()

        def _datatype_str(c: sqlite3.Row) -> str:
            base = (c["type"] or "").strip() or "UNKNOWN"
            if bool(c["notnull"]):
                base += " NOT NULL"
            if c["dflt_value"] is not None:
                base += f" DEFAULT {c['dflt_value']}"
            return base

        # If __doc_join is empty/missing, fall back to declared FK relationships as joins
        if not join_hints:
            try:
                derived: List[Dict[str, str]] = []
                for t in tables:
                    cur.execute(f"PRAGMA foreign_key_list({quote_ident(t)})")
                    for fk in (cur.fetchall() or []):
                        derived.append(
                            {
                                "left_table": t,
                                "left_column": fk["from"],
                                "right_table": fk["table"],
                                "right_column": fk["to"],
                                "join_type": "INNER",
                                "relationship": "derived from declared foreign key",
                                "notes": "This join was inferred from SQLite foreign_key_list(). Consider adding __doc_join rows for richer guidance.",
                            }
                        )
                join_hints = derived  # type: ignore[assignment]
            except sqlite3.Error:
                join_hints = []

        # ---- Render structured context (LLM-oriented) ----
        lines: List[str] = []
        lines.append("-- dialect: sqlite")
        lines.append("")
        lines.append("List of all tables and column information")

        for table_name in tables:
            lines.append(f"- {table_name}")
            tdesc = table_docs.get(table_name)
            if tdesc:
                lines.append(f"  - Table description: {tdesc}")
            else:
                lines.append("  - Table description: No description.")

            cur.execute(f"PRAGMA table_info({quote_ident(table_name)})")
            cols = cur.fetchall() or []
            for c in cols:
                col = c["name"]
                dtype = _datatype_str(c)
                doc = col_docs.get((table_name, col))
                desc = doc[0] if doc else "No description."
                ex = doc[1] if doc else None
                ex_str = f" Example value in this column: {ex if ex else 'unknown'}"
                ex_str = ""
                # Use fully-qualified table.column to reduce ambiguity for the LLM
                lines.append(
                    f"  - {table_name}.{col} ({dtype}): {desc}{ex_str}"
                )

        if views:
            lines.append("")
            lines.append("List of all views (read-only shortcuts)")
            for view_name in views:
                lines.append(f"- {view_name}")
                vdesc = table_docs.get(view_name)
                if vdesc:
                    lines.append(f"  - View description: {vdesc}")
                else:
                    lines.append("  - View description: No description.")

                # PRAGMA table_info works for views too; types may be UNKNOWN/blank
                cur.execute(f"PRAGMA table_info({quote_ident(view_name)})")
                vcols = cur.fetchall() or []
                for c in vcols:
                    col = c["name"]
                    dtype = _datatype_str(c)
                    doc = col_docs.get((view_name, col))
                    desc = doc[0] if doc else "No description."
                    ex = doc[1] if doc else None
                    ex_str = ex if ex else "unknown"
                    lines.append(
                        f"  - {view_name}.{col} ({dtype}): {desc} Example value in this column: {ex_str}"
                    )

        lines.append("")
        lines.append("List of All Joins")
        if not join_hints:
            lines.append("- No join information found.")
        else:
            for j in join_hints:
                lt = j["left_table"]
                lc = j["left_column"]
                rt = j["right_table"]
                rc = j["right_column"]
                jt = (j["join_type"] or "INNER").upper()
                rel = (j.get("relationship") or "").strip() if isinstance(j, dict) else (j["relationship"] or "").strip()
                notes = (j.get("notes") or "").strip() if isinstance(j, dict) else (j["notes"] or "").strip()

                msg = f"- {lt}.{lc} -> {rt}.{rc} [{jt}]"
                if rel:
                    msg += f": {rel}"
                lines.append(msg)
                if notes:
                    lines.append(f"  - {notes}")

        return "\n".join(lines).strip()
    except sqlite3.Error as e:
        return f"-- dialect: sqlite\n\nSchema inspection failed: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass