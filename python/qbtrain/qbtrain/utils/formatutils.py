import re 

import ast
import json
import re
from typing import Any, Dict, Optional

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

def _extract_balanced_object(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None

def _repair_common_non_json(s: str) -> str:
    # fix invalid JSON escape sequence \' (apostrophes don't need escaping in JSON)
    s = s.replace("\\'", "'")
    # remove trailing commas like {"a":1,}
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s

def extract_json_object(text: Optional[str]) -> Dict[str, Any]:
    """
    Extract a JSON object from:
      - ```json { ... } ```
      - or any text containing a top-level { ... } object

    Also tolerates common non-JSON issues like \' inside strings and trailing commas.
    Raises ValueError if no object can be parsed.
    """
    if not text:
        raise ValueError("No input text")

    s = text.strip()

    m = _JSON_FENCE_RE.search(s)
    candidate = m.group(1).strip() if m else _extract_balanced_object(s)
    if not candidate:
        raise ValueError("No JSON object found")

    # 1) strict JSON
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed JSON is not an object")
    except json.JSONDecodeError:
        pass

    # 2) repaired JSON
    repaired = _repair_common_non_json(candidate)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed JSON is not an object")
    except json.JSONDecodeError:
        pass

    # 3) last resort: python-literal-ish dict
    # (handle single quotes, True/False/None, etc.)
    py_like = repaired
    try:
        obj = ast.literal_eval(py_like)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed object is not a dict")
    except Exception as e:
        raise ValueError(f"Could not parse object as JSON: {e}") from e


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
        raise ValueError(
            f"Multiple SQL statements detected; refusing to execute. Remainder starts with: {remainder[:120]}"
        )

    if s.rstrip().endswith(";"):
        return non_empty[0] + ";"
    return non_empty[0]