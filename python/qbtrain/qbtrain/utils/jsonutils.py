# qbtrain/utils/jsonutils.py
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def to_json_str(obj: Any) -> str:
    return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)


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
    s = s.replace("\\'", "'")
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s

def extract_first_json(text: Optional[str]) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from arbitrary text.
    Supports optional ```json ... ``` fences or raw embedded JSON.
    """
    if not text:
        raise ValueError("No input text")

    s = text.strip()

    # Prefer fenced JSON blocks if present.
    # Handles: ```json { ... } ``` or ``` { ... } ```
    for fence in ("```json", "```"):
        start = s.find(fence)
        if start != -1:
            end = s.find("```", start + len(fence))
            if end != -1:
                candidate = s[start + len(fence) : end].strip()
                if candidate:
                    # If the fenced block contains extra text, still scan inside it.
                    try:
                        obj = extract_first_json(candidate) if candidate[0] != "{" else json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass

    # Streaming/scan for first parseable JSON object via JSONDecoder.raw_decode.
    dec = json.JSONDecoder()
    n = len(s)
    i = 0
    while i < n:
        # Find next plausible object start.
        if s[i] != "{":
            i = s.find("{", i + 1)
            if i == -1:
                break
        try:
            obj, end = dec.raw_decode(s[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            i += 1
            continue
        # If we decoded something non-dict, keep scanning after the decoded segment.
        i += max(1, end)

    raise ValueError("No valid JSON object found")

def extract_json_object(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        raise ValueError("No input text")
    s = text.strip()
    m = _JSON_FENCE_RE.search(s)
    candidate = m.group(1).strip() if m else _extract_balanced_object(s)
    if not candidate:
        return {}
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed JSON is not an object")
    except json.JSONDecodeError:
        pass
    repaired = _repair_common_non_json(candidate)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed JSON is not an object")
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(repaired)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Parsed object is not a dict")
    except Exception as e:
        raise ValueError(f"Could not parse object as JSON: {e}") from e
