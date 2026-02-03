# qbtrain/utils/callutils.py
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args, get_origin, Literal, Union


def get_stored_procedure_signatures(functions: Dict[str, Callable]) -> Optional[str]:
    if not functions:
        return None
    sigs = []
    for name, func in functions.items():
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        params = sig.parameters
        args = [
            {
                "name": p.name,
                "kind": str(p.kind),
                "annotation": getattr(p.annotation, "__name__", str(p.annotation)),
                "default": None if p.default is inspect.Parameter.empty else p.default,
            }
            for p in params.values()
            if p.default is inspect.Parameter.empty
        ]
        kwargs = [
            {
                "name": p.name,
                "kind": str(p.kind),
                "annotation": getattr(p.annotation, "__name__", str(p.annotation)),
                "default": None if p.default is inspect.Parameter.empty else p.default,
            }
            for p in params.values()
            if p.default is not inspect.Parameter.empty
        ]
        sigs.append(
            f"Function Name: {name}\n"
            f"Signature: {sig}\n"
            f"Args: {args}\n"
            f"Kwargs: {kwargs}\n"
            f"Docstring: {doc}\n"
            "\n---\n"
        )
    return "".join(sigs).strip()


def unwrap_container_annotation(anno: Any) -> Any:
    origin = get_origin(anno)
    if origin in (list, tuple, dict, set, List, Tuple, Dict):
        args = get_args(anno)
        if not args:
            return None
        if origin in (dict, Dict) and len(args) >= 2:
            return args[1]
        return args[0]
    return None


def _is_union(anno: Any) -> bool:
    return get_origin(anno) is Union


def _union_members(anno: Any) -> Tuple[Any, ...]:
    return get_args(anno)


def coerce_value(value: Any, anno: Any) -> Any:
    if anno in (inspect.Parameter.empty, Any) or anno is None:
        return value

    origin = get_origin(anno)
    args = get_args(anno)

    if _is_union(anno):
        if value is None:
            return None
        members = tuple(a for a in _union_members(anno) if a is not type(None))
        if not members:
            return value
        last_err: Optional[Exception] = None
        for a in members:
            try:
                return coerce_value(value, a)
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return value

    if origin is Literal:
        lit_vals = list(args)
        if value in lit_vals:
            return value
        if lit_vals:
            target_type = type(lit_vals[0])
            return coerce_value(value, target_type)
        return value

    if origin in (list, List):
        elem_t = args[0] if args else Any
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                value = parsed
            except Exception:
                value = [value]
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value).__name__}")
        return [coerce_value(v, elem_t) for v in value]

    if origin in (dict, Dict):
        val_t = args[1] if len(args) >= 2 else Any
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                value = parsed
            except Exception:
                raise ValueError("Expected JSON object string for dict-typed argument.")
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")
        return {k: coerce_value(v, val_t) for k, v in value.items()}

    if anno is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "t", "yes", "y", "1"):
                return True
            if s in ("false", "f", "no", "n", "0"):
                return False
        raise ValueError(f"Cannot coerce {value!r} to bool")

    if anno is int:
        if isinstance(value, bool):
            raise ValueError(f"Cannot coerce bool {value!r} to int")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            return int(value.strip(), 10)
        return int(value)

    if anno is float:
        if isinstance(value, bool):
            raise ValueError(f"Cannot coerce bool {value!r} to float")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        return float(value)

    if anno is str:
        if value is None:
            return ""
        return str(value)

    if isinstance(anno, type):
        if isinstance(value, anno):
            return value
        try:
            return anno(value)
        except Exception:
            return value

    return value


def coerce_args_to_func(
    func: Callable[..., Any],
    args: List[Any],
    kwargs: Dict[str, Any],
) -> tuple[List[Any], Dict[str, Any]]:
    sig = inspect.signature(func)
    try:
        type_hints: Dict[str, Any] = typing_get_type_hints(func)
    except Exception:
        type_hints = {}
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError as e:
        raise ValueError(f"Invalid stored procedure arguments for {func.__name__}{sig}: {e}") from e
    coerced: Dict[str, Any] = {}
    for name, value in bound.arguments.items():
        param = sig.parameters.get(name)
        anno = type_hints.get(name, (param.annotation if param else inspect.Parameter.empty))
        if param and param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not isinstance(value, tuple):
                value = tuple(value) if isinstance(value, list) else (value, )
            elem_type = unwrap_container_annotation(anno) or inspect.Parameter.empty
            coerced[name] = tuple(coerce_value(v, elem_type) for v in value)
            continue
        if param and param.kind == inspect.Parameter.VAR_KEYWORD:
            if not isinstance(value, dict):
                raise ValueError(f"Invalid **kwargs for {func.__name__}: expected object, got {type(value).__name__}")
            elem_type = unwrap_container_annotation(anno) or inspect.Parameter.empty
            coerced[name] = {k: coerce_value(v, elem_type) for k, v in value.items()}
            continue
        coerced[name] = coerce_value(value, anno)
    for k, v in coerced.items():
        bound.arguments[k] = v
    return list(bound.args), dict(bound.kwargs)


def typing_get_type_hints(func: Callable[..., Any]) -> Dict[str, Any]:
    try:
        from typing import get_type_hints as _gth  # py39+
        return _gth(func)
    except Exception:
        return {}
    

def normalize_tool_result(result: Any) -> Any:
    if isinstance(result, dict):
        if ("columns" in result and "rows" in result) or ("status" in result and "rows_affected" in result):
            return result
        return {"columns": list(result.keys()), "rows": [result], "row_count": 1}
    if result is None:
        return {"status": "ok", "rows_affected": 0}
    if isinstance(result, (list, tuple)):
        seq = list(result)
        if not seq:
            return {"columns": [], "rows": [], "row_count": 0}
        if all(isinstance(x, dict) for x in seq):
            cols: List[str] = []
            seen = set()
            for row in seq:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        cols.append(k)
            rows = [{c: row.get(c, None) for c in cols} for row in seq]
            return {"columns": cols, "rows": rows, "row_count": len(rows)}
        if all(isinstance(x, (list, tuple)) for x in seq):
            max_len = max(len(x) for x in seq)
            cols = [f"col{i+1}" for i in range(max_len)]
            rows = []
            for r in seq:
                rr = list(r)
                rr += [None] * (max_len - len(rr))
                rows.append(dict(zip(cols, rr)))
            return {"columns": cols, "rows": rows, "row_count": len(rows)}
        return {"value": seq}
    return {"value": result}
