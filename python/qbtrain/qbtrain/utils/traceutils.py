# qbtrain/utils/traceutils.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

TraceItem = Dict[str, Any]
TracePayload = Dict[str, Any]


class TraceState:
    def __init__(self) -> None:
        self.start_idx: int = 0
        self.last_seen_trace: Optional[TraceItem] = None
        self.last_yielded_payload: Optional[TracePayload] = None


def get_traces(tracer: Any) -> List[TraceItem]:
    if tracer is None:
        return []
    get_traces_fn = getattr(tracer, "get_traces", None)
    if callable(get_traces_fn):
        try:
            out = get_traces_fn()
            return out if isinstance(out, list) else []
        except Exception:
            return []
    steps = getattr(tracer, "trace_steps", None)
    return steps if isinstance(steps, list) else []


def mark_trace_start(tracer: Any, state: TraceState) -> None:
    state.start_idx = len(get_traces(tracer))
    state.last_seen_trace = None
    state.last_yielded_payload = None


def traces_since_start(tracer: Any, state: TraceState) -> List[TraceItem]:
    traces = get_traces(tracer)
    if state.start_idx <= 0:
        return traces
    if state.start_idx >= len(traces):
        return []
    return traces[state.start_idx :]


def last_trace_item(tracer: Any, state: TraceState) -> TraceItem:
    traces = traces_since_start(tracer, state)
    if not traces:
        return {}
    last = traces[-1]
    return last if isinstance(last, dict) else {}


def trace_event_if_changed(state: TraceState, payload: TracePayload, *, event_type: str = "trace") -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    if payload == state.last_yielded_payload:
        return None
    state.last_yielded_payload = payload
    return {"type": event_type, "content": payload}


def emit_trace_if_new(tracer: Any, state: TraceState, *, event_type: str = "trace") -> Optional[Dict[str, Any]]:
    if tracer is None:
        return None
    it = getattr(tracer, "iter_new_traces_since", None)
    if callable(it):
        newest: Optional[TraceItem] = None
        try:
            gen = it(state.last_seen_trace, False)  # type: ignore[misc]
        except TypeError:
            try:
                gen = it(state.last_seen_trace, redact=False)  # type: ignore[misc]
            except TypeError:
                gen = it(state.last_seen_trace)
        for t in gen:
            if isinstance(t, dict):
                newest = t
        if newest is None:
            return None
        state.last_seen_trace = newest
        return trace_event_if_changed(state, newest, event_type=event_type)
    payload = last_trace_item(tracer, state)
    return trace_event_if_changed(state, payload, event_type=event_type)
