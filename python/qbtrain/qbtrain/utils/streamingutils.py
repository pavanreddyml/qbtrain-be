# qbtrain/utils/streamingutils.py
from __future__ import annotations

from typing import Dict, Generator, Iterable, List, Tuple

Event = Dict[str, object]


def _emit_message_events(
    text: str,
    *,
    buf: str,
    min_chars: int = 20,
    final: bool = False,
) -> Tuple[List[Event], str]:
    events: List[Event] = []
    if text:
        buf += text
    while len(buf) >= min_chars:
        events.append({"type": "message", "content": buf[:min_chars]})
        buf = buf[min_chars:]
    if final and buf:
        events.append({"type": "message", "content": buf})
        buf = ""
    return events, buf


def stream_message_events(
    chunks: Iterable[str],
    *,
    min_chars: int = 20,
) -> Generator[Event, None, None]:
    buf = ""
    for chunk in chunks:
        events, buf = _emit_message_events(chunk, buf=buf, min_chars=min_chars, final=False)
        for e in events:
            yield e
    events, _ = _emit_message_events("", buf=buf, min_chars=min_chars, final=True)
    for e in events:
        yield e
