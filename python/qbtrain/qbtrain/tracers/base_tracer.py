from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional


TraceItem = Dict[str, Any]


class Tracer(ABC):
    def __init__(self) -> None:
        self.trace_steps: List[TraceItem] = []
        self._next_id: int = 0

    @abstractmethod
    def trace(self, agent_name: str, __type__: str, **kwargs: Any) -> None:
        pass

    def reset(self) -> None:
        self.trace_steps.clear()
        self._next_id = 0

    def get_traces(self) -> List[TraceItem]:
        return self.trace_steps

    def iter_new_traces_since(
        self,
        last_seen: Optional[TraceItem],
    ) -> Generator[TraceItem, None, None]:
        traces = self.trace_steps
        if not traces:
            return

        if last_seen is not None and traces[-1] == last_seen:
            return

        start_idx = 0
        if last_seen is not None:
            for i in range(len(traces) - 1, -1, -1):
                if traces[i] == last_seen:
                    start_idx = i + 1
                    break

        for item in traces[start_idx:]:
            yield item
