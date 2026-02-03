from . import Tracer

from typing import Any

class AgentTracer(Tracer):
    def trace(self, agent_name: str, __type__: str, **kwargs: Any) -> None:
        self.trace_steps.append({"id": self._next_id, "agent_name": agent_name, "__type__": __type__, **kwargs})
        self._next_id += 1