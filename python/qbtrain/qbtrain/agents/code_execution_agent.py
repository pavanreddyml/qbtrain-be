# qbtrain/agents/code_execution_agent.py
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
)

from pydantic import BaseModel, Field

from ..ai.llm import LLMClient
from ..exceptions.exceptions import DenylistViolationError
from ..tracers import AgentTracer, Tracer
from ..utils.streamingutils import stream_message_events
from ..utils.traceutils import (
    TraceState,
    mark_trace_start,
    trace_event_if_changed,
    traces_since_start,
    emit_trace_if_new,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CodeExecutionPrompts(BaseModel):
    """Prompt bundle for CodeExecutionAgent."""

    code_gen_system_prompt: str = Field(..., min_length=1)
    code_gen_user_prompt_template: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class GeneratedCodeResponse(BaseModel):
    python_code: str
    explanation: str = ""

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Denylist scanner — runs OUTSIDE the LLM, pure string analysis
# ---------------------------------------------------------------------------

def scan_denylist(code: str, denylist: Set[str]) -> List[str]:
    """
    Scan generated Python source for denied libraries / commands.

    Checks:
      - import statements  (import X, from X import …)
      - subprocess / os.system / os.popen string arguments
      - shlex-style command tokens

    Returns a list of matched denylist entries found in *code*.
    """
    if not denylist:
        return []

    violations: List[str] = []
    lower_deny = {d.lower() for d in denylist}

    # --- import scanning ---
    import_pattern = re.compile(
        r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE
    )
    for m in import_pattern.finditer(code):
        module_chain = m.group(1).lower()
        parts = module_chain.split(".")
        for i in range(len(parts)):
            prefix = ".".join(parts[: i + 1])
            if prefix in lower_deny:
                violations.append(m.group(1))

    # --- shell command scanning ---
    string_pattern = re.compile(r"""(?:"|'|"{3}|'{3})(.*?)(?:"|'|"{3}|'{3})""", re.DOTALL)
    for m in string_pattern.finditer(code):
        content = m.group(1).lower()
        tokens = re.split(r"[\s;|&]+", content)
        for token in tokens:
            token_clean = token.strip().split("/")[-1]
            if token_clean in lower_deny:
                violations.append(token_clean)

    # --- generic token scan for anything remaining ---
    word_pattern = re.compile(r"\b([\w.]+)\b")
    for m in word_pattern.finditer(code):
        word = m.group(1).lower()
        if word in lower_deny and word not in [v.lower() for v in violations]:
            violations.append(m.group(1))

    return list(dict.fromkeys(violations))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CodeExecutionAgent:
    """
    Takes a natural-language query, generates a Python script via an LLM,
    enforces an externally-provided denylist, then executes the script.

    **All guardrails are external.** The agent itself does NOT restrict what
    the LLM can generate. The *only* gate is the denylist — a set of library
    names and shell commands that are scanned for *after* generation and
    *before* execution.  If any denied token is found, execution is blocked
    with a ``DenylistViolationError`` and no code is run.

    Events emitted (same contract as SQLAgent):
      - ``{"type": "action",  "content": "…"}``   — status updates
      - ``{"type": "message", "content": "…"}``   — streamed result text
      - ``{"type": "trace",   "content": {…}}``   — observability traces
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: CodeExecutionPrompts,
        denylist: Optional[Set[str]] = None,
        execution_timeout: int = 30,
        tracer: Optional[Tracer] = None,
        stream: bool = False,
    ):
        if prompts is None:
            raise ValueError("prompts must not be None.")
        self.prompts = prompts
        self.llm_client = llm_client
        self.denylist: Set[str] = set(denylist) if denylist else set()
        self.execution_timeout = execution_timeout
        self.tracer: Optional[Tracer] = tracer if tracer else AgentTracer()
        self._trace_state: TraceState = TraceState()
        self.stream = stream
        self._last_streamed_trace_item: Optional[Dict[str, Any]] = None

    # ---- tracing helpers (mirror SQLAgent) ----
    def _safe_trace(self, __type__: str, **kwargs: Any) -> None:
        try:
            if self.tracer:
                self.tracer.trace(agent_name="CodeExecutionAgent", __type__=__type__, **kwargs)
        except Exception:
            pass

    def _drain_stream_traces(self) -> Generator[Dict[str, Any], None, None]:
        if not self.stream or not self.tracer:
            return
        for item in self.tracer.iter_new_traces_since(self._last_streamed_trace_item):
            self._last_streamed_trace_item = item
            yield {"type": "trace", "content": item}

    # ---- public API ----
    def act(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        start_total = time.monotonic()
        mark_trace_start(self.tracer, self._trace_state)
        self._last_streamed_trace_item = None

        yield from self._act_generate_and_execute(user_query=user_query, start_total=start_total)

    def run(self, user_query: str) -> Dict[str, Any]:
        message_parts: List[str] = []
        final_trace: Dict[str, Any] = {}
        for ev in self.act(user_query=user_query):
            if ev.get("type") == "message":
                message_parts.append(ev.get("content", "") or "")
            elif ev.get("type") == "trace":
                final_trace = ev.get("content") or {}
        return {"message": "".join(message_parts).strip(), "trace": final_trace}

    # ---- internals ----
    def _emit_final_trace(self, start_total: float) -> Optional[Dict[str, Any]]:
        payload = {
            "model": getattr(self.llm_client, "model", None),
            "calls": [t for t in traces_since_start(self.tracer, self._trace_state) if isinstance(t, dict)],
            "total_latency_ms": int((time.monotonic() - start_total) * 1000),
        }
        return trace_event_if_changed(self._trace_state, payload)

    def _emit_result_message(self, *, code: str, stdout: str, stderr: str, returncode: int) -> Generator[Dict[str, Any], None, None]:
        payload = {
            "__code_execution_result__": True,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
        }
        import json
        raw = json.dumps(payload, ensure_ascii=False, default=str)
        for evm in stream_message_events([raw], min_chars=20):
            yield evm

    def _generate_code(self, user_query: str) -> str:
        user_prompt = self.prompts.code_gen_user_prompt_template.format(
            user_query=user_query,
        )
        out = self.llm_client.json_response(
            prompt=user_prompt,
            system_prompt=self.prompts.code_gen_system_prompt,
            schema=GeneratedCodeResponse,
            tracer=self.tracer,
            temperature=0,
            top_k=20,
        )
        if not isinstance(out, dict):
            raise ValueError("LLM did not return a JSON object for code generation.")
        code = out.get("python_code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("LLM returned invalid response: missing 'python_code' string.")

        self._safe_trace(
            __type__="agent",
            operation="Code Generation Output",
            output=out,
        )
        return code

    def _enforce_denylist(self, code: str) -> None:
        violations = scan_denylist(code, self.denylist)
        if violations:
            self._safe_trace(
                __type__="error",
                operation="Denylist Violation",
                violations=violations,
            )
            raise DenylistViolationError(
                f"Blocked: generated code contains denied tokens: {violations}"
            )

    def _execute_code(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {self.execution_timeout}s",
                "returncode": -1,
            }
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _act_generate_and_execute(
        self, user_query: str, *, start_total: float
    ) -> Generator[Dict[str, Any], None, None]:
        try:
            # --- Step 1: generate code ---
            if self.stream:
                yield {"type": "action", "content": "Generating Python script"}

            code = self._generate_code(user_query)

            if self.stream:
                yield from self._drain_stream_traces()

            # --- Step 2: enforce denylist (external gate) ---
            if self.stream:
                yield {"type": "action", "content": "Scanning denylist"}

            self._enforce_denylist(code)

            if self.stream:
                yield from self._drain_stream_traces()

            # --- Step 3: execute ---
            if self.stream:
                yield {"type": "action", "content": "Executing script"}

            exec_result = self._execute_code(code)

            self._safe_trace(
                __type__="agent",
                operation="Code Execution Result",
                returncode=exec_result["returncode"],
                stdout_len=len(exec_result["stdout"]),
                stderr_len=len(exec_result["stderr"]),
            )

            if self.stream:
                yield from self._drain_stream_traces()

            # --- Step 4: emit result ---
            yield from self._emit_result_message(
                code=code,
                stdout=exec_result["stdout"],
                stderr=exec_result["stderr"],
                returncode=exec_result["returncode"],
            )

            ev = self._emit_final_trace(start_total)
            if ev:
                yield ev

        except DenylistViolationError:
            yield from self._emit_result_message(
                code="",
                stdout="",
                stderr="Execution blocked: generated code contains denied libraries or commands.",
                returncode=-1,
            )
            if self.stream:
                yield from self._drain_stream_traces()
            ev = self._emit_final_trace(start_total)
            if ev:
                yield ev

        except Exception as e:
            msg = f"Agent error: {e}"
            for evm in stream_message_events([msg], min_chars=20):
                yield evm
            ev = self._emit_final_trace(start_total)
            if ev:
                yield ev
