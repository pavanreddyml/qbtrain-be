# qbtrain/agents/sql_agent.py
from __future__ import annotations

import re

from typing import (
    Literal,
    Optional,
    Callable,
    Dict,
    List,
    Any,
    Iterable,
    Generator,
    Tuple,
)

import json
import time

from pydantic import BaseModel, RootModel, Field

from qbtrain.tracers import AgentTracer, Tracer

from ..ai.llm import LLMClient
from ..exceptions.exceptions import PermissionError
from qbtrain.utils.authutils import Authorizer

from ..utils.streamingutils import stream_message_events
from ..utils.callutils import (
    coerce_args_to_func,
    normalize_tool_result,
    get_stored_procedure_signatures,
)
from ..utils.sqlutils import (
    execute_sql,
    analyze_sql,
    extract_single_sql_statement,
)
from ..utils.jsonutils import to_json_str
from ..utils.traceutils import (
    TraceState,
    mark_trace_start,
    trace_event_if_changed,
    traces_since_start,
    emit_trace_if_new,
)


class SQLAgentPrompts(BaseModel):
    """
    Prompt bundle for SQLAgent.
    All fields are required and must be non-empty strings.
    """

    # Planning
    planner_system_prompt_template: str = Field(..., min_length=1)
    planner_user_prompt_template: str = Field(..., min_length=1)

    # SQL generation (receives only the plan, no user query/schema)
    sql_gen_system_prompt_template: str = Field(..., min_length=1)
    sql_gen_user_prompt_template: str = Field(..., min_length=1)

    # Stored procedure selection
    stored_proc_system_prompt_template: str = Field(..., min_length=1)
    stored_proc_user_prompt_template: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class StoredProcCallModel(BaseModel):
    function_name: str
    signature: Optional[str] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        extra = "forbid"


class StoredProcTerminateModel(BaseModel):
    terminate: Literal[True]
    reason: str

    class Config:
        extra = "forbid"

class StoredProcResponseModel(RootModel[StoredProcCallModel | StoredProcTerminateModel]):
    pass


class SQLJoinModel(BaseModel):
    left: str
    right: str
    type: Literal["INNER", "LEFT", "RIGHT", "FULL"] = "INNER"

    class Config:
        extra = "forbid"


class SQLPlanSpecModel(BaseModel):
    action: Literal["SELECT", "INSERT", "UPDATE", "DELETE"]
    tables: List[str]
    columns: List[str] = []
    joins: List[SQLJoinModel] = []
    filters: List[str] = []
    aggregations: List[str] = []
    group_by: List[str] = []
    order_by: List[str] = []
    limit: Optional[int] = None
    additional_operations: List[str] = []
    notes: List[str] = []

    class Config:
        extra = "forbid"


class SQLPlanTerminateModel(BaseModel):
    terminate: Literal[True]
    reason: str

    class Config:
        extra = "forbid"


class SQLPlanResponseModel(RootModel[SQLPlanSpecModel | SQLPlanTerminateModel]):
    pass


class SQLGenResponseModel(BaseModel):
    sql: str

    class Config:
        extra = "forbid"


EventType = Literal["action", "message", "trace"]


class SQLAgent:
    """
    SQLAgent: query -> planner -> SQL generator -> execute -> return response.

    The planner receives the user query, schema, and permissions. It outputs
    a complete plan with all info needed for SQL generation.

    The SQL generator receives ONLY the plan. No user query, no schema,
    no permissions. It strictly translates the plan into SQL.

    Permission enforcement is handled by the planner (prompt-based) and
    the authorizer (execution-level).
    """

    def __init__(
        self,
        db_path: str,
        llm_client: LLMClient,
        authorizer: Authorizer,
        prompts: SQLAgentPrompts,
        agent_permissions: Optional[Iterable[str]] = None,
        user_permissions: Optional[Iterable[str]] = None,
        stored_procedures: Optional[Dict[str, Callable]] = None,
        max_steps: int = 5,
        tracer: Optional[Tracer] = None,
        stream: bool = False,
        **kwargs: Any,
    ):
        if prompts is None:
            raise ValueError("prompts must not be None.")
        self.prompts = prompts

        self.llm_client = llm_client
        self.authorizer = authorizer
        self.agent_permissions: List[str] = list(agent_permissions) if agent_permissions else []
        self.user_permissions: List[str] = list(user_permissions) if user_permissions else []
        self.db_path = db_path
        self.max_steps = max(1, max_steps)
        self.stored_procedures = stored_procedures if stored_procedures else {}
        self.tracer: Optional[Tracer] = tracer if tracer else AgentTracer()
        self._trace_state: TraceState = TraceState()
        self.stream = stream
        self._last_streamed_trace_item: Optional[Dict[str, Any]] = None

    def _safe_trace(self, __type__: str, **kwargs: Any) -> None:
        try:
            if self.tracer:
                self.tracer.trace(agent_name="SQLAgent", __type__=__type__, **kwargs)
        except Exception:
            pass

    def _drain_stream_traces(self) -> Generator[Dict[str, Any], None, None]:
        if not self.stream or not self.tracer:
            return
        for item in self.tracer.iter_new_traces_since(self._last_streamed_trace_item):
            self._last_streamed_trace_item = item
            yield {"type": "trace", "content": item}

    # ---------- Public API ----------
    def act(
        self,
        user_query: str,
        exc_method: Literal['full_access', 'in_prompt', 'granular', 'delegated', 'stored_proc'] = "full_access",
    ) -> Generator[Dict[str, Any], None, None]:
        start_total = time.monotonic()
        mark_trace_start(self.tracer, self._trace_state)
        self._last_streamed_trace_item = None

        if exc_method in ['full_access', 'in_prompt', 'granular', 'delegated']:
            yield from self._act_plan_and_execute(user_query=user_query, start_total=start_total)
            return

        if exc_method == "stored_proc":
            yield from self._act_stored_procedures(user_query=user_query, start_total=start_total)
            return

        raise ValueError(f"Unknown execution method: {exc_method}")

    def run(
        self,
        user_query: str,
        exc_method: Literal['full_access', 'in_prompt', 'granular', 'delegated', 'stored_proc'] = "full_access",
    ) -> Dict[str, Any]:
        message_parts: List[str] = []
        final_trace: Dict[str, Any] = {}
        for ev in self.act(user_query=user_query, exc_method=exc_method):
            if ev.get("type") == "message":
                message_parts.append(ev.get("content", "") or "")
            elif ev.get("type") == "trace":
                final_trace = ev.get("content") or {}
        return {"message": "".join(message_parts).strip(), "trace": final_trace}

    def execute_sql_with_permissions(self, sql: str) -> Any:
        access, resources, stmt = analyze_sql(sql, db_uri=self.db_path)
        access = self.authorizer.authorize(access=access, resources=resources, permissions=self.agent_permissions)
        mode = "ro" if access == "read" else "rw"
        return execute_sql(self.db_path, stmt, mode=mode)

    # ---------- Internals ----------
    def _emit_final_trace(self, start_total: float) -> Optional[Dict[str, Any]]:
        payload = {
            "model": getattr(self.llm_client, "model", None),
            "calls": [t for t in traces_since_start(self.tracer, self._trace_state) if isinstance(t, dict)],
            "total_latency_ms": int((time.monotonic() - start_total) * 1000),
        }
        return trace_event_if_changed(self._trace_state, payload)

    def _emit_raw_result_message(self, *, sql: str, results: Any) -> Generator[Dict[str, Any], None, None]:
        payload = {"__sql_agent_result__": True, "sql": sql, "results": results}
        raw = to_json_str(payload)
        for evm in stream_message_events([raw], min_chars=20):
            yield evm

    def _act_plan_and_execute(self, user_query: str, *, start_total: float) -> Generator[Dict[str, Any], None, None]:
        previous_sql: Optional[str] = None
        previous_error: Optional[str] = None

        for attempt in range(self.max_steps):
            # Step 1: Plan
            try:
                if self.stream:
                    yield {"type": "action", "content": f"Planning query (attempt {attempt + 1})"}

                if self.stream:
                    plan = yield from self._plan(
                        user_query=user_query,
                        previous_sql=previous_sql,
                        previous_error=previous_error,
                    )
                else:
                    plan = self._plan(
                        user_query=user_query,
                        previous_sql=previous_sql,
                        previous_error=previous_error,
                    )

                if self.stream:
                    yield from self._drain_stream_traces()
            except Exception as e:
                previous_error = str(e)
                self._safe_trace(__type__="error", operation=f"Plan attempt {attempt + 1}", error=str(e))
                if self.stream:
                    yield from self._drain_stream_traces()
                continue

            # Handle termination from planner (permission denial, injection, etc.)
            # This is outside try/except so a terminate is NEVER retried.
            if isinstance(plan, dict) and plan.get("terminate"):
                reason = str(plan.get("reason", "I can't help with that request."))
                sql = f"SELECT '{reason}'"
                results = {"columns": [reason], "rows": [{reason: reason}], "row_count": 1}
                yield from self._emit_raw_result_message(sql=sql, results=results)
                ev = self._emit_final_trace(start_total)
                if ev:
                    yield ev
                return

            # Step 2+3: Generate SQL and Execute
            try:
                if self.stream:
                    yield {"type": "action", "content": f"Generating SQL (attempt {attempt + 1})"}

                # Step 2: Generate SQL from plan only
                plan_text = self._plan_to_text(plan)
                if self.stream:
                    sql = yield from self._generate_sql(plan_text=plan_text)
                else:
                    sql = self._generate_sql(plan_text=plan_text)

                if self.stream:
                    yield from self._drain_stream_traces()
                    yield {"type": "action", "content": f"Executing query (attempt {attempt + 1})"}

                # Step 3: Execute
                results = self.execute_sql_with_permissions(sql)

                if self.stream:
                    yield from self._drain_stream_traces()
                    yield {"type": "action", "content": "Generating response"}

                yield from self._emit_raw_result_message(sql=sql, results=results)

                if self.stream:
                    yield from self._drain_stream_traces()

                ev = self._emit_final_trace(start_total)
                if ev:
                    yield ev
                return

            except PermissionError:
                self._safe_trace(
                    __type__="error",
                    operation="Execute SQL",
                    message=f"SQL Agent does not have permission to execute the query.\n{sql}",
                )
                yield from self._emit_raw_result_message(
                    sql="",
                    results={
                        "__error__": True,
                        "type": "permission",
                        "message": "I cannot answer that question as the SQL Agent does not have the required permissions",
                    },
                )
                if self.stream:
                    yield from self._drain_stream_traces()
                ev = self._emit_final_trace(start_total)
                if ev:
                    yield ev
                return

            except Exception as e:
                previous_sql = sql if 'sql' in dir() else None
                previous_error = str(e)
                self._safe_trace(
                    __type__="error",
                    operation=f"Attempt {attempt + 1}",
                    error=str(e),
                )
                if self.stream:
                    yield from self._drain_stream_traces()
                # Continue to next attempt

        # All attempts exhausted
        final_msg = f"SQL execution failed after {self.max_steps} attempt(s): {previous_error or 'unknown error'}"
        for evm in stream_message_events([final_msg], min_chars=20):
            yield evm
        ev = self._emit_final_trace(start_total)
        if ev:
            yield ev

    def _act_stored_procedures(self, user_query: str, *, start_total: float) -> Generator[Dict[str, Any], None, None]:
        try:
            signatures = get_stored_procedure_signatures(self.stored_procedures)
            if signatures is None:
                raise ValueError("No stored procedures are registered.")

            if self.stream:
                yield {"type": "action", "content": "Selecting stored procedure"}

            sys_prompt = self.prompts.stored_proc_system_prompt_template
            user_prompt = self.prompts.stored_proc_user_prompt_template.format(
                user_query=str(user_query),
                signatures=str(signatures),
            )

            if self.stream:
                ev = emit_trace_if_new(self.tracer, self._trace_state)
                if ev:
                    yield ev

            tool_call = self.llm_client.json_response(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                schema=StoredProcResponseModel,
                tracer=self.tracer,
                temperature=0,
                top_k=20,
            )

            if tool_call.get("terminate"):
                terminate_info = tool_call.get(
                    "reason", "The AI terminated the operation and provided no reason for termination."
                )
                for evm in stream_message_events([str(terminate_info)], min_chars=20):
                    yield evm
                ev = self._emit_final_trace(start_total)
                if ev:
                    yield ev
                return

            if not isinstance(tool_call, dict):
                raise ValueError("LLM did not return a stored procedure call.")

            func_name = tool_call.get("function_name")
            kwargs = tool_call.get("kwargs", {})
            kwargs["permissions"] = self.agent_permissions

            self._safe_trace(
                __type__="agent",
                operation="Stored Procedure Output",
                func_name=func_name,
                args=[],
                kwargs=kwargs,
            )
            if self.stream:
                ev = emit_trace_if_new(self.tracer, self._trace_state)
                if ev:
                    yield ev

            if not isinstance(func_name, str) or not func_name:
                raise ValueError("Invalid stored procedure call: missing function_name.")
            if not isinstance(kwargs, dict):
                raise ValueError("Invalid stored procedure call: kwargs must be a dict.")

            func = self.stored_procedures.get(func_name)
            if func is None:
                raise ValueError(f"Stored procedure {func_name} not found.")

            # Pass permissions as first positional arg, everything else as kwargs
            coerced_args, coerced_kwargs = coerce_args_to_func(func=func, args=[], kwargs=kwargs)

            if self.stream:
                yield {"type": "action", "content": "Executing stored procedure"}

            try:
                raw_result = func(*coerced_args, **coerced_kwargs)
            except PermissionError as pe:
                self._safe_trace(
                    __type__="error",
                    operation="Execute Stored Procedure",
                    message=f"SQL Agent does not have permission to execute the stored procedure.\n{func_name}",
                )
                yield from self._emit_raw_result_message(
                    sql="",
                    results={
                        "__error__": True,
                        "type": "permission",
                        "message": "I cannot execute that stored procedure as the SQL Agent does not have the required permissions",
                    },
                )
                ev = self._emit_final_trace(start_total)
                if ev:
                    yield ev
                return
            except Exception as e:
                raise RuntimeError(f"Stored procedure execution failed: {e}") from e

            if self.stream:
                ev = emit_trace_if_new(self.tracer, self._trace_state)
                if ev:
                    yield ev
                yield {"type": "action", "content": "Generating response"}

            normalized_result = normalize_tool_result(raw_result)
            yield from self._emit_raw_result_message(sql="Stored Procedure", results=normalized_result)

            if self.stream:
                ev = emit_trace_if_new(self.tracer, self._trace_state)
                if ev:
                    yield ev

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

    # ---------- Planning ----------
    def _plan(
        self,
        user_query: str,
        previous_sql: Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, Optional[Dict[str, Any]]]:
        # Build previous execution block
        previous_block = "-"
        if previous_sql or previous_error:
            parts = []
            if previous_sql:
                parts.append(f"Last attempted SQL:\n{previous_sql.strip()}")
            if previous_error:
                parts.append(f"Last execution error:\n{previous_error.strip()}")
            previous_block = "\n".join(parts)

        user_prompt = self.prompts.planner_user_prompt_template.format(
            user_permissions_block=", ".join(self.user_permissions) or "-",
            user_query=str(user_query),
            previous_block=previous_block,
        )

        try:
            plan = self.llm_client.json_response(
                prompt=user_prompt,
                system_prompt=self.prompts.planner_system_prompt_template,
                schema=SQLPlanResponseModel,
                tracer=self.tracer,
                temperature=0,
                top_k=20,
            )
            self._safe_trace(__type__="agent", operation="Planner Output", output=plan)
            yield from self._drain_stream_traces()
            return plan if isinstance(plan, dict) else None
        except Exception as e:
            self._safe_trace(__type__="error", operation="Planner Error", error=str(e))
            yield from self._drain_stream_traces()
            return None

    # ---------- SQL Generation (from plan only) ----------
    def _generate_sql(
        self,
        plan_text: str,
    ) -> Generator[Dict[str, Any], None, str]:
        user_prompt = self.prompts.sql_gen_user_prompt_template.format(
            plan_text=plan_text,
        )

        out = self.llm_client.json_response(
            prompt=user_prompt,
            system_prompt=self.prompts.sql_gen_system_prompt_template,
            schema=SQLGenResponseModel,
            tracer=self.tracer,
            temperature=0,
            top_k=20,
        )
        if not isinstance(out, dict):
            raise ValueError("LLM did not return a JSON object for SQL generation.")
        sql = out.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("LLM returned invalid SQL JSON: missing 'sql' string.")
        extracted = extract_single_sql_statement(sql)
        self._safe_trace(__type__="agent", operation="SQL Gen Output", output=out, extracted_sql=extracted)
        yield from self._drain_stream_traces()
        return extracted

    def _plan_to_text(self, plan: Optional[Dict[str, Any]]) -> str:
        """Convert plan dict to a structured text block for the SQL generator."""
        if not plan or not isinstance(plan, dict):
            return "-"

        lines: List[str] = []
        for key in (
            "action",
            "tables",
            "columns",
            "joins",
            "filters",
            "aggregations",
            "group_by",
            "order_by",
            "limit",
            "additional_operations",
            "notes",
        ):
            if key in plan and plan[key] not in (None, "", [], {}):
                try:
                    val = plan[key]
                    val_s = (
                        json.dumps(val, ensure_ascii=False, default=str)
                        if isinstance(val, (dict, list))
                        else str(val)
                    )
                    lines.append(f"{key}: {val_s}")
                except Exception:
                    continue
        return "\n".join(lines) if lines else "-"
