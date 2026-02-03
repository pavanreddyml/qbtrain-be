# qbtrain/agents/sql_agent.py
from __future__ import annotations

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
    shared_user_prompt_template: str = Field(..., min_length=1)

    # SQL generation
    sql_gen_system_prompt_template: str = Field(..., min_length=1)

    # Stored procedure selection
    stored_proc_system_prompt_template: str = Field(..., min_length=1)
    stored_proc_user_prompt_template: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


def _build_shared_user_prompt(
    *,
    template: str,
    user_query: str,
    user_permissions: Optional[Iterable[str]] = None,
    plan_hints: Optional[str] = None,
    previous_sql: Optional[str] = None,
    previous_error: Optional[str] = None,
    include_permissions: bool = False,
    include_hints: bool = False,
    include_previous: bool = False,
    return_line: str = "Return the JSON object now:",
) -> str:
    user_permissions = user_permissions or []
    user_permissions_block = "-"
    if include_permissions:
        user_permissions_block = ", ".join(list(user_permissions or [])) or "-"

    hints_block = "-"
    if include_hints:
        hints_block = str(plan_hints or "-")

    previous_block = "-"
    if include_previous:
        prev_parts: List[str] = []
        if (previous_sql or "").strip():
            prev_parts.append("Last attempted SQL:\n" + str(previous_sql).strip() + "\n")
        if (previous_error or "").strip():
            prev_parts.append("Last execution error:\n" + str(previous_error).strip() + "\n")
        previous_block = "\n".join([p.strip() for p in prev_parts if p.strip()]) or "-"

    return template.format(
        user_permissions_block=user_permissions_block or "-",
        hints_block=hints_block or "-",
        user_query=str(user_query),
        previous_block=previous_block or "-",
        return_line=return_line,
    )



class StoredProcCallModel(BaseModel):
    function_name: str
    signature: Optional[str] = None
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        extra = "forbid"


class StoredProcTerminateModel(BaseModel):
    __terminate__: bool = True
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
    required_permissions: List[str] = []
    user_permissions: List[str] = []

    class Config:
        extra = "forbid"


class SQLPlanTerminateModel(BaseModel):
    __terminate__: bool = True
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
    SQLAgent executes generated SQL against a database URI, and enforces
    externally-provided permissions via Authorizer.

    Response generation/formatting is intentionally NOT handled here.
    SQLAgent emits a single "message" event containing a JSON payload:
      {"__sql_agent_result__": true, "sql": "...", "results": <any>}
    Consumers should turn this into a customer-facing message elsewhere.
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
        use_cot: bool = True,
        cot_max_steps: int = 5,
        cot_flow_through_permissions: bool = True,
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
        self.use_cot = use_cot
        self.cot_max_steps = cot_max_steps
        self.stored_procedures = stored_procedures if stored_procedures else {}
        self.tracer: Optional[Tracer] = tracer if tracer else AgentTracer()
        self._trace_state: TraceState = TraceState()
        self.stream = stream
        self._last_streamed_trace_item: Optional[Dict[str, Any]] = None

        self.flow_through_permissions: bool = kwargs.get("flow_through_permissions", True)
        self.cot_flow_through_permissions: bool = cot_flow_through_permissions

    def _safe_trace(self, __type__: str, **kwargs: Any) -> None:
        """
        Best-effort trace emission. Never fail the agent because tracing failed.
        """
        try:
            if self.tracer:
                self.tracer.trace(agent_name="SQLAgent", __type__=__type__, **kwargs)
        except Exception as e:
            pass

    def _drain_stream_traces(self) -> Generator[Dict[str, Any], None, None]:
        """
        Emit NEW trace steps one-by-one as stream events.
        Only active when self.stream is True.
        """
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
            if self.use_cot:
                yield from self._act_plan_and_execute(user_query=user_query, start_total=start_total)
                return
            yield from self._act_single_and_execute(user_query=user_query, start_total=start_total)
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
        """
        Returns:
          {"message": "<raw sql+results json payload as string>", "trace": <dict>}
        Consumers should format "message" externally.
        """
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

    # ---------- Non-streaming/streaming unified internals ----------
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

    def _act_single_and_execute(self, user_query: str, *, start_total: float) -> Generator[Dict[str, Any], None, None]:
        try:
            if self.stream:
                yield {"type": "action", "content": "Generating query"}

            # When streaming, _generate_sql will yield trace steps; when not, it returns as usual.
            sql, _ = yield from self.get_query(user_query=user_query, method="single") if self.stream else self.get_query(user_query=user_query, method="single")

            if self.stream:
                yield from self._drain_stream_traces()
                yield {"type": "action", "content": "Executing query"}
    
            results = self.execute_sql_with_permissions(sql)

            if self.stream:
                yield from self._drain_stream_traces()
                yield {"type": "action", "content": "Generating response"}

            # Emit raw payload (consumers format externally)
            yield from self._emit_raw_result_message(sql=sql, results=results)

            if self.stream:
                yield from self._drain_stream_traces()

            ev = self._emit_final_trace(start_total)
            if ev:
                yield ev

        except PermissionError:
            self._safe_trace(
                __type__="error",
                operation="Execute SQL",
                message=f"SQL Agent does not have permission to execute the query.\n{sql}",
            )

            # Keep message payload shape consistent for the consumer.
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
            msg = f"Agent error: {e}"
            for evm in stream_message_events([msg], min_chars=20):
                yield evm
            ev = self._emit_final_trace(start_total)
            if ev:
                yield ev

    def _act_plan_and_execute(self, user_query: str, *, start_total: float) -> Generator[Dict[str, Any], None, None]:
        previous_sql: Optional[str] = None
        previous_error: Optional[str] = None
        last_exc: Optional[Exception] = None

        attempts = max(1, self.cot_max_steps)
        for attempt in range(attempts):
            try:
                if self.stream:
                    yield {"type": "action", "content": f"Generating query (attempt {attempt + 1})"}

                if self.stream:
                    sql, _terminate = yield from self.get_query(
                        user_query=user_query,
                        method="cot",
                        previous_sql=previous_sql,
                        previous_sql_error=previous_error,
                    )
                else:
                    sql, _terminate = self.get_query(
                        user_query=user_query,
                        method="cot",
                        previous_sql=previous_sql,
                        previous_sql_error=previous_error,
                    )

                if self.stream:
                    yield from self._drain_stream_traces()
                    yield {"type": "action", "content": f"Executing query (attempt {attempt + 1})"}

                results = self.execute_sql_with_permissions(sql)

                if self.stream:
                    yield from self._drain_stream_traces()
                    yield {"type": "action", "content": f"Generating response (attempt {attempt + 1})"}

                # Emit raw payload (consumers format externally)
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

                # Keep message payload shape consistent for the consumer.
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
                msg = f"Agent error: {e}"
                for evm in stream_message_events([msg], min_chars=20):
                    yield evm
                ev = self._emit_final_trace(start_total)
                if ev:
                    yield ev

        final_msg = f"SQL execution failed after {attempts} attempt(s): {str(last_exc) if last_exc else 'unknown error'}"
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

            if tool_call.get("__terminate__"):
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
            args = tool_call.get("args", [])
            kwargs = tool_call.get("kwargs", {})

            self._safe_trace(
                __type__="agent",
                operation="Stored Procedure Output",
                func_name=func_name,
                args=args,
                kwargs=kwargs,
            )
            if self.stream:
                ev = emit_trace_if_new(self.tracer, self._trace_state)
                if ev:
                    yield ev


            if not isinstance(func_name, str) or not func_name:
                raise ValueError("Invalid stored procedure call: missing function_name.")
            if not isinstance(args, list):
                raise ValueError("Invalid stored procedure call: args must be a list.")
            if not isinstance(kwargs, dict):
                raise ValueError("Invalid stored procedure call: kwargs must be a dict.")

            func = self.stored_procedures.get(func_name)
            if func is None:
                raise ValueError(f"Stored procedure {func_name} not found.")

            coerced_args, coerced_kwargs = coerce_args_to_func(func=func, args=args, kwargs=kwargs)
            coerced_kwargs["permissions"] = self.agent_permissions

            if self.stream:
                yield {"type": "action", "content": "Executing stored procedure"}

            try:
                raw_result = func(*coerced_args, **coerced_kwargs)
            except Exception as e:
                raise RuntimeError(f"Stored procedure execution failed: {e}") from e

            if self.stream:
                ev = emit_trace_if_new(self.tracer, self._trace_state)
                if ev:
                    yield ev
                yield {"type": "action", "content": "Generating response"}

            normalized_result = normalize_tool_result(raw_result)

            # Emit raw payload (consumers format externally)
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

    # ---------- Planning / Generation ----------
    def get_query(
        self,
        user_query: str,
        method: Literal["single", "cot"],
        previous_sql: Optional[str] = None,
        previous_sql_error: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, Tuple[str, bool]]:
        if method == "single":
            if self.stream:
                sql = yield from self._generate_sql(user_query=user_query)  # yields trace steps
            else:
                sql = self._generate_sql(user_query=user_query)
            return (sql, True)

        if method == "cot":
            if self.stream:
                plan = yield from self._plan(user_query=user_query, previous_sql=previous_sql, previous_error=previous_sql_error)
            else:
                plan = self._plan(user_query=user_query, previous_sql=previous_sql, previous_error=previous_sql_error)
            if isinstance(plan, dict) and plan.get("__terminate__"):
                terminate_info = plan.get("reason", "I can’t help with that request.")
                terminate_info = str(terminate_info).replace('"', '""')
                return f'SELECT "{terminate_info}"', True
            
            try:
                plan_permissions = plan.get("user_permissions", []) if isinstance(plan, dict) else []
                plan_permissions = plan_permissions if len(plan_permissions) > 0 else None
            except Exception:
                plan_permissions = None
                
            if self.stream:
                sql = yield from self._generate_sql(
                    user_query=user_query,
                    plan=plan,
                    previous_sql=previous_sql,
                    previous_error=previous_sql_error,
                    user_permissions=plan_permissions if self.cot_flow_through_permissions else self.user_permissions,
                )
            else:
                sql = self._generate_sql(
                    user_query=user_query,
                    plan=plan,
                    previous_sql=previous_sql,
                    previous_error=previous_sql_error,
                    user_permissions=plan_permissions if self.cot_flow_through_permissions else self.user_permissions,
                )
            return sql, False

        raise ValueError(f"Unknown method: {method}")

    def _plan(
        self,
        user_query: str,
        previous_sql: Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, Optional[Dict[str, Any]]]:
        user_prompt = _build_shared_user_prompt(
            template=self.prompts.shared_user_prompt_template,
            user_query=user_query,
            user_permissions=self.user_permissions,
            include_permissions=True,
            include_previous=True,
            previous_sql=previous_sql,
            previous_error=previous_error,
            return_line="Return the strict JSON plan:",
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

    def _generate_sql(
        self,
        user_query: str,
        plan: Optional[Dict[str, Any]] = None,
        previous_sql: Optional[str] = None,
        previous_error: Optional[str] = None,
        user_permissions: Optional[Iterable[str]] = None,
    ) -> Generator[Dict[str, Any], None, str]:
        plan_hints = self._plan_to_hints(plan, previous_sql=previous_sql, previous_error=previous_error)

        has_hints = bool(plan_hints and str(plan_hints).strip() and str(plan_hints).strip() != "-")
        user_prompt = _build_shared_user_prompt(
            template=self.prompts.shared_user_prompt_template,
            user_query=user_query,
            user_permissions=user_permissions,
            plan_hints=str(plan_hints),
            include_hints=has_hints,
            include_permissions=True,   # was: not has_hints
            include_previous=False,
            return_line="Return the JSON object now:",
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

    def _plan_to_hints(
        self,
        plan: Optional[Dict[str, Any]] = None,
        previous_sql: Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> str:
        hints: List[str] = []
        if plan and isinstance(plan, dict):
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
                "notes",
                "required_permissions",
                "user_permissions"
            ):
                if key in plan and plan[key] not in (None, "", [], {}):
                    try:
                        val = plan[key]
                        val_s = (
                            json.dumps(val, ensure_ascii=False, default=str)
                            if isinstance(val, (dict, list))
                            else str(val)
                        )
                        hints.append(f"{key}: {val_s}")
                    except Exception:
                        continue
        if previous_sql:
            hints.append(f"previous_sql: {previous_sql}")
        if previous_error:
            hints.append(f"previous_error: {previous_error}")
        return "\n".join(hints) if hints else "-"
