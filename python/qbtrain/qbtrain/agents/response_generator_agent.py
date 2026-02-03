# qbtrain/agents/response_generator_agent.py
from __future__ import annotations

from typing import Any, Generator, Optional

from pydantic import BaseModel, Field

from ..ai.llm import LLMClient
from ..utils.jsonutils import to_json_str


class ResponseGeneratorPrompts(BaseModel):
    system_prompt_template: str = Field(..., min_length=1)
    user_prompt_template: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class ResponseGeneratorAgent:
    """
    Turns (user_query, executed_sql, results) into a customer-facing message.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompts: ResponseGeneratorPrompts,
    ):
        if prompts is None:
            raise ValueError("prompts must not be None.")
        self.llm_client = llm_client
        self.prompts = prompts

    def _render_system_prompt(self) -> str:
        return (
            self.prompts.system_prompt_template
        )

    def _render_user_prompt(self, *, user_query: str, sql: str, results: Any) -> str:
        results_str = to_json_str(results)
        return self.prompts.user_prompt_template.format(user_query=user_query, sql=sql, results=results_str)

    def generate(self, *, user_query: str, sql: str, results: Any, tracer=None) -> str:
        sys_prompt = self._render_system_prompt()
        user_prompt = self._render_user_prompt(user_query=user_query, sql=sql, results=results)
        out = self.llm_client.response(prompt=user_prompt, system_prompt=sys_prompt, tracer=tracer, temperature=0, top_k=1)
        if not isinstance(out, str) or not out.strip():
            raise ValueError("ResponseGeneratorAgent failed to produce a response.")
        return out.strip()

    def generate_stream(self, *, user_query: str, sql: str, results: Any, tracer=None) -> Generator[str, None, None]:
        sys_prompt = self._render_system_prompt()
        user_prompt = self._render_user_prompt(user_query=user_query, sql=sql, results=results)
        try:
            for chunk in self.llm_client.response_stream(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                tracer=tracer,
                temperature=0,
                top_k=1,
            ):
                if chunk:
                    yield chunk
        except Exception:
            out = self.llm_client.response(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                tracer=tracer,
                temperature=0,
                top_k=1,
            )
            if isinstance(out, str) and out:
                yield out
