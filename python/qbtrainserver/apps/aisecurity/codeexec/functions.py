"""
Business logic for code execution app

Uses qbtrain LLM client infrastructure for provider-agnostic LLM integration.
"""
import json
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

from qbtrain.ai.llm import LLMClientRegistry
from qbtrain.tracers.agent_tracer import AgentTracer

from . import prompts


# ======================== Exceptions ========================

class CodeExecError(Exception):
    """Base exception for code execution errors"""
    pass


class RouteDecisionError(CodeExecError):
    """Error during route decision"""
    pass


class CodeGenerationError(CodeExecError):
    """Error during code generation"""
    pass


class CodeExecutionError(CodeExecError):
    """Error during code execution"""
    pass


class OutputEvaluationError(CodeExecError):
    """Error during output evaluation"""
    pass


# ======================== Helpers ========================

def make_script_dir() -> Path:
    """Create and return the scripts directory"""
    script_dir = Path(__file__).parent / "tmp" / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    return script_dir


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation (words / 0.75 ≈ tokens)"""
    words = len(text.split())
    return max(1, int(words / 0.75))


def _build_client(client_config: Dict[str, Any]):
    """
    Build LLM client from configuration.

    client_config should have:
    {
        "type": "openai" | "ollama" | "azure_foundry" | etc.,
        "model": "model_name",
        "temperature": 0.7,
        "max_tokens": 2048,
        ... other provider-specific params
    }
    """
    client_type = client_config.get('type', 'openai')

    # Get client class from registry
    try:
        client_class = LLMClientRegistry.get(client_type)
    except KeyError:
        raise CodeExecError(f"Unknown LLM client type: {client_type}")

    # Extract init parameters (provider-specific)
    init_params = {k: v for k, v in client_config.items()
                   if k not in ['type', 'temperature', 'max_tokens', 'top_p', 'top_k',
                                'frequency_penalty', 'presence_penalty']}

    # Build client
    try:
        client = client_class(**init_params)
    except Exception as e:
        raise CodeExecError(f"Failed to initialize LLM client: {e}")

    return client


def _build_call_kwargs(client_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build per-call kwargs from client config.
    These are the dynamic parameters passed to each client.response() call.
    """
    kwargs = {}

    # Add per-call parameters
    for key in ['temperature', 'max_output_tokens', 'top_p', 'top_k',
                'frequency_penalty', 'presence_penalty']:
        if key in client_config and client_config[key] is not None:
            # Map max_tokens -> max_output_tokens for consistency
            if key == 'max_tokens':
                kwargs['max_output_tokens'] = client_config[key]
            else:
                kwargs[key] = client_config[key]

    return kwargs


def _call_llm(prompt: str, system_prompt: Optional[str],
              conversation_history: Optional[list],
              client_config: Dict[str, Any],
              tracer: Optional[AgentTracer] = None) -> str:
    """
    Call LLM using qbtrain client infrastructure.
    """
    try:
        client = _build_client(client_config)
        call_kwargs = _build_call_kwargs(client_config)

        if system_prompt:
            call_kwargs['system_prompt'] = system_prompt

        if conversation_history:
            # Trim to last N messages using client's utility
            call_kwargs['conversation_history'] = client.trim_conversation_history(
                conversation_history
            )

        if tracer:
            call_kwargs['tracer'] = tracer

        response = client.response(prompt=prompt, **call_kwargs)
        return response

    except Exception as e:
        raise CodeExecError(f"LLM call failed: {e}")


# ======================== Core Functions ========================

def decide_route(
    question: str,
    has_csv: bool,
    has_document: bool,
    conversation_history: list,
    client_config: Dict[str, Any],
    tracer: Optional[AgentTracer] = None
) -> Tuple[str, str]:
    """
    Decide which route to take for answering the question.
    Returns: (route, reasoning)
    """
    history_summary = ""
    if conversation_history:
        history_summary = "\n".join([f"- {m.get('type', 'unknown')}: {m.get('content', '')[:100]}"
                                    for m in conversation_history[-3:]])

    prompt = prompts.ROUTE_DECISION_PROMPT.format(
        question=question,
        has_csv=has_csv,
        has_document=has_document,
        history_summary=history_summary or "(empty)"
    )

    try:
        response = _call_llm(
            prompt=prompt,
            system_prompt=None,
            conversation_history=None,
            client_config=client_config,
            tracer=tracer
        )

        data = json.loads(response)
        route = data.get('route', 'python')
        reasoning = data.get('reasoning', '')

        # Validate route
        valid_routes = {'direct', 'request', 'csv', 'document', 'python'}
        if route not in valid_routes:
            route = 'python'

        return route, reasoning
    except json.JSONDecodeError:
        # Fallback: default to python
        return 'python', 'Failed to parse route decision'
    except Exception as e:
        raise RouteDecisionError(f"Failed to decide route: {e}")


def generate_code(
    question: str,
    route: str,
    context: str = "",
    previous_output: str = "",
    iteration: int = 1,
    script_dir: Optional[Path] = None,
    client_config: Optional[Dict[str, Any]] = None,
    tracer: Optional[AgentTracer] = None
) -> Tuple[str, Path]:
    """
    Generate Python code for the given route.
    Returns: (code_string, path_to_script)
    """
    if script_dir is None:
        script_dir = make_script_dir()

    if client_config is None:
        client_config = {}

    prompt = prompts.CODE_GEN_PROMPT.format(
        route=route,
        question=question,
        context=context[:1000] if context else "(no context)",
        previous_output=previous_output[:500] if previous_output else "(no previous output)",
        iteration=iteration
    )

    try:
        code = _call_llm(
            prompt=prompt,
            system_prompt=None,
            conversation_history=None,
            client_config=client_config,
            tracer=tracer
        )

        # Clean up markdown code blocks if present
        code = re.sub(r'^```python\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        code = code.strip()

        # Save code to file
        timestamp = int(time.time() * 1000)
        script_path = script_dir / f"query_{timestamp}_iter{iteration}.py"

        with open(script_path, 'w') as f:
            f.write(code)

        return code, script_path
    except Exception as e:
        raise CodeGenerationError(f"Failed to generate code: {e}")


def execute_code(
    script_path: Path,
    run_as_admin: bool = False
) -> Tuple[str, str, int]:
    """
    Execute a Python script and capture output.
    Returns: (stdout, stderr, returncode)
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent)
        )

        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        raise CodeExecutionError("Code execution timed out (30s limit)")
    except Exception as e:
        raise CodeExecutionError(f"Failed to execute code: {e}")


def parse_script_output(stdout: str) -> Optional[Dict]:
    """
    Parse JSON output from script.
    Expected format: {"values": ..., "files": [...]}
    """
    try:
        # Try to find JSON in stdout
        lines = stdout.strip().split('\n')
        for line in reversed(lines):  # Try last line first (most likely)
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    if 'values' in data:
                        return data
                except json.JSONDecodeError:
                    continue
        return None
    except Exception:
        return None


def is_output_sufficient(
    question: str,
    output: str,
    client_config: Optional[Dict[str, Any]] = None,
    tracer: Optional[AgentTracer] = None
) -> Tuple[bool, str]:
    """
    Evaluate if the output is sufficient to answer the question.
    Returns: (is_sufficient, reasoning)
    """
    if client_config is None:
        client_config = {}

    # If output is empty, it's definitely not sufficient
    if not output or not output.strip():
        return False, "Output is empty"

    prompt = prompts.SUFFICIENCY_PROMPT.format(
        question=question,
        output=output[:1000],  # Limit output length for LLM evaluation
        iteration=1
    )

    try:
        response = _call_llm(
            prompt=prompt,
            system_prompt=None,
            conversation_history=None,
            client_config=client_config,
            tracer=tracer
        )

        data = json.loads(response)
        sufficient = data.get('sufficient', False)
        reason = data.get('reason', '')
        return sufficient, reason
    except (json.JSONDecodeError, Exception):
        # If evaluation fails, assume output is sufficient
        return True, "Evaluation completed"


def pick_next_route(
    question: str,
    current_route: str,
    previous_output: str,
    has_csv: bool,
    has_document: bool,
    tried_routes: list,
    client_config: Optional[Dict[str, Any]] = None,
    tracer: Optional[AgentTracer] = None
) -> str:
    """
    Pick the next best route to try for the next iteration.
    Returns: route name
    """
    if client_config is None:
        client_config = {}

    # Simple fallback logic if LLM fails
    fallback_order = {
        'direct': ['python', 'request'],
        'request': ['python', 'csv'] if has_csv else ['python', 'document'],
        'csv': ['python', 'document'] if has_document else ['python', 'request'],
        'document': ['python', 'csv'] if has_csv else ['python', 'request'],
        'python': ['request', 'csv'] if has_csv else ['request', 'document'],
    }

    # Get next routes from fallback order
    candidates = fallback_order.get(current_route, ['python'])
    next_route = None
    for candidate in candidates:
        if candidate not in tried_routes:
            next_route = candidate
            break

    # If all candidates tried, default to remaining routes
    if not next_route:
        all_routes = ['direct', 'request', 'csv', 'document', 'python']
        for route in all_routes:
            if route not in tried_routes:
                next_route = route
                break

    return next_route or 'python'


def generate_direct_answer(
    question: str,
    conversation_history: list,
    client_config: Optional[Dict[str, Any]] = None,
    tracer: Optional[AgentTracer] = None
) -> str:
    """
    Generate a direct answer without code execution.
    Returns: answer text
    """
    if client_config is None:
        client_config = {}

    prompt = prompts.DIRECT_ANSWER_PROMPT.format(
        question=question,
        history="(empty)"
    )

    try:
        answer = _call_llm(
            prompt=prompt,
            system_prompt=None,
            conversation_history=conversation_history,
            client_config=client_config,
            tracer=tracer
        )
        return answer
    except Exception as e:
        raise CodeExecError(f"Failed to generate direct answer: {e}")


def generate_final_answer(
    question: str,
    output: str,
    route: str,
    client_config: Optional[Dict[str, Any]] = None,
    tracer: Optional[AgentTracer] = None
) -> str:
    """
    Synthesize code output into a final answer.
    Returns: final answer text
    """
    if client_config is None:
        client_config = {}

    prompt = prompts.FINAL_ANSWER_PROMPT.format(
        question=question,
        output=output[:2000],  # Limit output for LLM
        route=route
    )

    try:
        answer = _call_llm(
            prompt=prompt,
            system_prompt=None,
            conversation_history=None,
            client_config=client_config,
            tracer=tracer
        )
        return answer
    except Exception as e:
        # Fallback: return output as-is
        return f"Output:\n{output}"


def validate_inputs(
    question: str,
    file_content: Optional[str] = None,
    file_type: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate user inputs.
    Returns: (is_valid, error_message)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty"

    if len(question) > 10000:
        return False, "Question too long (max 10000 chars)"

    if file_content and len(file_content) > 1000000:
        return False, "File content too large (max 1MB)"

    return True, ""
