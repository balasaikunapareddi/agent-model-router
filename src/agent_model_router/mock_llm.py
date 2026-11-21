"""Deterministic mock LLM for the ADK agent loop (no key, no network).

Follows the same contract as the system prompt: every task goes through the
`route_task` tool exactly once, spend questions go to `ledger_report`, and
classification questions go to `classify_task`. After the tool returns, it
formats the final reply. This makes the *agent loop itself* testable
offline — the same reason the eligibility-claims repo ships a scripted
model.
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types


def _last_user_text(request: LlmRequest) -> str:
    for content in reversed(request.contents or []):
        if content.role == "user":
            for part in content.parts or []:
                if part.text:
                    return part.text
    return ""


def _last_function_response(request: LlmRequest) -> tuple[str, dict] | None:
    for content in reversed(request.contents or []):
        for part in content.parts or []:
            if part.function_response is not None:
                return part.function_response.name, dict(part.function_response.response or {})
    return None


def _pick_tool(user_text: str) -> str:
    lowered = user_text.lower()
    if any(k in lowered for k in ("spend", "savings", "saved", "ledger", "cost report", "how much have")):
        return "ledger_report"
    if "classif" in lowered:
        return "classify_task"
    return "route_task"


def _format_final(tool_name: str, result: dict) -> str:
    if tool_name == "route_task":
        esc = " (escalated)" if result.get("escalations") else ""
        return (
            f"{result.get('answer', '')}\n"
            f"[{result.get('model_used')} · {result.get('difficulty')} · "
            f"quality {result.get('quality_score')}{esc} · ${result.get('total_cost_usd')}]"
        )
    if tool_name == "ledger_report":
        return (
            f"Routed {result.get('tasks')} tasks ({result.get('attempts')} attempts, "
            f"{result.get('escalations')} escalations). Spend ${result.get('total_cost_usd')} "
            f"vs always-big baseline ${result.get('baseline_cost_usd')} — "
            f"saved {result.get('savings_pct')}%."
        )
    if tool_name == "classify_task":
        return (
            f"That classifies as {result.get('difficulty')} ({result.get('task_type')}, "
            f"score {result.get('score')}). Signals: {', '.join(result.get('signals') or []) or 'none'}."
        )
    return str(result)


class MockLlm(BaseLlm):
    """A BaseLlm that deterministically drives the routing tools."""

    model: str = "mock-router-llm"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"mock-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        tool_result = _last_function_response(llm_request)
        if tool_result is not None:
            tool_name, result = tool_result
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=_format_final(tool_name, result))],
                ),
                turn_complete=True,
            )
            return

        user_text = _last_user_text(llm_request)
        tool_name = _pick_tool(user_text)
        args = {} if tool_name == "ledger_report" else {"task": user_text}
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name=tool_name, args=args)
                    )
                ],
            ),
            turn_complete=True,
        )
