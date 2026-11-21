"""The ADK agent: system prompt + typed tools + agent loop.

Mock mode (default): the agent runs on MockLlm, so the full loop — prompt,
tool call, tool execution, final answer — works with no API key and no
network. Live mode (``live=True`` with GEMINI_API_KEY set) swaps in a real
Gemini model for the agent and the live model pool for routing.
"""

from __future__ import annotations

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from .ledger import Ledger
from .mock_llm import MockLlm
from .models import live_pool
from .prompts import SYSTEM_PROMPT
from .router import Router
from .tools import classify_task, ledger_report, route_task, set_default_router

APP_NAME = "agent_model_router"
LIVE_AGENT_MODEL = os.environ.get("ROUTER_AGENT_MODEL", "gemini-3.5-flash-lite")


def build_agent(live: bool = False, ledger: Ledger | None = None) -> LlmAgent:
    """Build the router agent. Mock by default; live needs GEMINI_API_KEY."""
    if live:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set; live mode unavailable")
        set_default_router(Router(pool=live_pool(), ledger=ledger or Ledger("router_ledger.jsonl")))
        model: str | MockLlm = LIVE_AGENT_MODEL
    else:
        set_default_router(Router(ledger=ledger if ledger is not None else Ledger()))
        model = MockLlm()

    return LlmAgent(
        name="model_router",
        description="Routes each task to the cheapest model that can do it well, with a cost ledger.",
        model=model,
        instruction=SYSTEM_PROMPT,
        tools=[
            FunctionTool(classify_task),
            FunctionTool(route_task),
            FunctionTool(ledger_report),
        ],
    )


async def _run_async(agent: LlmAgent, message: str) -> str:
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = await session_service.create_session(app_name=APP_NAME, user_id="demo")
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final_text = ""
    async for event in runner.run_async(
        user_id="demo", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text
    return final_text


def run_agent(message: str, live: bool = False, ledger: Ledger | None = None) -> str:
    """Run one message through the agent loop and return the final text."""
    agent = build_agent(live=live, ledger=ledger)
    return asyncio.run(_run_async(agent, message))
