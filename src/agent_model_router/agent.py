"""The ADK agent: system prompt + typed tools + agent loop.

Mock mode (default): the agent runs on MockLlm, so the full loop — prompt,
tool call, tool execution, final answer — works with no API key and no
network. Live mode (``live=True`` with GEMINI_API_KEY set) swaps in a real
Gemini model for the agent and the live model pool for routing.

The agent itself is :class:`ModelRouterAgent` below: it owns the model,
the system prompt (from ``prompts.py``), the typed tools, and the runner
loop. The module-level ``build_agent`` / ``run_agent`` helpers are thin
wrappers kept for backward compatibility (demo and tests use them).
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


class ModelRouterAgent:
    """Model-routing agent: cheapest capable model, quality gate, cost ledger.

    Wraps a Google ADK ``LlmAgent`` whose instruction is ``SYSTEM_PROMPT``
    (see ``prompts.py`` — behavior only; all routing decisions live in
    ``classifier`` / ``router`` / ``grader`` code). The agent exposes three
    typed tools — ``classify_task``, ``route_task``, ``ledger_report`` —
    and drives them through an ADK ``Runner`` with an in-memory session.

    Mock mode (default) runs the loop on the deterministic ``MockLlm`` and
    the mock model pool: no API key, no network. Live mode (``live=True``,
    requires ``GEMINI_API_KEY``) swaps in a real Gemini model for the agent
    and the live model pool for routing; everything else is unchanged.

    Attributes:
        live: Whether live (Gemini) mode is enabled.
        ledger: Cost/quality ledger shared with the router.
        router: The ``Router`` the tools delegate to.
        model: Agent model — ``MockLlm`` instance or Gemini model name.
        tools: ADK ``FunctionTool`` wrappers exposed to the agent.
        system_prompt: The instruction text (``SYSTEM_PROMPT``).
        llm_agent: The underlying ADK ``LlmAgent``.
    """

    name = "model_router"
    description = "Routes each task to the cheapest model that can do it well, with a cost ledger."
    system_prompt = SYSTEM_PROMPT

    def __init__(self, live: bool = False, ledger: Ledger | None = None):
        """Build the router agent. Mock by default; live needs GEMINI_API_KEY."""
        self.live = live
        if live:
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is not set; live mode unavailable")
            self.ledger = ledger or Ledger("router_ledger.jsonl")
            self.router = Router(pool=live_pool(), ledger=self.ledger)
            self.model: str | MockLlm = LIVE_AGENT_MODEL
        else:
            self.ledger = ledger if ledger is not None else Ledger()
            self.router = Router(ledger=self.ledger)
            self.model = MockLlm()
        set_default_router(self.router)
        self.tools = [
            FunctionTool(classify_task),
            FunctionTool(route_task),
            FunctionTool(ledger_report),
        ]
        self.llm_agent = LlmAgent(
            name=self.name,
            description=self.description,
            model=self.model,
            instruction=self.system_prompt,
            tools=self.tools,
        )

    async def run_async(self, message: str) -> str:
        """Run one message through the agent loop and return the final text."""
        return await _run_async(self.llm_agent, message)

    def run(self, message: str) -> str:
        """Synchronous wrapper around :meth:`run_async`."""
        return asyncio.run(self.run_async(message))


def build_agent(live: bool = False, ledger: Ledger | None = None) -> LlmAgent:
    """Build the router agent. Mock by default; live needs GEMINI_API_KEY."""
    return ModelRouterAgent(live=live, ledger=ledger).llm_agent


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
    return ModelRouterAgent(live=live, ledger=ledger).run(message)
