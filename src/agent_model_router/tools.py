"""Typed tool definitions exposed to the ADK agent.

Plain functions with type hints and docstrings — ADK turns them into tool
schemas. All decision logic stays in classifier/router/grader/ledger; these
are thin, typed wrappers.
"""

from __future__ import annotations

from .classifier import classify
from .ledger import Ledger
from .router import Router

_default_router: Router | None = None


def set_default_router(router: Router) -> None:
    """Inject the router the tools use (tests and demo set their own)."""
    global _default_router
    _default_router = router


def get_default_router() -> Router:
    global _default_router
    if _default_router is None:
        _default_router = Router(ledger=Ledger("router_ledger.jsonl"))
    return _default_router


def classify_task(task: str) -> dict:
    """Classify a task's difficulty and type without running it.

    Args:
        task: The full task text to classify.

    Returns:
        Dict with difficulty (simple/moderate/complex), task_type, score,
        and the signals that produced the classification.
    """
    return classify(task).to_dict()


def route_task(task: str) -> dict:
    """Route a task to the cheapest model that can do it well and answer it.

    Classifies the task, runs the appropriate model, grades the answer, and
    escalates to a bigger model if the grade fails. Every attempt is
    recorded in the cost ledger.

    Args:
        task: The full task text to answer.

    Returns:
        Dict with the answer, difficulty, task type, model used, quality
        score, escalation count, total estimated cost in USD, and the
        per-attempt breakdown.
    """
    return get_default_router().route(task).to_dict()


def ledger_report() -> dict:
    """Summarize routing spend vs an always-big-model baseline.

    Returns:
        Dict with tasks routed, attempts, escalations, actual spend,
        baseline spend, savings in USD and percent, attempts by model, and
        average final quality score.
    """
    return get_default_router().ledger.report()
