"""One-command demo: a batch of mixed tasks through the router.

    python -m agent_model_router.demo

Prints the routing decision for each task (difficulty, model path,
escalations, quality, cost), one end-to-end ADK agent-loop call, and the
ledger report vs an always-big-model baseline. Mock mode: no API key, no
network, fully deterministic.
"""

from __future__ import annotations

from .agent import run_agent
from .ledger import Ledger, format_report
from .router import Router

CONTRACT_TEXT = (
    "Review this vendor contract for risks. Section 4 covers termination: either "
    "party may terminate with 90 days notice, but early termination triggers a fee. "
    "Section 7 sets a liability cap at fees paid in the prior 12 months. Section 9 "
    "covers indemnification for third-party claims arising from the services. "
    "Section 11 requires confidentiality of all shared data for five years after "
    "the term ends. Section 13 describes payment terms of net 45 with late fees. "
    "Section 15 allows automatic renewal unless notice is given 60 days ahead. "
    "Section 18 states governing law as the State of Minnesota."
)

SUMMARY_SOURCE = (
    "Solar panels convert sunlight into electricity for homes and businesses. "
    "Modern solar panels are cheaper and more efficient than panels from a decade ago. "
    "Battery storage lets solar homes use stored electricity at night. "
    "Battery storage also helps the grid during peak demand hours. "
    "The grid benefits when solar and battery systems share excess electricity. "
    "Installers say solar plus battery is now the most popular home setup."
)

DEMO_TASKS = [
    "hi",
    "What is 12 + 30?",
    "What is 250 * 4?",
    "Explain why the sky is blue in one sentence.",
    f"Summarize this: {SUMMARY_SOURCE}",
    "Review the payment terms in this section.",
    "Compare solar and wind for a home setup and recommend one, step by step.",
    CONTRACT_TEXT,
]


def run_demo() -> dict:
    """Route the demo batch; return results + report (used by tests too)."""
    ledger = Ledger()
    router = Router(ledger=ledger)
    results = [router.route(task) for task in DEMO_TASKS]
    return {"results": results, "report": ledger.report()}


def main() -> int:
    print("=" * 72)
    print("MODEL ROUTER DEMO — cheapest model that can actually do the job")
    print("=" * 72)

    outcome = run_demo()
    for r in outcome["results"]:
        path = " → ".join(
            f"{a.response.model} (q={a.grade.score:.2f})" for a in r.attempts
        )
        task_preview = r.task if len(r.task) <= 64 else r.task[:61] + "..."
        print(f"\nTASK: {task_preview}")
        print(
            f"  {r.classification.task_type} · {r.classification.difficulty} · "
            f"signals: {', '.join(r.classification.signals) or 'none'}"
        )
        print(f"  route: {path}")
        print(
            f"  answer: {r.answer[:110]}{'...' if len(r.answer) > 110 else ''}"
        )
        print(
            f"  cost: ${r.total_cost_usd:.4f} · escalations: {r.escalations} · "
            f"passed: {r.passed}"
        )

    print("\n" + "-" * 72)
    print("ADK AGENT LOOP (same router, through the agent):")
    print(run_agent("What is 250 * 4?", ledger=Ledger()))

    print("\n" + "-" * 72)
    print("LEDGER REPORT")
    print(format_report(outcome["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
