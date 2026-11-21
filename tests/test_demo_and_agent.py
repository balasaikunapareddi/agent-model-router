from agent_model_router.agent import run_agent
from agent_model_router.demo import DEMO_TASKS, run_demo
from agent_model_router.ledger import Ledger


def test_demo_batch_all_pass():
    outcome = run_demo()
    assert len(outcome["results"]) == len(DEMO_TASKS)
    assert all(r.passed for r in outcome["results"])


def test_demo_shows_escalations_and_savings():
    outcome = run_demo()
    assert sum(r.escalations for r in outcome["results"]) >= 2
    report = outcome["report"]
    assert report["savings_usd"] > 0
    assert report["savings_pct"] > 0
    assert report["avg_final_quality"] >= 0.8


def test_demo_uses_all_three_tiers():
    outcome = run_demo()
    tiers = {r.tier_used for r in outcome["results"]}
    assert tiers == {"small", "medium", "large"}


def test_agent_loop_routes_task():
    reply = run_agent("What is 12 + 30?", ledger=Ledger())
    assert "42" in reply
    assert "mock-small" in reply


def test_agent_loop_escalates_and_reports():
    reply = run_agent("What is 250 * 4?", ledger=Ledger())
    assert "1000" in reply
    assert "escalated" in reply


def test_agent_loop_ledger_report():
    ledger = Ledger()
    run_agent("hi", ledger=ledger)
    reply = run_agent("How much have we spent so far?", ledger=ledger)
    assert "Routed 1 tasks" in reply
