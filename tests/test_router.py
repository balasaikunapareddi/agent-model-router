from agent_model_router.ledger import Ledger
from agent_model_router.router import Router


def _router():
    return Router(ledger=Ledger())


def test_greeting_uses_small_no_escalation():
    r = _router().route("hi")
    assert r.tier_used == "small"
    assert r.escalations == 0
    assert r.passed


def test_easy_math_uses_small():
    r = _router().route("What is 12 + 30?")
    assert r.tier_used == "small"
    assert "42" in r.answer


def test_small_math_slip_escalates_to_medium():
    r = _router().route("What is 250 * 4?")
    assert r.escalations == 1
    assert r.tier_used == "medium"
    assert "1000" in r.answer
    assert r.passed
    # Both attempts recorded, first one failed.
    assert len(r.attempts) == 2
    assert not r.attempts[0].grade.passed
    assert r.attempts[1].grade.passed


def test_shallow_analysis_escalates():
    r = _router().route("Review the payment terms in this section.")
    assert r.escalations >= 1
    assert "payment terms checked" in r.answer


def test_complex_goes_straight_to_large():
    task = (
        "Review this contract. The termination clause, liability cap, and "
        "indemnification terms all need checking against our compliance rules."
    )
    r = _router().route(task)
    assert r.tier_used == "large"
    assert r.escalations == 0
    assert r.passed


def test_every_attempt_lands_in_ledger():
    ledger = Ledger()
    router = Router(ledger=ledger)
    router.route("What is 250 * 4?")  # fails small, passes medium
    entries = ledger.entries()
    assert len(entries) == 2
    assert entries[0].model == "mock-small" and not entries[0].passed
    assert entries[1].model == "mock-medium" and entries[1].passed
    assert entries[1].escalated and entries[1].escalated_from == "small"


def test_result_dict_shape():
    d = _router().route("hi").to_dict()
    for key in ("answer", "difficulty", "task_type", "model_used",
                "quality_score", "passed", "escalations", "total_cost_usd", "attempts"):
        assert key in d
