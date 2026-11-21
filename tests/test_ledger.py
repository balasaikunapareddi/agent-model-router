from agent_model_router.ledger import Ledger, LedgerEntry
from agent_model_router.router import Router


def _entry(**overrides):
    base = dict(
        task_id="abc123",
        task_type="qa",
        difficulty="simple",
        model="mock-small",
        tier="small",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100,
        cost_usd=0.01,
        quality_score=1.0,
        passed=True,
        baseline_cost_usd=0.10,
    )
    base.update(overrides)
    return LedgerEntry(**base)


def test_empty_report():
    report = Ledger().report()
    assert report["tasks"] == 0
    assert report["total_cost_usd"] == 0.0


def test_report_math():
    ledger = Ledger()
    ledger.record(_entry(cost_usd=0.01, baseline_cost_usd=0.10))
    ledger.record(_entry(task_id="def456", cost_usd=0.02, baseline_cost_usd=0.10))
    report = ledger.report()
    assert report["tasks"] == 2
    assert report["total_cost_usd"] == 0.03
    assert report["baseline_cost_usd"] == 0.20
    assert report["savings_usd"] == 0.17
    assert report["savings_pct"] == 85.0


def test_failed_attempts_count_toward_spend_not_baseline():
    ledger = Ledger()
    ledger.record(_entry(passed=False, quality_score=0.0, cost_usd=0.01, baseline_cost_usd=0.0))
    ledger.record(
        _entry(model="mock-medium", tier="medium", escalated=True,
               escalated_from="small", cost_usd=0.05, baseline_cost_usd=0.10)
    )
    report = ledger.report()
    assert report["tasks"] == 1
    assert report["attempts"] == 2
    assert report["escalations"] == 1
    assert report["total_cost_usd"] == 0.06
    assert report["baseline_cost_usd"] == 0.10


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record(_entry())
    ledger.record(_entry(task_id="zzz999"))
    reloaded = Ledger(path)
    assert len(reloaded.entries()) == 2
    assert reloaded.report()["tasks"] == 2


def test_router_savings_positive_on_mixed_batch():
    ledger = Ledger()
    router = Router(ledger=ledger)
    for task in ["hi", "What is 12 + 30?", "thanks!", "What is 250 * 4?"]:
        router.route(task)
    report = ledger.report()
    assert report["savings_usd"] > 0
    assert report["savings_pct"] > 50
