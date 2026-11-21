from agent_model_router.classifier import classify
from agent_model_router.grader import grade


def _grade(task, answer):
    return grade(task, classify(task), answer)


def test_math_correct_passes():
    g = _grade("What is 12 + 30?", "The answer is 42.")
    assert g.passed and g.score == 1.0


def test_math_wrong_fails():
    g = _grade("What is 12 + 30?", "The answer is 41.")
    assert not g.passed and g.score == 0.0


def test_greeting_passes():
    g = _grade("hi", "Hello! How can I help you today?")
    assert g.passed


def test_empty_answer_fails():
    g = _grade("hi", "")
    assert not g.passed


def test_summary_coverage_scoring():
    source = (
        "Solar panels make electricity. Solar panels are cheaper now. "
        "Battery storage helps at night. Battery storage helps the grid."
    )
    task = f"Summarize this: {source}"
    good = _grade(task, "Solar panels make electricity and are cheaper; battery storage helps at night and the grid.")
    bad = _grade(task, "Solar panels make electricity.")
    assert good.score > bad.score
    assert good.passed
    assert not bad.passed


def test_analysis_findings_scoring():
    task = "Review the payment terms and termination clause in this contract."
    full = _grade(task, "Findings: payment terms checked; termination clause reviewed.")
    partial = _grade(task, "Findings: payment terms checked.")
    assert full.passed
    assert not partial.passed


def test_analysis_natural_wording_passes():
    # Real models don't emit the mock finding phrases; grade topic coverage.
    task = "Review the termination and liability clauses in this contract."
    answer = (
        "Termination: 30 days' notice is short for a critical vendor. "
        "Liability is capped at 3 months of fees, which is low."
    )
    g = _grade(task, answer)
    assert g.passed


def test_qa_refusal_scores_low():
    g = _grade("Explain why the sky is blue in one sentence.", "As an AI, I cannot answer that.")
    assert not g.passed
