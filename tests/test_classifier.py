from agent_model_router.classifier import classify


def test_greeting_is_simple():
    c = classify("hi")
    assert c.task_type == "greeting"
    assert c.difficulty == "simple"


def test_thanks_is_simple():
    c = classify("thanks!")
    assert c.difficulty == "simple"


def test_easy_math_is_simple():
    c = classify("What is 12 + 30?")
    assert c.task_type == "math"
    assert c.difficulty == "simple"


def test_large_operand_math_still_simple_but_scored():
    c = classify("What is 250 * 4?")
    assert c.task_type == "math"
    assert c.difficulty == "simple"
    assert c.score == 1  # large-operand signal


def test_multi_step_math_is_moderate():
    c = classify("What is 347 * 28 - 19?")
    assert c.task_type == "math"
    assert c.difficulty == "moderate"


def test_summary_is_moderate():
    text = "Summarize this: " + ("Solar panels make electricity. " * 10)
    c = classify(text)
    assert c.task_type == "summary"
    assert c.difficulty == "moderate"


def test_contract_review_is_complex():
    text = (
        "Review this contract. The termination clause, liability cap, and "
        "indemnification terms all need checking against our compliance rules."
    )
    c = classify(text)
    assert c.task_type == "analysis"
    assert c.difficulty == "complex"


def test_compare_and_recommend_is_complex():
    c = classify("Compare solar and wind for a home setup and recommend one, step by step.")
    assert c.difficulty == "complex"


def test_short_qa_is_simple():
    c = classify("Explain why the sky is blue in one sentence.")
    assert c.task_type == "qa"
    assert c.difficulty == "simple"


def test_signals_are_reported():
    c = classify("Analyze this contract clause")
    assert any("contract" in s for s in c.signals)
