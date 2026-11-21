from agent_model_router.classifier import classify
from agent_model_router.models import default_pool, estimate_tokens


def _run(tier, task):
    pool = default_pool()
    return pool.get(tier).generate(task, classify(task))


def test_small_model_easy_math_correct():
    r = _run("small", "What is 12 + 30?")
    assert "42" in r.text


def test_small_model_slips_on_large_operands():
    r = _run("small", "What is 250 * 4?")
    assert "1000" not in r.text  # deterministic slip: 1010


def test_medium_model_large_operands_correct():
    r = _run("medium", "What is 250 * 4?")
    assert "1000" in r.text


def test_large_model_multi_step_correct():
    r = _run("large", "What is 347 * 28 - 19?")
    assert "9697" in r.text


def test_cost_ordering():
    task = "What is 12 + 30?"
    small = _run("small", task).cost_usd
    medium = _run("medium", task).cost_usd
    large = _run("large", task).cost_usd
    assert small < medium < large


def test_deterministic_output():
    task = "Explain why the sky is blue in one sentence."
    assert _run("small", task).text == _run("small", task).text


def test_small_analysis_is_shallow():
    task = "Review the payment terms in this section."
    r = _run("small", task)
    assert "payment terms checked" not in r.text


def test_large_analysis_complete():
    task = "Review the payment terms and termination clause in this contract."
    r = _run("large", task)
    assert "payment terms checked" in r.text
    assert "termination clause reviewed" in r.text


def test_token_estimate_positive():
    assert estimate_tokens("hello world") >= 1
