"""The model pool: mock models with distinct cost/quality profiles.

Mock mode is the default and needs no key and no network. Each mock model is
deterministic: the same task always produces the same answer, the same token
counts, and the same simulated latency — so routing behavior is testable.

Quality profiles (the whole point of the exercise):
- ``mock-small``  — cheap and fast; fine for greetings and easy QA, slips on
  multi-step arithmetic, truncates summaries, hand-waves analysis.
- ``mock-medium`` — mid cost; solid summaries and arithmetic, partial on
  analysis.
- ``mock-large``  — expensive; correct arithmetic, full-coverage summaries,
  complete analysis findings.

Live mode (optional) maps the same three tiers onto Gemini models. It is
only constructed when explicitly requested with GEMINI_API_KEY set.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .classifier import Classification
from .matheval import format_number, solve_task

TIER_ORDER = ["small", "medium", "large"]


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (~1.33 tokens per word)."""
    return max(1, round(len(text.split()) * 1.33))


@dataclass
class ModelProfile:
    name: str
    tier: str  # small | medium | large
    input_cost_per_1k: float
    output_cost_per_1k: float
    base_latency_ms: int
    latency_per_output_token_ms: float


@dataclass
class ModelResponse:
    text: str
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float


def _cost(profile: ModelProfile, in_tokens: int, out_tokens: int) -> float:
    return round(
        (in_tokens / 1000) * profile.input_cost_per_1k
        + (out_tokens / 1000) * profile.output_cost_per_1k,
        6,
    )


def _latency(profile: ModelProfile, out_tokens: int) -> int:
    return int(profile.base_latency_ms + out_tokens * profile.latency_per_output_token_ms)


# Clause keywords an analysis task may mention, mapped to the finding a good
# answer must contain. Shared with the grader via `required_findings`.
CLAUSE_FINDINGS = {
    "termination": "termination clause reviewed",
    "liability": "liability cap identified",
    "indemnification": "indemnification obligations noted",
    "confidentiality": "confidentiality obligations noted",
    "payment": "payment terms checked",
    "renewal": "renewal terms checked",
    "governing law": "governing law identified",
    "intellectual property": "IP ownership clarified",
}


def required_findings(task: str) -> list[str]:
    lowered = task.lower()
    return [finding for kw, finding in CLAUSE_FINDINGS.items() if kw in lowered]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _source_text(task: str) -> str:
    """The document part of a summarize task (after the instruction)."""
    for sep in (":", "—", "-"):
        if sep in task:
            head, tail = task.split(sep, 1)
            if any(k in head.lower() for k in ("summar", "tldr")) and len(tail) > 40:
                return tail.strip()
    return task


class MockModel:
    """A deterministic stand-in for a real LLM with a fixed quality profile."""

    def __init__(self, profile: ModelProfile):
        self.profile = profile

    # -- per-task-type generators -------------------------------------------

    def _greeting(self, task: str) -> str:
        return "Hello! How can I help you today?"

    def _math(self, task: str) -> str:
        answer = solve_task(task)
        if answer is None:
            return "I could not parse that calculation."
        tier = self.profile.tier
        operators = len(re.findall(r"[+\-*/x×]", task))
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", task)]
        hard = operators >= 2 or (numbers and max(numbers) >= 100)
        very_hard = operators >= 3 and numbers and max(numbers) >= 100
        if tier == "small" and hard:
            # Deterministic slip: small models drop a step on multi-step math.
            answer = answer + 10
        elif tier == "medium" and very_hard:
            answer = answer - 1
        return f"The answer is {format_number(answer)}."

    def _summary(self, task: str) -> str:
        source = _source_text(task)
        sents = _sentences(source)
        if not sents:
            return "Nothing to summarize."
        tier = self.profile.tier
        if tier == "small":
            chosen = sents[:1]
        elif tier == "medium":
            chosen = sents[:3]
        else:
            # Large: greedy coverage of the source's content words.
            chosen = []
            covered: set[str] = set()
            for s in sents:
                words = set(re.findall(r"[a-z]{4,}", s.lower()))
                if words - covered or len(chosen) < 2:
                    chosen.append(s)
                    covered |= words
                if len(chosen) >= 5:
                    break
        return " ".join(chosen)

    def _analysis(self, task: str) -> str:
        findings = required_findings(task)
        tier = self.profile.tier
        if tier == "small":
            return "The document looks fairly standard overall. No major issues spotted."
        if tier == "medium":
            half = findings[: max(1, len(findings) // 2)]
            body = "; ".join(half) if half else "general structure reviewed"
            return f"Partial review: {body}."
        body = "; ".join(findings) if findings else "general structure reviewed"
        return f"Full review complete: {body}. Overall risk is manageable with the noted items addressed."

    def _code(self, task: str) -> str:
        if self.profile.tier == "small":
            return "You could write a loop for that."
        return "```python\ndef solve(items):\n    return [x for x in items if x]\n```"

    def _qa(self, task: str) -> str:
        lowered = task.lower()
        if "sky" in lowered and "blue" in lowered:
            return "The sky is blue because air molecules scatter blue sunlight more than other colors (Rayleigh scattering)."
        if "capital of france" in lowered:
            return "The capital of France is Paris."
        if self.profile.tier == "small":
            return "Short answer: it depends on the details, but the main point is covered above."
        return (
            "Here is a considered answer: the key factors are the context, the "
            "constraints, and the trade-offs involved, and weighing them points "
            "to the straightforward option."
        )

    # -- public API ----------------------------------------------------------

    def generate(self, task: str, classification: Classification) -> ModelResponse:
        generators = {
            "greeting": self._greeting,
            "math": self._math,
            "summary": self._summary,
            "analysis": self._analysis,
            "code": self._code,
            "qa": self._qa,
        }
        text = generators.get(classification.task_type, self._qa)(task)
        in_tokens = estimate_tokens(task)
        out_tokens = estimate_tokens(text)
        return ModelResponse(
            text=text,
            model=self.profile.name,
            tier=self.profile.tier,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_ms=_latency(self.profile, out_tokens),
            cost_usd=_cost(self.profile, in_tokens, out_tokens),
        )


class ModelPool:
    """Ordered pool of models by tier, cheapest first."""

    def __init__(self, models: list[MockModel]):
        self._by_tier = {m.profile.tier: m for m in models}

    def get(self, tier: str) -> MockModel:
        return self._by_tier[tier]

    def next_tier(self, tier: str) -> str | None:
        idx = TIER_ORDER.index(tier)
        return TIER_ORDER[idx + 1] if idx + 1 < len(TIER_ORDER) else None

    def largest(self) -> MockModel:
        return self._by_tier["large"]

    def baseline_cost(self, task: str, final_answer: str) -> float:
        """What the large model alone would have cost for this task."""
        large = self.largest().profile
        return _cost(large, estimate_tokens(task), estimate_tokens(final_answer))


def default_pool() -> ModelPool:
    """The mock pool: three tiers with distinct cost/quality profiles."""
    return ModelPool(
        [
            MockModel(ModelProfile("mock-small", "small", 0.10, 0.40, 120, 2.0)),
            MockModel(ModelProfile("mock-medium", "medium", 0.50, 1.50, 350, 5.0)),
            MockModel(ModelProfile("mock-large", "large", 3.00, 15.00, 900, 12.0)),
        ]
    )


# ---------------------------------------------------------------------------
# Optional live mode (Gemini). Only constructed on explicit request.
# ---------------------------------------------------------------------------

LIVE_MODELS = {
    "small": os.environ.get("ROUTER_MODEL_SMALL", "gemini-2.0-flash-lite"),
    "medium": os.environ.get("ROUTER_MODEL_MEDIUM", "gemini-2.0-flash"),
    "large": os.environ.get("ROUTER_MODEL_LARGE", "gemini-2.5-pro"),
}


class LiveModel:
    """Same interface as MockModel, backed by a real Gemini model."""

    def __init__(self, profile: ModelProfile, model_name: str):
        self.profile = profile
        self.model_name = model_name

    def generate(self, task: str, classification: Classification) -> ModelResponse:
        from google import genai  # lazy: mock mode never imports this

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; live mode unavailable")
        client = genai.Client(api_key=api_key)
        import time

        start = time.monotonic()
        result = client.models.generate_content(model=self.model_name, contents=task)
        latency_ms = int((time.monotonic() - start) * 1000)
        text = (result.text or "").strip()
        in_tokens = estimate_tokens(task)
        out_tokens = estimate_tokens(text)
        return ModelResponse(
            text=text,
            model=self.model_name,
            tier=self.profile.tier,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_ms=latency_ms,
            cost_usd=_cost(self.profile, in_tokens, out_tokens),
        )


def live_pool() -> ModelPool:
    """Pool of live Gemini models with the same tier profiles as the mocks."""
    pool = default_pool()
    models = [
        LiveModel(pool.get(tier).profile, LIVE_MODELS[tier]) for tier in TIER_ORDER
    ]
    return ModelPool(models)  # type: ignore[arg-type]
