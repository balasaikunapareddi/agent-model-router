"""Deterministic difficulty classifier.

Routing decisions must be explainable and testable, so difficulty comes from
code signals — task type, length, keywords, arithmetic shape — never from a
prompt. In live mode an LLM classifier may *refine* this result, but the
deterministic baseline always runs first and mock mode uses it as-is.

Difficulties: ``simple`` / ``moderate`` / ``complex``.
Task types: ``greeting`` / ``math`` / ``summary`` / ``analysis`` / ``code`` / ``qa``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GREETING_WORDS = {
    "hi", "hello", "hey", "thanks", "thank", "bye", "goodbye",
    "morning", "afternoon", "evening",
}

COMPLEX_KEYWORDS = [
    "contract", "clause", "analyze", "analyse", "analysis", "compare",
    "evaluate", "prove", "step by step", "step-by-step", "legal",
    "risk assessment", "indemnification", "liability", "termination",
    "confidentiality", "audit", "compliance", "recommend",
]

MODERATE_KEYWORDS = [
    "summarize", "summary", "explain", "describe", "list", "draft",
    "write", "outline", "translate",
]

ANALYSIS_KEYWORDS = [
    "contract", "clause", "analyze", "analyse", "analysis", "review",
    "compare", "evaluate", "legal", "risk", "audit", "compliance",
    "indemnification", "liability", "termination", "confidentiality",
]

CODE_KEYWORDS = ["write a function", "write code", "debug", "python", "script"]

_MATH_EXPR = re.compile(r"\d+(?:\.\d+)?\s*(?:[+\-*/x×]\s*\d+(?:\.\d+)?\s*)+")
_WORD = re.compile(r"[a-z']+")


@dataclass
class Classification:
    difficulty: str  # simple | moderate | complex
    task_type: str  # greeting | math | summary | analysis | code | qa
    score: int
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "score": self.score,
            "signals": list(self.signals),
        }


def _detect_task_type(text: str, lowered: str) -> str:
    words = _WORD.findall(lowered)
    if len(words) <= 4 and any(w in GREETING_WORDS for w in words):
        return "greeting"
    if _MATH_EXPR.search(text) or "calculate" in lowered or "how much is" in lowered:
        return "math"
    if any(k in lowered for k in ("summarize", "summarise", "summary", "tldr", "tl;dr")):
        return "summary"
    if any(k in lowered for k in ANALYSIS_KEYWORDS):
        return "analysis"
    if any(k in lowered for k in CODE_KEYWORDS):
        return "code"
    return "qa"


def _math_complexity(text: str) -> tuple[int, list[str]]:
    """Points for arithmetic shape: operators and operand size."""
    match = _MATH_EXPR.search(text)
    if not match:
        return 0, []
    expr = match.group(0)
    operators = len(re.findall(r"[+\-*/x×]", expr))
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", expr)]
    points, signals = 0, []
    if operators >= 2:
        points += 2
        signals.append(f"multi-step arithmetic ({operators} operators)")
    if numbers and max(numbers) >= 100:
        points += 1
        signals.append("large operands")
    return points, signals


def classify(task: str) -> Classification:
    """Classify a task's difficulty from deterministic signals."""
    text = (task or "").strip()
    lowered = text.lower()
    task_type = _detect_task_type(text, lowered)

    if task_type == "greeting":
        return Classification("simple", task_type, 0, ["greeting"])

    score = 0
    signals: list[str] = []

    # Length: long inputs usually carry more work.
    if len(text) > 1200:
        score += 3
        signals.append("very long input (>1200 chars)")
    elif len(text) > 500:
        score += 2
        signals.append("long input (>500 chars)")
    elif len(text) > 200:
        score += 1
        signals.append("medium-length input (>200 chars)")

    # Keywords.
    for kw in COMPLEX_KEYWORDS:
        if kw in lowered:
            score += 2
            signals.append(f"complex keyword: {kw}")
    for kw in MODERATE_KEYWORDS:
        if kw in lowered:
            score += 1
            signals.append(f"moderate keyword: {kw}")

    # Arithmetic shape.
    if task_type == "math":
        pts, sigs = _math_complexity(text)
        score += pts
        signals.extend(sigs)

    # Multi-part questions.
    if text.count("?") >= 2:
        score += 1
        signals.append("multi-part question")

    if score >= 4:
        difficulty = "complex"
    elif score >= 2:
        difficulty = "moderate"
    else:
        difficulty = "simple"

    return Classification(difficulty, task_type, score, signals)
