"""Deterministic quality gate.

The router trusts a cheap answer only after this grader passes it. Grading
is code, not vibes: arithmetic is recomputed, summaries are checked for
key-term coverage, analyses for required findings. A failed grade triggers
escalation to the next model tier (see router.py).

Thresholds rise with difficulty: a greeting just has to be polite, a
contract review has to be nearly complete.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .classifier import Classification
from .matheval import solve_task
from .models import _source_text, required_findings

THRESHOLDS = {"simple": 0.6, "moderate": 0.75, "complex": 0.8}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "will",
    "are", "was", "were", "been", "into", "than", "then", "them", "they",
    "their", "there", "these", "those", "about", "after", "before",
}


@dataclass
class GradeResult:
    score: float  # 0.0 .. 1.0
    passed: bool
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 3), "passed": self.passed, "checks": self.checks}


def _key_terms(text: str, limit: int = 10) -> list[str]:
    words = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS]
    return [w for w, _ in Counter(words).most_common(limit)]


def _grade_greeting(answer: str) -> tuple[float, dict]:
    ok = bool(answer.strip()) and any(
        w in answer.lower() for w in ("hello", "hi", "help", "welcome", "hey")
    )
    return (1.0 if ok else 0.0), {"polite_nonempty": ok}


def _grade_math(task: str, answer: str) -> tuple[float, dict]:
    expected = solve_task(task)
    if expected is None:
        return 1.0, {"ungradeable": True}
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if not numbers:
        return 0.0, {"expected": expected, "found": None}
    found = float(numbers[-1])
    ok = abs(found - expected) < 1e-9
    return (1.0 if ok else 0.0), {"expected": expected, "found": found}


def _grade_summary(task: str, answer: str) -> tuple[float, dict]:
    terms = _key_terms(_source_text(task))
    if not terms:
        return 1.0, {"ungradeable": True}
    lowered = answer.lower()
    covered = [t for t in terms if t in lowered]
    score = len(covered) / len(terms)
    return score, {"key_terms": terms, "covered": covered}


def _grade_analysis(task: str, answer: str) -> tuple[float, dict]:
    findings = required_findings(task)
    if not findings:
        ok = len(answer.split()) >= 8
        return (1.0 if ok else 0.0), {"substantive": ok}
    lowered = answer.lower()
    present = [f for f in findings if f in lowered]
    score = len(present) / len(findings)
    return score, {"required": findings, "present": present}


def _grade_code(answer: str) -> tuple[float, dict]:
    ok = "def " in answer or "```" in answer
    return (1.0 if ok else 0.0), {"contains_code": ok}


def _grade_qa(answer: str) -> tuple[float, dict]:
    words = len(answer.split())
    refusal = any(p in answer.lower() for p in ("i cannot", "i can't", "as an ai"))
    score = 1.0 if words >= 5 and not refusal else (0.5 if words >= 1 else 0.0)
    return score, {"word_count": words, "refusal": refusal}


def grade(task: str, classification: Classification, answer: str) -> GradeResult:
    """Grade an answer for a task. Deterministic; no model calls."""
    graders = {
        "greeting": lambda: _grade_greeting(answer),
        "math": lambda: _grade_math(task, answer),
        "summary": lambda: _grade_summary(task, answer),
        "analysis": lambda: _grade_analysis(task, answer),
        "code": lambda: _grade_code(answer),
        "qa": lambda: _grade_qa(answer),
    }
    score, checks = graders.get(classification.task_type, lambda: _grade_qa(answer))()
    threshold = THRESHOLDS[classification.difficulty]
    return GradeResult(score=score, passed=score >= threshold, checks=checks)
