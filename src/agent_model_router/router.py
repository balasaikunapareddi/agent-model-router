"""The routing engine: classify → run → grade → escalate → record.

Policy (all in code, none in the prompt):
- simple tasks start on the small model, moderate on medium, complex on large.
- every answer goes through the deterministic quality gate.
- a failed grade escalates one tier up and retries, up to the large model.
- every attempt — passes and failures alike — lands in the ledger, because
  failed attempts still cost money.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .classifier import Classification, classify
from .grader import GradeResult, grade
from .ledger import Ledger, LedgerEntry, task_id_for
from .models import ModelPool, ModelResponse, default_pool

STARTING_TIER = {"simple": "small", "moderate": "medium", "complex": "large"}


@dataclass
class Attempt:
    response: ModelResponse
    grade: GradeResult
    escalated: bool


@dataclass
class RouteResult:
    task: str
    classification: Classification
    answer: str
    model_used: str
    tier_used: str
    quality_score: float
    passed: bool
    escalations: int
    total_cost_usd: float
    attempts: list[Attempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "difficulty": self.classification.difficulty,
            "task_type": self.classification.task_type,
            "model_used": self.model_used,
            "tier_used": self.tier_used,
            "quality_score": round(self.quality_score, 3),
            "passed": self.passed,
            "escalations": self.escalations,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "attempts": [
                {
                    "model": a.response.model,
                    "tier": a.response.tier,
                    "quality_score": round(a.grade.score, 3),
                    "passed": a.grade.passed,
                    "cost_usd": a.response.cost_usd,
                    "escalated": a.escalated,
                }
                for a in self.attempts
            ],
        }


class Router:
    def __init__(self, pool: ModelPool | None = None, ledger: Ledger | None = None):
        self.pool = pool or default_pool()
        self.ledger = ledger if ledger is not None else Ledger()

    def route(self, task: str) -> RouteResult:
        classification = classify(task)
        tier = STARTING_TIER[classification.difficulty]
        attempts: list[Attempt] = []
        escalated_from: str | None = None

        while True:
            model = self.pool.get(tier)
            response = model.generate(task, classification)
            result = grade(task, classification, response.text)
            escalated = escalated_from is not None
            attempts.append(Attempt(response, result, escalated))

            next_tier = self.pool.next_tier(tier)
            will_escalate = not result.passed and next_tier is not None

            self.ledger.record(
                LedgerEntry(
                    task_id=task_id_for(task),
                    task_type=classification.task_type,
                    difficulty=classification.difficulty,
                    model=response.model,
                    tier=response.tier,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    quality_score=round(result.score, 3),
                    passed=result.passed,
                    escalated=escalated,
                    escalated_from=escalated_from,
                    # Baseline is attached to the final attempt only.
                    baseline_cost_usd=(
                        0.0 if will_escalate
                        else self.pool.baseline_cost(task, response.text)
                    ),
                )
            )

            if not will_escalate:
                break
            escalated_from = tier
            tier = next_tier

        final = attempts[-1]
        return RouteResult(
            task=task,
            classification=classification,
            answer=final.response.text,
            model_used=final.response.model,
            tier_used=final.response.tier,
            quality_score=final.grade.score,
            passed=final.grade.passed,
            escalations=sum(1 for a in attempts if a.escalated),
            total_cost_usd=round(sum(a.response.cost_usd for a in attempts), 6),
            attempts=attempts,
        )
