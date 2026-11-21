"""Append-only cost/quality ledger (JSONL).

Every model attempt — including failed ones that triggered an escalation —
is recorded: route, model, tokens, latency, estimated cost, quality score,
and whether the attempt was an escalation. The report compares actual spend
against an always-big-model baseline so the savings are a number, not a
feeling.

Run ``python -m agent_model_router.ledger <ledger.jsonl>`` (or the
``model-router-report`` script) to print a report for a ledger file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def task_id_for(task: str) -> str:
    return hashlib.sha256(task.strip().encode()).hexdigest()[:12]


@dataclass
class LedgerEntry:
    task_id: str
    task_type: str
    difficulty: str
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    quality_score: float
    passed: bool
    escalated: bool = False
    escalated_from: str | None = None
    baseline_cost_usd: float = 0.0  # large-model-only cost for the final answer
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)


class Ledger:
    """JSONL-backed ledger. ``path=None`` keeps entries in memory only."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._entries: list[LedgerEntry] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._entries.append(LedgerEntry(**json.loads(line)))

    def record(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def report(self) -> dict:
        """Summarize spend vs the always-big-model baseline."""
        entries = self._entries
        if not entries:
            return {
                "tasks": 0, "attempts": 0, "total_cost_usd": 0.0,
                "baseline_cost_usd": 0.0, "savings_usd": 0.0, "savings_pct": 0.0,
                "escalations": 0, "by_model": {}, "avg_final_quality": 0.0,
            }
        total_cost = round(sum(e.cost_usd for e in entries), 6)
        # Baseline counts each task once (its final, accepted attempt).
        finals: dict[str, LedgerEntry] = {}
        for e in entries:
            finals[e.task_id] = e
        baseline = round(sum(e.baseline_cost_usd for e in finals.values()), 6)
        savings = round(baseline - total_cost, 6)
        savings_pct = round(100 * savings / baseline, 1) if baseline else 0.0
        by_model: dict[str, int] = {}
        for e in entries:
            by_model[e.model] = by_model.get(e.model, 0) + 1
        avg_quality = round(
            sum(e.quality_score for e in finals.values()) / len(finals), 3
        )
        return {
            "tasks": len(finals),
            "attempts": len(entries),
            "total_cost_usd": total_cost,
            "baseline_cost_usd": baseline,
            "savings_usd": savings,
            "savings_pct": savings_pct,
            "escalations": sum(1 for e in entries if e.escalated),
            "by_model": by_model,
            "avg_final_quality": avg_quality,
        }


def format_report(report: dict) -> str:
    lines = [
        f"Tasks routed:        {report['tasks']} ({report['attempts']} model attempts)",
        f"Escalations:         {report['escalations']}",
        f"Actual spend:        ${report['total_cost_usd']:.4f}",
        f"Always-big baseline: ${report['baseline_cost_usd']:.4f}",
        f"Savings:             ${report['savings_usd']:.4f} ({report['savings_pct']}%)",
        f"Avg final quality:   {report['avg_final_quality']}",
        f"Attempts by model:   {report['by_model']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: model-router-report <ledger.jsonl>", file=sys.stderr)
        return 2
    ledger = Ledger(argv[0])
    print(format_report(ledger.report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
