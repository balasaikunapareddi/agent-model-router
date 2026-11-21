"""Model-router agent: route each task to the cheapest model that can do it.

Mock-first by design — the full pipeline (classifier, model pool, quality
gate, ledger, ADK agent loop) runs with no API key and no network. Set
GEMINI_API_KEY and pass live=True for the optional live mode.
"""

from .classifier import Classification, classify
from .grader import GradeResult, grade
from .ledger import Ledger, LedgerEntry
from .models import MockModel, ModelPool, ModelProfile, default_pool
from .router import RouteResult, Router

__all__ = [
    "Classification",
    "classify",
    "GradeResult",
    "grade",
    "Ledger",
    "LedgerEntry",
    "MockModel",
    "ModelPool",
    "ModelProfile",
    "default_pool",
    "RouteResult",
    "Router",
]

__version__ = "0.1.0"
