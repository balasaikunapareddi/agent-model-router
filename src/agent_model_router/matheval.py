"""Safe arithmetic evaluation shared by the mock models and the grader.

Only +, -, *, /, parentheses and numeric constants are allowed. Anything
else raises ValueError — the grader treats an unparseable task as
ungradeable rather than guessing.
"""

from __future__ import annotations

import ast
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_EXPR = re.compile(r"[\d\.\s\+\-\*/x×\(\)]+")


def extract_expression(text: str) -> str | None:
    """Pull the arithmetic expression out of a natural-language task."""
    best: str | None = None
    for match in _EXPR.finditer(text):
        expr = match.group(0).strip().replace("x", "*").replace("×", "*")
        # Must contain at least one operator and one digit.
        if not re.search(r"\d", expr) or not re.search(r"[+\-*/]", expr):
            continue
        if best is None or len(expr) > len(best):
            best = expr
    return best


def evaluate(expr: str) -> float:
    """Evaluate an arithmetic expression string safely."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def solve_task(text: str) -> float | None:
    """Extract and evaluate the arithmetic in a task, or None if there is none."""
    expr = extract_expression(text)
    if expr is None:
        return None
    try:
        return evaluate(expr)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None


def format_number(value: float) -> str:
    """Format a result the way a model would print it (no trailing .0)."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")
