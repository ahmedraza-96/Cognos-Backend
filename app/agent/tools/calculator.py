"""Calculator + datetime utility tools.

`safe_eval` evaluates arithmetic via an AST allowlist (never Python `eval`), so
the LLM can't smuggle in attribute access, names, or function calls.
"""
import ast
import operator
from datetime import datetime, timezone

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def safe_eval(expression: str):
    """Safely evaluate a basic arithmetic expression. Raises ValueError on misuse."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4) ** 2'."""
    try:
        return str(safe_eval(expression))
    except ValueError as exc:
        return f"Error: {exc}"


@tool
def current_datetime() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
