"""Safe arithmetic evaluator over named numeric variables.

NEVER calls Python ``eval``. Uses the ``ast`` module to walk the expression
tree and only accept a closed set of node types (BinOp, UnaryOp, constant,
parenthesized expressions, name lookup). Anything else → reject.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..stores.original_store import OriginalTable, _to_float
from .cell_extractor import ExtractedCell, ExtractedPlan


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Safe built-in functions allowed in expressions.
# Each maps a function-name string to a Python callable.
_SAFE_FUNCS = {
    "max": max,
    "min": min,
    "abs": abs,
    "int": lambda x: float(int(x)),
    "round": lambda x, n=0: float(round(x, int(n))),
    "sum": lambda *args: float(sum(args)),
}


def _safe_eval(expr: str, env: Dict[str, float]) -> float:
    """AST-based eval — no builtins, no calls, no attribute access."""
    tree = ast.parse(expr, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"non-numeric constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise KeyError(f"undefined variable {node.id!r}")
            return env[node.id]
        if isinstance(node, ast.BinOp):
            op_cls = type(node.op)
            if op_cls not in _BINOPS:
                raise ValueError(f"binop {op_cls.__name__} not allowed")
            return _BINOPS[op_cls](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op_cls = type(node.op)
            if op_cls not in _UNARYOPS:
                raise ValueError(f"unaryop {op_cls.__name__} not allowed")
            return _UNARYOPS[op_cls](walk(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"call target {type(node.func).__name__} not allowed")
            fname = node.func.id
            if fname not in _SAFE_FUNCS:
                raise ValueError(f"function {fname!r} not allowed")
            args = [walk(a) for a in node.args]
            return float(_SAFE_FUNCS[fname](*args))
        raise ValueError(f"node {type(node).__name__} not allowed")

    return walk(tree)


@dataclass
class SymbolicResult:
    ok: bool
    value: Optional[float]
    expression: str
    resolved_cells: List[Dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "value": self.value,
            "expression": self.expression,
            "resolved_cells": self.resolved_cells,
            "error": self.error,
        }


def evaluate_plan(plan: ExtractedPlan, table: OriginalTable) -> SymbolicResult:
    if not plan.parse_ok or not plan.cells or not plan.expression:
        return SymbolicResult(ok=False, value=None, expression=plan.expression,
                              error="empty_or_unparsed_plan")

    env: Dict[str, float] = {}
    resolved: List[Dict] = []
    for c in plan.cells:
        if not c.var:
            return SymbolicResult(False, None, plan.expression,
                                  resolved_cells=resolved, error="missing_var")
        hit = table.resolve(c.row_header, c.col_header)
        if hit is None:
            resolved.append({"var": c.var, "row_header": c.row_header,
                             "col_header": c.col_header, "row": None, "col": None,
                             "value": None})
            return SymbolicResult(False, None, plan.expression,
                                  resolved_cells=resolved,
                                  error=f"unresolved_cell:{c.var}")
        r, col, val = hit
        fval = _to_float(val)
        resolved.append({"var": c.var, "row_header": c.row_header,
                         "col_header": c.col_header, "row": r, "col": col,
                         "value": fval, "raw": val})
        if fval is None:
            return SymbolicResult(False, None, plan.expression,
                                  resolved_cells=resolved,
                                  error=f"non_numeric_cell:{c.var}={val!r}")
        env[c.var] = fval

    try:
        val = _safe_eval(plan.expression, env)
    except (KeyError, ValueError, ZeroDivisionError, SyntaxError) as e:
        return SymbolicResult(False, None, plan.expression,
                              resolved_cells=resolved, error=f"eval_error:{e}")
    return SymbolicResult(True, float(val), plan.expression, resolved_cells=resolved)
