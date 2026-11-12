"""
Row expressions for ``DataFrame.query`` and ``DataFrame.eval`` (ADR 83).

A plan filter or a data-quality rule is a predicate over column names. pandas
evaluates such a string with access to the caller's variables (``@name``) and to
every attribute of a column, so the string is held to an allowlist before pandas
sees it and is evaluated with empty variable scopes.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import pandas as pd

# Series methods a predicate may call, and the accessor it may go through.
ALLOWED_METHODS: frozenset[str] = frozenset(
    {
        "abs",
        "between",
        "contains",
        "endswith",
        "isin",
        "isna",
        "isnull",
        "len",
        "lower",
        "notna",
        "notnull",
        "round",
        "startswith",
        "strip",
        "upper",
    }
)
ALLOWED_ACCESSORS: frozenset[str] = frozenset({"str"})

MAX_EXPRESSION_LENGTH = 2000

_OPERATORS = (
    ast.And,
    ast.Or,
    ast.Not,
    ast.Invert,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Load,
)
_CONTAINERS = (ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.List, ast.Tuple)
_CONSTANT_TYPES = (str, int, float, bool, type(None))

_BACKTICK = re.compile(r"`[^`]+`")
_STRING_LITERAL = re.compile(r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*\"""")

_ALLOWED_SUMMARY = (
    "Write a comparison of column names and constants, joined by and / or / not "
    "(or & | ~), with in / not in and the methods "
    f"{', '.join(sorted(ALLOWED_METHODS))} (string methods through .str)."
)


def _reject(expression: str, what: str) -> ValueError:
    return ValueError(
        f"Row expression {expression!r} is not allowed: {what}. {_ALLOWED_SUMMARY}"
    )


def _check(node: ast.AST, expression: str, *, called: bool = False) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body, expression)
    elif isinstance(node, _OPERATORS):
        return
    elif isinstance(node, _CONTAINERS):
        for child in ast.iter_child_nodes(node):
            _check(child, expression)
    elif isinstance(node, ast.Name):
        if node.id.startswith("__"):
            raise _reject(expression, f"name '{node.id}'")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, _CONSTANT_TYPES):
            raise _reject(expression, f"constant of type {type(node.value).__name__}")
    elif isinstance(node, ast.Attribute):
        is_method = called and node.attr in ALLOWED_METHODS
        if not is_method and node.attr not in ALLOWED_ACCESSORS:
            raise _reject(expression, f"attribute '.{node.attr}'")
        _check(node.value, expression)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise _reject(expression, "a function call (only method calls on a column)")
        _check(node.func, expression, called=True)
        if node.func.attr == "contains":
            # A pattern is data, not a regex: a backtracking pattern can stall a run.
            literal = any(
                kw.arg == "regex"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if not literal or len(node.args) != 1:
                raise _reject(
                    expression,
                    "'.str.contains' without a literal match "
                    "(pass one pattern and regex=False)",
                )
        for arg in node.args:
            _check(arg, expression)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise _reject(expression, "keyword unpacking in a call")
            _check(keyword.value, expression)
    else:
        raise _reject(expression, f"{type(node).__name__}")


def validate_row_expression(expression: str) -> str:
    """
    Return ``expression`` when it is a row predicate; raise ``ValueError`` otherwise.

    Accepted: column names (backtick-quoted when not identifiers), string, numeric,
    boolean and null constants, lists and tuples of them, arithmetic, comparisons,
    ``in`` / ``not in``, ``and`` / ``or`` / ``not`` and ``& | ~``, and calls of
    ``ALLOWED_METHODS`` on a column or on its ``.str`` accessor. Everything else is
    rejected: ``@`` variable references, any other attribute, function calls,
    subscripts, lambdas, comprehensions, conditional expressions and assignments.
    """
    if not isinstance(expression, str):
        raise ValueError(
            f"Row expression must be a string, got {type(expression).__name__}."
        )
    if not expression.strip():
        raise ValueError("Row expression is empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise _reject(
            expression[:60] + "...",
            f"longer than {MAX_EXPRESSION_LENGTH} characters",
        )

    code_only = _BACKTICK.sub("", _STRING_LITERAL.sub("", expression))
    if "@" in code_only:
        raise _reject(expression, "'@' variable references")

    # A backtick-quoted column name stands for one identifier.
    parseable = _BACKTICK.sub("_column_", expression).strip()
    try:
        tree = ast.parse(parseable, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        reason = getattr(exc, "msg", None) or str(exc)
        raise _reject(expression, f"not a valid single expression ({reason})") from exc
    _check(tree, expression)
    return expression


def query_rows(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    """
    Rows of ``df`` selected by a validated row expression.

    The expression must evaluate to a boolean mask aligned with ``df.index``; a numeric
    or misaligned result raises instead of being read as row labels.
    """
    mask = eval_rows(df, expression)
    if (
        not isinstance(mask, pd.Series)
        or not pd.api.types.is_bool_dtype(mask)
        or not mask.index.equals(df.index)
    ):
        raise ValueError(
            f"Row expression {expression!r} must evaluate to a boolean value for every "
            "row (a comparison; add na=False to a .str method on a column with nulls)."
        )
    # The aligned mask goes to .loc as is, so a nullable boolean NA selects no row.
    return df.loc[mask]


def eval_rows(df: pd.DataFrame, expression: str) -> Any:
    """``df.eval`` for a validated row expression, with no access to caller variables."""
    validate_row_expression(expression)
    return df.eval(expression, local_dict={}, global_dict={})
