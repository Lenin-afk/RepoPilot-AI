"""Parse Python source into structured dictionaries using the stdlib ast module.

Python's ``ast`` module reads source as a tree of nodes (module, class,
function, etc.) instead of raw text. That lets us list functions, classes,
parameters, and docstrings without executing the code.
"""

from __future__ import annotations

import ast
from typing import Any


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def parse_python_source(source: str) -> dict[str, Any]:
    """Parse Python source and extract top-level functions and classes.

    Parameters
    ----------
    source:
        The full text of a Python file (or snippet). Empty text is valid
        and returns empty lists.

    Returns
    -------
    dict
        A dictionary with:

        - ``success``: ``True`` if the source parsed, ``False`` on syntax errors
        - ``error``: error message when parsing fails, otherwise ``None``
        - ``functions``: top-level functions (not methods, not nested helpers)
        - ``classes``: top-level classes, each with a ``methods`` list
    """
    if source is None:
        source = ""

    # ast.parse compiles the text into a tree. It does not run the code.
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "success": False,
            "error": _format_syntax_error(exc),
            "functions": [],
            "classes": [],
        }

    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

    # Only walk the module body so nested functions stay inside their parent.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_function_to_dict(node, source))
        elif isinstance(node, ast.ClassDef):
            classes.append(_class_to_dict(node, source))

    return {
        "success": True,
        "error": None,
        "functions": functions,
        "classes": classes,
    }


def _format_syntax_error(exc: SyntaxError) -> str:
    """Turn a SyntaxError into a short, readable message."""
    location = ""
    if exc.lineno is not None:
        location = f" (line {exc.lineno}"
        if exc.offset is not None:
            location += f", column {exc.offset}"
        location += ")"
    message = exc.msg or "invalid syntax"
    return f"{message}{location}"


def _function_to_dict(node: FunctionNode, source: str) -> dict[str, Any]:
    """Convert a function (or async function) AST node into a dictionary."""
    start_line, end_line = _line_span(node)
    return {
        "name": node.name,
        "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        "parameters": _extract_parameters(node),
        # ast.get_docstring reads the first string in the body, if any.
        "docstring": ast.get_docstring(node),
        "start_line": start_line,
        "end_line": end_line,
        "source": _extract_source(source, node),
    }


def _class_to_dict(node: ast.ClassDef, source: str) -> dict[str, Any]:
    """Convert a class AST node into a dictionary, including its methods."""
    start_line, end_line = _line_span(node)
    methods: list[dict[str, Any]] = []

    # Methods are function definitions sitting directly in the class body.
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_function_to_dict(child, source))

    return {
        "name": node.name,
        "docstring": ast.get_docstring(node),
        "start_line": start_line,
        "end_line": end_line,
        "source": _extract_source(source, node),
        "methods": methods,
    }


def _extract_parameters(node: FunctionNode) -> list[str]:
    """Return parameter names in definition order.

    Special forms keep a prefix so callers can tell them apart:

    - ``*args`` stays ``*args``
    - ``**kwargs`` stays ``**kwargs``
    """
    args = node.args
    parameters: list[str] = []

    # Positional-only args exist when the signature uses ``/``.
    for arg in args.posonlyargs:
        parameters.append(arg.arg)

    for arg in args.args:
        parameters.append(arg.arg)

    if args.vararg is not None:
        parameters.append(f"*{args.vararg.arg}")

    for arg in args.kwonlyargs:
        parameters.append(arg.arg)

    if args.kwarg is not None:
        parameters.append(f"**{args.kwarg.arg}")

    return parameters


def _line_span(node: ast.AST) -> tuple[int, int]:
    """Return 1-based start and end line numbers for an AST node.

    ``lineno`` is the first line of the node (the first decorator, if any).
    ``end_lineno`` is the last line of the node. Both exist on Python 3.8+.
    """
    start = getattr(node, "lineno", 1) or 1
    end = getattr(node, "end_lineno", None) or start
    return start, end


def _extract_source(source: str, node: ast.AST) -> str:
    """Return the original source slice for this node.

    ``ast.get_source_segment`` is preferred because it matches the exact
    text from the file. If it cannot find a segment (rare), we fall back
    to slicing by line numbers.
    """
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment

    start, end = _line_span(node)
    lines = source.splitlines(keepends=True)
    if not lines:
        return ""

    # AST line numbers are 1-based; list indexes are 0-based.
    start_index = max(start - 1, 0)
    return "".join(lines[start_index:end]).rstrip("\n")
