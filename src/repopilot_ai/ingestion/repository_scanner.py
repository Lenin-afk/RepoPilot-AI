"""Discover Python files in a local repository and parse them with ast.

This scanner only *reads* file text. It never imports or executes code
from the repository. Parsing is delegated to ``parse_python_source()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from repopilot_ai.ingestion.python_parser import parse_python_source

# Directory names we never descend into. Tooling, caches, and installs
# would add noise and are not part of the project's source.
IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "dist",
        "build",
    }
)


def scan_repository(repository_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Walk a local folder, find ``.py`` files, and parse each one.

    Parameters
    ----------
    repository_path:
        Path to the repository root on disk.

    Returns
    -------
    dict
        Structured scan result. ``success`` is ``False`` when the path is
        missing or not a directory. Individual file problems (unreadable
        or invalid syntax) do not fail the whole scan; they are counted
        in ``failed_files``.
    """
    root = Path(repository_path)

    if not root.exists():
        return _empty_result(
            repository_path=str(root),
            success=False,
            error=f"Repository path does not exist: {root}",
        )

    if not root.is_dir():
        return _empty_result(
            repository_path=str(root),
            success=False,
            error=f"Repository path is not a directory: {root}",
        )

    # resolve() turns relative paths into an absolute path, so later
    # relative_to() is always computed from one stable root.
    root = root.resolve()
    files: list[dict[str, Any]] = []

    for py_path in _discover_python_files(root):
        relative_path = py_path.relative_to(root).as_posix()
        files.append(_parse_python_file(py_path, relative_path))

    # Stable order makes tests and later UI output predictable.
    files.sort(key=lambda item: item["path"])

    successful = [item for item in files if item["success"]]
    failed = [item for item in files if not item["success"]]

    return {
        "success": True,
        "error": None,
        "repository_path": str(root),
        "total_discovered_files": len(files),
        "successfully_parsed_files": len(successful),
        "failed_files": len(failed),
        "files": files,
    }


def _discover_python_files(root: Path) -> list[Path]:
    """Recursively collect ``.py`` files, skipping ignored/hidden folders.

    ``os.walk`` lets us edit ``dirnames`` in place. Directories we remove
    here are never visited, so we do not even look at files inside
    ``.venv`` or ``.git``.
    """
    discovered: list[Path] = []

    for current_dir, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not _should_skip_directory(name)]

        for filename in filenames:
            if filename.endswith(".py"):
                discovered.append(Path(current_dir) / filename)

    return discovered


def _should_skip_directory(name: str) -> bool:
    """Return True for ignored names and any hidden directory (starts with '.')."""
    if name in IGNORED_DIRECTORY_NAMES:
        return True
    # Hidden folders such as ``.idea`` or ``.secret`` are skipped even if
    # they are not in the explicit ignore list.
    if name.startswith("."):
        return True
    return False


def _parse_python_file(absolute_path: Path, relative_path: str) -> dict[str, Any]:
    """Read one file as UTF-8 text and pass it to the AST parser.

    Reading can fail (permissions, invalid bytes). Parsing can fail
    (syntax errors). Both are reported as a failed file record, not a
    crash.
    """
    try:
        # errors is left at the default ('strict') so we notice binary
        # files that only happen to end in .py.
        source = absolute_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _file_record(
            path=relative_path,
            success=False,
            error=f"Could not read file: {exc}",
        )
    except UnicodeDecodeError as exc:
        return _file_record(
            path=relative_path,
            success=False,
            error=f"File is not valid UTF-8: {exc}",
        )

    parsed = parse_python_source(source)
    return _file_record(
        path=relative_path,
        success=parsed["success"],
        error=parsed["error"],
        functions=parsed["functions"],
        classes=parsed["classes"],
    )


def _file_record(
    path: str,
    success: bool,
    error: str | None,
    functions: list[dict[str, Any]] | None = None,
    classes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the per-file dictionary used in scan results."""
    return {
        "path": path,
        "success": success,
        "error": error,
        "functions": functions or [],
        "classes": classes or [],
    }


def _empty_result(
    repository_path: str,
    success: bool,
    error: str | None,
) -> dict[str, Any]:
    """Result used when the scan cannot start (missing/invalid root)."""
    return {
        "success": success,
        "error": error,
        "repository_path": repository_path,
        "total_discovered_files": 0,
        "successfully_parsed_files": 0,
        "failed_files": 0,
        "files": [],
    }
