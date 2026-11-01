"""Turn scanner output into semantic code chunks.

A *semantic* chunk is a real unit of Python (a function, a class, or
leftover module-level code). That is different from cutting a file
every N characters, which can split a function in the middle of a line.

This module never executes repository code. Function and class text
comes from the parser. Module-level leftovers are read as UTF-8 text
so we can keep imports and constants without repeating ``def`` / ``class``
bodies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Names used in chunk_type. Methods are not a type of their own: they
# already live inside the class chunk.
CHUNK_TYPE_FUNCTION = "function"
CHUNK_TYPE_CLASS = "class"
CHUNK_TYPE_FILE = "file"

FILE_SYMBOL_NAME = "__module__"


def chunk_repository(
    scan_result: dict[str, Any],
    max_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Build chunks from a ``scan_repository()`` result.

    Parameters
    ----------
    scan_result:
        The dictionary returned by the repository scanner.
    max_chunk_size:
        Optional maximum number of characters per chunk. ``None`` means
        "do not split". A symbol longer than this limit is split on
        **line boundaries**. A single line that is itself over the limit
        is kept whole so we never drop characters.

    Returns
    -------
    dict
        ``chunks`` plus a count. If the scan itself failed (missing
        folder), ``chunks`` is empty and ``error`` is forwarded.
    """
    if not scan_result.get("success", False):
        return {
            "success": False,
            "error": scan_result.get("error") or "Scan result is not usable",
            "repository_path": scan_result.get("repository_path"),
            "total_chunks": 0,
            "chunks": [],
        }

    repository_path = scan_result.get("repository_path")
    root = Path(repository_path) if repository_path else None
    chunks: list[dict[str, Any]] = []

    for file_record in scan_result.get("files", []):
        chunks.extend(
            _chunks_for_file(
                file_record=file_record,
                repository_root=root,
                max_chunk_size=max_chunk_size,
            )
        )

    chunks.sort(key=lambda item: (item["file_path"], item["start_line"], item["chunk_type"]))

    return {
        "success": True,
        "error": None,
        "repository_path": repository_path,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


def _chunks_for_file(
    file_record: dict[str, Any],
    repository_root: Path | None,
    max_chunk_size: int | None,
) -> list[dict[str, Any]]:
    """Create function, class, and module-level chunks for one scanned file."""
    file_path = file_record["path"]
    functions = file_record.get("functions") or []
    classes = file_record.get("classes") or []
    chunks: list[dict[str, Any]] = []

    # Top-level functions only. Class methods are already in class["source"].
    for function in functions:
        chunks.extend(
            _chunks_from_symbol(
                file_path=file_path,
                symbol_name=function["name"],
                chunk_type=CHUNK_TYPE_FUNCTION,
                start_line=function["start_line"],
                end_line=function["end_line"],
                content=function.get("source") or "",
                max_chunk_size=max_chunk_size,
            )
        )

    for cls in classes:
        chunks.extend(
            _chunks_from_symbol(
                file_path=file_path,
                symbol_name=cls["name"],
                chunk_type=CHUNK_TYPE_CLASS,
                start_line=cls["start_line"],
                end_line=cls["end_line"],
                content=cls.get("source") or "",
                max_chunk_size=max_chunk_size,
            )
        )

    source = _read_file_source(repository_root, file_path)
    if source is None:
        return chunks

    leftover_lines = _module_level_lines(source, functions, classes)
    if leftover_lines:
        first_line = leftover_lines[0][0]
        last_line = leftover_lines[-1][0]
        leftover_text = "".join(text for _, text in leftover_lines)
        chunks.extend(
            _split_numbered_lines(
                file_path=file_path,
                symbol_name=FILE_SYMBOL_NAME,
                chunk_type=CHUNK_TYPE_FILE,
                numbered_lines=leftover_lines,
                start_line=first_line,
                end_line=last_line,
                content=leftover_text,
                max_chunk_size=max_chunk_size,
            )
        )

    return chunks


def _read_file_source(repository_root: Path | None, relative_path: str) -> str | None:
    """Read the original file as UTF-8. Returns None if it cannot be read."""
    if repository_root is None:
        return None
    try:
        return (repository_root / relative_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _module_level_lines(
    source: str,
    functions: list[dict[str, Any]],
    classes: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    """Lines that are *not* part of a top-level function or class.

    We mark every line covered by a parsed symbol, then keep the rest
    (imports, constants, module docstring). Failed parses have no
    symbols, so the whole file becomes this leftover — we still keep
    the text instead of dropping it.
    """
    occupied: set[int] = set()
    for symbol in (*functions, *classes):
        start = int(symbol["start_line"])
        end = int(symbol["end_line"])
        occupied.update(range(start, end + 1))

    numbered = list(enumerate(source.splitlines(keepends=True), start=1))
    leftover = [(line_no, text) for line_no, text in numbered if line_no not in occupied]
    return _strip_blank_edges(leftover)


def _strip_blank_edges(numbered_lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop leading/trailing blank leftover lines, keep inner blanks."""
    start = 0
    end = len(numbered_lines)
    while start < end and numbered_lines[start][1].strip() == "":
        start += 1
    while end > start and numbered_lines[end - 1][1].strip() == "":
        end -= 1
    return numbered_lines[start:end]


def _chunks_from_symbol(
    file_path: str,
    symbol_name: str,
    chunk_type: str,
    start_line: int,
    end_line: int,
    content: str,
    max_chunk_size: int | None,
) -> list[dict[str, Any]]:
    """Turn one function or class into one or more chunks."""
    numbered_lines = _content_to_numbered_lines(content, start_line)
    return _split_numbered_lines(
        file_path=file_path,
        symbol_name=symbol_name,
        chunk_type=chunk_type,
        numbered_lines=numbered_lines,
        start_line=start_line,
        end_line=end_line,
        content=content,
        max_chunk_size=max_chunk_size,
    )


def _content_to_numbered_lines(content: str, start_line: int) -> list[tuple[int, str]]:
    """Pair each line of a symbol's source with its original file line number."""
    if content == "":
        return []
    lines = content.splitlines(keepends=True)
    return [(start_line + offset, line) for offset, line in enumerate(lines)]


def _split_numbered_lines(
    file_path: str,
    symbol_name: str,
    chunk_type: str,
    numbered_lines: list[tuple[int, str]],
    start_line: int,
    end_line: int,
    content: str,
    max_chunk_size: int | None,
) -> list[dict[str, Any]]:
    """Split only when needed, and only at line boundaries.

    All characters from ``content`` appear in the resulting chunks.
    We never slice a string in the middle of a line.
    """
    if not numbered_lines:
        if content == "":
            return []
        return [
            _make_chunk(
                file_path=file_path,
                symbol_name=symbol_name,
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        ]

    if max_chunk_size is None:
        joined = "".join(text for _, text in numbered_lines)
        return [
            _make_chunk(
                file_path=file_path,
                symbol_name=symbol_name,
                chunk_type=chunk_type,
                start_line=numbered_lines[0][0],
                end_line=numbered_lines[-1][0],
                content=joined,
            )
        ]

    parts: list[dict[str, Any]] = []
    bucket: list[tuple[int, str]] = []
    bucket_size = 0

    def flush() -> None:
        nonlocal bucket, bucket_size
        if not bucket:
            return
        parts.append(
            _make_chunk(
                file_path=file_path,
                symbol_name=symbol_name,
                chunk_type=chunk_type,
                start_line=bucket[0][0],
                end_line=bucket[-1][0],
                content="".join(text for _, text in bucket),
            )
        )
        bucket = []
        bucket_size = 0

    for line_no, text in numbered_lines:
        line_size = len(text)
        # A single line longer than the limit still becomes its own chunk.
        if bucket and bucket_size + line_size > max_chunk_size:
            flush()
        bucket.append((line_no, text))
        bucket_size += line_size
        if bucket_size >= max_chunk_size:
            flush()

    flush()
    return parts


def _make_chunk(
    file_path: str,
    symbol_name: str,
    chunk_type: str,
    start_line: int,
    end_line: int,
    content: str,
) -> dict[str, Any]:
    """Build the public chunk dictionary."""
    return {
        "chunk_id": f"{file_path}:{chunk_type}:{symbol_name}:{start_line}-{end_line}",
        "file_path": file_path,
        "symbol_name": symbol_name,
        "chunk_type": chunk_type,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
    }
