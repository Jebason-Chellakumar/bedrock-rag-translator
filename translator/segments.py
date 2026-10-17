"""Reading and writing translatable content: string tables (CSV) and documents."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

KEY_COLUMNS = ("key", "id")
SOURCE_COLUMNS = ("source", "en", "en-US", "English")


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    context: str = ""  # where it appears (button, email subject...), helps disambiguate


def read_strings(path: str | Path) -> list[Segment]:
    """CSV string table with a key column, an English source column and optional context."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        key = next((c for c in cols if c in KEY_COLUMNS), None)
        src = next((c for c in cols if c in SOURCE_COLUMNS), None)
        if key is None or src is None:
            raise ValueError(f"need a key column {KEY_COLUMNS} and a source column {SOURCE_COLUMNS}; got {cols}")
        return [
            Segment(row[key].strip(), row[src], (row.get("context") or "").strip())
            for row in reader
            if row[key].strip() and row[src].strip()
        ]


def write_strings(path: str | Path, segments: Sequence[Segment], targets: dict[str, dict[str, str]]) -> None:
    """Wide table: key, source, one column per locale. ``targets[locale][key] = text``."""
    locales = list(targets)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "source", *locales])
        for s in segments:
            writer.writerow([s.id, s.text, *(targets[loc].get(s.id, "") for loc in locales)])


def split_document(text: str) -> list[Segment]:
    """Split Markdown/plain text into paragraph segments; fenced code blocks are skipped."""
    segments, in_code = [], False
    blocks = re.split(r"\n\s*\n", text.strip())
    for i, block in enumerate(blocks, 1):
        fences = block.count("```")
        if in_code or fences:  # inside or touching a fenced code block: never translate
            if fences % 2:
                in_code = not in_code
            continue
        if block.strip():
            context = "heading" if block.lstrip().startswith("#") else "paragraph"
            segments.append(Segment(f"p{i}", block.strip(), context))
    return segments


def join_document(original: str, translated: dict[str, str]) -> str:
    """Rebuild a document, keeping untranslated blocks (e.g. code) as they were."""
    blocks = re.split(r"\n\s*\n", original.strip())
    return "\n\n".join(translated.get(f"p{i}", block.strip()) for i, block in enumerate(blocks, 1)) + "\n"
