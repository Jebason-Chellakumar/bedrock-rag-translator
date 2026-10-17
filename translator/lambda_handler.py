"""AWS Lambda entry point: translate string tables dropped into S3.

Upload ``incoming/<name>.csv`` to the bucket and the function writes
``translated/<name>.csv`` and ``translated/<name>.report.md``. The glossary and
translation memory are read from ``config/glossary.csv`` and ``config/memory.csv``.

Environment: ``LOCALES`` (comma-separated), ``MODEL_ID``, ``EMBEDDER`` (titan | hash).
"""

from __future__ import annotations

import os
import posixpath
import tempfile
import urllib.parse
from collections.abc import Callable
from typing import Any

from .embeddings import HashingEmbedder, TitanEmbedder
from .glossary import Glossary
from .memory import TranslationMemory
from .pipeline import NEEDS_REVIEW, Pipeline
from .report import build_report
from .segments import read_strings, write_strings

_s3 = None


def _client():
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client("s3")
    return _s3


def _download(s3, bucket: str, key: str, path: str) -> bool:
    try:
        s3.download_file(bucket, key, path)
        return True
    except Exception as err:  # missing optional config files are fine
        if getattr(err, "response", {}).get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise


def handler(
    event: dict,
    context: Any = None,
    s3: Any = None,
    translator_factory: Callable[[], Any] | None = None,
) -> dict:
    s3 = s3 or _client()
    locales = [loc.strip() for loc in os.environ.get("LOCALES", "de-DE,fr-FR").split(",") if loc.strip()]
    embedder = HashingEmbedder() if os.environ.get("EMBEDDER") == "hash" else TitanEmbedder()
    if translator_factory is None:
        from .llm import ClaudeTranslator

        def translator_factory():
            return ClaudeTranslator(model=os.environ.get("MODEL_ID", "anthropic.claude-opus-5-5"))

    processed = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        if not (key.startswith("incoming/") and key.endswith(".csv")):
            continue
        name = posixpath.splitext(posixpath.basename(key))[0]
        with tempfile.TemporaryDirectory() as tmp:
            src, out, rep = (os.path.join(tmp, f) for f in ("in.csv", "out.csv", "report.md"))
            gloss, mem = os.path.join(tmp, "glossary.csv"), os.path.join(tmp, "memory.csv")
            s3.download_file(bucket, key, src)
            glossary = Glossary.load(gloss) if _download(s3, bucket, "config/glossary.csv", gloss) else Glossary([])
            memory = (
                TranslationMemory.load(mem, embedder)
                if _download(s3, bucket, "config/memory.csv", mem)
                else TranslationMemory(embedder)
            )
            translator = translator_factory()
            segments = read_strings(src)
            results = Pipeline(translator, memory, glossary).run(segments, locales)
            write_strings(out, segments, {loc: {r.id: r.target for r in rows} for loc, rows in results.items()})
            with open(rep, "w", encoding="utf-8") as fh:
                fh.write(build_report(results, getattr(translator, "usage", None)))
            s3.upload_file(out, bucket, f"translated/{name}.csv")
            s3.upload_file(rep, bucket, f"translated/{name}.report.md")
        review = sum(1 for rows in results.values() for r in rows if r.status == NEEDS_REVIEW)
        processed.append({"key": key, "segments": len(segments), "locales": locales, "needs_review": review})
    return {"processed": processed}
