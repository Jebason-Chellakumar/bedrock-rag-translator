"""Command line: ``rag-translate strings | document``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .embeddings import HashingEmbedder, TitanEmbedder
from .glossary import Glossary
from .memory import Entry, TranslationMemory
from .pipeline import NEEDS_REVIEW, TRANSLATED, Pipeline
from .report import build_report
from .segments import join_document, read_strings, split_document, write_strings


def _pipeline(args, translator=None) -> Pipeline:
    embedder = HashingEmbedder() if args.embedder == "hash" else TitanEmbedder(region=args.region)
    memory = TranslationMemory.load(args.memory, embedder) if args.memory else TranslationMemory(embedder)
    glossary = Glossary.load(args.glossary) if args.glossary else Glossary([])
    if translator is None and not args.dry_run:
        from .llm import ClaudeTranslator

        translator = ClaudeTranslator(model=args.model, region=args.region)
    return Pipeline(translator, memory, glossary, k=args.k, batch_size=args.batch_size)


def _dry_run(pipeline: Pipeline, segments, locales) -> None:
    for locale in locales:
        print(f"\n=== {locale} ===")
        for p in pipeline.plan(segments, locale):
            print(f"\n[{p.segment.id}] {p.segment.text}")
            if p.exact:
                print(f"  memory: exact match, reused without a model call -> {p.exact.target}")
                continue
            for t in p.terms:
                print(f"  glossary: {t.term} => {t.term if t.dnt else t.translation}")
            for m in p.matches:
                print(f"  reference ({m.score:.2f}): {m.entry.source} -> {m.entry.target}")
            if not p.terms and not p.matches:
                print("  (no glossary terms or similar approved translations)")


def _finish(args, pipeline, results) -> None:
    report = build_report(results, getattr(pipeline.translator, "usage", None))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
    print(report, file=sys.stderr)
    if args.update_memory and args.memory:
        # Only new translations that passed QA; memory hits are already stored.
        approved = [
            Entry(r.source, r.locale, r.target) for rows in results.values() for r in rows if r.status == TRANSLATED
        ]
        pipeline.memory.add(approved)
        pipeline.memory.save(args.memory)
        print(f"Translation memory updated ({len(pipeline.memory)} entries)", file=sys.stderr)


def cmd_strings(args, translator=None) -> int:
    segments = read_strings(args.input)
    locales = [loc.strip() for loc in args.locales.split(",") if loc.strip()]
    pipeline = _pipeline(args, translator)
    if args.dry_run:
        _dry_run(pipeline, segments, locales)
        return 0
    results = pipeline.run(segments, locales)
    targets = {loc: {r.id: r.target for r in rows} for loc, rows in results.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_strings(args.out, segments, targets)
    print(f"Wrote {args.out}", file=sys.stderr)
    _finish(args, pipeline, results)
    return 1 if any(r.status == NEEDS_REVIEW for rows in results.values() for r in rows) else 0


def cmd_document(args, translator=None) -> int:
    original = Path(args.input).read_text(encoding="utf-8")
    segments = split_document(original)
    pipeline = _pipeline(args, translator)
    if args.dry_run:
        _dry_run(pipeline, segments, [args.locale])
        return 0
    results = pipeline.run(segments, [args.locale])
    translated = {r.id: r.target for r in results[args.locale] if r.target}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(join_document(original, translated), encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    _finish(args, pipeline, results)
    return 1 if any(r.status == NEEDS_REVIEW for r in results[args.locale]) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-translate", description="RAG translation with Claude on Bedrock")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--glossary", help="glossary CSV (term, locale, translation, dnt)")
        p.add_argument("--memory", help="translation memory CSV (source, locale, target)")
        p.add_argument("--embedder", choices=["titan", "hash"], default="titan", help="hash = offline, no AWS")
        p.add_argument("--region", help="AWS region for Bedrock (default: from AWS config)")
        p.add_argument("--model", default="anthropic.claude-opus-5-5", help="Bedrock model ID")
        p.add_argument("--k", type=int, default=3, help="similar approved translations per segment")
        p.add_argument("--batch-size", type=int, default=25)
        p.add_argument("--report", help="write the QA report (Markdown) here")
        p.add_argument("--update-memory", action="store_true", help="add passing translations to --memory")
        p.add_argument("--dry-run", action="store_true", help="show retrieved context; no model calls")

    p = sub.add_parser("strings", help="translate a CSV string table into several locales")
    p.add_argument("input")
    p.add_argument("--locales", required=True, help="comma-separated, e.g. de-DE,fr-FR,ja-JP")
    p.add_argument("--out", default="out/translated.csv")
    common(p)
    p.set_defaults(func=cmd_strings)

    p = sub.add_parser("document", help="translate a Markdown/text document paragraph by paragraph")
    p.add_argument("input")
    p.add_argument("--locale", required=True)
    p.add_argument("--out", default="out/translated.md")
    common(p)
    p.set_defaults(func=cmd_document)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
