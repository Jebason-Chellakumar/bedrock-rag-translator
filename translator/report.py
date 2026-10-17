"""Markdown QA report for translators and reviewers."""

from __future__ import annotations

from collections import Counter

from .pipeline import NEEDS_REVIEW, TM_EXACT, TRANSLATED, SegmentResult


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_report(results: dict[str, list[SegmentResult]], usage=None) -> str:
    lines = [
        "# Translation QA report",
        "",
        "| locale | segments | memory reuse | translated | needs review | warnings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for locale, rows in results.items():
        c = Counter(r.status for r in rows)
        warnings = sum(1 for r in rows for i in r.issues if i.severity == "warning")
        lines.append(f"| {locale} | {len(rows)} | {c[TM_EXACT]} | {c[TRANSLATED]} | {c[NEEDS_REVIEW]} | {warnings} |")

    review = [r for rows in results.values() for r in rows if r.status == NEEDS_REVIEW]
    lines += ["", "## Needs human review", ""]
    if review:
        lines += ["| locale | key | source | issues |", "|---|---|---|---|"]
        lines += [
            f"| {r.locale} | `{r.id}` | {_cell(r.source)} | {_cell('; '.join(i.message for i in r.issues))} |"
            for r in review
        ]
    else:
        lines.append("Nothing. Every segment passed QA.")

    warned = [r for rows in results.values() for r in rows if r.status != NEEDS_REVIEW and r.issues]
    if warned:
        lines += ["", "## Warnings", "", "| locale | key | translation | warning |", "|---|---|---|---|"]
        lines += [
            f"| {r.locale} | `{r.id}` | {_cell(r.target)} | {_cell('; '.join(i.message for i in r.issues))} |"
            for r in warned
        ]

    retried = sum(1 for rows in results.values() for r in rows if r.attempts > 1)
    lines += ["", f"Segments auto-corrected after QA feedback: {retried}"]
    if usage is not None:
        lines.append(
            f"Model usage: {usage.requests} requests, {usage.input_tokens:,} input tokens "
            f"({usage.cache_read_tokens:,} from cache), {usage.output_tokens:,} output tokens"
        )
    return "\n".join(lines) + "\n"
