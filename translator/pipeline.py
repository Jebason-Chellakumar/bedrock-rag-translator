"""Retrieval-augmented translation pipeline.

For each target locale:

1. **Exact memory hits** reuse the approved translation: no model call, no cost.
2. **Retrieve** the most similar approved translations and the glossary terms
   that occur in each remaining segment.
3. **Translate** in batches with Claude, placeholders masked as tokens.
4. **QA** every result. Segments with errors are retried once with the problems as
   feedback; anything still failing is marked ``needs_review`` for a human.

Locales run in parallel.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import qa
from .glossary import Glossary
from .llm import BatchItem, BatchTooLarge, TranslationRefused
from .memory import Entry, Match, TranslationMemory
from .placeholders import Masked, mask, unmask
from .segments import Segment

TM_EXACT, TRANSLATED, NEEDS_REVIEW = "tm_exact", "translated", "needs_review"


@dataclass
class SegmentResult:
    id: str
    locale: str
    source: str
    target: str
    status: str
    issues: list[qa.Issue] = field(default_factory=list)
    references: int = 0
    attempts: int = 0


@dataclass
class Plan:
    """What the pipeline would send for one segment (``--dry-run``)."""

    segment: Segment
    exact: Entry | None
    matches: list[Match]
    terms: list


class Pipeline:
    def __init__(
        self,
        translator,
        memory: TranslationMemory,
        glossary: Glossary,
        k: int = 3,
        batch_size: int = 25,
        max_workers: int = 4,
    ):
        self.translator = translator
        self.memory = memory
        self.glossary = glossary
        self.k = k
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._failures: dict[tuple, str] = {}  # (locale, segment id) -> why no translation came back

    # ------------------------------------------------------------ public API

    def run(self, segments: Sequence[Segment], locales: Sequence[str]) -> dict[str, list[SegmentResult]]:
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(locales)) or 1) as pool:
            results = pool.map(lambda loc: self._run_locale(segments, loc), locales)
            return dict(zip(locales, results))

    def plan(self, segments: Sequence[Segment], locale: str) -> list[Plan]:
        matches = self.memory.search([s.text for s in segments], locale, self.k)
        return [
            Plan(s, self.memory.exact(s.text, locale), m, self.glossary.matches(s.text, locale))
            for s, m in zip(segments, matches)
        ]

    # ------------------------------------------------------------ per locale

    def _run_locale(self, segments: Sequence[Segment], locale: str) -> list[SegmentResult]:
        results: dict[str, SegmentResult] = {}
        todo: list[Segment] = []
        for seg in segments:
            hit = self.memory.exact(seg.text, locale)
            if hit is not None:
                results[seg.id] = SegmentResult(seg.id, locale, seg.text, hit.target, TM_EXACT, references=1)
            else:
                todo.append(seg)

        by_id = {s.id: s for s in todo}
        masks = {s.id: mask(s.text) for s in todo}
        items = self._items(todo, locale, masks)
        feedback: dict[str, str] = {}
        for attempt in (1, 2):
            if not items:
                break
            outputs = self._translate(locale, items)
            retry = []
            for item in items:
                seg = by_id[item.id]
                result = self._evaluate(seg, locale, masks[seg.id], outputs.get(seg.id), item, attempt)
                results[seg.id] = result
                if result.status == NEEDS_REVIEW and attempt == 1 and seg.id in outputs:
                    feedback[seg.id] = "; ".join(i.message for i in result.issues if i.severity == qa.ERROR)
                    retry.append(_with_feedback(item, feedback[seg.id]))
            items = retry
        return [results[s.id] for s in segments]

    def _items(self, segments: Sequence[Segment], locale: str, masks: dict[str, Masked]) -> list[BatchItem]:
        matches = self.memory.search([s.text for s in segments], locale, self.k)
        items = []
        for seg, found in zip(segments, matches):
            items.append(
                BatchItem(
                    id=seg.id,
                    text=masks[seg.id].text,
                    context=seg.context,
                    references=tuple((mask(m.entry.source).text, m.entry.target, m.score) for m in found),
                    terms=tuple((t.term, t.translation, t.dnt) for t in self.glossary.matches(seg.text, locale)),
                )
            )
        return items

    def _evaluate(self, seg, locale, masked, output, item, attempt) -> SegmentResult:
        if output is None:
            reason = self._failures.get((locale, seg.id), "no translation returned")
            return SegmentResult(
                seg.id, locale, seg.text, "", NEEDS_REVIEW, [qa.Issue(qa.ERROR, reason)], len(item.references), attempt
            )
        target = unmask(output, masked)
        issues = qa.check(seg.text, masked, output, target, locale, self.glossary)
        status = NEEDS_REVIEW if qa.has_errors(issues) else TRANSLATED
        return SegmentResult(seg.id, locale, seg.text, target, status, issues, len(item.references), attempt)

    # ------------------------------------------------------------ model calls

    def _translate(self, locale: str, items: list[BatchItem]) -> dict[str, str]:
        out: dict[str, str] = {}
        for start in range(0, len(items), self.batch_size):
            out.update(self._call(locale, items[start : start + self.batch_size]))
        return out

    def _call(self, locale: str, batch: list[BatchItem]) -> dict[str, str]:
        try:
            result = self.translator.translate(locale, batch)
        except BatchTooLarge:
            if len(batch) == 1:
                self._fail(locale, batch, "translation exceeded the output limit")
                return {}
            mid = len(batch) // 2  # split and retry each half
            return {**self._call(locale, batch[:mid]), **self._call(locale, batch[mid:])}
        except TranslationRefused as err:
            self._fail(locale, batch, f"model declined: {err}")
            return {}
        wanted = {i.id for i in batch}
        return {k: v for k, v in result.items() if k in wanted}

    def _fail(self, locale: str, batch: list[BatchItem], reason: str) -> None:
        for item in batch:
            self._failures[(locale, item.id)] = reason


def _with_feedback(item: BatchItem, feedback: str) -> BatchItem:
    return BatchItem(item.id, item.text, item.context, item.references, item.terms, feedback)
