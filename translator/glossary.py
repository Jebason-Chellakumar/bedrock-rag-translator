"""Terminology: approved translations and do-not-translate terms.

CSV columns: ``term, locale, translation, dnt``. A ``dnt=true`` row (locale ``*``)
means the term must appear unchanged in every language (product names, brands).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .placeholders import PATTERN as PLACEHOLDER


@dataclass(frozen=True)
class Term:
    term: str
    locale: str  # "*" for do-not-translate terms
    translation: str
    dnt: bool = False

    @property
    def pattern(self) -> re.Pattern:
        # Whole words, allowing a regular English plural ("course" also matches "courses").
        return re.compile(rf"(?<!\w){re.escape(self.term)}(?:s|es)?(?!\w)", re.IGNORECASE)


class Glossary:
    def __init__(self, terms: list[Term]):
        self.terms = terms

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        with open(path, newline="", encoding="utf-8") as fh:
            return cls(
                [
                    Term(
                        term=row["term"].strip(),
                        locale=row.get("locale", "*").strip() or "*",
                        translation=(row.get("translation") or row["term"]).strip(),
                        dnt=(row.get("dnt") or "").strip().lower() in ("1", "true", "yes"),
                    )
                    for row in csv.DictReader(fh)
                    if row.get("term", "").strip()
                ]
            )

    def matches(self, source: str, locale: str) -> list[Term]:
        """Glossary entries that occur in ``source`` and apply to ``locale``.

        Longer terms first, so "safety course" is preferred over "course". Placeholders,
        markup and URLs are ignored: ``{course}`` is a variable, not the word "course".
        """
        source = PLACEHOLDER.sub(" ", source)
        hits = [t for t in self.terms if (t.dnt or t.locale.lower() == locale.lower()) and t.pattern.search(source)]
        return sorted(hits, key=lambda t: -len(t.term))

    def violations(self, source: str, target: str, locale: str) -> list[str]:
        problems = []
        lowered = target.lower()
        for t in self.matches(source, locale):
            expected = t.term if t.dnt else t.translation
            if expected.lower() not in lowered:
                kind = "do-not-translate term" if t.dnt else "glossary term"
                problems.append(f"{kind} '{t.term}' must be rendered as '{expected}'")
        return problems
