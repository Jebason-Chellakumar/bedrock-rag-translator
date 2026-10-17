"""Automatic quality checks on every translation.

Errors block a segment (it is retried once with the problems as feedback, then sent to
human review). Warnings are reported but don't block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .glossary import Glossary
from .placeholders import PATTERN, Masked, token_problems

ERROR, WARNING = "error", "warning"
CJK_LOCALES = ("ja", "zh", "ko")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass(frozen=True)
class Issue:
    severity: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


def check(
    source: str,
    masked: Masked,
    translated_masked: str,
    target: str,
    locale: str,
    glossary: Glossary,
) -> list[Issue]:
    if not target.strip():
        return [Issue(ERROR, "translation is empty")]

    issues = [Issue(ERROR, p) for p in token_problems(translated_masked, masked)]
    issues += [Issue(ERROR, p) for p in glossary.violations(source, target, locale)]

    plain_source, plain_target = PATTERN.sub(" ", source), PATTERN.sub(" ", target)
    for number in set(NUMBER.findall(plain_source)):
        variants = {number, number.replace(",", "."), number.replace(".", ",")}
        if not any(v in plain_target for v in variants):
            issues.append(Issue(WARNING, f"number {number} not found in translation"))

    words = re.findall(r"\w+", plain_source)
    if len(words) > 3 and target.strip() == source.strip():
        issues.append(Issue(WARNING, "translation is identical to the source"))

    ratio = len(plain_target.strip()) / max(len(plain_source.strip()), 1)
    low = 0.15 if locale.lower().startswith(CJK_LOCALES) else 0.4
    if len(plain_source.strip()) >= 20 and not low <= ratio <= 2.5:
        issues.append(Issue(WARNING, f"unusual length ratio {ratio:.2f} (target/source)"))
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == ERROR for i in issues)
