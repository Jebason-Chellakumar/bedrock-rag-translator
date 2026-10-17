"""Protect placeholders, markup and URLs from translation.

Before a segment goes to the model, every ``{{company_url}}``, ``{name}``, ``%s``,
``<b>`` or URL is swapped for an opaque token (``⟦P0⟧``). The model only ever sees
tokens, so it cannot translate, reorder the inside of, or "fix" them; after
translation the tokens are restored. QA then checks every token came back exactly once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERN = re.compile(
    r"\{\{\s*[\w.]+\s*\}\}"  # {{company_url}}
    r"|\{[\w.]+\}"  # {name}
    r"|%\(\w+\)[sdif]"  # %(count)d
    r"|%[sdif]"  # %s
    r"|</?[A-Za-z][^<>]*>"  # <b>, </a>, <br/>
    r"|https?://[^\s<>\"']+[^\s<>\"'.,;:!?)]"  # URLs (without trailing punctuation)
)
TOKEN = re.compile(r"⟦P(\d+)⟧")


@dataclass(frozen=True)
class Masked:
    text: str
    originals: dict[str, str]  # token -> original placeholder

    @property
    def tokens(self) -> list[str]:
        return list(self.originals)


def mask(text: str) -> Masked:
    originals: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        token = f"⟦P{len(originals)}⟧"
        originals[token] = m.group(0)
        return token

    return Masked(PATTERN.sub(repl, text), originals)


def unmask(text: str, masked: Masked) -> str:
    return TOKEN.sub(lambda m: masked.originals.get(m.group(0), m.group(0)), text)


def token_problems(translated: str, masked: Masked) -> list[str]:
    """Missing, duplicated or invented tokens in a translation (before unmasking)."""
    found = TOKEN.findall(translated)
    problems = []
    for token, original in masked.originals.items():
        n = found.count(TOKEN.match(token).group(1))
        if n == 0:
            problems.append(f"placeholder {original} is missing")
        elif n > 1:
            problems.append(f"placeholder {original} appears {n} times")
    for idx in set(found):
        if f"⟦P{idx}⟧" not in masked.originals:
            problems.append(f"unknown placeholder token ⟦P{idx}⟧")
    return problems
