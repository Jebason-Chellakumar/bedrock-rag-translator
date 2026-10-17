from pathlib import Path

import pytest

from translator.embeddings import HashingEmbedder
from translator.glossary import Glossary
from translator.llm import BatchTooLarge, TranslationRefused, Usage
from translator.memory import TranslationMemory

SAMPLE = Path(__file__).resolve().parent.parent / "sample"


class FakeTranslator:
    """Stands in for Claude: deterministic output that honours tokens and glossary terms.

    ``break_tokens`` drops placeholder tokens on a first attempt (no QA feedback yet);
    ``always_break`` drops them every time; ``max_batch`` raises BatchTooLarge above
    that size; ``refuse`` declines whole batches for those locales.
    """

    def __init__(self, break_tokens=(), always_break=(), max_batch=None, refuse=()):
        self.break_tokens, self.always_break = set(break_tokens), set(always_break)
        self.max_batch, self.refuse = max_batch, set(refuse)
        self.calls = []
        self.usage = Usage()

    def translate(self, locale, items):
        self.calls.append((locale, [i.id for i in items]))
        if locale in self.refuse:
            raise TranslationRefused("test refusal")
        if self.max_batch and len(items) > self.max_batch:
            raise BatchTooLarge("too big")
        out = {}
        for item in items:
            terms = " ".join(t if dnt else tr for t, tr, dnt in item.terms)
            text = f"«{locale}» {item.text} {terms}".strip()
            if item.id in self.always_break or (item.id in self.break_tokens and not item.feedback):
                text = text.replace("⟦P0⟧", "")
            out[item.id] = text
        return out


@pytest.fixture
def glossary():
    return Glossary.load(SAMPLE / "glossary.csv")


@pytest.fixture
def memory():
    return TranslationMemory.load(SAMPLE / "memory.csv", HashingEmbedder())
