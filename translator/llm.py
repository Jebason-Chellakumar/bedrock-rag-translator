"""Claude on Amazon Bedrock as the translation engine.

One request translates a batch of segments into one locale. The request carries the
retrieved context (similar approved translations + glossary terms) and asks for
schema-validated JSON (structured outputs), so parsing never depends on the model
formatting text correctly.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "anthropic.claude-opus-5-5"  # Bedrock model ID
MAX_TOKENS = 16000

# Register and conventions per locale (the most common source of "correct but wrong" output).
LOCALE_STYLE = {
    "de-DE": "German (Germany). Formal address (Sie). Gender-neutral where natural.",
    "fr-FR": "French (France). Formal address (vous). Non-breaking space before : ; ! ?",
    "es-MX": "Mexican Spanish. Formal address (usted). Latin American vocabulary (e.g. celular, computadora).",
    "es-ES": "Castilian Spanish (Spain). Formal address (usted). Peninsular vocabulary (e.g. móvil, ordenador).",
    "it-IT": "Italian. Formal address (Lei).",
    "pt-BR": "Brazilian Portuguese. Address the reader as você.",
    "pt-PT": "European Portuguese. Formal, impersonal register.",
    "nl-NL": "Dutch (Netherlands). Polite address (u).",
    "pl-PL": "Polish. Polite, impersonal register (prosimy, proszę).",
    "sv-SE": "Swedish. Address the reader as du.",
    "da-DK": "Danish. Address the reader as du.",
    "ru-RU": "Russian. Formal address (Вы).",
    "ja-JP": "Japanese. Polite です/ます style. Full-width punctuation (、。).",
}

SYSTEM = """\
You are a senior localization translator for a workplace training product. You translate \
English UI text, notifications and course content into one target locale per request.

Rules:
1. Tokens like ⟦P0⟧ stand for placeholders, links and markup. Copy each token exactly once, \
unchanged; move it only if the target grammar requires.
2. Glossary terms are mandatory: use the given translation exactly (inflect only when the \
language requires it). Do-not-translate terms stay exactly as written.
3. Reference translations are approved earlier work. Reuse their terminology and phrasing \
when the meaning matches, so the product reads consistently.
4. Follow the locale style notes for register and conventions.
5. Translate meaning faithfully: add nothing, omit nothing, keep the tone. Keep it natural \
for a native speaker, not word-for-word.
6. If a segment has QA feedback from a previous attempt, fix exactly those problems.
Return one translation for every segment id."""

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "translation": {"type": "string"}},
                "required": ["id", "translation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


class TranslationRefused(RuntimeError):
    pass


class BatchTooLarge(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchItem:
    id: str
    text: str  # masked source
    context: str = ""
    references: tuple[tuple[str, str, float], ...] = ()  # (masked source, target, similarity)
    terms: tuple[tuple[str, str, bool], ...] = ()  # (term, translation, do-not-translate)
    feedback: str = ""


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Any) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0


def build_prompt(locale: str, items: Sequence[BatchItem]) -> str:
    style = LOCALE_STYLE.get(locale, f"{locale}. Natural, professional register.")
    terms: dict[str, tuple[str, bool]] = {}
    refs: dict[str, tuple[str, float]] = {}
    for item in items:
        terms.update({t: (tr, dnt) for t, tr, dnt in item.terms})
        for src, tgt, score in item.references:
            if src not in refs or refs[src][1] < score:
                refs[src] = (tgt, score)

    parts = [f"<target_locale>{locale}</target_locale>", f"<style>{style}</style>"]
    if terms:
        lines = [
            f"- {t} => {t} (do not translate)" if dnt else f"- {t} => {tr}" for t, (tr, dnt) in sorted(terms.items())
        ]
        parts.append("<glossary>\n" + "\n".join(lines) + "\n</glossary>")
    if refs:
        ranked = sorted(refs.items(), key=lambda kv: -kv[1][1])[:30]
        lines = [json.dumps({"source": s, "approved_translation": t}, ensure_ascii=False) for s, (t, _) in ranked]
        parts.append("<reference_translations>\n" + "\n".join(lines) + "\n</reference_translations>")
    segments = [
        {k: v for k, v in {"id": i.id, "text": i.text, "context": i.context, "qa_feedback": i.feedback}.items() if v}
        for i in items
    ]
    parts.append("<segments>\n" + json.dumps(segments, ensure_ascii=False, indent=1) + "\n</segments>")
    return "\n\n".join(parts)


class ClaudeTranslator:
    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        region: str | None = None,
        effort: str = "medium",
    ):
        if client is None:
            from anthropic import AnthropicBedrockMantle

            client = AnthropicBedrockMantle(aws_region=region)
        self.client = client
        self.model = model
        self.effort = effort
        self.usage = Usage()

    def translate(self, locale: str, items: Sequence[BatchItem]) -> dict[str, str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(locale, items)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            cache_control={"type": "ephemeral"},
        )
        self.usage.add(response.usage)
        if response.stop_reason == "refusal":
            raise TranslationRefused(f"model declined this batch ({locale})")
        if response.stop_reason == "max_tokens":
            raise BatchTooLarge(f"batch of {len(items)} exceeded max_tokens")
        text = next(b.text for b in response.content if b.type == "text")
        return {t["id"]: t["translation"] for t in json.loads(text)["translations"]}
